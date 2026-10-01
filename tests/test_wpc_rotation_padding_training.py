from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import tools.finetune_wpc_rotation_padding as finetune_runner
from orion.experimental.wpc_cips_baseline import (
    CIPSConvCase,
    rotation_padded_reference,
)
from orion.experimental.wpc_rotation_padding_training import (
    WPCRotationPaddingConv2d,
    convert_model_to_wpc_rotation_padding,
    rotation_padding_conv2d,
    rotation_padding_module_names,
)
from orion.models.unet import UNet22PlusOutput
from tools.finetune_wpc_rotation_padding import (
    _NonFiniteEpochError,
    _build_result,
    _evaluate,
    _parser,
    _run_epoch_with_backoff,
    _should_select_candidate,
    _training_audits_complete,
    _validate_resume_checkpoint,
)


def test_rotation_padding_conv_matches_independent_numpy_oracle() -> None:
    generator = torch.Generator().manual_seed(20260930)
    value = torch.randn((2, 3, 4, 8), generator=generator, dtype=torch.float64)
    weight = torch.randn((4, 3, 3, 3), generator=generator, dtype=torch.float64)
    bias = torch.randn((4,), generator=generator, dtype=torch.float64)

    actual = rotation_padding_conv2d(value, weight, bias, padding=1)
    case = CIPSConvCase(
        slots=128,
        input_channels=3,
        output_channels=4,
        height=4,
        width=8,
        kernel_height=3,
        kernel_width=3,
        pad_height_before=1,
        pad_width_before=1,
    )
    expected = np.stack(
        [
            rotation_padded_reference(
                sample.detach().numpy(),
                weight.detach().numpy(),
                case,
            )
            + bias.detach().numpy()[:, None, None]
            for sample in value
        ]
    )
    assert np.allclose(actual.detach().numpy(), expected, rtol=0.0, atol=1e-12)


def test_rotation_padding_is_not_zero_padding_at_boundaries() -> None:
    value = torch.arange(1, 13, dtype=torch.float64).reshape(1, 1, 3, 4)
    weight = torch.ones((1, 1, 3, 3), dtype=torch.float64)
    rotation = rotation_padding_conv2d(value, weight, padding=1)
    zero = F.conv2d(value, weight, padding=1)
    assert not torch.equal(rotation, zero)
    assert float((rotation - zero).abs().max()) > 0.0


def test_rotation_padding_conv_is_differentiable() -> None:
    layer = WPCRotationPaddingConv2d(2, 3, 3, padding=1, bias=True)
    value = torch.randn((2, 2, 4, 4), requires_grad=True)
    loss = layer(value).square().mean()
    loss.backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None
    assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()


def test_unet_conversion_preserves_checkpoint_keys_and_weights() -> None:
    torch.manual_seed(7)
    model = UNet22PlusOutput(
        in_channels=1,
        out_channels=1,
        base_channels=4,
        activation="silu",
        silu_degree=7,
    )
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    conversions = convert_model_to_wpc_rotation_padding(model)
    after = model.state_dict()

    assert len(conversions) == 18
    assert len(rotation_padding_module_names(model)) == 18
    assert set(after) == set(before)
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert not isinstance(model.output, WPCRotationPaddingConv2d)
    assert tuple(model.output.kernel_size) == (1, 1)


def test_conversion_rejects_unsupported_stride_or_groups() -> None:
    with pytest.raises(ValueError, match="stride-one"):
        WPCRotationPaddingConv2d.from_conv2d(
            nn.Conv2d(2, 2, 3, stride=2, padding=1)
        )
    with pytest.raises(ValueError, match="groups=1"):
        WPCRotationPaddingConv2d.from_conv2d(
            nn.Conv2d(2, 2, 3, padding=1, groups=2)
        )


class _SelectiveNonfiniteModel(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = value[:, :1].clone()
        nonfinite = value[:, 0, 0, 0] > 0.5
        result[nonfinite] = float("nan")
        return result


def test_evaluate_accounts_for_nonfinite_samples_without_emitting_nan() -> None:
    images = torch.stack(
        [
            torch.zeros((1, 2, 2), dtype=torch.float32),
            torch.ones((1, 2, 2), dtype=torch.float32),
        ]
    )
    masks = torch.zeros_like(images)
    loader = DataLoader(TensorDataset(images, masks), batch_size=2, shuffle=False)

    metrics = _evaluate(
        _SelectiveNonfiniteModel(),
        loader,
        device=torch.device("cpu"),
        native_reference=nn.Identity(),
    )

    assert metrics["sample_count"] == 2
    assert metrics["finite_sample_count"] == 1
    assert metrics["nonfinite_sample_count"] == 1
    assert metrics["nonfinite_sample_indices"] == [1]
    assert metrics["fully_finite"] is False
    assert metrics["metric_scope"] == "finite_samples_only"
    assert metrics["comparison_sample_count"] == 1
    assert metrics["native_reference_nonfinite_sample_count"] == 0
    json.dumps(metrics, allow_nan=False)


def _validation_metrics(
    *,
    fully_finite: bool,
    dice: float,
    nonfinite_count: int = 0,
) -> dict[str, object]:
    finite_count = 2 - int(nonfinite_count)
    return {
        "loss": 0.25,
        "dice": float(dice),
        "iou": 0.8,
        "sample_count": 2,
        "finite_sample_count": finite_count,
        "nonfinite_sample_count": int(nonfinite_count),
        "nonfinite_sample_indices": [1] if nonfinite_count else [],
        "fully_finite": bool(fully_finite),
        "metric_scope": "all_samples" if fully_finite else "finite_samples_only",
        "logits_mae_vs_native": 1.0,
        "max_abs_logit_delta_vs_native": 2.0,
        "prob_mae_vs_native": 0.01,
        "prediction_flip_rate_vs_native": 0.02,
        "comparison_sample_count": finite_count,
        "native_reference_nonfinite_sample_count": 0,
        "native_reference_nonfinite_sample_indices": [],
    }


def test_result_accepts_finite_recovery_from_nonfinite_baseline(
    tmp_path: Path,
) -> None:
    best_path = tmp_path / "rotation_padding_best.pt"
    last_path = tmp_path / "rotation_padding_last.pt"
    best_path.write_bytes(b"best")
    last_path.write_bytes(b"last")
    (tmp_path / "rotation_padding_epoch_0000.pt").write_bytes(b"epoch zero")
    (tmp_path / "rotation_padding_epoch_0001.pt").write_bytes(b"epoch one")
    training_audit = _validation_metrics(fully_finite=True, dice=0.90)
    training_audit.update(
        {
            "sample_count": 2048,
            "finite_sample_count": 2048,
            "comparison_sample_count": 2048,
        }
    )
    args = argparse.Namespace(
        dataset="covid19",
        image_size=256,
        epochs=1,
        batch_size=1,
        lr=1.0e-6,
        resume_lr=None,
        max_epoch_retries=4,
        lr_backoff_factor=0.25,
        min_lr=1.0e-10,
        weight_decay=1.0e-4,
        grad_clip_norm=1.0,
        distill_weight=0.001,
        train_limit=2048,
        val_limit=512,
        num_workers=2,
        seed=0,
        device="cuda",
        resume_if_present=True,
        eval_only=False,
        result=tmp_path / "result.json",
    )

    result = _build_result(
        args=args,
        source_checkpoint=tmp_path / "source.pt",
        source_sha256="a" * 64,
        data_path=tmp_path / "covid19radio_512.npz",
        train_count=2048,
        val_count=2,
        conversions=[{"name": str(index)} for index in range(18)],
        conversion_preserved_state=True,
        activations_fully_polynomial=True,
        native_metrics=_validation_metrics(fully_finite=True, dice=0.91),
        pre_metrics=_validation_metrics(
            fully_finite=False,
            dice=0.80,
            nonfinite_count=1,
        ),
        best_metrics=_validation_metrics(fully_finite=True, dice=0.90),
        best_epoch=1,
        completed_epoch=1,
        history=[
            {
                "epoch": 1,
                "train": {"sample_count": 2048},
                "post_epoch_training_audit": training_audit,
            }
        ],
        best_path=best_path,
        last_path=last_path,
        checkpoint_reload_compatible=True,
        started=0.0,
    )

    assert result["schema_version"] == 4
    assert result["status"] == "ok"
    assert result["acceptance"]["valid"] is True
    assert result["metrics"]["numerical_stability_recovered_by_finetuning"] is True
    assert result["metrics"]["native_to_unfinetuned_rotation"]["comparable"] is False
    assert result["metrics"]["native_to_unfinetuned_rotation"]["dice_delta"] is None
    json.dumps(result, allow_nan=False)


def test_stable_server_protocol_defaults() -> None:
    args = _parser().parse_args([])
    assert args.batch_size == 1
    assert args.lr == 1.0e-6
    assert args.resume_lr is None
    assert args.max_epoch_retries == 4
    assert args.lr_backoff_factor == 0.25
    assert args.min_lr == 1.0e-10
    assert args.distill_weight == 0.001
    assert args.train_limit == 2048
    assert args.val_limit == 512
    assert args.seed == 0


def test_finite_resume_candidate_replaces_legacy_nonfinite_best() -> None:
    candidate = _validation_metrics(fully_finite=True, dice=0.90)
    legacy_best = {
        "loss": float("nan"),
        "dice": 0.95,
        "iou": 0.90,
        "sample_count": 2,
    }
    assert _should_select_candidate(candidate, legacy_best) is True

    better_finite_best = _validation_metrics(fully_finite=True, dice=0.91)
    assert _should_select_candidate(candidate, better_finite_best) is False


def test_epoch_retry_restores_safe_checkpoint_and_backs_off_lr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    native = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    safe_weight = model.weight.detach().clone()
    safe_path = tmp_path / "rotation_padding_last.pt"
    torch.save(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        safe_path,
    )
    calls = 0

    def fake_train_epoch(*args: object, **kwargs: object) -> dict[str, float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            with torch.no_grad():
                model.weight.fill_(99.0)
            raise _NonFiniteEpochError("synthetic non-finite batch")
        assert torch.equal(model.weight.detach(), safe_weight)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.25)
        return {"loss": 0.2, "segmentation_loss": 0.2, "distillation_loss": 0.0}

    monkeypatch.setattr(finetune_runner, "_train_epoch", fake_train_epoch)
    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        metrics = _validation_metrics(fully_finite=True, dice=0.9)
        if kwargs.get("native_reference") is None:
            metrics.update(
                {
                    "sample_count": 1,
                    "finite_sample_count": 1,
                    "nonfinite_sample_count": 0,
                }
            )
        return metrics

    monkeypatch.setattr(finetune_runner, "_evaluate", fake_evaluate)
    dataset = TensorDataset(torch.zeros((1, 1)), torch.zeros((1, 1)))
    validation_loader = DataLoader(dataset, batch_size=1)

    (
        train_metrics,
        training_audit,
        validation_metrics,
        accounting,
    ) = _run_epoch_with_backoff(
        model=model,
        native_teacher=native,
        train_set=dataset,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        epoch=2,
        batch_size=1,
        num_workers=0,
        seed=0,
        grad_clip_norm=1.0,
        distill_weight=0.001,
        safe_checkpoint_path=safe_path,
        max_epoch_retries=2,
        lr_backoff_factor=0.25,
        min_lr=1.0e-10,
    )

    assert calls == 2
    assert train_metrics["loss"] == 0.2
    assert training_audit["fully_finite"] is True
    assert validation_metrics["fully_finite"] is True
    assert accounting["attempt_count"] == 2
    assert accounting["effective_lr"] == pytest.approx(0.25)
    assert accounting["shuffle_seed"] == 1
    assert len(accounting["failed_attempts"]) == 1


def test_post_epoch_training_audit_failure_rolls_back_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    native = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    safe_path = tmp_path / "rotation_padding_last.pt"
    torch.save(
        {
            "epoch": 0,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        safe_path,
    )
    monkeypatch.setattr(
        finetune_runner,
        "_train_epoch",
        lambda *args, **kwargs: {
            "loss": 0.2,
            "segmentation_loss": 0.2,
            "distillation_loss": 0.0,
            "sample_count": 1,
        },
    )
    evaluation_count = 0

    def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal evaluation_count
        evaluation_count += 1
        if evaluation_count == 1:
            metrics = _validation_metrics(
                fully_finite=False,
                dice=0.0,
                nonfinite_count=1,
            )
            metrics.update({"sample_count": 1, "finite_sample_count": 0})
            return metrics
        metrics = _validation_metrics(fully_finite=True, dice=0.9)
        if kwargs.get("native_reference") is None:
            metrics.update(
                {
                    "sample_count": 1,
                    "finite_sample_count": 1,
                    "nonfinite_sample_count": 0,
                }
            )
        return metrics

    monkeypatch.setattr(finetune_runner, "_evaluate", fake_evaluate)
    dataset = TensorDataset(torch.zeros((1, 1)), torch.zeros((1, 1)))
    validation_loader = DataLoader(dataset, batch_size=1)

    _, training_audit, validation_metrics, accounting = _run_epoch_with_backoff(
        model=model,
        native_teacher=native,
        train_set=dataset,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        epoch=1,
        batch_size=1,
        num_workers=0,
        seed=0,
        grad_clip_norm=1.0,
        distill_weight=0.001,
        safe_checkpoint_path=safe_path,
        max_epoch_retries=1,
        lr_backoff_factor=0.25,
        min_lr=1.0e-10,
    )

    assert training_audit["fully_finite"] is True
    assert validation_metrics["fully_finite"] is True
    assert accounting["attempt_count"] == 2
    assert accounting["effective_lr"] == pytest.approx(0.25)
    assert "training-set audit" in accounting["failed_attempts"][0]["error"]


def test_legacy_resume_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "rotation_padding_last.pt"
    with pytest.raises(RuntimeError, match="predates mandatory"):
        _validate_resume_checkpoint(
            {"schema_version": 3, "epoch": 2, "history": []},
            checkpoint_path=checkpoint_path,
            train_count=2048,
        )


def test_audited_resume_checkpoint_requires_immutable_evidence(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "rotation_padding_last.pt"
    training_audit = _validation_metrics(fully_finite=True, dice=0.9)
    training_audit.update(
        {
            "sample_count": 2048,
            "finite_sample_count": 2048,
        }
    )
    checkpoint = {
        "schema_version": 4,
        "epoch": 1,
        "history": [
            {
                "epoch": 1,
                "post_epoch_training_audit": training_audit,
            }
        ],
    }
    assert _training_audits_complete(
        checkpoint["history"],
        completed_epoch=1,
        train_count=2048,
    )
    with pytest.raises(RuntimeError, match="missing immutable epoch evidence"):
        _validate_resume_checkpoint(
            checkpoint,
            checkpoint_path=checkpoint_path,
            train_count=2048,
        )

    (tmp_path / "rotation_padding_epoch_0000.pt").write_bytes(b"epoch zero")
    (tmp_path / "rotation_padding_epoch_0001.pt").write_bytes(b"epoch one")
    _validate_resume_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        train_count=2048,
    )
