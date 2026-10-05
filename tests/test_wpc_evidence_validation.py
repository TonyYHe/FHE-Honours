from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from orion.experimental.wpc_evidence_validation import (
    EvidenceValidationError, finite, load_checkpoint_bytes,
)
from tools.finetune_wpc_rotation_padding import _validate_evaluated_checkpoint


@pytest.mark.parametrize("value", [True, "2.0", None, float("nan"), float("inf"), -1])
def test_measurements_reject_non_numeric_nonfinite_and_negative_values(value):
    with pytest.raises(EvidenceValidationError):
        finite(value, name="measurement", minimum=0)


def test_hash_identifies_deserialized_bytes_even_if_path_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"weight": torch.tensor([1.])}}, path)
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    real_load = torch.load

    def replace_before_deserialize(handle, **kwargs):
        torch.save({"state_dict": {"weight": torch.tensor([2.])}}, path)
        return real_load(handle, **kwargs)

    monkeypatch.setattr(torch, "load", replace_before_deserialize)
    payload, digest = load_checkpoint_bytes(path)
    assert digest == original_hash
    assert digest != hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["state_dict"]["weight"].item() == 1


def _audited_checkpoint(path: Path):
    for epoch in range(2):
        (path.parent / f"rotation_padding_epoch_{epoch:04d}.pt").write_bytes(b"evidence")
    state = {}
    for index in range(18):
        state[f"act{index}.coeffs"] = torch.ones(8)
        state[f"act{index}.postscale_tensor"] = torch.tensor(1.)
        state[f"act{index}.prescale_tensor"] = torch.tensor(1.)
    return {
        "schema_version": 4, "epoch": 1, "source_checkpoint_sha256": "a" * 64,
        "model": {"padding_semantics": "wpc_flattened_spatial_rotation_padding"},
        "state_dict": state,
        "history": [{"epoch": 1, "post_epoch_training_audit": {
            "fully_finite": True, "sample_count": 2, "finite_sample_count": 2,
            "nonfinite_sample_count": 0, "nonfinite_sample_indices": [],
            "metric_scope": "all_samples", "loss": .1, "dice": .9, "iou": .8,
        }}],
    }


def test_eval_accepts_existing_audited_adapter_checkpoint_without_blend_parameters(tmp_path):
    path = tmp_path / "rotation_padding_best.pt"
    _validate_evaluated_checkpoint(_audited_checkpoint(path), path=path, source_sha256="a" * 64, train_count=2)


@pytest.mark.parametrize("corruption", ["source", "semantics", "weights", "audit", "scale", "legacy"])
def test_eval_only_does_not_bypass_checkpoint_provenance_and_audits(tmp_path, corruption):
    path = tmp_path / "rotation_padding_best.pt"
    saved = _audited_checkpoint(path)
    if corruption == "source":
        saved["source_checkpoint_sha256"] = "b" * 64
    elif corruption == "semantics":
        saved["model"]["padding_semantics"] = "zero_padding"
    elif corruption == "weights":
        saved["state_dict"]["act0.coeffs"][0] = float("nan")
    elif corruption == "audit":
        saved["history"] = []
    elif corruption == "scale":
        saved["state_dict"]["act0.prescale_tensor"] = torch.tensor(-1.)
    else:
        saved["schema_version"] = 3
    with pytest.raises(RuntimeError):
        _validate_evaluated_checkpoint(saved, path=path, source_sha256="a" * 64, train_count=2)
