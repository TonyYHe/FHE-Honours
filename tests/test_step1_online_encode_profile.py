from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import run_lattigo_e2e_compare as e2e
from tools import run_step1_online_encode_profile as step1


def _operator_breakdown(*, he_forward_s: float = 100.0) -> dict:
    return {
        "totals": {
            "he_forward_s": float(he_forward_s),
            "mvm_kernel_s": 40.0,
            "activation_s": 10.0,
            "bootstrap_s": 20.0,
            "lt_layer_cache_encode_s": 15.0,
            "lt_layer_cache_key_prepare_s": 2.0,
            "lt_layer_cache_evict_s": 3.0,
            "lt_layer_cache_turnover_s": 20.0,
            "wall_runtime_load_trim_s": 1.0,
            "wall_linear_wrapper_postprocess_s": 2.0,
            "wall_executor_overhead_s": 1.0,
            "wall_unattributed_he_forward_s": 6.0,
            "lt_runtime_stream_accumulate_s": 0.5,
            "linear_wrapper_accumulate_s": 0.25,
            "executor_accumulate_s": 0.25,
        },
        "module_wall": {
            "totals_by_category": {
                "add": {"module_wall_s": 0.75},
                "multiply": {"module_wall_s": 1.25},
            }
        },
    }


def _profile(*, encode_s: float = 15.0, he_forward_s: float = 100.0) -> dict:
    breakdown = _operator_breakdown(he_forward_s=he_forward_s)
    breakdown["totals"]["lt_layer_cache_encode_s"] = float(encode_s)
    breakdown["totals"]["lt_layer_cache_turnover_s"] = float(encode_s + 5.0)
    return e2e._build_step1_online_encode_profile(
        operator_breakdown=breakdown,
        lt_profile_counters={"diag_terms": 20, "baby_rotation_count": 3},
        lt_profile_seconds={
            "transform_mul_accum_s": 12.0,
            "pre_rotate_automorphism_s": 1.0,
            "transform_giant_moddown_s": 2.0,
            "transform_giant_keyswitch_s": 3.0,
            "transform_giant_auto_s": 4.0,
        },
        bootstrap_profile={"available": True, "totals": {"total_s": 20.0}},
        runtime_fairness_mode="single_slot_layer_cache",
        backend="lattigo",
        clear_backend_enabled=False,
    )


def test_lattigo_profile_seconds_are_named_and_zero_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = SimpleNamespace(GetLinearTransformEvaluationProfileSeconds=lambda: [1.5, 2.5])
    monkeypatch.setattr(e2e.scheme, "backend", backend, raising=False)

    result = e2e._collect_lattigo_lt_profile_seconds()

    assert result["shared_buffer_s"] == 1.5
    assert result["decompose_s"] == 2.5
    assert result["transform_mul_accum_s"] == 0.0
    assert set(result) == set(e2e._LT_PROFILE_SECONDS_NAMES)


def test_step1_profile_reports_online_encode_and_operator_percentages() -> None:
    profile = _profile()

    assert profile["valid"] is True
    assert profile["online_encode_s"] == 15.0
    assert profile["online_encode_pct_of_he_forward"] == 15.0
    assert profile["major_wall_categories"]["bootstrap"]["percent_of_he_forward"] == 20.0
    assert profile["major_wall_categories"]["other_he_forward"]["seconds"] == 16.0
    assert profile["major_wall_categories_accounting"]["valid"] is True
    assert profile["major_wall_categories_accounting"]["category_sum_s"] == 100.0
    assert profile["major_wall_categories_metadata"]["additive"] is True
    assert profile["operator_microprofile_metadata"]["additive"] is False
    assert profile["operator_microprofile"]["lt_fused_multiply_accumulate"]["seconds"] == 12.0
    assert profile["operator_microprofile"]["lt_rotation"]["seconds"] == 10.0
    assert profile["operator_microprofile"]["elementwise_add_module_wall"]["seconds"] == 0.75
    assert profile["operation_counts"]["baby_rotation_count"] == 3


def test_step1_profile_rejects_clear_or_legacy_streaming_runs() -> None:
    profile = e2e._build_step1_online_encode_profile(
        operator_breakdown=_operator_breakdown(),
        lt_profile_counters={"diag_terms": 20},
        lt_profile_seconds={"transform_mul_accum_s": 12.0},
        bootstrap_profile={"available": True, "totals": {"total_s": 20.0}},
        runtime_fairness_mode="streaming_eval_encode",
        backend="lattigo",
        clear_backend_enabled=True,
    )

    assert profile["valid"] is False
    assert any("CLEAR_BACKEND" in value for value in profile["validation_errors"])
    assert any("single_slot_layer_cache" in value for value in profile["validation_errors"])


def test_step1_profile_uses_reset_backend_bootstrap_wall_time() -> None:
    breakdown = _operator_breakdown()
    breakdown["totals"]["bootstrap_s"] = 250.0

    profile = e2e._build_step1_online_encode_profile(
        operator_breakdown=breakdown,
        lt_profile_counters={"diag_terms": 20},
        lt_profile_seconds={"transform_mul_accum_s": 12.0},
        bootstrap_profile={"available": True, "totals": {"total_s": 20.0}},
        runtime_fairness_mode="single_slot_layer_cache",
        backend="lattigo",
        clear_backend_enabled=False,
    )

    assert profile["valid"] is True
    assert profile["major_wall_categories"]["bootstrap"]["seconds"] == 20.0
    assert profile["operator_microprofile"]["bootstrap"]["seconds"] == 20.0


def test_step1_profile_rejects_missing_bootstrap_profile() -> None:
    profile = e2e._build_step1_online_encode_profile(
        operator_breakdown=_operator_breakdown(),
        lt_profile_counters={"diag_terms": 20},
        lt_profile_seconds={"transform_mul_accum_s": 12.0},
        bootstrap_profile=None,
        runtime_fairness_mode="single_slot_layer_cache",
        backend="lattigo",
        clear_backend_enabled=False,
    )

    assert profile["valid"] is False
    assert profile["major_wall_categories_accounting"]["valid"] is False
    assert any("bootstrap wall profile is missing" in value for value in profile["validation_errors"])


def test_step1_profile_rejects_negative_wall_residual() -> None:
    profile = e2e._build_step1_online_encode_profile(
        operator_breakdown=_operator_breakdown(he_forward_s=50.0),
        lt_profile_counters={"diag_terms": 20},
        lt_profile_seconds={"transform_mul_accum_s": 12.0},
        bootstrap_profile={"available": True, "totals": {"total_s": 20.0}},
        runtime_fairness_mode="single_slot_layer_cache",
        backend="lattigo",
        clear_backend_enabled=False,
    )

    assert profile["valid"] is False
    assert profile["major_wall_categories_accounting"]["valid"] is False
    assert any("negative" in value for value in profile["validation_errors"])


def test_bootstrap_runtime_profiles_reset_per_forward() -> None:
    bootstrap = e2e.Bootstrap(0.0, 1.0, 3)
    bootstrap._bootstrap_runtime_profile = [{"timing_s": {"forward_total_inner": 1.0}}]
    bootstrap._bootstrap_runtime_call_index = 7
    net = e2e.torch.nn.Sequential(bootstrap)

    assert e2e._reset_bootstrap_runtime_profiles(net) == 1
    assert bootstrap._bootstrap_runtime_profile == []
    assert bootstrap._bootstrap_runtime_call_index == 0


def test_activation_profiler_hooks_only_outermost_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyActivation(e2e.Module):
        def __init__(self, child=None):
            super().__init__()
            if child is not None:
                self.child = child

        def forward(self, value):
            child = getattr(self, "child", None)
            return value if child is None else child(value)

    net = e2e.torch.nn.Sequential(DummyActivation(DummyActivation()))
    monkeypatch.setattr(
        e2e,
        "_module_category",
        lambda module: "activation" if isinstance(module, DummyActivation) else "other",
    )

    snapshot, remove = e2e._install_activation_breakdown_profiler(net)
    try:
        assert snapshot()["profiled_module_count"] == 1
    finally:
        remove()


def test_activation_breakdown_subtracts_all_descendant_bootstraps() -> None:
    activation = e2e._collect_activation_runtime_breakdown(
        {
            "enabled": True,
            "profiled_module_count": 1,
            "active_module_count": 1,
            "rows": [
                {
                    "module_path": "act",
                    "elapsed_s": 30.0,
                    "call_count": 1,
                    "bootstrapper_paths": ["act.first", "act.second"],
                }
            ],
        },
        {"by_name": {"act.first": 10.0, "act.second": 5.0}},
    )

    assert activation["totals"]["activation_inclusive_s"] == 30.0
    assert activation["totals"]["bootstrap_child_s"] == 15.0
    assert activation["totals"]["activation_s"] == 15.0


def test_step1_measured_summary_recomputes_ratio_from_mean_times() -> None:
    first = _profile(encode_s=10.0, he_forward_s=100.0)
    second = _profile(encode_s=20.0, he_forward_s=200.0)
    attempts = [
        {"step1_online_encode_profile": first},
        {"step1_online_encode_profile": second},
    ]

    summary = e2e._mean_step1_online_encode_profiles(attempts)

    assert summary["valid"] is True
    assert summary["measured_attempt_count"] == 2
    assert summary["he_forward_s"] == 150.0
    assert summary["online_encode_s"] == 15.0
    assert summary["online_encode_pct_of_he_forward"] == 10.0
    assert summary["operation_counts_mean_per_forward"]["diag_terms"] == 20.0
    assert summary["operation_counts_total"]["diag_terms"] == 40


def test_step1_measured_summary_rejects_missing_requested_attempt() -> None:
    summary = e2e._mean_step1_online_encode_profiles(
        [{"step1_online_encode_profile": _profile()}],
        expected_attempt_count=2,
    )

    assert summary["valid"] is False
    assert summary["requested_measured_attempt_count"] == 2
    assert any("1 of 2" in value for value in summary["validation_errors"])


def test_step1_launcher_forces_real_fhe_single_slot_configuration(tmp_path) -> None:
    env = step1._profile_environment(
        {
            "ORION_LATTIGO_CLEAR_BACKEND": "1",
            "ORION_LATTIGO_STREAMING_LT": "1",
        },
        encode_workers=4,
    )
    args = SimpleNamespace(
        mode="provider",
        network="u22_64_base32",
        seed=7,
        forward_runs=3,
        warmup_runs=1,
        provider_mode=None,
        activation=None,
        ckks_preset=None,
        logn_override=None,
        trace_forward_memory=False,
    )

    command = step1._runner_command(args, tmp_path / "result.json")

    assert env["ORION_LATTIGO_CLEAR_BACKEND"] == "0"
    assert env["ORION_SINGLE_SLOT_LAYER_CACHE"] == "1"
    assert env["ORION_LATTIGO_STREAMING_LT"] == "0"
    assert env["ORION_LATTIGO_LEGACY_CHUNK_STREAMING_LT"] == "0"
    assert env["ORION_SINGLE_SLOT_ENCODE_WORKERS"] == "4"
    assert command[command.index("--backend") + 1] == "lattigo"
    assert "--profile-lt" in command
    assert "--operator-breakdown" in command
    assert "--profile-modules" in command
    assert command[command.index("--io-mode") + 1] == "none"


def test_step1_launcher_result_gate_requires_all_runs_and_correct_shape() -> None:
    payload = {
        "status": "ok",
        "forward_runs": 2,
        "mae_vs_clear": {"shape_match": True},
        "step1_online_encode_profile": {
            "schema_version": 2,
            "valid": True,
            "measured_attempt_count": 2,
            "profile_count": 2,
            "major_wall_categories_accounting": {"valid": True},
        },
    }
    assert step1._result_errors(payload) == []

    payload["step1_online_encode_profile"]["measured_attempt_count"] = 1
    payload["mae_vs_clear"]["shape_match"] = False
    errors = step1._result_errors(payload)

    assert "not every requested measured forward succeeded" in errors
    assert "final decrypted output does not match the clear output shape" in errors


def test_step1_launcher_rejects_obsolete_unvalidated_accounting() -> None:
    payload = {
        "status": "ok",
        "forward_runs": 1,
        "mae_vs_clear": {"shape_match": True},
        "step1_online_encode_profile": {
            "schema_version": 1,
            "valid": True,
            "measured_attempt_count": 1,
            "profile_count": 1,
        },
    }

    errors = step1._result_errors(payload)

    assert any("obsolete" in error for error in errors)
    assert any("additive wall-time accounting" in error for error in errors)
