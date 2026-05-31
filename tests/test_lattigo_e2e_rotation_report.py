from __future__ import annotations

import json
from types import SimpleNamespace

from tools import describe_lt_sharing_oracle as sharing
from tools import run_lattigo_e2e_compare as runner


class NativeAlignedHaloNoRIConvExecutor:
    use_ct_pt_hybrid_packing = False


class NativeHaloStripeNoRIConvExecutor:
    use_ct_pt_hybrid_packing = False


class InputPairConvRuntimeExecutor:
    use_ct_pt_hybrid_packing = False


class TconvK2S2PythonRuntimeExecutor:
    requires_compact_source = True
    use_ct_pt_hybrid_packing = False


class _HaloFacade:
    def __init__(self, delegate, *, same_shape_spec=None, force_input_pair: bool = False) -> None:
        self._delegate = delegate
        self.same_shape_spec = same_shape_spec
        self.force_input_pair = bool(force_input_pair)

    @property
    def delegate(self):
        return self._delegate


class _FakeBackend:
    def __init__(self, keys_by_id: dict[int, list[int]]) -> None:
        self.keys_by_id = dict(keys_by_id)

    def GetLinearTransformRotationKeys(self, transform_id: int) -> list[int]:
        return list(self.keys_by_id[int(transform_id)])

    def GetLinearTransformRotationEvalCount(self, transform_id: int) -> int:
        return len([key for key in self.keys_by_id[int(transform_id)] if int(key) != 1])


def test_provider_rotation_stats_treats_galois_one_as_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.scheme,
        "backend",
        _FakeBackend({1: [1, 5], 2: [1, 5]}),
        raising=False,
    )
    executor = SimpleNamespace(groups_by_input_chunk=[SimpleNamespace(unified_ids=[1, 2])])

    stats = runner._provider_rotation_stats(executor)

    assert stats["transform_rotation_key_count_total"] == 2
    assert stats["unique_rotation_keys"] == [5]
    assert stats["rotation_eval_count_estimate"] == 1


def test_provider_rotation_stats_counts_no_hybrid_same_shape_input_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        runner.scheme,
        "backend",
        _FakeBackend(
            {
                1: [1, 5, 7],
                2: [5, 9],
                3: [1, 11],
            }
        ),
        raising=False,
    )
    executor = SimpleNamespace(
        groups_by_input_chunk=[
            SimpleNamespace(unified_ids=[1, 2]),
            SimpleNamespace(unified_ids=[3]),
        ]
    )

    stats = runner._provider_rotation_stats(executor)

    assert stats["source"] == "compiled_backend_unified_transform_rotation_keys"
    assert stats["group_count"] == 2
    assert stats["transform_count"] == 3
    assert stats["transform_rotation_key_count_total"] == 5
    assert stats["shared_rotation_eval_count_total"] == 4
    assert stats["rotation_eval_count_estimate"] == 4


def test_provider_rotation_stats_counts_individual_runtime_groups(monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    monkeypatch.setattr(
        runner.scheme,
        "backend",
        _FakeBackend(
            {
                10: [1, 3, 5],
                11: [1, 3],
                12: [1, 7, 9, 11],
            }
        ),
        raising=False,
    )
    executor = SimpleNamespace(
        base_executor=SimpleNamespace(
            runtime_groups=[
                SimpleNamespace(input_index=0, group=SimpleNamespace(unified_ids=[10])),
                SimpleNamespace(input_index=0, group=SimpleNamespace(unified_ids=[11])),
                SimpleNamespace(input_index=1, group=SimpleNamespace(unified_ids=[12])),
            ]
        )
    )

    stats = runner._provider_rotation_stats(executor)

    assert stats["source"] == "compiled_backend_unified_transform_rotation_keys"
    assert stats["group_count"] == 3
    assert stats["transform_count"] == 3
    assert stats["transform_rotation_key_count_total"] == 6
    assert stats["shared_rotation_eval_count_total"] == 6
    assert stats["rotation_eval_count_mode"] == "independent_transform_bsgs"
    assert stats["rotation_eval_count_estimate"] == 6


def test_provider_rotation_stats_walks_wrapper_base_executor_before_delegate(monkeypatch) -> None:
    class DelegatingWrapper:
        def __init__(self, base_executor):
            self.base_executor = base_executor

        def __getattr__(self, name):
            return getattr(self.base_executor, name)

    monkeypatch.setenv("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    monkeypatch.setattr(
        runner.scheme,
        "backend",
        _FakeBackend(
            {
                20: [1, 2, 4],
                21: [1, 6],
            }
        ),
        raising=False,
    )
    native_delegate = SimpleNamespace(runtime_groups=[])
    base_executor = SimpleNamespace(
        delegate=native_delegate,
        runtime_groups=[
            SimpleNamespace(input_index=0, group=SimpleNamespace(unified_ids=[20])),
            SimpleNamespace(input_index=0, group=SimpleNamespace(unified_ids=[21])),
        ],
    )

    stats = runner._provider_rotation_stats(DelegatingWrapper(base_executor))

    assert stats["source"] == "compiled_backend_unified_transform_rotation_keys"
    assert stats["group_count"] == 2
    assert stats["transform_count"] == 2
    assert stats["transform_rotation_key_count_total"] == 3
    assert stats["rotation_eval_count_estimate"] == 3


def test_provider_rotation_stats_walks_private_delegate_runtime_groups(monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    monkeypatch.setattr(
        runner.scheme,
        "backend",
        _FakeBackend(
            {
                30: [1, 2, 4],
                31: [1, 6, 8],
            }
        ),
        raising=False,
    )
    native_delegate = SimpleNamespace(
        runtime_groups=[
            SimpleNamespace(input_index=0, group=SimpleNamespace(unified_ids=[30])),
            SimpleNamespace(input_index=1, group=SimpleNamespace(unified_ids=[31])),
        ]
    )
    halo_facade = SimpleNamespace(_delegate=native_delegate)
    executor = SimpleNamespace(base_executor=halo_facade)

    stats = runner._provider_rotation_stats(executor)

    assert stats["source"] == "compiled_backend_unified_transform_rotation_keys"
    assert stats["group_count"] == 2
    assert stats["transform_count"] == 2
    assert stats["transform_rotation_key_count_total"] == 4
    assert stats["rotation_eval_count_estimate"] == 4


def test_collect_rotation_report_deduplicates_module_and_provider_group(monkeypatch) -> None:
    class FakeLinearTransform:
        pass

    group = SimpleNamespace(unified_ids=[40, 41])
    module = FakeLinearTransform()
    module.group = group
    module.region_output_id = "enc1b"
    module.region_runtime = SimpleNamespace(
        executable=True,
        supports_scheme=lambda _scheme: True,
        executor=SimpleNamespace(runtime_groups=[SimpleNamespace(group=group)]),
        stage="provider",
        conv_nodes=("enc1b",),
    )

    class FakeNet:
        def named_modules(self):
            return [("enc1b", module)]

    monkeypatch.setattr(runner, "LinearTransform", FakeLinearTransform)
    monkeypatch.setenv("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    monkeypatch.setattr(
        runner.scheme,
        "backend",
        _FakeBackend(
            {
                40: [1, 2, 4],
                41: [1, 6],
            }
        ),
        raising=False,
    )

    report = runner._collect_rotation_report(FakeNet(), mode="provider")

    assert report["row_count"] == 2
    assert report["unique_rotation_stats_row_count"] == 1
    assert report["duplicate_rotation_stats_row_count"] == 1
    assert report["total_rotation_eval_count_estimate"] == 3
    assert [row["kind"] for row in report["rows"]] == [
        "FakeLinearTransform_module_unified",
        "provider_region",
    ]
    assert [bool(row["duplicate_rotation_stats"]) for row in report["rows"]] == [False, True]


def test_provider_rotation_stats_falls_back_to_runtime_io_native_halo_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    executor = SimpleNamespace(
        runtime_groups=[],
        last_runtime_counts={"partial_count": 138},
        last_runtime_timing={"built_transform_count": 138.0},
        last_runtime_io={
            "native_c_only_rotations": 1600,
            "native_cb_shared_rotations": 410,
            "native_plan_c_only_rotations": 533,
            "native_plan_cb_shared_rotations": 241,
            "runtime_group_count": 138,
            "runtime_transform_count": 138,
            "provider_lt_grouping_mode": "individual",
            "provider_disable_shared_rotation": True,
            "native_halo_channel_fold_mode": "per_stripe",
            "native_output_storage_layout": "tight_compact",
        },
    )

    stats = runner._provider_rotation_stats(executor)

    assert stats["source"] == "runtime_io_native_halo_rotation_estimate"
    assert stats["group_count"] == 138
    assert stats["transform_count"] == 138
    assert stats["rotation_eval_count_mode"] == "independent_transform_bsgs"
    assert stats["rotation_eval_count_estimate"] == 1600
    assert stats["shared_rotation_eval_count_total"] == 410
    assert stats["native_plan_c_only_rotation_estimate"] == 533


def test_provider_rotation_stats_falls_back_to_runtime_io_unified_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("ORION_UNIFIED_LT_INDIVIDUAL_EVAL", "1")
    executor = SimpleNamespace(
        groups_by_pair=[],
        last_runtime_io={
            "source": "runtime_io_unified_rotation_snapshot",
            "runtime_group_count": 2,
            "runtime_transform_count": 3,
            "transform_rotation_key_count_total": 17,
            "shared_rotation_eval_count_total": 9,
            "unique_rotation_key_count": 6,
            "rotation_eval_count_estimate": 17,
            "rotation_eval_count_mode": "independent_transform_bsgs",
            "unique_rotation_keys": [2, 4, 8, 16, 32, 64],
            "runtime_rotation_groups": [
                {"group_index": 0, "transform_count": 1, "rotation_key_count_total": 5},
                {"group_index": 1, "transform_count": 2, "rotation_key_count_total": 12},
            ],
        },
    )

    stats = runner._provider_rotation_stats(executor)

    assert stats["source"] == "runtime_io_unified_rotation_snapshot"
    assert stats["group_count"] == 2
    assert stats["transform_count"] == 3
    assert stats["rotation_eval_count_mode"] == "independent_transform_bsgs"
    assert stats["rotation_eval_count_estimate"] == 17
    assert stats["shared_rotation_eval_count_total"] == 9
    assert stats["unique_rotation_keys"] == [2, 4, 8, 16, 32, 64]


def test_provider_executor_role_distinguishes_native_halo_from_input_pair_executor() -> None:
    native = sharing._provider_executor_role(
        SimpleNamespace(
            base_executor=_HaloFacade(
                NativeAlignedHaloNoRIConvExecutor(),
                same_shape_spec=object(),
            )
        )
    )
    fallback = sharing._provider_executor_role(
        SimpleNamespace(
            base_executor=_HaloFacade(
                InputPairConvRuntimeExecutor(),
                same_shape_spec=None,
                force_input_pair=True,
            )
        )
    )
    tconv = sharing._provider_executor_role(TconvK2S2PythonRuntimeExecutor())
    generic_native = sharing._provider_executor_role(_HaloFacade(NativeHaloStripeNoRIConvExecutor()))

    assert native["provider_lowering"] == "native_aligned_halo_no_ri"
    assert native["path_family"] == "HaloED native halo local programs"
    assert native["native_halo_programs"] is True
    assert native["same_shape_spec_present"] is True
    assert native["delegate_executor"] == "NativeAlignedHaloNoRIConvExecutor"

    assert generic_native["provider_lowering"] == "native_halo_stripe_no_ri"
    assert generic_native["path_family"] == "HaloED native halo local programs"
    assert generic_native["native_halo_programs"] is True
    assert generic_native["delegate_executor"] == "NativeHaloStripeNoRIConvExecutor"

    assert fallback["provider_lowering"] == "input_pair_no_ri"
    assert fallback["path_family"] == "Provider input-pair programs"
    assert fallback["native_halo_programs"] is False
    assert fallback["delegate_executor"] == "InputPairConvRuntimeExecutor"

    assert tconv["provider_lowering"] == "local_output_placement"
    assert tconv["path_family"] == "HaloED local output-placement programs"


def _e2e_payload(*, he_forward: float, resident: float | None, serving: float, mode: str) -> dict:
    return {
        "status": "ok",
        "network": "toy",
        "backend": "lattigo",
        "label": "toy",
        "activation": "square",
        "timing_s": {
            "compile": 10.0,
            "encrypt": 1.0,
            "he_forward": float(he_forward),
            "decrypt_decode": 1.0,
        },
        "measured_forward_mean_timing_s": {
            "encrypt": 1.0,
            "he_forward": float(he_forward),
            "decrypt_decode": 1.0,
        },
        "decoded": {"values": [1.0, 2.0]},
        "clear": {"values": [1.0, 2.0]},
        "runtime_fairness_mode": str(mode),
        "resident_compute_s": resident,
        "serving_hot_s": float(serving),
        "artifact_read_s": 0.25,
        "artifact_load_s": 0.5,
        "artifact_unload_s": 0.125,
        "trim_s": 0.0625,
        "rotation_report_after_forward": {"total_rotation_eval_count_estimate": 8},
        "bootstrap_report_after_forward": {"count": 0},
    }


def test_lattigo_e2e_summary_uses_resident_compute_for_fair_runtime_speedup(tmp_path) -> None:
    dense_path = tmp_path / "dense.json"
    provider_path = tmp_path / "provider.json"
    out_path = tmp_path / "summary.json"
    dense = _e2e_payload(he_forward=100.0, resident=20.0, serving=200.0, mode="resident_compute")
    provider = _e2e_payload(
        he_forward=20.0,
        resident=5.0,
        serving=100.0,
        mode="memory_bounded_load_eval",
    )
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    provider_path.write_text(json.dumps(provider), encoding="utf-8")

    summary = runner._summarize(dense_path=dense_path, provider_path=provider_path, out_path=out_path)

    assert summary["dense"]["resident_compute_s"] == 20.0
    assert summary["provider"]["artifact_load_s"] == 0.5
    assert summary["provider"]["runtime_fairness_mode"] == "memory_bounded_load_eval"
    assert summary["ratios"]["runtime_speedup_metric"] == "resident_compute_s"
    assert summary["ratios"]["runtime_dense_over_provider"] == 4.0
    assert summary["ratios"]["resident_compute_dense_over_provider"] == 4.0
    assert summary["ratios"]["serving_hot_dense_over_provider"] == 2.0
    assert summary["ratios"]["artifact_runtime_dense_over_provider"] == 102.0 / 22.0


def _node_bench_result(*, hot: float, resident: float | None, mode: str = "resident_compute") -> dict:
    return {
        "status": "ok",
        "compile_s": 10.0,
        "hot_run_mean_s": float(hot),
        "resident_compute_s": resident,
        "serving_hot_s": float(hot),
        "artifact_read_s": 0.1,
        "artifact_load_s": 0.2,
        "artifact_unload_s": 0.3,
        "trim_s": 0.4,
        "runtime_fairness_mode": str(mode),
        "runtime_fairness_timing": {
            "resident_compute_s": resident,
            "serving_hot_s": float(hot),
            "runtime_fairness_mode": str(mode),
        },
        "rotation_eval_count": 8,
        "runtime_rotation_eval_count": 8,
    }


def test_node_benchmark_summary_and_csv_use_runtime_fairness_fields() -> None:
    from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench

    case = {
        "case": "toy_case",
        "node": "toy_node",
        "op": "conv",
        "stage": "toy",
        "multiplicity": 1,
        "paths": {
            "dense": {"status": "ok", "result": _node_bench_result(hot=200.0, resident=20.0)},
            "provider": {
                "status": "ok",
                "result": _node_bench_result(
                    hot=100.0,
                    resident=5.0,
                    mode="memory_bounded_load_eval",
                ),
            },
        },
    }

    bench._summarize_case(case)
    rows = bench._flatten_csv_rows(
        {
            "networks": [
                {
                    "backend": "lattigo",
                    "network": "toy",
                    "label": "toy",
                    "cases": [case],
                    "config": {},
                }
            ]
        }
    )

    assert case["speedup"]["time_speedup_metric"] == "resident_compute_s"
    assert case["speedup"]["time_dense_over_provider"] == 4.0
    assert case["speedup"]["resident_compute_dense_over_provider"] == 4.0
    assert case["speedup"]["serving_hot_dense_over_provider"] == 2.0
    assert "resident_compute_s" in bench.CSV_COLUMNS
    assert "runtime_fairness_timing_json" in bench.CSV_COLUMNS
    assert rows[0]["resident_compute_s"] == 20.0
    assert rows[0]["runtime_fairness_mode"] == "resident_compute"
    assert rows[1]["resident_compute_s"] == 5.0
    assert rows[1]["runtime_fairness_mode"] == "memory_bounded_load_eval"
