from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from orion.experimental.wpc_periodicity import (
    PROFILE_ENV,
    PROFILE_JSONL_ENV,
    PROFILE_SUMMARY_ENV,
    PeriodicityCollector,
    PeriodicityObservation,
    aggregate_periodicity,
    analyze_slot_periodicity,
    canonicalize_slot_value,
    encoded_plaintext_bytes,
    flush_periodicity_profile,
    record_flattened_diagonals,
    record_payload,
    reset_global_periodicity_collector,
    slot_payload_sha256,
)


@pytest.mark.parametrize(
    ("payload", "period", "classification", "candidate"),
    [
        ([3.0] * 8, 1, "constant_nonzero", True),
        ([1.0, 2.0] * 4, 2, "periodic", True),
        ([1.0, 2.0, 3.0, 4.0] * 2, 4, "periodic", True),
        (list(range(8)), 8, "aperiodic", False),
        ([0.0] * 8, 1, "all_zero", False),
    ],
)
def test_exact_minimal_power_of_two_period(payload, period, classification, candidate) -> None:
    result = analyze_slot_periodicity(payload)

    assert result.minimal_period == period
    assert result.classification == classification
    assert result.wpc_candidate is candidate
    assert result.compression_ratio == len(payload) // period


def test_complex_and_interleaved_complex_payloads_match() -> None:
    native = [1.0 + 2.0j, -3.0 + 4.0j] * 4
    interleaved = [component for value in native for component in (value.real, value.imag)]

    native_result = analyze_slot_periodicity(native, payload_format="complex")
    interleaved_result = analyze_slot_periodicity(
        interleaved,
        payload_format="interleaved_complex",
    )

    assert native_result == interleaved_result
    assert native_result.minimal_period == 2
    assert slot_payload_sha256(native) == slot_payload_sha256(
        interleaved,
        payload_format="interleaved_complex",
    )


def test_signed_zero_is_canonical_for_classification_and_hashing() -> None:
    value = canonicalize_slot_value(complex(-0.0, -0.0))
    assert math.copysign(1.0, value.real) == 1.0
    assert math.copysign(1.0, value.imag) == 1.0

    signed = [0.0, -0.0, complex(0.0, -0.0), complex(-0.0, 0.0)]
    positive = [0.0] * 4
    result = analyze_slot_periodicity(signed)
    assert result.all_zero is True
    assert result.minimal_period == 1
    assert slot_payload_sha256(signed) == slot_payload_sha256(positive)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, complex(1.0, math.nan)])
def test_nonfinite_values_are_rejected(bad) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        analyze_slot_periodicity([bad, 0.0])


@pytest.mark.parametrize("payload", [[], [1.0, 2.0, 3.0]])
def test_invalid_slot_lengths_are_rejected(payload) -> None:
    with pytest.raises(ValueError):
        analyze_slot_periodicity(payload)


def test_aggregation_excludes_zero_and_weights_count_bytes_and_time() -> None:
    rows = [
        PeriodicityObservation(analyze_slot_periodicity([2.0] * 8), 800, 2.0),
        PeriodicityObservation(analyze_slot_periodicity([1.0, 2.0, 3.0, 4.0] * 2), 800, 3.0),
        PeriodicityObservation(analyze_slot_periodicity(list(range(8))), 800, 5.0),
        PeriodicityObservation(analyze_slot_periodicity([0.0] * 8), 800, 7.0),
    ]

    result = aggregate_periodicity(rows)

    assert result.observed_count == 4
    assert result.nonzero_count == 3
    assert result.periodic_count == 2
    assert result.all_zero_count == 1
    assert result.count_coverage == pytest.approx(2.0 / 3.0)
    assert result.byte_coverage == pytest.approx(2.0 / 3.0)
    assert result.encode_time_coverage == pytest.approx(0.5)
    assert result.hybrid_storage_bytes == 1300
    assert result.partial_storage_compression_ratio == pytest.approx(2400.0 / 1300.0)


def test_collector_flushes_unique_and_occurrence_records(tmp_path) -> None:
    collector = PeriodicityCollector()
    metadata = {
        "model": "resnet20",
        "mode": "dense",
        "checkpoint_hash": "abc",
        "module_name": "conv1",
        "operator_type": "Conv2d",
        "transform_id": 7,
        "block_row": 0,
        "block_col": 0,
        "diagonal_index": 3,
    }
    for _ in range(2):
        collector.record_payload(
            [1.0, 2.0] * 4,
            metadata=metadata,
            full_encoded_bytes=800,
            baseline_encode_s=0.25,
        )

    jsonl_path = tmp_path / "periodicity.jsonl"
    summary_path = tmp_path / "periodicity-summary.json"
    summary = collector.flush(jsonl_path, summary_path=summary_path)
    records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]

    assert [record["record_type"] for record in records].count("slot_periodicity_unique") == 1
    assert [record["record_type"] for record in records].count("slot_periodicity_occurrence") == 2
    assert summary["unique_diagonal"]["periodic_count"] == 1
    assert summary["occurrence"]["periodic_count"] == 2
    assert json.loads(summary_path.read_text())["scope"] == "slot_message_only"


def test_global_collector_is_noop_when_disabled(monkeypatch) -> None:
    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("disabled collector touched the payload")

    monkeypatch.delenv(PROFILE_ENV, raising=False)
    reset_global_periodicity_collector()
    try:
        assert record_payload(MustNotIterate()) is None
        assert flush_periodicity_profile("unused.jsonl") is None
    finally:
        reset_global_periodicity_collector()


def test_env_gated_global_collector_records_interleaved_payload(monkeypatch, tmp_path) -> None:
    jsonl_path = tmp_path / "global.jsonl"
    summary_path = tmp_path / "global-summary.json"
    monkeypatch.setenv(PROFILE_ENV, "1")
    monkeypatch.setenv(PROFILE_JSONL_ENV, str(jsonl_path))
    monkeypatch.setenv(PROFILE_SUMMARY_ENV, str(summary_path))
    reset_global_periodicity_collector()
    try:
        record = record_payload(
            [1.0, 2.0, 3.0, 4.0] * 4,
            payload_format="interleaved_complex",
            metadata={"module_name": "conv"},
        )
        assert record is not None
        assert record["slot_min_period_t"] == 2
        summary = flush_periodicity_profile()
        assert summary is not None
        assert jsonl_path.exists()
        assert summary_path.exists()
    finally:
        reset_global_periodicity_collector()


def test_encoded_plaintext_bytes_counts_active_q_and_p_limbs() -> None:
    assert encoded_plaintext_bytes(ring_degree=16, level_q=2, level_p=1) == 16 * 8 * 5
    assert encoded_plaintext_bytes(ring_degree=16, level_q=0, level_p=-1) == 16 * 8


def test_flattened_real_payload_records_each_diagonal(monkeypatch, tmp_path) -> None:
    jsonl_path = tmp_path / "flat-real.jsonl"
    monkeypatch.setenv(PROFILE_ENV, "1")
    monkeypatch.setenv(PROFILE_JSONL_ENV, str(jsonl_path))
    reset_global_periodicity_collector()
    try:
        count = record_flattened_diagonals(
            [3, 9],
            [1.0, 2.0] * 4 + list(range(8)),
            metadata={"module_name": "conv"},
            full_encoded_bytes=800,
        )
        assert count == 2
        summary = flush_periodicity_profile()
        assert summary is not None
        assert summary["occurrence"]["observed_count"] == 2
        assert summary["occurrence"]["periodic_count"] == 1
        records = [
            json.loads(line)
            for line in jsonl_path.read_text().splitlines()
            if json.loads(line)["record_type"] == "slot_periodicity_occurrence"
        ]
        assert [record["diagonal_index"] for record in records] == [3, 9]
    finally:
        reset_global_periodicity_collector()


def test_flattened_interleaved_payload_uses_decoded_slot_count(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(PROFILE_ENV, "1")
    monkeypatch.setenv(PROFILE_JSONL_ENV, str(tmp_path / "flat-complex.jsonl"))
    reset_global_periodicity_collector()
    try:
        # Two complex diagonals, each with four decoded slots.
        diagonal = [1.0, 2.0, 3.0, 4.0] * 2
        assert record_flattened_diagonals(
            [1, 2],
            diagonal + diagonal,
            has_complex=True,
        ) == 2
        summary = flush_periodicity_profile()
        assert summary is not None
        assert summary["occurrence"]["observed_count"] == 2
        assert summary["occurrence"]["periodic_count"] == 2
    finally:
        reset_global_periodicity_collector()


def test_flattened_payload_is_noop_without_profiling(monkeypatch) -> None:
    class MustNotTouch:
        @property
        def size(self):
            raise AssertionError("disabled flattened recorder touched payload data")

    monkeypatch.delenv(PROFILE_ENV, raising=False)
    reset_global_periodicity_collector()
    try:
        assert record_flattened_diagonals([1], MustNotTouch()) == 0
    finally:
        reset_global_periodicity_collector()


def test_dense_runtime_hook_records_complete_metadata(monkeypatch, tmp_path) -> None:
    from orion.backend.python.lt_evaluator import NewEvaluator

    monkeypatch.setenv(PROFILE_ENV, "1")
    monkeypatch.setenv(PROFILE_JSONL_ENV, str(tmp_path / "dense-hook.jsonl"))
    monkeypatch.setenv("ORION_WPC_MODEL", "resnet20_cifar10")
    monkeypatch.setenv("ORION_WPC_MODE", "dense")
    reset_global_periodicity_collector()
    try:
        fake_evaluator = SimpleNamespace(
            params=SimpleNamespace(
                get_logp=lambda: [60],
                get_ring_degree=lambda: 8,
            )
        )
        layer = SimpleNamespace(name="conv1", level=2, bsgs_ratio=2.0)
        payloads = [
            (
                0,
                1,
                np.asarray([3, 7], dtype=np.int32),
                np.asarray([1.0, 2.0] * 2 + list(range(4)), dtype=np.float32),
            )
        ]
        recorded = NewEvaluator._record_wpc_dense_payloads(fake_evaluator, layer, payloads)
        assert recorded == 2
        summary = flush_periodicity_profile()
        assert summary is not None
        assert summary["occurrence"]["observed_count"] == 2
        assert summary["metadata_complete_occurrence_count"] == 2
        assert summary["occurrence"]["full_encoded_bytes"] == 2 * 8 * 8 * 4
    finally:
        reset_global_periodicity_collector()


def test_provider_runtime_hook_records_complete_metadata(monkeypatch, tmp_path) -> None:
    from orion.nn.unified_transform import UnifiedTransformGroup

    monkeypatch.setenv(PROFILE_ENV, "1")
    monkeypatch.setenv(PROFILE_JSONL_ENV, str(tmp_path / "provider-hook.jsonl"))
    monkeypatch.setenv("ORION_WPC_MODEL", "u22_64_base32")
    monkeypatch.setenv("ORION_WPC_MODE", "provider")
    reset_global_periodicity_collector()
    try:
        params = SimpleNamespace(get_logp=lambda: [60], get_ring_degree=lambda: 8)
        transform = SimpleNamespace(
            name="encoder_conv",
            scheme=SimpleNamespace(params=params),
            N1=2,
        )
        group = UnifiedTransformGroup([transform])
        payloads = [
            (
                np.asarray([5], dtype=np.int32),
                np.asarray([1.0, 2.0] * 2, dtype=np.float32),
                2,
            )
        ]
        recorded = group._record_wpc_provider_payloads(payloads, has_complex=False)
        assert recorded == 1
        summary = flush_periodicity_profile()
        assert summary is not None
        assert summary["occurrence"]["observed_count"] == 1
        assert summary["metadata_complete_occurrence_count"] == 1
        assert summary["occurrence"]["full_encoded_bytes"] == 8 * 8 * 4
    finally:
        reset_global_periodicity_collector()


def test_provider_group_identity_prevents_false_payload_changes(monkeypatch, tmp_path) -> None:
    from orion.nn.unified_transform import UnifiedTransformGroup

    monkeypatch.setenv(PROFILE_ENV, "1")
    monkeypatch.setenv(PROFILE_JSONL_ENV, str(tmp_path / "provider-groups.jsonl"))
    monkeypatch.setenv("ORION_WPC_MODEL", "u22_64_base32")
    monkeypatch.setenv("ORION_WPC_MODE", "provider")
    reset_global_periodicity_collector()
    try:
        params = SimpleNamespace(get_logp=lambda: [60], get_ring_degree=lambda: 8)

        def group_with_payload(values):
            transform = SimpleNamespace(
                name="reused_transform_name",
                scheme=SimpleNamespace(params=params),
                N1=2,
            )
            group = UnifiedTransformGroup([transform])
            payload = (
                np.asarray([5], dtype=np.int32),
                np.asarray(values, dtype=np.float32),
                2,
            )
            assert group._record_wpc_provider_payloads([payload], has_complex=False) == 1

        group_with_payload([1.0, 2.0] * 2)
        group_with_payload([3.0, 4.0] * 2)
        summary = flush_periodicity_profile()
        assert summary is not None
        assert summary["occurrence"]["observed_count"] == 2
        assert summary["unique_diagonal"]["observed_count"] == 2
        assert summary["logical_diagonal_payload_change_count"] == 0
    finally:
        reset_global_periodicity_collector()
