#!/usr/bin/env python3
"""Descriptor-only LT sharing oracle for Orion vs HaloED paths."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion.nn.unified_transform import UnifiedTransformGroup
from tools import benchmark_node_specific_lattigo_provider_vs_dense as bench


DEFAULT_CASES = (
    "r34_imgnet:stage1_same",
    "r34_imgnet:stage2_same",
    "r34_imgnet:stage4_same",
    "u22_256_base32:up2",
)


def _diag_indices_from_mapping(diagonals: Any) -> set[int]:
    indices: set[int] = set()
    for block in dict(diagonals or {}).values():
        indices.update(int(value) for value in dict(block or {}).keys())
    return indices


def _diag_indices_from_transform(transform: Any) -> set[int]:
    return _diag_indices_from_mapping(getattr(transform, "diagonals", {}) or {})


def _orion_transform_entries(module: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for (row, col), block in sorted(dict(getattr(module, "diagonals", {}) or {}).items()):
        entries.append(
            {
                "row": int(row),
                "col": int(col),
                "transform_id": int(len(entries)),
                "diag_indices": {int(value) for value in dict(block or {}).keys()},
            }
        )
    return entries


def _rows_cols(entries: list[dict[str, Any]]) -> tuple[int, int]:
    if not entries:
        return 0, 0
    rows = max(int(entry["row"]) for entry in entries) + 1
    cols = max(int(entry["col"]) for entry in entries) + 1
    return int(rows), int(cols)


def _best_n1_for_entry(entry: dict[str, Any], *, slots: int, log_ratio: int) -> int:
    return bench._lattigo_find_best_bsgs_n1(
        {int(value) for value in entry.get("diag_indices", ())},
        slots=int(slots),
        log_max_ratio=int(log_ratio),
    )


def _sum_independent(entries: list[dict[str, Any]], *, slots: int, n1_by_index: dict[int, int]) -> int:
    total = 0
    for index, entry in enumerate(entries):
        cost = bench._shared_cache_bsgs_group_cost(
            [entry],
            slots=int(slots),
            n1s=[int(n1_by_index[int(index)])],
        )
        total += int(cost["actual_rotation_callback_count"])
    return int(total)


def _sum_shared_signature(
    entries: list[dict[str, Any]],
    *,
    slots: int,
    n1_by_index: dict[int, int],
) -> tuple[int, int, list[int]]:
    groups: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for index, entry in enumerate(entries):
        key = (int(entry["col"]), int(n1_by_index[int(index)]))
        groups.setdefault(key, []).append((int(index), entry))
    total = 0
    schedule_n1s: list[int] = []
    for (_col, n1), group_entries in sorted(groups.items()):
        payload = bench._shared_cache_bsgs_group_cost(
            [entry for _index, entry in group_entries],
            slots=int(slots),
            n1s=[int(n1)] * len(group_entries),
        )
        total += int(payload["actual_rotation_callback_count"])
        schedule_n1s.append(int(n1))
    return int(total), int(len(groups)), sorted(set(int(value) for value in schedule_n1s))


def _sum_shared_source_upper(
    entries: list[dict[str, Any]],
    *,
    slots: int,
    n1_by_index: dict[int, int],
) -> tuple[int, int]:
    by_col: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, entry in enumerate(entries):
        by_col.setdefault(int(entry["col"]), []).append((int(index), entry))
    total = 0
    for _col, group_entries in sorted(by_col.items()):
        payload = bench._shared_cache_bsgs_group_cost(
            [entry for _index, entry in group_entries],
            slots=int(slots),
            n1s=[int(n1_by_index[int(index)]) for index, _entry in group_entries],
        )
        total += int(payload["actual_rotation_callback_count"])
    return int(total), int(len(by_col))


def _dense_orion_descriptors(network: str, case_name: str) -> list[dict[str, Any]]:
    bench._init_scheme(str(network), backend="lattigo")
    dag, _audit = bench._prepare_dag(str(network), provider=False)
    case = bench._case_by_name(str(network), str(case_name))
    module = dag.nodes[str(case["node"])]["module"]
    module.generate_diagonals(last=False)
    entries = _orion_transform_entries(module)
    rows, cols = _rows_cols(entries)
    slots = int(bench.scheme.params.get_slots())
    log_ratio = bench._dense_bsgs_log_ratio(module)
    n1_by_index = {
        int(index): _best_n1_for_entry(entry, slots=int(slots), log_ratio=int(log_ratio))
        for index, entry in enumerate(entries)
    }
    recovery_rotations = int(rows * int(getattr(module, "output_rotations", 0) or 0))
    independent = _sum_independent(entries, slots=int(slots), n1_by_index=n1_by_index)
    shared_signature, signature_groups, signature_n1s = _sum_shared_signature(
        entries,
        slots=int(slots),
        n1_by_index=n1_by_index,
    )
    shared_upper, source_groups = _sum_shared_source_upper(
        entries,
        slots=int(slots),
        n1_by_index=n1_by_index,
    )
    common = {
        "backend": "lattigo",
        "network": str(network),
        "network_label": str(bench.NETWORK_SPECS[str(network)]["label"]),
        "case": str(case_name),
        "node": str(case["node"]),
        "op": str(case["op"]),
        "path_family": "Orion block-LT",
        "lt_tasks": int(len(entries)),
        "source_ct_count": int(cols),
        "recovery_rotation_eval_count": int(recovery_rotations),
        "bsgs_log_ratio": int(log_ratio),
    }
    return [
        {
            **common,
            "variant": "Orion independent BSGS",
            "sharing_group_count": int(len(entries)),
            "distinct_bsgs_schedule_count": int(len(set(n1_by_index.values()))),
            "distinct_bsgs_n1s": sorted(set(int(value) for value in n1_by_index.values())),
            "source_shared_rotation_eval_count": int(independent),
            "oracle_rotation_eval_count": int(independent + recovery_rotations),
            "note": "current block-Toeplitz tasks; no source-side BSGS sharing",
        },
        {
            **common,
            "variant": "Orion+Shared BSGS",
            "sharing_group_count": int(signature_groups),
            "distinct_bsgs_schedule_count": int(len(signature_n1s)),
            "distinct_bsgs_n1s": list(signature_n1s),
            "source_shared_rotation_eval_count": int(shared_signature),
            "oracle_rotation_eval_count": int(shared_signature + recovery_rotations),
            "note": "same Orion tasks; share baby rotations within (source_ct, BSGS n1)",
        },
        {
            **common,
            "variant": "Orion+Shared BSGS upper",
            "sharing_group_count": int(source_groups),
            "distinct_bsgs_schedule_count": int(len(set(n1_by_index.values()))),
            "distinct_bsgs_n1s": sorted(set(int(value) for value in n1_by_index.values())),
            "source_shared_rotation_eval_count": int(shared_upper),
            "oracle_rotation_eval_count": int(shared_upper + recovery_rotations),
            "note": "diagnostic upper bound: share all same-source baby rotations even across n1 signatures",
        },
    ]


@contextmanager
def _fake_unified_compile() -> Any:
    original = UnifiedTransformGroup.compile_unified
    counter = itertools.count(10_000_000)

    def fake_compile_unified(self: UnifiedTransformGroup, backend: Any) -> None:
        ids: list[int] = []
        self._diag_indices_by_transform = {}
        self._required_keys_by_transform = {}
        for transform in list(getattr(self, "transforms", []) or []):
            transform_id = int(next(counter))
            ids.append(int(transform_id))
            self._diag_indices_by_transform[int(transform_id)] = tuple(
                sorted(_diag_indices_from_transform(transform))
            )
        self.unified_ids = list(ids)
        self.is_compiled = True

    UnifiedTransformGroup.compile_unified = fake_compile_unified  # type: ignore[method-assign]
    try:
        yield
    finally:
        UnifiedTransformGroup.compile_unified = original  # type: ignore[method-assign]


def _provider_executor_role(executor: Any) -> dict[str, Any]:
    """Classify the provider executor without confusing facades with lowering."""

    base = getattr(executor, "base_executor", executor)
    delegate = base
    for attr in ("delegate", "_delegate"):
        value = getattr(base, attr, None)
        if value is not None:
            delegate = value
            break

    wrapper_name = type(executor).__name__ if executor is not None else ""
    base_name = type(base).__name__ if base is not None else ""
    delegate_name = type(delegate).__name__ if delegate is not None else ""
    same_shape_spec_present = bool(getattr(base, "same_shape_spec", None) is not None)
    force_input_pair = bool(getattr(base, "force_input_pair", False))
    requires_compact_source = bool(
        getattr(delegate, "requires_compact_source", getattr(base, "requires_compact_source", False))
    )
    ct_pt_hybrid_packing = bool(
        getattr(delegate, "use_ct_pt_hybrid_packing", getattr(base, "use_ct_pt_hybrid_packing", False))
    )

    if delegate_name in {"NativeAlignedHaloNoRIConvExecutor", "NativeHaloStripeNoRIConvExecutor"}:
        lowering = (
            "native_aligned_halo_no_ri"
            if delegate_name == "NativeAlignedHaloNoRIConvExecutor"
            else "native_halo_stripe_no_ri"
        )
        path_family = "HaloED native halo local programs"
        native_halo_programs = True
        note_suffix = "native halo no-RI lowering"
    elif delegate_name == "InputPairConvRuntimeExecutor":
        lowering = "input_pair_no_ri"
        path_family = "Provider input-pair programs"
        native_halo_programs = False
        note_suffix = "input-pair executor, not native halo-local geometry"
    elif requires_compact_source or delegate_name == "TconvK2S2PythonRuntimeExecutor":
        lowering = "local_output_placement"
        path_family = "HaloED local output-placement programs"
        native_halo_programs = False
        note_suffix = "local output-placement lowering"
    else:
        lowering = "provider_executor"
        path_family = "Provider local programs"
        native_halo_programs = False
        note_suffix = "provider lowering"

    return {
        "path_family": path_family,
        "provider_lowering": lowering,
        "native_halo_programs": bool(native_halo_programs),
        "executor": wrapper_name,
        "base_executor": base_name,
        "delegate_executor": delegate_name,
        "same_shape_spec_present": bool(same_shape_spec_present),
        "force_input_pair": bool(force_input_pair),
        "requires_compact_source": bool(requires_compact_source),
        "ct_pt_hybrid_packing": bool(ct_pt_hybrid_packing),
        "note_suffix": note_suffix,
    }


def _provider_group_descriptors(network: str, case_name: str) -> dict[str, Any]:
    bench._init_scheme(str(network), backend="lattigo")
    dag, _audit = bench._prepare_dag(str(network), provider=True)
    case = bench._case_by_name(str(network), str(case_name))
    module = dag.nodes[str(case["node"])]["module"]
    module.generate_diagonals(last=False)
    with _fake_unified_compile():
        module.compile()
    runtime = getattr(module, "region_runtime", None)
    executor = getattr(runtime, "executor", None)
    role = _provider_executor_role(executor)
    groups = bench._executor_unified_groups(executor)
    slots = int(bench.scheme.params.get_slots())
    per_group: list[dict[str, Any]] = []
    source_shared_total = 0
    individual_total = 0
    transform_count = 0
    n1s: list[int] = []
    for group_index, group in enumerate(groups):
        entries: list[dict[str, Any]] = []
        for transform_index, transform in enumerate(list(getattr(group, "transforms", []) or [])):
            entries.append(
                {
                    "transform_index": int(transform_index),
                    "transform_id": int(transform_index),
                    "diag_indices": _diag_indices_from_transform(transform),
                }
            )
        if not entries:
            continue
        n1, cost = bench._best_unified_common_n1(entries, slots=int(slots))
        n1s.append(int(n1))
        transform_count += int(len(entries))
        source_shared_total += int(cost["actual_rotation_callback_count"])
        individual_total += int(cost["sum_per_transform_rotation_count"])
        per_group.append(
            {
                "group_index": int(group_index),
                "transform_count": int(len(entries)),
                "unified_n1": int(n1),
                "shared_rotation_eval_count": int(cost["actual_rotation_callback_count"]),
                "individual_rotation_eval_count": int(cost["sum_per_transform_rotation_count"]),
                "diag_count_total": int(sum(len(entry["diag_indices"]) for entry in entries)),
            }
        )
    output_rotations = int(
        getattr(executor, "output_fold_rotations", getattr(executor, "output_rotations", 0)) or 0
    )
    output_ct_count = int(
        getattr(executor, "output_block_count", getattr(executor, "bank_count", 0)) or 0
    )
    recovery_rotations = int(output_rotations * output_ct_count)
    return {
        "backend": "lattigo",
        "network": str(network),
        "network_label": str(bench.NETWORK_SPECS[str(network)]["label"]),
        "case": str(case_name),
        "node": str(case["node"]),
        "op": str(case["op"]),
        "path_family": str(role["path_family"]),
        "variant": "HaloED Full",
        "lt_tasks": int(transform_count),
        "source_ct_count": int(bench._source_count(module, path_kind="provider")),
        "sharing_group_count": int(len(per_group)),
        "distinct_bsgs_schedule_count": int(len(set(n1s))),
        "distinct_bsgs_n1s": sorted(set(int(value) for value in n1s)),
        "source_shared_rotation_eval_count": int(source_shared_total),
        "oracle_rotation_eval_count": int(source_shared_total + recovery_rotations),
        "recovery_rotation_eval_count": int(recovery_rotations),
        "individual_rotation_eval_count": int(individual_total + recovery_rotations),
        "hybrid_pair_count": int(getattr(executor, "hybrid_pair_count", 0) or 0),
        "executor": str(role["executor"]),
        "base_executor": str(role["base_executor"]),
        "delegate_executor": str(role["delegate_executor"]),
        "provider_lowering": str(role["provider_lowering"]),
        "native_halo_programs": bool(role["native_halo_programs"]),
        "same_shape_spec_present": bool(role["same_shape_spec_present"]),
        "force_input_pair": bool(role["force_input_pair"]),
        "requires_compact_source": bool(role["requires_compact_source"]),
        "ct_pt_hybrid_packing": bool(role["ct_pt_hybrid_packing"]),
        "note": f"full provider path; {role['note_suffix']}",
        "per_group": per_group,
    }


def _describe_case(network: str, case_name: str) -> list[dict[str, Any]]:
    rows = _dense_orion_descriptors(str(network), str(case_name))
    rows.append(_provider_group_descriptors(str(network), str(case_name)))
    gc.collect()
    return rows


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "backend",
        "network",
        "network_label",
        "case",
        "node",
        "op",
        "path_family",
        "variant",
        "lt_tasks",
        "source_ct_count",
        "sharing_group_count",
        "distinct_bsgs_schedule_count",
        "distinct_bsgs_n1s",
        "source_shared_rotation_eval_count",
        "recovery_rotation_eval_count",
        "oracle_rotation_eval_count",
        "individual_rotation_eval_count",
        "hybrid_pair_count",
        "executor",
        "base_executor",
        "delegate_executor",
        "provider_lowering",
        "native_halo_programs",
        "same_shape_spec_present",
        "force_input_pair",
        "requires_compact_source",
        "ct_pt_hybrid_packing",
        "note",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, (list, dict, tuple)):
            value = json.dumps(value, separators=(",", ":"), sort_keys=True)
        out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument("--ckks-profile", choices=("e2e", "kernel"), default="e2e")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    os.environ[bench.CKKS_PROFILE_ENV] = str(args.ckks_profile)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    bench._require_backend("lattigo")

    rows: list[dict[str, Any]] = []
    for token in args.cases:
        if ":" not in str(token):
            raise ValueError(f"case must be network:case, got {token!r}")
        network, case_name = str(token).split(":", 1)
        rows.extend(_describe_case(str(network), str(case_name)))

    payload = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ckks_profile": str(args.ckks_profile),
        "cases": list(args.cases),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_out = Path(args.csv_out) if args.csv_out is not None else args.out.with_suffix(".csv")
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        flat = [_flatten_row(row) for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()) if flat else [])
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"status": "ok", "out": str(args.out), "csv_out": str(csv_out), "row_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
