"""Opt-in exact WPC storage in Orion's unchanged dense cache transforms.

No CIPS layout, lossy periodicity, or resident fallback-diagonal bank is used.
This adapter retains only empty shells, identities and eligible encoded periods.
The original runtime builder still supplies full float32 payloads each time.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import time

import numpy as np


POLICY_ENV = "ORION_WPC_SELECTIVE_POLICY"
FIELDS = (
    "schema_version", "diagonal_count", "eligible_count", "compressed_count",
    "full_payload_bytes", "eligible_full_payload_bytes", "compressed_payload_bytes",
    "metadata_bytes", "offline_embed_calls", "online_embed_calls",
    "online_eligible_embed_calls", "materialization_count", "materialized_bytes",
    "peak_materialized_bytes", "last_prepare_nanoseconds", "last_encode_nanoseconds",
    "last_eligible_encode_nanoseconds", "last_decompress_nanoseconds",
    "total_prepare_nanoseconds", "total_encode_nanoseconds",
    "total_eligible_encode_nanoseconds", "total_decompress_nanoseconds",
)
APIS = (
    "GenerateWPCSelectiveLinearTransform", "MaterializeWPCSelectiveLinearTransform",
    "ReleaseWPCSelectiveLinearTransform", "VerifyWPCSelectiveLinearTransformExact",
    "GetWPCSelectiveLinearTransformStats",
)


def selective_policy() -> str | None:
    policy = os.environ.get(POLICY_ENV, "off").strip().lower()
    if policy in ("", "off", "0"):
        return None
    if policy not in ("online", "hybrid"):
        raise ValueError(f"{POLICY_ENV} must be off, online or hybrid")
    return policy


def _buffers(indices, data, slots):
    raw_indices = np.asarray(indices)
    if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu" or raw_indices.size == 0:
        raise ValueError("selective indices must be a nonempty integer vector")
    if np.any(raw_indices < -slots) or np.any(raw_indices >= slots):
        raise ValueError("selective index out of range")
    if len(set(map(lambda x: int(x) % slots, raw_indices))) != raw_indices.size:
        raise ValueError("duplicate normalized selective index")
    indices = np.ascontiguousarray(raw_indices, dtype=np.int32)
    data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    if data.size != indices.size * slots or not np.isfinite(data).all():
        raise ValueError("selective payload must contain finite full-slot float32 diagonals")
    return indices, data


def _args(indices, data):
    return (indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), int(indices.size),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(data.size))


class SelectiveOrionTransform:
    def __init__(self, backend, indices, data, *, slots, level, bsgs_ratio, policy):
        if policy not in ("online", "hybrid"):
            raise ValueError("selective policy must be online or hybrid")
        if any(not callable(getattr(backend, name, None)) for name in APIS):
            raise RuntimeError("selective storage requires a rebuilt real Lattigo library")
        if isinstance(slots, bool) or int(slots) < 1 or int(slots) & (int(slots) - 1):
            raise ValueError("selective slot count must be a positive power of two")
        if not np.isfinite(bsgs_ratio) or float(bsgs_ratio) <= 0:
            raise ValueError("selective BSGS ratio must be finite and positive")
        self.backend, self.slots, self.policy = backend, int(slots), policy
        indices, data = _buffers(indices, data, self.slots)
        digest = hashlib.sha256()
        for offset, key in enumerate(indices):
            digest.update(struct.pack("<I", int(key) % self.slots))
            digest.update(data[offset*self.slots:(offset+1)*self.slots].astype("<f4", copy=False).tobytes())
        self.payload_sha256 = digest.hexdigest()
        self.id = int(backend.GenerateWPCSelectiveLinearTransform(
            *_args(indices, data), int(level), float(bsgs_ratio), int(policy == "hybrid")))
        if self.id < 0:
            raise RuntimeError("selective compilation rejected the payload or encoded copy map")
        self.closed = False

    def stats(self):
        if self.closed:
            raise RuntimeError("selective transform is closed")
        values = list(self.backend.GetWPCSelectiveLinearTransformStats(self.id))
        if len(values) != len(FIELDS) or values[0] != 1:
            raise RuntimeError("unexpected selective backend statistics schema")
        return dict(zip(FIELDS, map(int, values)))

    def materialize(self, indices, data):
        if self.closed:
            raise RuntimeError("selective transform is closed")
        indices, data = _buffers(indices, data, self.slots)
        if int(self.backend.MaterializeWPCSelectiveLinearTransform(self.id, *_args(indices, data))) != 1:
            raise RuntimeError("selective materialization rejected changed payload identity or active state")

    def release(self):
        if not self.closed:
            self.backend.ReleaseWPCSelectiveLinearTransform(self.id)

    def close(self):
        if not self.closed:
            self.backend.DeleteLinearTransform(self.id)
            self.closed = True


def prepare_layer(evaluator, layer, *, level, bsgs_ratio):
    """Offline scan/Encode, bounded to one block if that builder is available."""
    policy = selective_policy()
    if policy is None:
        return
    if (not evaluator.single_slot_layer_cache_enabled() or evaluator.io_mode != "none"
            or os.environ.get("ORION_LATTIGO_CLEAR_BACKEND", "0").lower() not in ("", "0", "false", "off", "no")
            or os.environ.get("ORION_SINGLE_SLOT_ENCODE_WORKERS", "1") != "1"):
        raise RuntimeError("selective cache requires real FHE, io-mode none, single-slot cache and one Encode worker")
    if getattr(layer, "_wpc_selective_plans", None):
        raise RuntimeError("selective layer was already compiled")
    plans = {}
    started = time.perf_counter()
    try:
        blocks = sorted(layer._dense_layer_cache_diag_indices_by_block)
        block_builder = (callable(getattr(layer, "_dense_layer_cache_build_block_payloads", None))
                         or callable(getattr(layer, "_dense_layer_cache_build_block_diagonals", None)))
        batches = (evaluator._dense_layer_cache_build_payloads_for_blocks(layer, (key,)) for key in blocks) if block_builder else (
            evaluator._dense_layer_cache_build_payloads(layer),)
        for payloads in batches:
            for row, col, indices, data in payloads:
                key = (int(row), int(col))
                if key not in blocks or key in plans:
                    raise RuntimeError("selective compile builder returned unexpected blocks")
                plans[key] = SelectiveOrionTransform(evaluator.backend, indices, data,
                    slots=evaluator.params.get_slots(), level=level, bsgs_ratio=bsgs_ratio, policy=policy)
        if set(plans) != set(blocks):
            raise RuntimeError("selective compile builder omitted blocks")
    except BaseException:
        for plan in plans.values():
            plan.close()
        raise
    layer._wpc_selective_plans = plans
    layer._wpc_selective_policy = policy
    layer._wpc_selective_offline_s = time.perf_counter() - started


def materialize_payloads(evaluator, layer, payloads):
    results, materialized = [], []
    try:
        for row, col, indices, data in payloads:
            plan = layer._wpc_selective_plans[(int(row), int(col))]
            plan.materialize(indices, data)
            materialized.append(plan)
            results.append((int(row), int(col), indices, plan.id))
    except BaseException:
        for plan in materialized:
            plan.release()
        raise
    return results


def release_transform(evaluator, layer, transform_id):
    for plan in getattr(layer, "_wpc_selective_plans", {}).values():
        if plan.id == transform_id:
            plan.release()
            return
    remove = getattr(evaluator.backend, "RemovePlaintextDiagonals", None)
    if callable(remove):
        remove(int(transform_id))
    evaluator.backend.DeleteLinearTransform(int(transform_id))


def snapshot_model(model):
    """Outside-forward audit snapshot; covers only opted-in cache transforms."""
    rows, seen = [], set()
    for name, module in model.named_modules():
        owners = [(name, module)] + [(f"{name}/concat_input_{i}", proxy) for i, proxy in enumerate(
            getattr(module, "_concat_transform_sources_by_input", []) or [])]
        for owner_name, owner in owners:
            for block, plan in getattr(owner, "_wpc_selective_plans", {}).items():
                if plan.id in seen:
                    continue
                seen.add(plan.id)
                rows.append({"module": owner_name, "operator": type(module).__name__,
                    "block": list(block), "policy": plan.policy, "slot_payload_sha256": plan.payload_sha256,
                    "stats": plan.stats()})
    totals = {key: sum(row["stats"][key] for row in rows) for key in FIELDS
              if key not in ("schema_version", "peak_materialized_bytes") and not key.startswith("last_")}
    return {"schema_version": 1, "scope": "opted_in_dense_layer_cache_transforms_only",
            "transform_count": len(rows), "totals": totals, "transforms": rows,
            "peak_tracking_note": "per-transform maxima are not summed into a global allocation peak"}


def runtime_delta(before, after, *, forward_s):
    if before["transform_count"] != after["transform_count"]:
        raise ValueError("selective transforms changed during a forward")
    static_fields = ("schema_version", "diagonal_count", "eligible_count", "compressed_count",
        "full_payload_bytes", "eligible_full_payload_bytes", "compressed_payload_bytes",
        "metadata_bytes", "offline_embed_calls")
    def identity(snapshot):
        if len(snapshot["transforms"]) != snapshot["transform_count"]:
            raise ValueError("selective transform count disagrees with raw rows")
        identities = []
        for row in snapshot["transforms"]:
            if row["policy"] not in ("online", "hybrid"):
                raise ValueError("invalid selective snapshot policy")
            if any(type(row["stats"].get(key)) is not int or row["stats"][key] < 0 for key in FIELDS):
                raise ValueError("selective counters must be nonnegative integers")
            identities.append((row["module"], row["operator"], tuple(row["block"]),
                row["policy"], row["slot_payload_sha256"],
                tuple(row["stats"][key] for key in static_fields)))
        if len({(row[0], row[2]) for row in identities}) != len(identities):
            raise ValueError("duplicate selective transform identity")
        for key, value in snapshot["totals"].items():
            if value != sum(row["stats"][key] for row in snapshot["transforms"]):
                raise ValueError("selective totals disagree with raw transform rows")
        return sorted(identities)
    if before["scope"] != after["scope"] or identity(before) != identity(after):
        raise ValueError("selective transform identity changed during a forward")
    keys = ("online_embed_calls", "online_eligible_embed_calls", "materialization_count",
            "total_prepare_nanoseconds", "total_encode_nanoseconds",
            "total_eligible_encode_nanoseconds", "total_decompress_nanoseconds")
    delta = {key: after["totals"][key] - before["totals"][key] for key in keys}
    if any(value < 0 for value in delta.values()) or forward_s <= 0 or not np.isfinite(forward_s):
        raise ValueError("invalid selective runtime counter delta")
    if (delta["online_eligible_embed_calls"] > delta["online_embed_calls"]
            or delta["total_eligible_encode_nanoseconds"] > delta["total_encode_nanoseconds"]):
        raise ValueError("eligible counters exceed total Encode counters")
    encode = delta["total_encode_nanoseconds"] / 1e9
    eligible = delta["total_eligible_encode_nanoseconds"] / 1e9
    prepare = delta["total_prepare_nanoseconds"] / 1e9
    decompress = delta["total_decompress_nanoseconds"] / 1e9
    if encode + prepare + decompress > forward_s + max(1e-8, 1e-6*forward_s):
        raise ValueError("selective backend timers exceed the forward wall clock")
    return {"counter_delta": delta, "backend_encode_s": encode,
        "backend_prepare_s": prepare, "decompression_s": decompress,
        "backend_encode_pct_of_he_forward": 100*encode/forward_s,
        "baseline_eligible_encode_time_coverage_pct": (100*eligible/encode if encode else None)
            if all(row["policy"] == "online" for row in after["transforms"]) else None,
        "materialized_bytes_after_forward": after["totals"]["materialized_bytes"],
        "scope": after["scope"],
        "timing_note": "CKKS per-diagonal Embed timer; excludes Python recipe building, bias and FFI; not historical Step-1 Encode"}
