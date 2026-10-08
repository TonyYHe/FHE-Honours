"""Matched-function decoder control using Orion's native diagonal packers.

This is an experimental square-embedding control, not the automatic whole-model
compiler. Conv2d changes only the boundary function to flattened cyclic padding;
native channel-first/multiplexed packing, diagonal construction, BSGS and CKKS
evaluation remain Orion's. Online weights are regenerated one block at a time.
"""
from __future__ import annotations

import time
import ctypes
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from orion.backend.python.tensors import CipherTensor
from orion.core.packing import direct_diagonalize_conv2d, direct_diagonalize_conv_transpose2d
from orion.experimental.wpc_cips_checkpoint import WPCCIPSTrainedActivationBootstrap

SIGNATURE_ATTRIBUTE = "_orion_layout_control_signature"


def native_signature(shape, slots: int, gap: int = 1):
    n, channels, height, width = map(int, shape)
    slots, gap = int(slots), int(gap)
    area = height * width
    if n != 1 or gap not in (1, 2) or channels % (gap * gap):
        raise ValueError("native control requires N=1, gap 1/2 and complete multiplex groups")
    if min(channels, height, width, slots) <= 0 or slots % (area * gap * gap):
        raise ValueError("native control requires complete spatial planes per ciphertext")
    capacity = slots // area
    ranges = tuple((i, min(i + capacity, channels)) for i in range(0, channels, capacity))
    return (f"orion_multiplex_gap_{gap}", slots, channels, height, width, ranges)


def pack_native(values, signature):
    _, slots, channels, height, width, _ranges = signature
    gap = int(signature[0].rsplit("_", 1)[1])
    expected = native_signature((1, channels, height, width), slots, gap)
    if signature != expected:
        raise ValueError("invalid native packing signature")
    value = torch.tensor(np.asarray(values), dtype=torch.float64)
    if tuple(value.shape) == (channels, height, width):
        value = value[None]
    if tuple(value.shape) != (1, channels, height, width):
        raise ValueError("native input shape mismatch")
    flat = F.pixel_shuffle(value, gap).numpy().reshape(-1) if gap > 1 else value.numpy().reshape(-1)
    padded = np.zeros(len(signature[5]) * slots, dtype=np.float64)
    padded[:len(flat)] = flat
    return padded.reshape(-1, slots)


def unpack_native(messages, signature):
    _, slots, channels, height, width, _ranges = signature
    gap = int(signature[0].rsplit("_", 1)[1])
    native_signature((1, channels, height, width), slots, gap)
    values = np.asarray(messages, dtype=np.float64)
    if values.shape != (len(signature[5]), slots):
        raise ValueError("native decoded message shape mismatch")
    physical = torch.tensor(values.reshape(-1)[:channels * height * width].copy()).reshape(
        1, channels // (gap * gap), height * gap, width * gap)
    logical = F.pixel_unshuffle(physical, gap) if gap > 1 else physical
    return logical.numpy()[0]


def encrypt_native(scheme, values, signature, level):
    plaintext = scheme.encode(torch.tensor(pack_native(values, signature), dtype=torch.float32), level=int(level))
    try:
        result = scheme.encrypt(plaintext)
    finally:
        plaintext.release()
    setattr(result, SIGNATURE_ATTRIBUTE, signature)
    return result


class OrionLayoutTrainedActivationBootstrap(WPCCIPSTrainedActivationBootstrap):
    signature_attribute = SIGNATURE_ATTRIBUTE

    def _normalise_signature(self, signature, *, logical_shape):
        expected = native_signature(logical_shape, int(signature[1]), 1)
        if signature != expected:
            raise ValueError("native activation requires channel-first gap-one packing")
        return expected

    def _active_slot_mask(self):
        area = self.logical_shape[2] * self.logical_shape[3]
        masks = []
        for start, end in self.packing_signature[5]:
            mask = torch.zeros(self.slots, dtype=torch.bool)
            mask[:(end - start) * area] = True
            masks.append(mask)
        return torch.cat(masks)


class OrionLayoutConvPlan:
    """Explicit full/online policy over ordinary Orion native transforms."""

    def __init__(self, layer, input_shape, scheme, *, storage_mode="full", transpose=False):
        if storage_mode not in ("full", "online_encode"):
            raise ValueError("native control supports full and online_encode, not WPC compression")
        self.layer, self.scheme = layer, scheme
        self.layer_name, self.storage_mode = str(layer.name), storage_mode
        self.transpose = bool(transpose)
        expected_kernel, expected_stride, expected_padding = ((2, 2), (2, 2), (0, 0)) if transpose else ((3, 3), (1, 1), (1, 1))
        if (tuple(layer.kernel_size), tuple(layer.stride), tuple(layer.padding), tuple(layer.dilation), layer.groups) != (
                expected_kernel, expected_stride, expected_padding, (1, 1), 1):
            raise ValueError("native decoder control supports only 2x2 stride-two upsample or 3x3 same-shape convolution")
        if transpose and tuple(layer.output_padding) != (0, 0):
            raise ValueError("native upsample control requires output_padding=0")
        self.input_shape = tuple(map(int, input_shape))
        if self.input_shape[1] != layer.in_channels:
            raise ValueError("native layer input channels mismatch")
        n, _channels, height, width = self.input_shape
        self.output_shape = (n, layer.out_channels, height * (2 if transpose else 1), width * (2 if transpose else 1))
        self.level, self.output_level = int(layer.level), int(layer.level) - 1
        slots = scheme.params.get_slots()
        self.input_packing_signature = native_signature(self.input_shape, slots, 2 if transpose else 1)
        self.output_packing_signature = native_signature(self.output_shape, slots, 1)
        self.case = SimpleNamespace(slots=slots, input_group_count=len(self.input_packing_signature[5]),
                                    output_group_count=len(self.output_packing_signature[5]))
        layer.input_shape, layer.output_shape = torch.Size(self.input_shape), torch.Size(self.output_shape)
        layer.input_gap, layer.output_gap = (2 if transpose else 1), 1
        layer.fhe_input_shape = torch.Size((n, layer.in_channels // layer.input_gap**2,
                                          height * layer.input_gap, width * layer.input_gap))
        layer.fhe_output_shape = torch.Size(self.output_shape)
        layer.on_input_shape, layer.on_output_shape = layer.fhe_input_shape, layer.fhe_output_shape
        self.full_control_transform_ids, self.compressed_transform_ids = {}, {}
        self.transform_rows, self.runtime_rows, self.indices = {}, {}, {}
        self.record_sequence, self.last_evaluation = True, {}
        self.current_materialized_count = 0
        self.cleaned = False
        self.bias_plaintext = None
        self.bias_plaintext_payload_bytes = 0
        try:
            self._compile()
        except BaseException:
            self.cleanup()
            raise

    def _build(self, blocks=None):
        if self.transpose:
            diagonals, rotations = direct_diagonalize_conv_transpose2d(
                self.layer, self.case.slots, "square", False, allow_hybrid=False, allowed_blocks=blocks)
        else:
            diagonals, rotations = direct_diagonalize_conv2d(
                self.layer, self.layer.on_weight, self.case.slots, "square", False,
                allow_hybrid=False, allowed_blocks=blocks, padding_semantics="flattened_spatial_cyclic")
        if rotations:
            raise RuntimeError("square native control must not require output rotations")
        return diagonals

    def _encode(self, block):
        indices = sorted(block)
        data = np.concatenate([np.asarray(block[i], dtype=np.float32) for i in indices])
        return self._encode_data(indices, data)

    def _encode_data(self, indices, data):
        # The binding expands Python lists, not float32 ndarrays. Pass an
        # explicit pointer/length pair and keep its owned buffer alive here.
        return int(self.scheme.backend.GenerateLinearTransform(
            indices, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(data.size),
            self.level, float(self.layer.bsgs_ratio), "none"))

    def _compile(self):
        backend = self.scheme.backend
        if not hasattr(backend, "GetLinearTransformPayloadStats"):
            raise RuntimeError("native payload statistics API missing; rebuild Lattigo")
        diagonals = self._build()
        n = 2 * self.case.slots
        p_limbs = len(self.scheme.params.get_logp())
        for key, block in sorted(diagonals.items()):
            self.indices[key] = tuple(sorted(block))
            expected = len(block) * n * (self.level + 1 + p_limbs) * 8
            row = {"diagonal_count": len(block), "full_payload_bytes": expected,
                   "payload_bytes_measured": False, "actual_qp_stats": None, "exact_qp_match": None}
            self.transform_rows[f"out{key[0]}_in{key[1]}"] = row
            if self.storage_mode == "full":
                tid = self._encode(block)
                self.full_control_transform_ids[key] = tid
                version, count, _q, _p, total = map(int, backend.GetLinearTransformPayloadStats(tid))
                if version != 1 or count != len(block) or total != expected:
                    raise RuntimeError("native actual Q/P bytes differ from the limb estimate")
                row["payload_bytes_measured"] = True
                row["actual_qp_stats"] = {"schema_version": version, "diagonal_count": count,
                                          "q_bytes": _q, "p_bytes": _p, "total_bytes": total}
                self.scheme.lt_evaluator.generate_rotation_keys(tid)
            else:
                requests = self.scheme.lt_evaluator._plan_linear_transform_rotation_key_requests(
                    self.indices[key], level=self.level, bsgs_ratio=float(self.layer.bsgs_ratio))
                self.scheme.lt_evaluator.generate_rotation_key_requests(requests)
        self.case.transform_count = len(self.indices)
        del diagonals
        if self.layer.on_bias is not None:
            bias = np.broadcast_to(self.layer.on_bias.detach().cpu().numpy()[None, :, None, None], self.output_shape)
            messages = pack_native(bias, self.output_packing_signature)
            self.bias_plaintext = self.scheme.encode(torch.tensor(messages, dtype=torch.float32), level=self.output_level)
            self.bias_plaintext_payload_bytes = self.case.output_group_count * n * (self.output_level + 1) * 8

    def encrypt_input(self, source):
        return encrypt_native(self.scheme, source, self.input_packing_signature, self.level)

    def clear_reference(self, source):
        value = torch.tensor(np.asarray(source), dtype=torch.float64)
        weight = self.layer.on_weight.detach().cpu().to(torch.float64)
        bias = self.layer.on_bias.detach().cpu().to(torch.float64) if self.layer.on_bias is not None else None
        if self.transpose:
            return F.conv_transpose2d(value, weight, bias, stride=2).numpy()[0]
        flat = value.flatten(2)
        result = torch.zeros((1, self.layer.out_channels, flat.shape[-1]), dtype=torch.float64)
        for kh in range(3):
            for kw in range(3):
                selected = torch.roll(flat, -((kh - 1) * self.input_shape[3] + kw - 1), -1)
                result += torch.einsum("oi,nih->noh", weight[:, :, kh, kw], selected)
        result = result.reshape(self.output_shape)
        if bias is not None:
            result += bias[None, :, None, None]
        return result.numpy()[0]

    def evaluate(self, value):
        if self.cleaned or getattr(value, SIGNATURE_ATTRIBUTE, None) != self.input_packing_signature:
            raise ValueError("native plan input packing mismatch or plan already released")
        backend = self.scheme.backend
        if len(value.ids) != self.case.input_group_count or any(
            int(backend.GetCiphertextLevel(i)) != self.level for i in value.ids):
            raise ValueError("native plan input count/level mismatch")
        output_ids, sequence = [], []
        self.runtime_rows = {}
        full_time = 0.0
        try:
            for row in range(self.case.output_group_count):
                accumulator = None
                try:
                    for key in sorted(k for k in self.indices if k[0] == row):
                        online = self.storage_mode == "online_encode"
                        tid, partial = None, None
                        prep_s = encode_s = 0.0
                        try:
                            if online:
                                started = time.perf_counter()
                                block = self._build({key})[key]
                                # Include Python flattening in preparation, not Encode.
                                indices = sorted(block)
                                data = np.concatenate([np.asarray(block[i], dtype=np.float32) for i in indices])
                                prep_s = time.perf_counter() - started
                                started = time.perf_counter()
                                tid = self._encode_data(indices, data)
                                encode_s = time.perf_counter() - started
                                self.current_materialized_count = 1
                                del block, data
                            else:
                                tid = self.full_control_transform_ids[key]
                            started = time.perf_counter()
                            partial = int(backend.EvaluateLinearTransform(tid, int(value.ids[key[1]])))
                            evaluate_s = time.perf_counter() - started
                            full_time += evaluate_s
                            self.runtime_rows[f"out{key[0]}_in{key[1]}"] = {
                                "last_prepare_nanoseconds": round(prep_s * 1e9),
                                "last_encode_nanoseconds": round(encode_s * 1e9),
                                "last_evaluate_nanoseconds": round(evaluate_s * 1e9)}
                            if accumulator is None:
                                accumulator, partial = partial, None
                            else:
                                backend.AddCiphertext(accumulator, partial)
                        finally:
                            if partial is not None:
                                backend.DeleteCiphertext(partial)
                            if online and tid is not None:
                                backend.DeleteLinearTransform(tid)
                                self.current_materialized_count = 0
                        if self.record_sequence:
                            sequence.append({"block": list(key), "materialized_transforms_after": self.current_materialized_count})
                    if accumulator is None:
                        raise RuntimeError("native output block has no nonzero transforms")
                    rescaled = int(self.scheme.evaluator.rescale(accumulator, in_place=False))
                    output_ids.append(rescaled)
                    if self.bias_plaintext is not None:
                        backend.AddPlaintext(rescaled, int(self.bias_plaintext.ids[row]))
                finally:
                    if accumulator is not None:
                        backend.DeleteCiphertext(accumulator)
            output = CipherTensor(self.scheme, output_ids, torch.Size((len(output_ids), self.case.slots)))
            setattr(output, SIGNATURE_ATTRIBUTE, self.output_packing_signature)
            self.last_evaluation = {"evaluation_sequence": sequence, "full_transform_evaluate_call_s": full_time}
            return output
        except BaseException:
            for identifier in output_ids:
                backend.DeleteCiphertext(identifier)
            raise

    def decrypt_unpack(self, value):
        if getattr(value, SIGNATURE_ATTRIBUTE, None) != self.output_packing_signature:
            raise ValueError("native output packing mismatch")
        plaintext = self.scheme.decrypt(value)
        try:
            return unpack_native(np.asarray(self.scheme.decode(plaintext)), self.output_packing_signature)
        finally:
            plaintext.release()

    __call__ = evaluate

    def online_stats(self):
        return self.runtime_rows

    def compressed_stats(self):
        return {}

    def storage_summary(self):
        full = sum(row["full_payload_bytes"] for row in self.transform_rows.values())
        resident = full if self.storage_mode == "full" else 0
        recipes = self.layer.on_weight.numel() * self.layer.on_weight.element_size() if self.storage_mode == "online_encode" else 0
        metadata = 4 * sum(len(indices) for indices in self.indices.values()) + 16 * len(self.indices)
        stored = resident + recipes + metadata + self.bias_plaintext_payload_bytes
        return {"storage_mode": self.storage_mode, "transform_count": self.case.transform_count,
                "full_weight_qp_payload_bytes": full, "resident_weight_qp_payload_bytes": resident,
                "compressed_weight_qp_payload_bytes": 0, "unencoded_recipe_payload_bytes": recipes,
                "weight_metadata_bytes": metadata, "uncompressed_bias_q_payload_bytes": self.bias_plaintext_payload_bytes,
                "full_weight_plus_bias_payload_bytes": full + self.bias_plaintext_payload_bytes,
                "stored_weight_plus_metadata_plus_bias_bytes": stored,
                "payload_accounting_scope": "Q/P limb arrays; float32 kernel and int32 indices online; excludes Python/Go object overhead",
                "layer_plaintext_storage_compression_ratio": (full + self.bias_plaintext_payload_bytes) / stored}

    def cleanup(self):
        if self.cleaned:
            return
        self.cleaned = True
        for tid in self.full_control_transform_ids.values():
            self.scheme.backend.DeleteLinearTransform(tid)
        self.full_control_transform_ids = {}
        if self.bias_plaintext is not None:
            self.bias_plaintext.release()
            self.bias_plaintext = None


class OrionAlignedConcatPlan:
    def __init__(self, scheme, first, second, consumer):
        self.scheme, self.first, self.second, self.consumer = scheme, first, second, consumer
        if first.output_packing_signature[0] != "orion_multiplex_gap_1" or second.output_packing_signature != first.output_packing_signature:
            raise ValueError("native concat requires identical gap-one branch layouts")
        if first.output_shape[1] * first.output_shape[2] * first.output_shape[3] % scheme.params.get_slots():
            raise ValueError("native concat control currently requires ciphertext-aligned branches")
        expected = native_signature((1, 2 * first.output_shape[1], *first.output_shape[2:]), scheme.params.get_slots(), 1)
        if expected != consumer.input_packing_signature:
            raise ValueError("native concat consumer mismatch")

    def evaluate(self, first, second):
        ids = []
        try:
            for value, signature in ((first, self.first.output_packing_signature), (second, self.second.output_packing_signature)):
                if getattr(value, SIGNATURE_ATTRIBUTE, None) != signature:
                    raise ValueError("native concat input packing mismatch")
                if len(value.ids) != len(signature[5]):
                    raise ValueError("native concat input ciphertext count mismatch")
                for identifier in value.ids:
                    if self.scheme.backend.GetCiphertextLevel(identifier) != self.consumer.level + 1:
                        raise ValueError("native concat input level mismatch")
                    ids.append(int(self.scheme.backend.CopyWPCLayoutCiphertext(identifier, self.consumer.level)))
            output = CipherTensor(self.scheme, ids, torch.Size((len(ids), self.scheme.params.get_slots())))
            setattr(output, SIGNATURE_ATTRIBUTE, self.consumer.input_packing_signature)
            return output
        except BaseException:
            for identifier in ids:
                self.scheme.backend.DeleteCiphertext(identifier)
            raise

    def storage_summary(self):
        return {"full_qp_payload_bytes": 0, "transform_count": 0,
                "storage_mode": "ciphertext_aligned_native_concat_copy"}

    def cleanup(self):
        pass

    __call__ = evaluate
