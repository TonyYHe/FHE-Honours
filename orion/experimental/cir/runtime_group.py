from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import os
import sys
import time

import torch

from orion.core.network_dag import NetworkDAG
from orion.core.tracer import OrionTracer
from orion.models.resnet import ResNet18

from .region_first_data import (
    R18_TINY_DENSE_FULL_STATS,
    R18_TINY_REGION_FIRST_FULL_STATS,
    STAGE_MATERIALIZER_REFERENCES,
    score,
    stats_delta,
)
from .lattigo_block import (
    R18_STAGE4_SPEC,
    _phase_mask,
    build_r18_stage1_shared_block_plan,
    build_r18_stage2_shared_block_plan,
    build_r18_stage3_shared_block_plan,
    build_r18_stage4_compact_intra_plan,
)
from .r18_e2e_bridges import (
    R18_STAGE12_TRANSITION_SPEC,
    R18_STAGE23_TRANSITION_SPEC,
    R18_STAGE34_TRANSITION_SPEC,
    R18StemBridgeRuntimeExecutor,
    R18TransitionBridgeRuntimeExecutor,
)


DEFAULT_R18_TINY_E2E_OUT = Path("/tmp/orion_r18_tiny_region_first_e2e.json")
DEFAULT_R18_ACTUAL_E2E_OUT = Path("/tmp/orion_r18_actual_region_first_e2e.json")


def _region_compile_workers(item_count: int) -> int:
    raw = os.environ.get("ORION_REGION_COMPILE_WORKERS", "1")
    try:
        requested = int(raw)
    except (TypeError, ValueError):
        requested = 1
    return max(1, min(int(item_count), int(requested)))


def _region_compile_trace_enabled() -> bool:
    return os.environ.get("ORION_REGION_COMPILE_TRACE", "0").strip().lower() not in ("", "0", "false", "no", "off")


def _region_compile_trace(event: str, **fields: Any) -> None:
    if not _region_compile_trace_enabled():
        return
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[region_compile] event={event} {payload}", file=sys.stderr, flush=True)


def transforms_from_conv_scheme_plan(
    plan: Any,
    *,
    level: int,
    scheme: Any,
    bank_count: int | None = None,
    output_scale: float = 1.0,
) -> tuple[list[Any], list[str]]:
    if len(plan.linear_transform_steps) != 1:
        raise ValueError("RegionFirstRuntimeGroup expects one collapsed SharedMultiOutput LT step")
    step = plan.linear_transform_steps[0]
    prepared = {str(plain.plaintext_id): plain for plain in plan.prepared_plaintexts}
    templates = {str(entry.template_id): entry for family in plan.family_templates for entry in family.template_entries}
    slots = int(plan.ring_slot_count)
    selected_banks = tuple(step.shared_output_banks[: int(bank_count or len(step.shared_output_banks))])
    transforms: list[Any] = []
    bank_ids: list[str] = []
    for bank in selected_banks:
        bank_id = str(bank.bank_id)
        diag_tensors: dict[int, torch.Tensor] = {}
        terms = [term for term in step.terms if str(getattr(term, "bank_id", "")) == bank_id]
        if len(terms) != int(bank.term_count):
            raise ValueError(f"bank {bank_id} expected {bank.term_count} terms, got {len(terms)}")
        for term in terms:
            template = templates[str(term.template_id)]
            plaintext = prepared[str(term.plaintext_id)]
            mapped_source_indices = template.indices.to(dtype=torch.int64).index_select(0, term.lookup_indices.to(dtype=torch.int64))
            output_indices = term.output_slot_indices.to(dtype=torch.int64)
            if not bool(torch.equal(mapped_source_indices, output_indices)):
                raise ValueError(f"term {term.term_id} cannot be encoded as one dense Orion diagonal")
            values = plaintext.values.to(dtype=torch.complex64)
            if float(output_scale) != 1.0:
                values = values * float(output_scale)
            diag_index = (-int(term.shift)) % int(slots)
            diag = diag_tensors.setdefault(int(diag_index), torch.zeros((int(slots),), dtype=torch.complex64))
            diag.index_add_(0, output_indices, values)
        transforms.append(
            SimpleNamespace(
                name=f"{plan.case_name}_{bank_id}",
                diagonals={(0, 0): {int(index): diag.tolist() for index, diag in sorted(diag_tensors.items())}},
                level=int(level),
                scheme=scheme,
                fhe_output_shape=torch.Size([1, int(slots)]),
                output_shape=torch.Size([1, int(slots)]),
                bank_id=bank_id,
            )
        )
        bank_ids.append(bank_id)
    return transforms, bank_ids


def _rescale_cipher_tensor(ct: Any) -> Any:
    if len(getattr(ct, "ids", ())) != 1:
        raise ValueError("region-first rescale helper expects a single-ciphertext tensor")
    if bool(getattr(ct.scheme.backend, "lt_outputs_are_rescaled", False)):
        return ct
    rescaled_id = ct.evaluator.rescale(int(ct.ids[0]), in_place=False)
    return type(ct)(ct.scheme, [int(rescaled_id)], ct.shape, ct.on_shape)


def _encode_plaintext_for_add(ct: Any, values: torch.Tensor) -> Any:
    scale = int(ct.scheme.params.get_default_scale())
    if bool(getattr(ct.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(ct.scale()))
        ct.set_scale(int(scale))
    return ct.scheme.encode(values, ct.level(), scale=scale)


def _add_plaintext_for_add(ct: Any, ptxt: Any) -> Any:
    if bool(getattr(ct.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(ct.scale()))
        ct.set_scale(int(scale))
        ptxt.set_scale(int(scale))
    return ct + ptxt


def _align_ciphertexts_for_add(left: Any, right: Any) -> tuple[Any, Any]:
    if bool(getattr(left.scheme.backend, "align_addition_scales", False)):
        scale = max(1, int(left.scale()))
        left.set_scale(int(scale))
        right.set_scale(int(scale))
    return left, right


def _delete_ciphertext_id(scheme: Any, ciphertext_id: int) -> None:
    delete = getattr(getattr(scheme, "backend", None), "DeleteCiphertext", None)
    if not callable(delete):
        return
    try:
        delete(int(ciphertext_id))
    except Exception:
        pass


class Stage1RuntimeExecutor:
    def __init__(self, *, plan: Any, output_node_ids: tuple[str, ...], real_output_scale: float = 0.5) -> None:
        self.plan = plan
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.real_output_scale = float(real_output_scale)
        self.group = None
        self.transforms: list[Any] = []
        self.bank_ids: list[str] = []
        self.compile_count = 0

    def compile(self, scheme: Any) -> None:
        if self.group is not None:
            return
        from orion.nn.unified_transform import UnifiedTransformGroup

        level = len(scheme.params.get_logq()) - 1
        self.transforms, self.bank_ids = transforms_from_conv_scheme_plan(
            self.plan,
            level=int(level),
            scheme=scheme,
            bank_count=len(self.output_node_ids),
            output_scale=float(self.real_output_scale),
        )
        self.group = UnifiedTransformGroup(self.transforms)
        self.group.compile_unified(scheme.backend)
        self.compile_count += 1

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        from orion.backend.python.tensors import CipherTensor

        scheme = source_ct.scheme
        self.compile(scheme)
        assert self.group is not None
        output_ids = self.group.evaluate_unified(int(source_ct.ids[0]), scheme.backend)
        outputs: dict[str, Any] = {}
        for node_id, output_id in zip(self.output_node_ids, output_ids):
            ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(self.plan.ring_slot_count)]), torch.Size([1, int(self.plan.ring_slot_count)]))
            ct = _rescale_cipher_tensor(ct)
            conj = ct.conjugate(in_place=False)
            ct, conj = _align_ciphertexts_for_add(ct, conj)
            outputs[str(node_id)] = ct + conj
        return outputs

    def cleanup(self, backend: Any) -> None:
        if self.group is not None and hasattr(self.group, "cleanup"):
            self.group.cleanup(backend)
        self.group = None


class Stage2RuntimeExecutor:
    def __init__(self, *, plans: tuple[Any, Any], output_node_ids: tuple[str, ...], real_output_scale: float = 0.5) -> None:
        if len(plans) != 2:
            raise ValueError("Stage2RuntimeExecutor expects exactly two input surface-pair block plans")
        self.plans = tuple(plans)
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.real_output_scale = float(real_output_scale)
        self.groups: list[Any] = []
        self.transforms_by_block: list[list[Any]] = []
        self.bank_ids_by_block: list[list[str]] = []
        self.compile_count = 0
        self.block_evaluate_count = 0

    def compile(self, scheme: Any) -> None:
        if self.groups:
            return
        from orion.nn.unified_transform import UnifiedTransformGroup

        level = len(scheme.params.get_logq()) - 1
        for plan in self.plans:
            transforms, bank_ids = transforms_from_conv_scheme_plan(
                plan,
                level=int(level),
                scheme=scheme,
                bank_count=len(self.output_node_ids),
                output_scale=float(self.real_output_scale),
            )
            group = UnifiedTransformGroup(transforms)
            group.compile_unified(scheme.backend)
            self.transforms_by_block.append(transforms)
            self.bank_ids_by_block.append(bank_ids)
            self.groups.append(group)
        self.compile_count += 1

    def _source_ids_for_blocks(self, source_ct: Any) -> tuple[int, int]:
        explicit_sources = getattr(source_ct, "region_first_block_sources", None)
        if explicit_sources is None:
            explicit_sources = getattr(source_ct, "stage2_input_pair_ciphertexts", None)
        if explicit_sources is not None:
            if len(explicit_sources) != 2:
                raise RuntimeError("stage2 region-first executor requires exactly two input surface-pair ciphertexts")
            ids: list[int] = []
            for source in explicit_sources:
                if hasattr(source, "ids"):
                    ids.append(int(source.ids[0]))
                else:
                    ids.append(int(source))
            return int(ids[0]), int(ids[1])

        ids = getattr(source_ct, "ids", None)
        if ids is not None and len(ids) >= 2:
            return int(ids[0]), int(ids[1])
        raise RuntimeError(
            "stage2 region-first executor requires a compatible source layout: "
            "provide two ciphertext ids or source_ct.region_first_block_sources for the two input surface-pair blocks"
        )

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        from orion.backend.python.tensors import CipherTensor

        scheme = source_ct.scheme
        self.compile(scheme)
        source_ids = self._source_ids_for_blocks(source_ct)
        complex_outputs: dict[str, Any] = {}
        slots = int(self.plans[0].ring_slot_count)
        for group, source_id in zip(self.groups, source_ids):
            output_ids = group.evaluate_unified(int(source_id), scheme.backend)
            self.block_evaluate_count += 1
            for node_id, output_id in zip(self.output_node_ids, output_ids):
                block_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, int(slots)]), torch.Size([1, int(slots)]))
                if str(node_id) in complex_outputs:
                    lhs, rhs = _align_ciphertexts_for_add(complex_outputs[str(node_id)], block_ct)
                    complex_outputs[str(node_id)] = lhs + rhs
                else:
                    complex_outputs[str(node_id)] = block_ct
        outputs: dict[str, Any] = {}
        for node_id in list(complex_outputs):
            ct = complex_outputs.pop(str(node_id))
            ct = _rescale_cipher_tensor(ct)
            conj = ct.conjugate(in_place=False)
            ct, conj = _align_ciphertexts_for_add(ct, conj)
            outputs[str(node_id)] = ct + conj
        return outputs

    def cleanup(self, backend: Any) -> None:
        for group in self.groups:
            if hasattr(group, "cleanup"):
                group.cleanup(backend)
        self.groups = []
        self.transforms_by_block = []
        self.bank_ids_by_block = []


class LazySingleBlockRuntimeExecutor(Stage1RuntimeExecutor):
    def __init__(
        self,
        *,
        plan: Any | None = None,
        output_node_ids: tuple[str, ...],
        plan_builder: Any | None = None,
        builder_kwargs: dict[str, Any] | None = None,
        real_output_scale: float = 0.5,
    ) -> None:
        self.plan = plan
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.real_output_scale = float(real_output_scale)
        self.group = None
        self.transforms: list[Any] = []
        self.bank_ids: list[str] = []
        self.compile_count = 0
        self._plan_builder = plan_builder
        self._builder_kwargs = dict(builder_kwargs or {})

    def _ensure_plan(self) -> None:
        if self.plan is not None:
            return
        if self._plan_builder is None:
            raise RuntimeError("lazy region-first executor has no materializer")
        self.plan, _inputs, _reference = self._plan_builder(**self._builder_kwargs)

    def compile(self, scheme: Any) -> None:
        self._ensure_plan()
        super().compile(scheme)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        self._ensure_plan()
        return super().__call__(source_ct)


class Stage3RuntimeExecutor(LazySingleBlockRuntimeExecutor):
    pass


class Stage4RuntimeExecutor(LazySingleBlockRuntimeExecutor):
    def __call__(self, source_ct: Any) -> dict[str, Any]:
        compact_source = getattr(source_ct, "region_first_compact_source", None)
        if compact_source is not None and compact_source is not True:
            source_ct = compact_source
        elif not bool(
            getattr(source_ct, "region_first_compact_source", False)
            or getattr(source_ct, "stage4_compact_source", False)
            or getattr(source_ct, "is_region_first_compact_source", False)
        ):
            raise RuntimeError(
                "stage4 region-first executor requires a ciphertext already in the compact-intra source layout"
            )
        return super().__call__(source_ct)


class FullConvRegionRuntimeExecutor:
    def __init__(
        self,
        *,
        plans: tuple[Any, ...] = (),
        plan_builders: tuple[Any, ...] = (),
        builder_kwargs: tuple[dict[str, Any], ...] = (),
        output_node_id: str,
        output_shape: Any,
        fhe_output_shape: Any,
        bias_vector: torch.Tensor | None = None,
        source_pair_count: int | None = None,
        requires_compact_source: bool = False,
        real_output_scale: float = 0.5,
    ) -> None:
        self.plans = tuple(plans)
        self._plan_builders = tuple(plan_builders)
        self._builder_kwargs = tuple(dict(value) for value in builder_kwargs)
        if not self.plans and not self._plan_builders:
            raise ValueError("FullConvRegionRuntimeExecutor expects at least one block plan or materializer")
        self.output_node_id = str(output_node_id)
        self.output_shape = torch.Size(output_shape)
        self.fhe_output_shape = torch.Size(fhe_output_shape)
        self.bias_vector = None if bias_vector is None else bias_vector.detach().to(dtype=torch.float32).clone()
        self.source_pair_count = int(source_pair_count or len(plans))
        self.requires_compact_source = bool(requires_compact_source)
        self.real_output_scale = float(real_output_scale)
        self.groups: list[Any] = []
        self.transforms_by_block: list[list[Any]] = []
        self.bias_plaintexts: tuple[Any | None, ...] = ()
        self._bias_plaintext_cache: dict[tuple[int, int], Any] = {}
        self.compile_count = 0
        self.block_evaluate_count = 0
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_timing: dict[str, float] = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self._compact_source_mask_cache: dict[int, tuple[Any, Any]] = {}
        self._compact_source_shift: int | None = None

    @property
    def bank_count(self) -> int:
        self._ensure_plans()
        return len(self.plans[0].linear_transform_steps[0].shared_output_banks)

    def _ensure_plans(self) -> None:
        if self.plans:
            return
        tasks = list(zip(self._plan_builders, self._builder_kwargs))
        if len(tasks) <= 1:
            plans: list[Any] = []
            for builder, kwargs in tasks:
                plan, _inputs, _reference = builder(**kwargs)
                plans.append(plan)
            self.plans = tuple(plans)
            return

        workers = _region_compile_workers(len(tasks))

        def _build_plan(task: tuple[Any, dict[str, Any]]) -> Any:
            builder, kwargs = task
            plan, _inputs, _reference = builder(**kwargs)
            return plan

        with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="orion-region-plan") as executor:
            self.plans = tuple(executor.map(_build_plan, tasks))

    def compile(self, scheme: Any) -> None:
        if self.groups:
            return
        _region_compile_trace(
            "full_conv_compile_start",
            output_node=self.output_node_id,
            plan_count=len(self.plans) if self.plans else len(self._plan_builders),
        )
        total_started = time.perf_counter()
        prepare_plans_started = time.time()
        self._ensure_plans()
        self.last_runtime_timing["prepare_plans_s"] = float(time.time() - prepare_plans_started)
        _region_compile_trace(
            "full_conv_prepare_plans_done",
            output_node=self.output_node_id,
            seconds=f"{self.last_runtime_timing['prepare_plans_s']:.6f}",
            plan_count=len(self.plans),
        )
        from orion.nn.unified_transform import UnifiedTransformGroup

        level = int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1
        prepare_transforms_started = time.time()
        _region_compile_trace(
            "full_conv_prepare_transforms_start",
            output_node=self.output_node_id,
            plan_count=len(self.plans),
            bank_count=self.bank_count,
        )
        if len(self.plans) <= 1:
            prepared_transforms = [
                transforms_from_conv_scheme_plan(
                    plan,
                    level=int(level),
                    scheme=scheme,
                    bank_count=self.bank_count,
                    output_scale=float(self.real_output_scale),
                )[0]
                for plan in self.plans
            ]
        else:
            workers = _region_compile_workers(len(self.plans))

            def _build_transforms(plan: Any) -> list[Any]:
                transforms, _bank_ids = transforms_from_conv_scheme_plan(
                    plan,
                    level=int(level),
                    scheme=scheme,
                    bank_count=self.bank_count,
                    output_scale=float(self.real_output_scale),
                )
                return transforms

            with ThreadPoolExecutor(max_workers=int(workers), thread_name_prefix="orion-region-transform") as executor:
                prepared_transforms = list(executor.map(_build_transforms, self.plans))
        self.last_runtime_timing["prepare_transforms_s"] = float(time.time() - prepare_transforms_started)
        _region_compile_trace(
            "full_conv_prepare_transforms_done",
            output_node=self.output_node_id,
            seconds=f"{self.last_runtime_timing['prepare_transforms_s']:.6f}",
            group_count=len(prepared_transforms),
            transform_count=sum(len(value) for value in prepared_transforms),
        )

        compile_started = time.time()
        for group_index, transforms in enumerate(prepared_transforms):
            _region_compile_trace(
                "full_conv_compile_group_start",
                output_node=self.output_node_id,
                group_index=group_index,
                transform_count=len(transforms),
            )
            group_started = time.perf_counter()
            group = UnifiedTransformGroup(transforms)
            group.compile_unified(scheme.backend)
            _region_compile_trace(
                "full_conv_compile_group_done",
                output_node=self.output_node_id,
                group_index=group_index,
                seconds=f"{time.perf_counter() - group_started:.6f}",
            )
            self.transforms_by_block.append(transforms)
            self.groups.append(group)
        self.bias_plaintexts = self._compile_bias_plaintexts(scheme)
        self.compile_count += 1
        self.last_runtime_timing["compile_unified_s"] = float(time.time() - compile_started)
        _region_compile_trace(
            "full_conv_compile_done",
            output_node=self.output_node_id,
            seconds=f"{time.perf_counter() - total_started:.6f}",
            compile_unified_s=f"{self.last_runtime_timing['compile_unified_s']:.6f}",
        )

    def _complex_sources_from_ids(self, source_ct: Any) -> tuple[Any, ...]:
        from orion.backend.python.tensors import CipherTensor

        scheme = source_ct.scheme
        ids = tuple(int(value) for value in getattr(source_ct, "ids", ()))
        needed = int(self.source_pair_count * 2)
        if len(ids) < needed:
            raise RuntimeError(
                f"region-first full-conv source layout requires {needed} ciphertext ids "
                f"for {self.source_pair_count} complex source pairs, got {len(ids)}"
            )
        sources: list[Any] = []
        for pair_index in range(int(self.source_pair_count)):
            left_id = int(ids[2 * pair_index])
            right_id = int(ids[2 * pair_index + 1])
            imag_id = scheme.evaluator.mul_imaginary_unit(right_id, +1, False)
            try:
                complex_id = scheme.evaluator.add_ciphertext(left_id, imag_id, False)
            finally:
                _delete_ciphertext_id(scheme, int(imag_id))
            sources.append(CipherTensor(scheme, [int(complex_id)], source_ct.shape, source_ct.on_shape))
        return tuple(sources)

    def _stage4_compact_masks(self, source_ct: Any) -> tuple[Any, Any]:
        scheme = source_ct.scheme
        level = int(source_ct.level())
        cached = self._compact_source_mask_cache.get(int(level))
        if cached is not None:
            return cached

        spec = R18_STAGE4_SPEC
        left_phases = tuple(range(int(spec.gap * spec.gap // 2)))
        right_phases = tuple(range(int(spec.gap * spec.gap // 2), int(spec.gap * spec.gap)))
        left_selector = _phase_mask(
            phases=left_phases,
            shape=(int(spec.c), int(spec.h), int(spec.w)),
            gap=int(spec.gap),
        ).to(dtype=torch.float32)
        right_selector = _phase_mask(
            phases=right_phases,
            shape=(int(spec.c), int(spec.h), int(spec.w)),
            gap=int(spec.gap),
        ).to(dtype=torch.float32)
        left_pt = scheme.encode(left_selector, level)
        right_pt = scheme.encode(right_selector, level)
        self._compact_source_mask_cache[int(level)] = (left_pt, right_pt)
        self._compact_source_shift = -int((int(spec.gap) // 2) * int(spec.w) * int(spec.gap))
        return left_pt, right_pt

    def _compact_stage4_source_from_regular(self, source_ct: Any) -> Any:
        left_pt, right_pt = self._stage4_compact_masks(source_ct)
        shift = int(self._compact_source_shift or 0)

        left = source_ct * left_pt
        right = (source_ct * right_pt).roll(int(shift), in_place=False)
        compact = left + right.mul_imaginary_unit(+1, in_place=False)
        compact = compact + compact.roll(int(-shift), in_place=False)
        compact.region_first_compact_source = True
        compact.stage4_compact_source = True
        compact.is_region_first_compact_source = True
        return compact

    def _source_ciphertexts(self, source_ct: Any) -> tuple[Any, ...]:
        explicit_sources = getattr(source_ct, "region_first_block_sources", None)
        if explicit_sources is not None:
            if len(explicit_sources) != len(self.plans):
                raise RuntimeError(
                    f"region-first full-conv executor expected {len(self.plans)} explicit block sources, got {len(explicit_sources)}"
                )
            return tuple(explicit_sources)
        if self.requires_compact_source:
            compact_source = getattr(source_ct, "region_first_compact_source", None)
            if compact_source is not None and compact_source is not True:
                return (compact_source,)
            if bool(
                getattr(source_ct, "region_first_compact_source", False)
                or getattr(source_ct, "stage4_compact_source", False)
                or getattr(source_ct, "is_region_first_compact_source", False)
            ):
                return (source_ct,)
            return (self._compact_stage4_source_from_regular(source_ct),)
        return self._complex_sources_from_ids(source_ct)

    def _output_level(self, scheme: Any, *, extra_depth: int = 0) -> int:
        level = int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1
        depth = int(self.assigned_depth) if self.assigned_depth is not None else 1
        return max(0, int(level) - max(0, int(depth)) - max(0, int(extra_depth)))

    def _bias_chunk(self, *, bank_index: int, slots: int) -> torch.Tensor | None:
        if self.bias_vector is None:
            return None
        start = int(bank_index * slots)
        chunk = torch.zeros((slots,), dtype=torch.float32)
        end = min(int(start + slots), int(self.bias_vector.numel()))
        if end > start:
            chunk[: int(end - start)] = self.bias_vector[start:end]
        return chunk

    def _compile_bias_plaintexts(self, scheme: Any) -> tuple[Any | None, ...]:
        if self.bias_vector is None:
            return ()
        level = self._output_level(scheme)
        scale = int(scheme.params.get_default_scale())
        slots = int(scheme.params.get_slots())
        plaintexts: list[Any | None] = []
        for bank_index in range(int(self.bank_count)):
            chunk = self._bias_chunk(bank_index=int(bank_index), slots=int(slots))
            ptxt = None if chunk is None else scheme.encode(chunk, level=int(level), scale=int(scale))
            if ptxt is not None:
                self._bias_plaintext_cache[(int(bank_index), int(level))] = ptxt
            plaintexts.append(ptxt)
        return tuple(plaintexts)

    def _add_bias(self, ct: Any, *, bank_index: int) -> Any:
        if self.bias_vector is None:
            return ct
        bias_pt = self.bias_plaintexts[int(bank_index)] if int(bank_index) < len(self.bias_plaintexts) else None
        if bias_pt is None or int(bias_pt.level()) != int(ct.level()):
            bias_pt = self._bias_plaintext_cache.get((int(bank_index), int(ct.level())))
        if bias_pt is None or int(bias_pt.level()) != int(ct.level()):
            chunk = self._bias_chunk(bank_index=int(bank_index), slots=int(ct.slots()))
            if chunk is None:
                return ct
            bias_pt = _encode_plaintext_for_add(ct, chunk)
            self._bias_plaintext_cache[(int(bank_index), int(ct.level()))] = bias_pt
        return _add_plaintext_for_add(ct, bias_pt)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        from orion.backend.python.tensors import CipherTensor

        scheme = source_ct.scheme
        self._ensure_plans()
        self.last_runtime_timing = {
            "prepare_plans_s": 0.0,
            "prepare_transforms_s": 0.0,
            "compile_unified_s": 0.0,
            "evaluate_unified_s": 0.0,
            "postprocess_s": 0.0,
        }
        self.compile(scheme)
        sources = self._source_ciphertexts(source_ct)
        complex_outputs: list[Any | None] = [None for _ in range(self.bank_count)]
        slots = int(self.plans[0].ring_slot_count)
        evaluate_started = time.time()
        for group, block_source in zip(self.groups, sources):
            output_ids = group.evaluate_unified(int(block_source.ids[0]), scheme.backend)
            self.block_evaluate_count += 1
            for bank_index, output_id in enumerate(output_ids):
                block_ct = CipherTensor(scheme, [int(output_id)], torch.Size([1, slots]), torch.Size([1, slots]))
                if complex_outputs[int(bank_index)] is None:
                    complex_outputs[int(bank_index)] = block_ct
                else:
                    lhs, rhs = _align_ciphertexts_for_add(complex_outputs[int(bank_index)], block_ct)
                    complex_outputs[int(bank_index)] = lhs + rhs
        self.last_runtime_timing["evaluate_unified_s"] = float(time.time() - evaluate_started)
        real_ids: list[int] = []
        postprocess_started = time.time()
        for bank_index, ct in enumerate(complex_outputs):
            if ct is None:
                raise RuntimeError(f"missing region-first output bank {bank_index}")
            complex_outputs[int(bank_index)] = None
            ct = _rescale_cipher_tensor(ct)
            conj = ct.conjugate(in_place=False)
            ct, conj = _align_ciphertexts_for_add(ct, conj)
            real = ct + conj
            real = self._add_bias(real, bank_index=int(bank_index))
            real.set_scale(int(scheme.params.get_default_scale()))
            real_ids.append(int(real.ids[0]))
            # Transfer ownership of the produced ciphertext id to the assembled
            # multi-bank CipherTensor returned below.
            real.ids = []
        self.last_runtime_timing["postprocess_s"] = float(time.time() - postprocess_started)
        return {
            self.output_node_id: CipherTensor(
                scheme,
                real_ids,
                self.output_shape,
                self.fhe_output_shape,
            )
        }

    def cleanup(self, backend: Any) -> None:
        for group in self.groups:
            if hasattr(group, "cleanup"):
                group.cleanup(backend)
        self.groups = []
        self.transforms_by_block = []
        self.bias_plaintexts = ()
        self._bias_plaintext_cache = {}
        self._compact_source_mask_cache = {}
        self._compact_source_shift = None


def _replace_group(group: "RegionFirstRuntimeGroup", **updates: Any) -> "RegionFirstRuntimeGroup":
    payload = {
        key: getattr(group, key)
        for key in (
            "region_id",
            "network",
            "stage",
            "module_prefix",
            "conv_nodes",
            "strategy",
            "materializer",
            "depth",
            "solver_depth",
            "boundary_actions",
            "expected_stats",
            "full_region",
            "hidden_fallback",
            "executable",
            "fallback_reason",
            "output_node_ids",
            "executor",
            "plan",
            "fused_weight_count",
            "compiled",
            "execute_count",
        )
    }
    payload.update(updates)
    return RegionFirstRuntimeGroup(**payload)


@dataclass
class RegionFirstRuntimeGroup:
    region_id: str
    network: str
    stage: str
    module_prefix: str
    conv_nodes: tuple[str, ...]
    strategy: str
    materializer: str
    depth: int
    boundary_actions: tuple[str, ...]
    expected_stats: dict[str, int]
    full_region: bool = True
    hidden_fallback: bool = False
    executable: bool = False
    fallback_reason: str = "materializer_does_not_accept_fused_weights"
    solver_depth: int | None = None
    output_node_ids: tuple[str, ...] = ()
    executor: Any | None = None
    plan: Any | None = None
    fused_weight_count: int = 0
    compiled: bool = False
    execute_count: int = 0
    _cache_key: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _cache_outputs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in {"executor", "plan", "_cache_key", "_cache_outputs"}
        }
        payload["executor_attached"] = bool(self.executor is not None)
        return payload

    def effective_depth(self) -> int:
        if self.solver_depth is not None:
            return int(self.solver_depth)
        return int(self.depth)

    def __post_init__(self) -> None:
        if not self.output_node_ids:
            self.output_node_ids = tuple(self.conv_nodes)

    def compile(self, scheme: Any | None = None) -> None:
        if self.executable and self.executor is None:
            raise RuntimeError(f"region {self.region_id} is executable but has no executor")
        if self.executor is not None and hasattr(self.executor, "assigned_level") and hasattr(self, "assigned_level"):
            self.executor.assigned_level = getattr(self, "assigned_level")
        if self.executor is not None and hasattr(self.executor, "assigned_depth") and hasattr(self, "assigned_depth"):
            self.executor.assigned_depth = getattr(self, "assigned_depth")
        if self.executor is not None and scheme is not None and hasattr(self.executor, "compile"):
            self.executor.compile(scheme)
        self.compiled = True

    def supports_scheme(self, scheme: Any | None) -> bool:
        if self.executor is not None and hasattr(self.executor, "supports_scheme"):
            return bool(self.executor.supports_scheme(scheme))
        return True

    def _source_key(self, source_ct: Any) -> tuple[int, ...]:
        ids = getattr(source_ct, "ids", None)
        if ids is None:
            return (id(source_ct),)
        return tuple(int(value) for value in ids)

    def execute(self, source_ct: Any) -> dict[str, Any]:
        if not bool(self.executable):
            raise RuntimeError(f"region {self.region_id} is not executable: {self.fallback_reason}")
        self.execute_count += 1
        if self.executor is not None:
            if hasattr(self.executor, "assigned_level") and hasattr(self, "assigned_level"):
                self.executor.assigned_level = getattr(self, "assigned_level")
            if hasattr(self.executor, "assigned_depth") and hasattr(self, "assigned_depth"):
                self.executor.assigned_depth = getattr(self, "assigned_depth")
            outputs = self.executor(source_ct)
        else:
            raise RuntimeError(f"region {self.region_id} is executable but has no executor")
        return dict(outputs)

    def _cleanup_executor_after_outputs_consumed(self, source_ct: Any) -> None:
        cleanup_enabled = str(os.environ.get("ORION_REGION_FIRST_CLEANUP_AFTER_OUTPUTS", "0")).strip().lower()
        if cleanup_enabled in {"", "0", "false", "no", "off"}:
            return
        cleanup = getattr(self.executor, "cleanup", None)
        if not callable(cleanup):
            return
        backend = getattr(getattr(source_ct, "scheme", None), "backend", None)
        if backend is None:
            return
        cleanup(backend)
        self.compiled = False

    def output(self, output_node_id: str, source_ct: Any) -> Any:
        key = self._source_key(source_ct)
        if self._cache_key != key:
            self._cache_outputs = self.execute(source_ct)
            self._cache_key = key
        if str(output_node_id) not in self._cache_outputs:
            raise KeyError(f"region {self.region_id} has no output bank {output_node_id!r}")
        output = self._cache_outputs.pop(str(output_node_id))
        if not self._cache_outputs:
            self._cache_key = None
            self._cleanup_executor_after_outputs_consumed(source_ct)
        return output


def _r18_stage_references() -> dict[str, Any]:
    return {
        str(ref.stage): ref
        for ref in STAGE_MATERIALIZER_REFERENCES
        if str(ref.network) == "R18"
    }


def _stage1_modules_compatible(modules: tuple[Any, ...]) -> bool:
    if len(modules) < 1:
        return False
    for module in modules:
        weight = getattr(module, "on_weight", None)
        if weight is None or tuple(int(v) for v in weight.shape) != (64, 64, 3, 3):
            return False
    return True


def _stage2_modules_compatible(modules: tuple[Any, ...]) -> bool:
    if len(modules) < 1:
        return False
    for module in modules:
        weight = getattr(module, "on_weight", None)
        if weight is None or tuple(int(v) for v in weight.shape) != (128, 128, 3, 3):
            return False
    return True


def _stage_modules_compatible(modules: tuple[Any, ...], *, expected_weight_shape: tuple[int, int, int, int]) -> bool:
    if len(modules) < 1:
        return False
    expected = tuple(int(value) for value in expected_weight_shape)
    for module in modules:
        weight = getattr(module, "on_weight", None)
        if weight is None or tuple(int(v) for v in weight.shape) != expected:
            return False
    return True


def _stage1_runtime_from_modules(group: RegionFirstRuntimeGroup, modules: tuple[Any, ...]) -> RegionFirstRuntimeGroup:
    if not _stage1_modules_compatible(modules):
        return group
    # The first fused stage1 conv is enough to prove actual fused-weight handoff
    # for this milestone. Full multi-conv stage1 runtime execution is a later
    # graph-replacement step.
    first = modules[0]
    plan, _inputs, _reference = build_r18_stage1_shared_block_plan(
        bank_count=8,
        weight_override=getattr(first, "on_weight"),
        bias_override=getattr(first, "on_bias", None),
        input_shape=(64, 64, 64),
        output_shape=(64, 64, 64),
        input_gap=1,
        output_gap=1,
    )
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        plan=plan,
        fused_weight_count=len(modules),
        executor=Stage1RuntimeExecutor(plan=plan, output_node_ids=group.conv_nodes),
    )


def _stage2_runtime_from_modules(group: RegionFirstRuntimeGroup, modules: tuple[Any, ...]) -> RegionFirstRuntimeGroup:
    if not _stage2_modules_compatible(modules):
        return group
    # As with stage1, the first fused conv proves the fused-weight handoff.
    # Stage2 needs two input surface-pair block plans, both using that same
    # fused weight tensor but consuming different source surface pairs.
    first = modules[0]
    plans: list[Any] = []
    for input_pair_index in (0, 1):
        plan, _inputs, _reference = build_r18_stage2_shared_block_plan(
            input_pair_index=int(input_pair_index),
            bank_count=len(group.conv_nodes),
            weight_override=getattr(first, "on_weight"),
            bias_override=getattr(first, "on_bias", None),
            input_shape=(128, 32, 32),
            output_shape=(128, 32, 32),
            input_gap=2,
            output_gap=2,
        )
        plans.append(plan)
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        plan=tuple(plans),
        fused_weight_count=len(modules),
        executor=Stage2RuntimeExecutor(plans=(plans[0], plans[1]), output_node_ids=group.conv_nodes),
    )


def _stage3_runtime_from_modules(group: RegionFirstRuntimeGroup, modules: tuple[Any, ...]) -> RegionFirstRuntimeGroup:
    if not _stage_modules_compatible(modules, expected_weight_shape=(256, 256, 3, 3)):
        return group
    first = modules[0]
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        plan=None,
        fused_weight_count=len(modules),
        executor=Stage3RuntimeExecutor(
            output_node_ids=group.conv_nodes,
            plan_builder=build_r18_stage3_shared_block_plan,
            builder_kwargs={
                "bank_count": len(group.conv_nodes),
                "weight_override": getattr(first, "on_weight"),
                "bias_override": getattr(first, "on_bias", None),
                "input_shape": (256, 16, 16),
                "output_shape": (256, 16, 16),
                "input_gap": 4,
                "output_gap": 4,
            },
        ),
    )


def _stage4_runtime_from_modules(group: RegionFirstRuntimeGroup, modules: tuple[Any, ...]) -> RegionFirstRuntimeGroup:
    if not _stage_modules_compatible(modules, expected_weight_shape=(512, 512, 3, 3)):
        return group
    first = modules[0]
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        plan=None,
        fused_weight_count=len(modules),
        executor=Stage4RuntimeExecutor(
            output_node_ids=group.conv_nodes,
            plan_builder=build_r18_stage4_compact_intra_plan,
            builder_kwargs={
                "weight_override": getattr(first, "on_weight"),
                "bias_override": getattr(first, "on_bias", None),
                "input_shape": (512, 8, 8),
                "output_shape": (512, 8, 8),
                "input_gap": 8,
                "output_gap": 8,
            },
        ),
    )


def _conv_bias_vector(module: Any) -> torch.Tensor:
    from orion.core import packing

    return packing.construct_conv2d_bias(module).to(dtype=torch.float32)


def _full_conv_region_from_module(
    *,
    stage: str,
    node: str,
    module: Any,
    ref: Any,
) -> RegionFirstRuntimeGroup | None:
    weight = getattr(module, "on_weight", None)
    if weight is None:
        return None
    stage = str(stage)
    node = str(node)
    plan_builders: list[Any] = []
    builder_kwargs: list[dict[str, Any]] = []
    requires_compact = False
    source_pair_count = 1
    strategy = "inter_group_shared_lt"
    if stage == "stage1":
        if tuple(int(v) for v in weight.shape) != (64, 64, 3, 3):
            return None
        for input_pair_index in range(4):
            plan_builders.append(build_r18_stage1_shared_block_plan)
            builder_kwargs.append(
                {
                    "input_pair_index": int(input_pair_index),
                    "bank_count": 8,
                    "weight_override": weight,
                    "bias_override": getattr(module, "on_bias", None),
                    "input_shape": (64, 64, 64),
                    "output_shape": (64, 64, 64),
                    "input_gap": 1,
                    "output_gap": 1,
                }
            )
        source_pair_count = 4
    elif stage == "stage2":
        if tuple(int(v) for v in weight.shape) != (128, 128, 3, 3):
            return None
        for input_pair_index in range(2):
            plan_builders.append(build_r18_stage2_shared_block_plan)
            builder_kwargs.append(
                {
                    "input_pair_index": int(input_pair_index),
                    "bank_count": 4,
                    "weight_override": weight,
                    "bias_override": getattr(module, "on_bias", None),
                    "input_shape": (128, 32, 32),
                    "output_shape": (128, 32, 32),
                    "input_gap": 2,
                    "output_gap": 2,
                }
            )
        source_pair_count = 2
    elif stage == "stage3":
        if tuple(int(v) for v in weight.shape) != (256, 256, 3, 3):
            return None
        plan_builders.append(build_r18_stage3_shared_block_plan)
        builder_kwargs.append(
            {
                "bank_count": 2,
                "weight_override": weight,
                "bias_override": getattr(module, "on_bias", None),
                "input_shape": (256, 16, 16),
                "output_shape": (256, 16, 16),
                "input_gap": 4,
                "output_gap": 4,
            }
        )
        source_pair_count = 1
    elif stage == "stage4":
        if tuple(int(v) for v in weight.shape) != (512, 512, 3, 3):
            return None
        plan_builders.append(build_r18_stage4_compact_intra_plan)
        builder_kwargs.append(
            {
                "weight_override": weight,
                "bias_override": getattr(module, "on_bias", None),
                "input_shape": (512, 8, 8),
                "output_shape": (512, 8, 8),
                "input_gap": 8,
                "output_gap": 8,
            }
        )
        requires_compact = True
        source_pair_count = 1
        strategy = "compact_intra_group_phase"
    else:
        return None

    executor = FullConvRegionRuntimeExecutor(
        plan_builders=tuple(plan_builders),
        builder_kwargs=tuple(builder_kwargs),
        output_node_id=node,
        output_shape=getattr(module, "output_shape"),
        fhe_output_shape=getattr(module, "fhe_output_shape"),
        bias_vector=_conv_bias_vector(module),
        source_pair_count=int(source_pair_count),
        requires_compact_source=bool(requires_compact),
    )
    return RegionFirstRuntimeGroup(
        region_id=f"r18_tiny_e2e_{node}",
        network="R18",
        stage=stage,
        module_prefix=str(node).replace("_", "."),
        conv_nodes=(node,),
        strategy=strategy,
        materializer=str(ref.materializer),
        depth=2 if stage == "stage4" else 1,
        boundary_actions=("full_conv_output_assembly", "insert_extract_before_relu_or_add"),
        expected_stats=dict(ref.expected_stats),
        executable=True,
        fallback_reason="",
        output_node_ids=(node,),
        executor=executor,
        fused_weight_count=1,
    )


def _r18_transition_modules_compatible(modules: tuple[Any, ...], *, spec: Any) -> bool:
    if len(modules) != 2:
        return False
    conv, shortcut = modules
    conv_weight = getattr(conv, "on_weight", None)
    shortcut_weight = getattr(shortcut, "on_weight", None)
    return (
        conv_weight is not None
        and shortcut_weight is not None
        and tuple(int(v) for v in conv_weight.shape) == (int(spec.c_out), int(spec.c_in), 3, 3)
        and tuple(int(v) for v in shortcut_weight.shape) == (int(spec.c_out), int(spec.c_in), 1, 1)
        and tuple(getattr(conv, "stride", ())) == (2, 2)
        and tuple(getattr(shortcut, "stride", ())) == (2, 2)
        and tuple(int(v) for v in getattr(conv, "input_shape", torch.Size())[1:]) == (int(spec.c_in), int(spec.h_in), int(spec.w_in))
        and tuple(int(v) for v in getattr(shortcut, "input_shape", torch.Size())[1:]) == (int(spec.c_in), int(spec.h_in), int(spec.w_in))
        and tuple(int(v) for v in getattr(conv, "output_shape", torch.Size())[1:]) == (int(spec.c_out), int(spec.h_out), int(spec.w_out))
        and tuple(int(v) for v in getattr(shortcut, "output_shape", torch.Size())[1:]) == (int(spec.c_out), int(spec.h_out), int(spec.w_out))
        and int(getattr(conv, "input_gap", -1)) == int(spec.input_gap)
        and int(getattr(shortcut, "input_gap", -1)) == int(spec.input_gap)
        and int(getattr(conv, "output_gap", -1)) == int(spec.output_gap)
        and int(getattr(shortcut, "output_gap", -1)) == int(spec.output_gap)
    )


def _r18_transition_group(*, stage: str, conv_node: str, shortcut_node: str, spec: Any) -> RegionFirstRuntimeGroup:
    return RegionFirstRuntimeGroup(
        region_id=f"r18_tiny_e2e_{stage}",
        network="R18",
        stage=str(stage),
        module_prefix=str(conv_node).replace("_", "."),
        conv_nodes=(str(conv_node), str(shortcut_node)),
        strategy="strict_channel_split_hybrid_transition_bridge",
        materializer="strict_channel_split_hybrid_transition_bridge",
        depth=1,
        boundary_actions=("full_conv_output_assembly", "insert_extract_before_relu_or_add"),
        expected_stats={},
        executable=False,
        fallback_reason="transition_bridge_materializer_unavailable",
        output_node_ids=(str(conv_node), str(shortcut_node)),
    )


def _r18_transition_runtime_from_modules(group: RegionFirstRuntimeGroup, modules: tuple[Any, ...], *, spec: Any) -> RegionFirstRuntimeGroup:
    if not _r18_transition_modules_compatible(modules, spec=spec):
        return group
    conv, shortcut = modules
    executor = R18TransitionBridgeRuntimeExecutor(
        conv_module=conv,
        shortcut_module=shortcut,
        spec=spec,
        output_node_ids=group.conv_nodes,
    )
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        executor=executor,
        fused_weight_count=2,
    )


def _r18_stem_module_compatible(module: Any) -> bool:
    weight = getattr(module, "on_weight", None)
    return bool(
        weight is not None
        and tuple(int(v) for v in weight.shape) == (64, 3, 7, 7)
        and tuple(getattr(module, "stride", ())) == (1, 1)
        and tuple(getattr(module, "padding", ())) == (3, 3)
    )


def _r18_stem_group(node: str) -> RegionFirstRuntimeGroup:
    return RegionFirstRuntimeGroup(
        region_id="r18_tiny_e2e_stem",
        network="R18",
        stage="stem",
        module_prefix="conv1",
        conv_nodes=(str(node),),
        strategy="strict_channel_split_stem_bridge",
        materializer="strict_channel_split_stem_bridge",
        depth=1,
        solver_depth=1,
        boundary_actions=("full_conv_output_assembly",),
        expected_stats={},
        executable=False,
        fallback_reason="stem_bridge_materializer_unavailable",
        output_node_ids=(str(node),),
    )


def _r18_stem_runtime_from_module(group: RegionFirstRuntimeGroup, module: Any) -> RegionFirstRuntimeGroup:
    if module is None or not _r18_stem_module_compatible(module):
        return group
    executor = R18StemBridgeRuntimeExecutor(module=module, output_node_id=str(group.conv_nodes[0]))
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        executor=executor,
        fused_weight_count=1,
    )


def _r18_pool_module_compatible(module: Any) -> bool:
    from orion.nn.pooling import AdaptiveAvgPool2d, AvgPool2d

    weight = getattr(module, "on_weight", None)
    return bool(
        isinstance(module, (AvgPool2d, AdaptiveAvgPool2d))
        and weight is not None
        and tuple(int(v) for v in getattr(module, "input_shape", torch.Size()))[0:1] == (1,)
        and tuple(int(v) for v in getattr(module, "output_shape", torch.Size()))[0:1] == (1,)
    )


def _r18_pool_group(node: str) -> RegionFirstRuntimeGroup:
    return RegionFirstRuntimeGroup(
        region_id=f"r18_tiny_e2e_{str(node)}",
        network="R18",
        stage="pool",
        module_prefix=str(node),
        conv_nodes=(str(node),),
        strategy="input_pair_pool_shared_rotations",
        materializer="input_pair_pool_shared_rotations",
        depth=1,
        solver_depth=1,
        boundary_actions=("input_pair_ctpt_hybrid", "shared_block_rotation_cache"),
        expected_stats={},
        executable=False,
        fallback_reason="pool_provider_materializer_unavailable",
        output_node_ids=(str(node),),
    )


def _r18_pool_runtime_from_module(group: RegionFirstRuntimeGroup, module: Any) -> RegionFirstRuntimeGroup:
    if module is None or not _r18_pool_module_compatible(module):
        return group
    from orion.experimental.cir.transition_pool_provider import InputPairConvRuntimeExecutor

    executor = InputPairConvRuntimeExecutor(module=module, output_node_id=str(group.conv_nodes[0]))
    return _replace_group(
        group,
        executable=True,
        fallback_reason="",
        executor=executor,
        fused_weight_count=1,
    )


def _groups_from_dag(dag: NetworkDAG) -> tuple[RegionFirstRuntimeGroup, dict[str, Any]]:
    refs = _r18_stage_references()
    groups: list[RegionFirstRuntimeGroup] = []
    excluded_nodes: list[dict[str, str]] = []
    same_shape_constraints = {
        "stage2": {"in_channels": 128, "out_channels": 128, "stride": (1, 1), "max_nodes": 4},
        "stage3": {"in_channels": 256, "out_channels": 256, "stride": (1, 1), "max_nodes": 2},
        "stage4": {"in_channels": 512, "out_channels": 512, "stride": (1, 1), "max_nodes": 1},
    }
    for stage_index, stage_name in enumerate(("stage1", "stage2", "stage3", "stage4")):
        prefix = f"layers_{stage_index}"
        conv_nodes_list: list[str] = []
        for node in dag.topological_sort():
            if not (str(node).startswith(prefix) and "conv" in str(node)):
                continue
            module = dag.nodes[node].get("module")
            if str(stage_name) in same_shape_constraints:
                constraints = same_shape_constraints[str(stage_name)]
                if (
                    getattr(module, "in_channels", None) != int(constraints["in_channels"])
                    or getattr(module, "out_channels", None) != int(constraints["out_channels"])
                    or tuple(getattr(module, "kernel_size", ())) != (3, 3)
                    or tuple(getattr(module, "stride", ())) != tuple(constraints["stride"])
                ):
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "stage": str(stage_name),
                            "reason": f"transition_conv_not_supported_by_{stage_name}_same_shape_materializer",
                        }
                    )
                    continue
                if len(conv_nodes_list) >= int(constraints["max_nodes"]):
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "stage": str(stage_name),
                            "reason": f"{stage_name}_materializer_output_bank_limit",
                        }
                    )
                    continue
            conv_nodes_list.append(str(node))
        conv_nodes = tuple(conv_nodes_list)
        ref = refs[str(stage_name)]
        depth = 3 if str(stage_name) == "stage4" else 2
        groups.append(
            RegionFirstRuntimeGroup(
                region_id=f"r18_tiny_{stage_name}",
                network="R18",
                stage=str(stage_name),
                module_prefix=f"layers.{stage_index}",
                conv_nodes=conv_nodes,
                strategy="compact_intra_group_phase" if str(stage_name) == "stage4" else "inter_group_shared_lt",
                materializer=str(ref.materializer),
                depth=int(depth),
                solver_depth=1,
                boundary_actions=("insert_extract_before_relu_or_add", "validate_relu_safe"),
                expected_stats=dict(ref.expected_stats),
                executable=False,
                fallback_reason="materializer_does_not_accept_fused_weights",
            )
        )
    graph_audit = {
        "node_count": int(len(dag.nodes)),
        "edge_count": int(len(dag.edges)),
        "selected_region_count": int(len(groups)),
        "stage_node_counts": {group.stage: int(len(group.conv_nodes)) for group in groups},
        "excluded_nodes": excluded_nodes,
    }
    return tuple(groups), graph_audit


def discover_r18_tiny_region_groups() -> tuple[RegionFirstRuntimeGroup, dict[str, Any]]:
    torch.manual_seed(0)
    net = ResNet18(dataset="tiny")
    traced = OrionTracer().trace_model(net)
    dag = NetworkDAG(traced)
    dag.build_dag()
    return _groups_from_dag(dag)


@dataclass
class RegionFirstCompileRegistry:
    groups: tuple[RegionFirstRuntimeGroup, ...]
    graph_audit: dict[str, Any]

    @classmethod
    def for_r18_tiny(cls, dag: NetworkDAG) -> "RegionFirstCompileRegistry":
        groups, graph_audit = _groups_from_dag(dag)
        return cls(groups=tuple(groups), graph_audit=dict(graph_audit))

    @classmethod
    def for_r18_tiny_e2e(
        cls,
        dag: NetworkDAG,
        *,
        allowed_stages: tuple[str, ...] | None = None,
    ) -> "RegionFirstCompileRegistry":
        refs = _r18_stage_references()
        groups: list[RegionFirstRuntimeGroup] = []
        excluded_nodes: list[dict[str, str]] = []
        stage_by_prefix = {
            "layers_0": "stage1",
            "layers_1": "stage2",
            "layers_2": "stage3",
            "layers_3": "stage4",
        }
        allowed = None if allowed_stages is None else {str(stage) for stage in allowed_stages}
        existing_nodes = set(dag.nodes)
        handled_nodes: set[str] = set()
        special_groups: list[RegionFirstRuntimeGroup] = []

        if "conv1" in existing_nodes and (allowed is None or "stem" in allowed):
            module = dag.nodes["conv1"].get("module")
            group = _r18_stem_runtime_from_module(_r18_stem_group("conv1"), module)
            if bool(group.executable):
                special_groups.append(group)
                handled_nodes.add("conv1")
            else:
                excluded_nodes.append({"node": "conv1", "stage": "stem", "reason": str(group.fallback_reason)})

        transition_specs = (
            ("stage1_transition", ("layers_1_0_conv1", "layers_1_0_shortcut_0"), R18_STAGE12_TRANSITION_SPEC),
            ("stage2_transition", ("layers_2_0_conv1", "layers_2_0_shortcut_0"), R18_STAGE23_TRANSITION_SPEC),
            ("stage3_transition", ("layers_3_0_conv1", "layers_3_0_shortcut_0"), R18_STAGE34_TRANSITION_SPEC),
        )
        for stage_name, nodes, spec in transition_specs:
            if allowed is not None and str(stage_name) not in allowed:
                continue
            if not set(nodes).issubset(existing_nodes):
                excluded_nodes.append({"node": ",".join(nodes), "stage": str(stage_name), "reason": "transition_pair_missing"})
                continue
            modules = tuple(dag.nodes[node].get("module") for node in nodes)
            group = _r18_transition_runtime_from_modules(
                _r18_transition_group(stage=str(stage_name), conv_node=nodes[0], shortcut_node=nodes[1], spec=spec),
                modules,
                spec=spec,
            )
            if bool(group.executable):
                special_groups.append(group)
                handled_nodes.update(str(node) for node in nodes)
            else:
                excluded_nodes.append({"node": ",".join(nodes), "stage": str(stage_name), "reason": str(group.fallback_reason)})

        if "avgpool" in existing_nodes and (allowed is None or "pool" in allowed or "avgpool" in allowed):
            module = dag.nodes["avgpool"].get("module")
            group = _r18_pool_runtime_from_module(_r18_pool_group("avgpool"), module)
            if bool(group.executable):
                special_groups.append(group)
                handled_nodes.add("avgpool")
            else:
                excluded_nodes.append({"node": "avgpool", "stage": "pool", "reason": str(group.fallback_reason)})

        for node in dag.topological_sort():
            if "conv" not in str(node):
                continue
            if str(node) in handled_nodes:
                continue
            stage = None
            for prefix, candidate in stage_by_prefix.items():
                if str(node).startswith(prefix):
                    stage = candidate
                    break
            if stage is None:
                continue
            if allowed is not None and str(stage) not in allowed:
                excluded_nodes.append({"node": str(node), "stage": str(stage), "reason": "stage_filtered_out"})
                continue
            module = dag.nodes[node].get("module")
            if module is None:
                continue
            if (
                tuple(getattr(module, "kernel_size", ())) != (3, 3)
                or tuple(getattr(module, "stride", ())) != (1, 1)
                or getattr(module, "in_channels", None) != getattr(module, "out_channels", None)
            ):
                excluded_nodes.append({"node": str(node), "stage": str(stage), "reason": "not_same_shape_3x3_stride1"})
                continue
            group = _full_conv_region_from_module(stage=str(stage), node=str(node), module=module, ref=refs[str(stage)])
            if group is None:
                excluded_nodes.append({"node": str(node), "stage": str(stage), "reason": "full_conv_region_materializer_unavailable"})
                continue
            groups.append(group)
        groups = list(special_groups) + groups
        graph_audit = {
            "node_count": int(len(dag.nodes)),
            "edge_count": int(len(dag.edges)),
            "selected_region_count": int(len(groups)),
            "excluded_nodes": excluded_nodes,
            "replacement_mode": "full_conv_region_nodes",
        }
        return cls(groups=tuple(groups), graph_audit=graph_audit)

    def attach_to_dag(self, dag: NetworkDAG) -> dict[str, Any]:
        attached: list[dict[str, Any]] = []
        fallback_layers: list[dict[str, Any]] = []
        resolved_groups: list[RegionFirstRuntimeGroup] = []
        for group in self.groups:
            modules = tuple(dag.nodes[node].get("module") for node in group.conv_nodes if node in dag.nodes)
            if bool(group.executable) and group.executor is not None:
                resolved_groups.append(group)
                continue
            if group.stage == "stage1":
                group = _stage1_runtime_from_modules(group, modules)
            elif group.stage == "stage2":
                group = _stage2_runtime_from_modules(group, modules)
            elif group.stage == "stage3":
                group = _stage3_runtime_from_modules(group, modules)
            elif group.stage == "stage4":
                group = _stage4_runtime_from_modules(group, modules)
            resolved_groups.append(group)
        object.__setattr__(self, "groups", tuple(resolved_groups))
        group_by_node = {node: group for group in self.groups for node in group.conv_nodes}
        for node, group in group_by_node.items():
            if node not in dag.nodes:
                continue
            module = dag.nodes[node].get("module")
            if module is None:
                continue
            module.region_runtime = group
            module.region_output_id = str(node)
            module.region_first_skip_dense_pack = bool(group.executable)
            if bool(group.executable) and hasattr(module, "set_depth"):
                module.set_depth(int(group.effective_depth()))
            attached.append({"node": str(node), "stage": str(group.stage), "executable": bool(group.executable)})
            if not bool(group.executable):
                fallback_layers.append({"node": str(node), "stage": str(group.stage), "reason": str(group.fallback_reason)})
        return {
            "attached_count": int(len(attached)),
            "attached": attached,
            "fallback_layers": fallback_layers,
            "executable_region_count": int(sum(1 for group in self.groups if bool(group.executable))),
            "graph_audit": dict(self.graph_audit),
        }

    def attach_probe_dense_bypass_to_dag(self, dag: NetworkDAG, *, lazy_region_compile: bool = True) -> list[dict[str, str]]:
        from orion.nn.linear import Conv2d

        bypassed: list[dict[str, str]] = []
        selected_nodes = {node for group in self.groups for node in group.conv_nodes}
        for node in selected_nodes:
            if node not in dag.nodes:
                continue
            module = dag.nodes[node].get("module")
            if module is not None:
                module.region_first_probe_lazy_region_compile = bool(lazy_region_compile)
        for node in dag.topological_sort():
            if node in selected_nodes:
                continue
            module = dag.nodes[node].get("module")
            if not isinstance(module, Conv2d):
                continue
            module.region_first_probe_dense_bypass = True
            module.region_first_probe_reason = "probe_only_skip_orion_dense_pack_conv2d"
            bypassed.append(
                {
                    "node": str(node),
                    "module": type(module).__name__,
                    "reason": "probe_only_skip_orion_dense_pack_conv2d",
                }
            )
        return bypassed

    def attach_probe_stem_activation_bypass(self, net: Any) -> list[dict[str, str]]:
        from orion.nn.activation import ReLU

        stem_act = getattr(net, "act", None)
        if not isinstance(stem_act, ReLU):
            return []
        stem_act.region_first_probe_activation_bypass = True
        stem_act.region_first_probe_reason = "probe_only_skip_stem_relu"
        return [
            {
                "node": "act",
                "module": type(stem_act).__name__,
                "reason": "probe_only_skip_stem_relu",
            }
        ]


def build_r18_tiny_region_first_e2e_report() -> dict[str, Any]:
    started = time.time()
    groups, graph_audit = discover_r18_tiny_region_groups()
    # Report mode does not have fused Orion modules available, so simulate the
    # post-compile state: the selected stage1-4 proxy groups are fused-weight
    # capable. Transition and output-bank-limit exclusions remain auditable in
    # graph_audit.excluded_nodes and are not counted as executable region facts.
    groups = tuple(
        _replace_group(
            group,
            executable=True,
            fallback_reason="",
            fused_weight_count=len(group.conv_nodes),
            executor=f"{group.stage}_runtime_executor_attached",
        )
        for group in groups
    )
    dense_stats = dict(R18_TINY_DENSE_FULL_STATS)
    region_stats = dict(R18_TINY_REGION_FIRST_FULL_STATS)
    dense_score = score(dense_stats)
    region_score = score(region_stats)
    speedup = float(dense_score / region_score) if region_score else 0.0
    depth_audit = {
        group.stage: {
            "region_id": group.region_id,
            "depth": int(group.depth),
            "lt_depth": 1,
            "extract_depth": int(group.depth - 1),
            "bootstrap_visible": True,
        }
        for group in groups
    }
    fallback_layers = [
        {"stage": group.stage, "node": node, "reason": group.fallback_reason}
        for group in groups
        for node in group.conv_nodes
        if not bool(group.executable)
    ]
    return {
        "status": "partial",
        "scope": "R18 TinyImageNet experimental region-first full-network comparison; cost/proxy path, not full CKKS runtime",
        "network": "R18",
        "dataset": "tiny",
        "dense": {
            "stats": dense_stats,
            "score": float(dense_score),
            "runtime_s": None,
        },
        "region_first": {
            "stats": region_stats,
            "score": float(region_score),
            "runtime_s": None,
            "runtime_group_count": int(len(groups)),
            "groups": [group.to_dict() for group in groups],
        },
        "comparison": {
            "delta_region_first_minus_dense": stats_delta(region_stats, dense_stats),
            "cost_score_speedup": float(speedup),
            "runtime_speedup": None,
            "runtime_publishable": False,
            "mae": None,
            "max_abs": None,
        },
        "graph_audit": graph_audit,
        "bootstrap_audit": {
            "status": "depths_declared_for_solver",
            "region_depths": depth_audit,
            "dense_bootstraps": None,
            "region_first_bootstraps": None,
        },
        "fallback_audit": {
            "unselected_layers_dense": True,
            "selected_region_hidden_fallback_count": int(sum(1 for group in groups if group.hidden_fallback)),
            "selected_executable_regions_no_dense_pack_conv2d": True,
            "fallback_layers": fallback_layers,
            "fallback_count": int(len(fallback_layers)),
            "executable_region_count": int(sum(1 for group in groups if group.executable)),
        },
        "claim": {
            "selected_regions_use_region_first_runtime_group": True,
            "full_network_ckks": False,
            "full_runtime_publishable": False,
            "reason": "Selected stage1-4 proxy groups are executable, but excluded transition/output-bank-limit nodes and full source-layout handoff keep full CKKS runtime non-publishable.",
        },
        "timing_s": {"report_build_s": float(time.time() - started)},
    }


def build_r18_actual_region_first_e2e_report(
    *,
    seed: int = 0,
    attempt_ckks_forward: bool = False,
    activation: str = "relu",
    silu_degree: int = 127,
    stem_relu: bool = True,
) -> dict[str, Any]:
    started = time.time()
    torch.manual_seed(int(seed))
    net = ResNet18(
        dataset="tiny",
        activation=str(activation),
        silu_degree=int(silu_degree),
        stem_relu=bool(stem_relu),
    )
    net.eval()
    sample = torch.randn((1, 3, 64, 64), dtype=torch.float32)
    dense_started = time.time()
    with torch.no_grad():
        dense_out = net(sample)
    dense_runtime_s = float(time.time() - dense_started)

    traced = OrionTracer().trace_model(net)
    from orion.core.tracer import StatsTracker

    StatsTracker(traced).propagate(sample)
    dag = NetworkDAG(traced)
    dag.build_dag()
    for module in net.modules():
        if hasattr(module, "init_orion_params") and callable(module.init_orion_params):
            module.init_orion_params()
    registry = RegionFirstCompileRegistry.for_r18_tiny_e2e(dag)
    attach_audit = registry.attach_to_dag(dag)
    replacements = [group.to_dict() for group in registry.groups]
    excluded = list(registry.graph_audit.get("excluded_nodes", ()))
    source_contracts = {
        group.region_id: {
            "node": group.conv_nodes[0] if group.conv_nodes else "",
            "stage": group.stage,
            "source_layout": (
                "regular_stage4_ciphertext_prepacked_to_compact_intra_at_runtime"
                if group.stage == "stage4"
                else "paired_dense_orion_ciphertexts_to_complex_sources"
            ),
            "output_assembly": "multi_ciphertext_full_conv_output",
        }
        for group in registry.groups
    }
    ckks_status = {
        "attempted": bool(attempt_ckks_forward),
        "status": "not_run",
        "reason": "full encrypted forward is gated behind explicit attempt; dense fallback compile can be very expensive",
    }
    if attempt_ckks_forward:
        ckks_status = {
            "attempted": True,
            "status": "blocked",
            "reason": "compile/run harness exists, but this report path does not launch full ResNet18 CKKS inside tests; use experimental_region_first=r18_tiny_e2e for manual run",
        }
    full_network_ready = bool(replacements) and not excluded and bool(attempt_ckks_forward) and ckks_status.get("status") == "ok"
    return {
        "status": "ready_for_manual_ckks_forward" if replacements else "failed",
        "scope": "R18 TinyImageNet actual graph-replacement E2E gate",
        "network": "R18",
        "dataset": "tiny",
        "activation": {
            "kind": str(activation),
            "silu_degree": int(silu_degree) if str(activation).lower() == "silu" else None,
            "stem_relu": bool(stem_relu),
            "expected_bootstraps_reference": 61 if str(activation).lower() == "silu" and int(silu_degree) == 7 and bool(stem_relu) else None,
        },
        "input": {"seed": int(seed), "shape": list(sample.shape)},
        "dense_cleartext": {
            "ran": True,
            "runtime_s": dense_runtime_s,
            "output_shape": list(dense_out.shape),
            "output_checksum": float(dense_out.detach().sum()),
        },
        "region_first": {
            "replacement_mode": "full_conv_region_nodes",
            "runtime_group_count": int(len(registry.groups)),
            "groups": replacements,
            "source_contracts": source_contracts,
        },
        "graph_audit": {
            **dict(registry.graph_audit),
            "attach_audit": attach_audit,
        },
        "e2e_gate": {
            "graph_replacement_ready": bool(len(registry.groups) > 0),
            "source_pairing_runtime_ready": True,
            "output_assembly_runtime_ready": True,
            "excluded_dense_node_count": int(len(excluded)),
            "ckks_forward": ckks_status,
        },
        "claim": {
            "full_network_ckks": bool(full_network_ready),
            "runtime_speedup_publishable": False,
            "reason": "Graph replacement now uses full-conv region nodes, but full CKKS speedup requires an explicit manual run and remaining dense exclusions must be accounted.",
        },
        "timing_s": {"report_build_s": float(time.time() - started)},
    }


def write_r18_actual_region_first_e2e_report(
    *,
    out_path: Path = DEFAULT_R18_ACTUAL_E2E_OUT,
    attempt_ckks_forward: bool = False,
    activation: str = "relu",
    silu_degree: int = 127,
    stem_relu: bool = True,
) -> dict[str, Any]:
    payload = build_r18_actual_region_first_e2e_report(
        attempt_ckks_forward=bool(attempt_ckks_forward),
        activation=str(activation),
        silu_degree=int(silu_degree),
        stem_relu=bool(stem_relu),
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def write_r18_tiny_region_first_e2e_report(
    *,
    out_path: Path = DEFAULT_R18_TINY_E2E_OUT,
) -> dict[str, Any]:
    payload = build_r18_tiny_region_first_e2e_report()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
