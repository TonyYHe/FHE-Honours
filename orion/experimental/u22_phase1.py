from __future__ import annotations

import math
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any

import torch

from orion.core import packing
from orion.experimental.cir.lattigo_block import _slot_index
from orion.experimental.cir.runtime_group import RegionFirstRuntimeGroup
from orion.nn.linear import ConvTranspose2d
from orion.nn.unified_transform import UnifiedTransformGroup


def _u22_tconv_module_supported(module: Any) -> bool:
    input_gap = int(getattr(module, "input_gap", -1))
    output_gap = int(getattr(module, "output_gap", -1))
    return bool(
        isinstance(module, ConvTranspose2d)
        and tuple(getattr(module, "kernel_size", ())) == (2, 2)
        and tuple(getattr(module, "stride", ())) == (2, 2)
        and tuple(getattr(module, "padding", ())) == (0, 0)
        and tuple(getattr(module, "output_padding", ())) == (0, 0)
        and tuple(getattr(module, "dilation", ())) == (1, 1)
        and int(getattr(module, "groups", 1)) == 1
        and int(input_gap) >= 2
        and int(input_gap) % 2 == 0
        and int(output_gap) * 2 == int(input_gap)
    )


class TconvK2S2PythonRuntimeExecutor:
    kernel_kind = "tconv_k2s2_gap_halving_experimental"

    def __init__(self, *, module: Any, output_node_id: str) -> None:
        if not _u22_tconv_module_supported(module):
            raise ValueError("U22 experimental tconv kernel only supports k=2, s=2, gap-halving ConvTranspose2d layers")
        self.module = module
        self.output_node_id = str(output_node_id)
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.groups: list[UnifiedTransformGroup] = []
        self.output_block_count = 0
        self.input_block_count = 0
        self._bias_vector: torch.Tensor | None = None
        self._bias_ptxt_cache: dict[tuple[int, int, float], Any] = {}
        self._compiled = False

    def supports_scheme(self, scheme: Any | None) -> bool:
        if scheme is None:
            return False
        backend = str(getattr(getattr(scheme, "params", None), "get_backend", lambda: "")())
        if backend not in {"python", "lattigo"}:
            return False
        slots = int(scheme.params.get_slots())
        input_plane = int(torch.Size(getattr(self.module, "fhe_input_shape"))[2:].numel())
        output_plane = int(torch.Size(getattr(self.module, "fhe_output_shape"))[2:].numel())
        return int(max(input_plane, output_plane)) <= int(slots)

    def _block_layout(self, *, scheme: Any) -> dict[str, int]:
        slots = int(scheme.params.get_slots())
        on_ci = int(getattr(self.module, "fhe_input_shape")[1])
        on_hi = int(getattr(self.module, "fhe_input_shape")[2])
        on_wi = int(getattr(self.module, "fhe_input_shape")[3])
        on_co = int(getattr(self.module, "fhe_output_shape")[1])
        on_ho = int(getattr(self.module, "fhe_output_shape")[2])
        on_wo = int(getattr(self.module, "fhe_output_shape")[3])
        input_plane = int(on_hi * on_wi)
        output_plane = int(on_ho * on_wo)
        input_channels_per_block = max(1, int(slots // input_plane))
        output_channels_per_block = max(1, int(slots // output_plane))
        return {
            "slots": int(slots),
            "input_channels_per_block": int(input_channels_per_block),
            "output_channels_per_block": int(output_channels_per_block),
            "input_block_count": int(math.ceil(int(on_ci) / int(input_channels_per_block))),
            "output_block_count": int(math.ceil(int(on_co) / int(output_channels_per_block))),
        }

    def compile(self, scheme: Any) -> None:
        if self._compiled:
            return
        if not self.supports_scheme(scheme):
            raise RuntimeError("U22 experimental tconv kernel requires a compatible Python or Lattigo backend")
        level = int(self.assigned_level) if self.assigned_level is not None else len(scheme.params.get_logq()) - 1
        transforms_by_source_block = self._build_transforms_by_source_block(scheme=scheme, level=int(level))
        self.groups = []
        for transforms in transforms_by_source_block:
            group = UnifiedTransformGroup(transforms)
            group.compile_unified(scheme.backend)
            self.groups.append(group)
        self._compiled = True

    def _build_transforms_by_source_block(self, *, scheme: Any, level: int) -> list[list[Any]]:
        c_in = int(getattr(self.module, "input_shape")[1])
        h_in = int(getattr(self.module, "input_shape")[2])
        w_in = int(getattr(self.module, "input_shape")[3])
        c_out = int(getattr(self.module, "output_shape")[1])
        h_out = int(getattr(self.module, "output_shape")[2])
        w_out = int(getattr(self.module, "output_shape")[3])
        input_gap = int(getattr(self.module, "input_gap"))
        weight = getattr(self.module, "on_weight").detach().to(dtype=torch.float32)
        output_gap = int(getattr(self.module, "output_gap"))
        layout = self._block_layout(scheme=scheme)
        slots = int(layout["slots"])
        self.input_block_count = int(layout["input_block_count"])
        self.output_block_count = int(layout["output_block_count"])

        diagonal_entries: list[list[dict[int, dict[int, float]]]] = [
            [
                {}
                for _output_block in range(int(self.output_block_count))
            ]
            for _input_block in range(int(self.input_block_count))
        ]
        for ic in range(int(c_in)):
            for ih in range(int(h_in)):
                for iw in range(int(w_in)):
                    src_slot = int(_slot_index(int(ic), int(ih), int(iw), h=int(h_in), w=int(w_in), gap=int(input_gap)))
                    source_block = int(src_slot // int(slots))
                    source_local = int(src_slot % int(slots))
                    for kh in range(2):
                        oh = int(ih * 2 + kh)
                        for kw in range(2):
                            ow = int(iw * 2 + kw)
                            for oc in range(int(c_out)):
                                coeff = float(weight[int(ic), int(oc), int(kh), int(kw)])
                                if coeff == 0.0:
                                    continue
                                out_slot = int(_slot_index(int(oc), int(oh), int(ow), h=int(h_out), w=int(w_out), gap=int(output_gap)))
                                output_block = int(out_slot // int(slots))
                                output_local = int(out_slot % int(slots))
                                diag_idx = (int(source_local) - int(output_local)) % int(slots)
                                row = diagonal_entries[int(source_block)][int(output_block)].setdefault(int(diag_idx), {})
                                row[int(output_local)] = float(row.get(int(output_local), 0.0) + coeff)

        transforms_by_source_block: list[list[Any]] = []
        for source_block in range(int(self.input_block_count)):
            block_transforms: list[Any] = []
            for output_block in range(int(self.output_block_count)):
                diagonals: dict[int, list[float]] = {}
                for diag_idx, slot_values in diagonal_entries[int(source_block)][int(output_block)].items():
                    diag = torch.zeros((int(slots),), dtype=torch.float32)
                    indices = torch.tensor(sorted(int(value) for value in slot_values.keys()), dtype=torch.int64)
                    values = torch.tensor([float(slot_values[int(index)]) for index in indices.tolist()], dtype=torch.float32)
                    diag.index_copy_(0, indices, values)
                    diagonals[int(diag_idx)] = diag.tolist()
                block_transforms.append(
                    SimpleNamespace(
                        name=f"{self.output_node_id}_experimental_tconv_src{int(source_block)}_out{int(output_block)}",
                        diagonals={(0, 0): diagonals or {0: [0.0] * int(slots)}},
                        level=int(level),
                        scheme=scheme,
                        fhe_output_shape=torch.Size([1, int(slots)]),
                        output_shape=torch.Size([1, int(slots)]),
                    )
                )
            transforms_by_source_block.append(block_transforms)
        return transforms_by_source_block

    def _bias_plaintext(self, *, scheme: Any, level: int, output_block: int):
        scale = int(scheme.params.get_default_scale())
        cache_key = (int(output_block), int(level), int(scale))
        cached = self._bias_ptxt_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._bias_vector is None:
            self._bias_vector = packing.construct_conv_transpose2d_bias(self.module).to(dtype=torch.float32)
        slots = int(scheme.params.get_slots())
        start = int(output_block) * int(slots)
        stop = min(int(start + int(slots)), int(self._bias_vector.numel()))
        chunk = torch.zeros((int(slots),), dtype=torch.float32)
        if int(stop) > int(start):
            chunk[: int(stop - start)] = self._bias_vector[int(start) : int(stop)]
        ptxt = scheme.encode(chunk, level=int(level), scale=int(scale))
        self._bias_ptxt_cache[cache_key] = ptxt
        return ptxt

    def _assemble_output(self, output_ids: list[int], *, scheme: Any):
        from orion.backend.python.tensors import CipherTensor

        block_ids: list[int] = []
        for output_block, output_id in enumerate(output_ids):
            block_ct = CipherTensor(
                scheme,
                [int(output_id)],
                torch.Size([1, int(scheme.params.get_slots())]),
                torch.Size([1, int(scheme.params.get_slots())]),
            )
            block_ct = block_ct + self._bias_plaintext(
                scheme=scheme,
                level=int(block_ct.level()),
                output_block=int(output_block),
            )
            block_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        return CipherTensor(
            scheme,
            block_ids,
            getattr(self.module, "output_shape"),
            getattr(self.module, "fhe_output_shape"),
        )

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        scheme = source_ct.scheme
        self.compile(scheme)
        ids = [int(value) for value in getattr(source_ct, "ids", ())]
        if len(ids) < int(self.input_block_count):
            raise RuntimeError(
                "U22 experimental tconv kernel requires one ciphertext per packed input block: "
                f"expected {self.input_block_count}, got {len(ids)}"
            )
        if len(self.groups) != int(self.input_block_count):
            raise RuntimeError("U22 experimental tconv kernel was not compiled")
        accumulated: list[Any | None] = [None for _ in range(int(self.output_block_count))]
        for input_block, group in enumerate(self.groups):
            output_ids = group.evaluate_unified(int(ids[int(input_block)]), scheme.backend)
            for output_block, output_id in enumerate(output_ids):
                from orion.backend.python.tensors import CipherTensor

                block_ct = CipherTensor(
                    scheme,
                    [int(output_id)],
                    torch.Size([1, int(scheme.params.get_slots())]),
                    torch.Size([1, int(scheme.params.get_slots())]),
                )
                if accumulated[int(output_block)] is None:
                    accumulated[int(output_block)] = block_ct
                else:
                    accumulated[int(output_block)] = accumulated[int(output_block)] + block_ct
        final_ids: list[int] = []
        for output_block, block_ct in enumerate(accumulated):
            if block_ct is None:
                raise RuntimeError(f"U22 experimental tconv kernel missing output block {output_block}")
            final_ids.append(int(block_ct.ids[0]))
            block_ct.ids = []
        return {self.output_node_id: self._assemble_output(final_ids, scheme=scheme)}


@dataclass
class U22CompileRegistry:
    groups: tuple[RegionFirstRuntimeGroup, ...]
    graph_audit: dict[str, Any]

    @classmethod
    def for_dag(cls, dag) -> "U22CompileRegistry":
        groups: list[RegionFirstRuntimeGroup] = []
        excluded_nodes: list[dict[str, str]] = []
        for node in dag.topological_sort():
            module = dag.nodes[node].get("module")
            if isinstance(module, ConvTranspose2d):
                if not _u22_tconv_module_supported(module):
                    excluded_nodes.append(
                        {
                            "node": str(node),
                            "reason": "experimental_tconv_requires_k2s2_gap_halving",
                        }
                    )
                    continue
                groups.append(
                    RegionFirstRuntimeGroup(
                        region_id=f"u22_tconv_{node}",
                        network="U22",
                        stage="decoder_up",
                        module_prefix=str(node),
                        conv_nodes=(str(node),),
                        strategy="tconv_k2s2_gap_halving_experimental",
                        materializer="tconv_k2s2_gap_halving_experimental",
                        depth=2,
                        solver_depth=2,
                        boundary_actions=("packed_slot_gather", "phase_halving_output_repack"),
                        expected_stats={},
                        executable=True,
                        fallback_reason="",
                        output_node_ids=(str(node),),
                        executor=TconvK2S2PythonRuntimeExecutor(module=module, output_node_id=str(node)),
                        fused_weight_count=1,
                    )
                )
        return cls(
            groups=tuple(groups),
            graph_audit={
                "node_count": int(len(dag.nodes)),
                "edge_count": int(len(dag.edges)),
                "selected_region_count": int(len(groups)),
                "excluded_nodes": excluded_nodes,
            },
        )

    def attach_to_dag(self, dag) -> dict[str, Any]:
        attached: list[dict[str, Any]] = []
        for group in self.groups:
            for node in group.conv_nodes:
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
                attached.append(
                    {
                        "node": str(node),
                        "stage": str(group.stage),
                        "executable": bool(group.executable),
                    }
                )
        return {
            "attached_count": int(len(attached)),
            "attached": attached,
            "executable_region_count": int(sum(1 for group in self.groups if bool(group.executable))),
            "graph_audit": dict(self.graph_audit),
        }
