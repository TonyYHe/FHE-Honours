from __future__ import annotations

from typing import Any

from orion.experimental.cir.transition_pool_provider import (
    BranchPairConvRuntimeExecutor,
    BranchPairNoHybridConvRuntimeExecutor,
    InputPairConvRuntimeExecutor,
)


class HaloLocalConvRuntimeExecutor:
    """Generic halo-local Conv2d/AvgPool provider facade.

    The public provider path is intentionally network-agnostic.  For the
    existing R34 same-shape kernels we delegate to the proven tile-local
    lowering internally, while direct Conv2d/AvgPool nodes use the shared
    input-pair provider.  This keeps R34/U22/U256 registries from depending on
    R34-specific executor classes.
    """

    kernel_kind = "halo_local_conv2d"
    use_ct_pt_hybrid_packing = True
    native_halo_input_capable = True
    native_halo_output_capable = True

    def __init__(
        self,
        *,
        module: Any,
        output_node_id: str,
        same_shape_spec: Any | None = None,
        force_input_pair: bool = False,
        use_real_imag_hybrid: bool = True,
        native_halo_input_capable: bool | None = None,
    ) -> None:
        self.module = module
        self.output_node_id = str(output_node_id)
        self.same_shape_spec = same_shape_spec
        self.force_input_pair = bool(force_input_pair)
        self.use_ct_pt_hybrid_packing = bool(use_real_imag_hybrid)
        self._native_halo_input_override = (
            None if native_halo_input_capable is None else bool(native_halo_input_capable)
        )
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.native_halo_input_capable = (
            bool(same_shape_spec is None or bool(force_input_pair))
            if self._native_halo_input_override is None
            else bool(self._native_halo_input_override)
        )
        self.native_halo_output_capable = bool(same_shape_spec is None or bool(force_input_pair))
        self.last_runtime_io: dict[str, Any] = {}
        self._delegate: Any | None = None

    @property
    def delegate(self) -> Any:
        if self._delegate is None:
            self._delegate = self._make_delegate()
        self._sync_delegate_assignment()
        return self._delegate

    def _make_delegate(self) -> Any:
        if self.same_shape_spec is not None and not bool(self.force_input_pair):
            from orion.experimental.cir.r34_orion_same_shape import (
                R34InterGroupHybridSameShapeRuntimeExecutor,
                R34NoHybridSameShapeRuntimeExecutor,
                R34Pack2SameShapeRuntimeExecutor,
            )

            policy = str(getattr(self.same_shape_spec, "policy", ""))
            if not bool(self.use_ct_pt_hybrid_packing):
                delegate = R34NoHybridSameShapeRuntimeExecutor(
                    module=self.module,
                    spec=self.same_shape_spec,
                    output_node_id=self.output_node_id,
                )
            elif policy == "inter_group_hybrid":
                delegate = R34InterGroupHybridSameShapeRuntimeExecutor(
                    module=self.module,
                    spec=self.same_shape_spec,
                    output_node_id=self.output_node_id,
                )
            else:
                delegate = R34Pack2SameShapeRuntimeExecutor(
                    module=self.module,
                    spec=self.same_shape_spec,
                    output_node_id=self.output_node_id,
                )
            # The R34 lowerer already materializes tile-local halo sections, but
            # it is not the row-offset native-halo path used by input-pair convs.
            self.native_halo_input_capable = False
            self.native_halo_output_capable = False
            return delegate
        self.native_halo_input_capable = (
            True if self._native_halo_input_override is None else bool(self._native_halo_input_override)
        )
        self.native_halo_output_capable = True
        return InputPairConvRuntimeExecutor(
            module=self.module,
            output_node_id=self.output_node_id,
            use_ct_pt_hybrid_packing=bool(self.use_ct_pt_hybrid_packing),
        )

    def _sync_delegate_assignment(self) -> None:
        if self._delegate is None:
            return
        if hasattr(self._delegate, "assigned_level"):
            self._delegate.assigned_level = self.assigned_level
        if hasattr(self._delegate, "assigned_depth"):
            self._delegate.assigned_depth = self.assigned_depth
        if hasattr(self._delegate, "output_shape") and hasattr(self.module, "output_shape"):
            self._delegate.output_shape = getattr(self.module, "output_shape")
        if hasattr(self._delegate, "fhe_output_shape") and hasattr(self.module, "fhe_output_shape"):
            self._delegate.fhe_output_shape = getattr(self.module, "fhe_output_shape")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def supports_scheme(self, scheme: Any | None) -> bool:
        return bool(getattr(self.delegate, "supports_scheme", lambda _scheme: True)(scheme))

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        payload = dict(metadata or {})
        delegate_payload = payload.get("delegate")
        if isinstance(delegate_payload, dict):
            payload = dict(delegate_payload)
        load = getattr(self.delegate, "load_compile_cache_metadata", None)
        if callable(load):
            load(payload)

    def compile_cache_metadata(self) -> dict[str, Any]:
        get_metadata = getattr(self.delegate, "compile_cache_metadata", None)
        delegate_metadata = dict(get_metadata()) if callable(get_metadata) else {}
        return {
            "kind": self.kernel_kind,
            "output_node_id": str(self.output_node_id),
            "delegate_kind": type(self.delegate).__name__,
            "native_halo_input_capable": bool(self.native_halo_input_capable),
            "use_ct_pt_hybrid_packing": bool(self.use_ct_pt_hybrid_packing),
            "delegate": delegate_metadata,
            **{
                key: value
                for key, value in delegate_metadata.items()
                if key not in {"kind"}
            },
        }

    def compile(self, scheme: Any) -> None:
        self._sync_delegate_assignment()
        self.delegate.compile(scheme)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        self._sync_delegate_assignment()
        outputs = dict(self.delegate(source_ct))
        base_io = dict(getattr(self.delegate, "last_runtime_io", {}) or {})
        self.last_runtime_io = {
            **base_io,
            "runtime_lowering": "halo_local_conv2d",
            "provider_executor": type(self).__name__,
            "delegate_executor": type(self.delegate).__name__,
            "native_halo_input_capable": bool(self.native_halo_input_capable),
            "use_ct_pt_hybrid_packing": bool(self.use_ct_pt_hybrid_packing),
        }
        return outputs

    def cleanup(self, backend: Any | None) -> None:
        cleanup = getattr(self.delegate, "cleanup", None)
        if callable(cleanup):
            cleanup(backend)


class HaloLocalBranchPairConvRuntimeExecutor:
    """Generic facade for transition Conv2d + shortcut Conv2d branch pairs."""

    kernel_kind = "halo_local_branch_pair_conv2d"
    native_halo_input_capable = False

    use_ct_pt_hybrid_packing = True

    def __init__(
        self,
        *,
        conv_module: Any,
        shortcut_module: Any,
        output_node_ids: tuple[str, str],
        use_real_imag_hybrid: bool = True,
    ) -> None:
        self.conv_module = conv_module
        self.shortcut_module = shortcut_module
        self.output_node_ids = tuple(str(value) for value in output_node_ids)
        self.use_ct_pt_hybrid_packing = bool(use_real_imag_hybrid)
        self.assigned_level: int | None = None
        self.assigned_depth: int | None = None
        self.last_runtime_io: dict[str, Any] = {}
        delegate_cls = (
            BranchPairConvRuntimeExecutor
            if bool(self.use_ct_pt_hybrid_packing)
            else BranchPairNoHybridConvRuntimeExecutor
        )
        self._delegate = delegate_cls(
            conv_module=conv_module,
            shortcut_module=shortcut_module,
            output_node_ids=self.output_node_ids,
        )

    def _sync_delegate_assignment(self) -> None:
        if hasattr(self._delegate, "assigned_level"):
            self._delegate.assigned_level = self.assigned_level
        if hasattr(self._delegate, "assigned_depth"):
            self._delegate.assigned_depth = self.assigned_depth

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def supports_scheme(self, scheme: Any | None) -> bool:
        return bool(self._delegate.supports_scheme(scheme))

    def load_compile_cache_metadata(self, metadata: dict[str, Any]) -> None:
        payload = dict(metadata or {})
        delegate_payload = payload.get("delegate")
        if isinstance(delegate_payload, dict):
            payload = dict(delegate_payload)
        self._delegate.load_compile_cache_metadata(payload)

    def compile_cache_metadata(self) -> dict[str, Any]:
        delegate_metadata = dict(self._delegate.compile_cache_metadata())
        return {
            "kind": self.kernel_kind,
            "delegate_kind": type(self._delegate).__name__,
            "use_ct_pt_hybrid_packing": bool(self.use_ct_pt_hybrid_packing),
            "delegate": delegate_metadata,
            **{
                key: value
                for key, value in delegate_metadata.items()
                if key not in {"kind"}
            },
        }

    def compile(self, scheme: Any) -> None:
        self._sync_delegate_assignment()
        self._delegate.compile(scheme)

    def __call__(self, source_ct: Any) -> dict[str, Any]:
        self._sync_delegate_assignment()
        outputs = dict(self._delegate(source_ct))
        base_io = dict(getattr(self._delegate, "last_runtime_io", {}) or {})
        self.last_runtime_io = {
            **base_io,
            "runtime_lowering": "halo_local_branch_pair_conv2d",
            "provider_executor": type(self).__name__,
            "delegate_executor": type(self._delegate).__name__,
            "use_ct_pt_hybrid_packing": bool(self.use_ct_pt_hybrid_packing),
        }
        return outputs

    def cleanup(self, backend: Any | None) -> None:
        self._delegate.cleanup(backend)
