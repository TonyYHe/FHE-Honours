"""C++ diagonal payload builder hooks.

The builder is an optional acceleration layer for host-side diagonal payload
materialization.  It deliberately returns the same payload ABI that the
existing Lattigo and clear-Lattigo bindings already consume.
"""

from .bindings import (
    DenseConv2DPayload,
    build_dense_conv2d_index_only,
    build_dense_conv2d_payloads,
    build_dense_conv2d_payloads_if_enabled,
    build_provider_compact_source_conv2d_payload,
    build_provider_native_source_conv2d_payload,
    dense_builder_enabled,
    load_library,
)

__all__ = [
    "DenseConv2DPayload",
    "build_dense_conv2d_index_only",
    "build_dense_conv2d_payloads",
    "build_dense_conv2d_payloads_if_enabled",
    "build_provider_compact_source_conv2d_payload",
    "build_provider_native_source_conv2d_payload",
    "dense_builder_enabled",
    "load_library",
]
