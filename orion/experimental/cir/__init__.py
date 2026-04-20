"""Vendored region-first CIR support for Orion experiments.

This package contains the minimal scripts/cir-derived data model and selected
R18/R34 selector/materializer surfaces needed by Orion's experimental
region-first report path. It is intentionally local to Orion and does not import
HaloED or mutate ``sys.path``.
"""

from .ir import (
    CanonicalTemplateEntry,
    ConvSchemePlan,
    ExecutionStats,
    FamilyTemplateBank,
    LinearTransformStep,
    LinearTransformTerm,
    PreparedPlaintext,
    SharedOutputBank,
    TensorRegion,
)
from .lattigo_block import build_r18_stage1_shared_block_plan

__all__ = [
    "CanonicalTemplateEntry",
    "ConvSchemePlan",
    "ExecutionStats",
    "FamilyTemplateBank",
    "LinearTransformStep",
    "LinearTransformTerm",
    "PreparedPlaintext",
    "SharedOutputBank",
    "TensorRegion",
    "build_r18_stage1_shared_block_plan",
]

