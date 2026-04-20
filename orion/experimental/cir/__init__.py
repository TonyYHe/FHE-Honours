"""Vendored region-first CIR support for Orion experiments.

This package contains the minimal region-first data model and selected R18/R34
selector/materializer surfaces needed by Orion's experimental report path. It is
intentionally self-contained inside Orion.
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
from .lattigo_block import (
    build_r18_stage1_shared_block_plan,
    build_r18_stage2_shared_block_plan,
    build_r18_stage3_shared_block_plan,
    build_r18_stage4_compact_intra_plan,
)
from .report import build_region_first_pipeline_report, write_region_first_pipeline_report
from .selector import build_region_first_full_selector, build_region_first_full_selector_summary
from .stage_matrix import build_stage_materialization_lattigo_matrix, write_stage_materialization_lattigo_matrix

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
    "build_r18_stage2_shared_block_plan",
    "build_r18_stage3_shared_block_plan",
    "build_r18_stage4_compact_intra_plan",
    "build_region_first_full_selector",
    "build_region_first_full_selector_summary",
    "build_region_first_pipeline_report",
    "build_stage_materialization_lattigo_matrix",
    "write_region_first_pipeline_report",
    "write_stage_materialization_lattigo_matrix",
]
