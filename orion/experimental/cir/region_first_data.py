from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


STATS_KEYS = ("rotations", "conjugations", "ct_pt_mults", "adds")
ORION_BASELINE = "orion_dense_bsgs_output_fold_lt"
REGION_FIRST = "generalized_inter_hsplit_full_native_shared_output_collapse"


@dataclass(frozen=True)
class RegionFirstFixture:
    model: str
    network: str
    region_id: str
    family: str
    representative: str
    candidate: str
    materializer: str
    orion_stats: dict[str, int]
    selected_stats: dict[str, int]
    parity: dict[str, float | bool]
    source_families: tuple[str, ...]


@dataclass(frozen=True)
class StageMaterializerReference:
    network: str
    stage: str
    family: str
    materializer: str
    expected_stats: dict[str, int]
    source: str
    status: Literal["ported", "missing_materializer"] = "missing_materializer"
    note: str = ""


R18_STAGE1_STAGE2 = RegionFirstFixture(
    model="resnet18_tiny_imagenet",
    network="R18",
    region_id="stage1_stage2_same_shape",
    family="stage1_stage2_same_shape",
    representative="stage1_same+stage2_same",
    candidate=REGION_FIRST,
    materializer=REGION_FIRST,
    orion_stats={"rotations": 7136, "conjugations": 0, "ct_pt_mults": 69840, "adds": 69840},
    selected_stats={"rotations": 572, "conjugations": 152, "ct_pt_mults": 34920, "adds": 35094},
    parity={"exact": True, "max_abs": 8.20159912109375e-05, "tolerance": 0.001},
    source_families=("stage1_same", "stage2_same"),
)


R34_STAGE1_STAGE2 = RegionFirstFixture(
    model="resnet34_imagenet",
    network="R34",
    region_id="stage1_stage2_same_shape",
    family="stage1_stage2_same_shape",
    representative="stage1_same_3x3_s1_gap4_to4+stage2_same_3x3_s1_gap8_to8",
    candidate=REGION_FIRST,
    materializer=REGION_FIRST,
    orion_stats={"rotations": 4558, "conjugations": 0, "ct_pt_mults": 120370, "adds": 120370},
    selected_stats={"rotations": 1928, "conjugations": 124, "ct_pt_mults": 48508, "adds": 48670},
    parity={"exact": True, "max_abs": 0.000125885009765625, "tolerance": 0.001},
    source_families=("stage1_same_3x3_s1_gap4_to4", "stage2_same_3x3_s1_gap8_to8"),
)


R18_R34_FIXTURES = (R18_STAGE1_STAGE2, R34_STAGE1_STAGE2)
EXCLUDED_SYNTHETIC_ROWS = ("R20:stage1_two_output_region",)


STAGE_MATERIALIZER_REFERENCES: tuple[StageMaterializerReference, ...] = (
    StageMaterializerReference(
        network="R18",
        stage="stage1",
        family="stage1_same",
        materializer="generalized_inter_hsplit_full_native_shared_output_collapse",
        expected_stats={"rotations": 20, "conjugations": 8, "ct_pt_mults": 1080, "adds": 1089},
        source="vendored R18 stage1 first input surface-pair full output-bank block",
        status="ported",
        note="original-size Lattigo proof covers all 8 output banks for block0",
    ),
    StageMaterializerReference(
        network="R18",
        stage="stage2",
        family="stage2_same",
        materializer="generalized_inter_hsplit_full_native_shared_output_collapse",
        expected_stats={"rotations": 84, "conjugations": 8, "ct_pt_mults": 5880, "adds": 5890},
        source="/tmp/region_first_end_to_end_orion_comparison.json per-instance stage2_same",
    ),
    StageMaterializerReference(
        network="R18",
        stage="stage3",
        family="stage3_same",
        materializer="inter_group_shared_lt",
        expected_stats={"rotations": 90, "conjugations": 2, "ct_pt_mults": 6750, "adds": 6753},
        source="/tmp/r18_shared_lt_selector.json stage3_same",
    ),
    StageMaterializerReference(
        network="R18",
        stage="stage4",
        family="stage4_same",
        materializer="compact_intra_group_phase",
        expected_stats={"rotations": 158, "conjugations": 1, "ct_pt_mults": 9767, "adds": 9768},
        source="/tmp/r18_shared_lt_selector.json stage4_same",
    ),
    StageMaterializerReference(
        network="R34",
        stage="stage1",
        family="stage1_same_3x3_s1_gap4_to4",
        materializer="generalized_inter_hsplit_full_native_shared_output_collapse",
        expected_stats={"rotations": 144, "conjugations": 16, "ct_pt_mults": 3600, "adds": 3620},
        source="/tmp/region_first_end_to_end_orion_comparison.json per-instance stage1_same_3x3_s1_gap4_to4",
    ),
    StageMaterializerReference(
        network="R34",
        stage="stage2",
        family="stage2_same_3x3_s1_gap8_to8",
        materializer="generalized_inter_hsplit_full_native_shared_output_collapse",
        expected_stats={"rotations": 152, "conjugations": 4, "ct_pt_mults": 3844, "adds": 3850},
        source="/tmp/region_first_end_to_end_orion_comparison.json per-instance stage2_same_3x3_s1_gap8_to8",
    ),
    StageMaterializerReference(
        network="R34",
        stage="stage3",
        family="stage3_same_3x3_s1_gap16_to16",
        materializer="inter_hsplit_native_shared_multi_output",
        expected_stats={"rotations": 204, "conjugations": 2, "ct_pt_mults": 7938, "adds": 7941},
        source="/tmp/r34_shared_lt_selected_families.json layer3_0_conv2_torch",
    ),
    StageMaterializerReference(
        network="R34",
        stage="stage4",
        family="stage4_same_3x3_s1_gap32_to32",
        materializer="missing_stage4_reference",
        expected_stats={"rotations": 0, "conjugations": 0, "ct_pt_mults": 0, "adds": 0},
        source="not locked in current vendored Orion reference table",
        note="requires reading/porting scripts/cir stage4 selected fixture",
    ),
)


def stats_sum(rows: list[dict[str, int]] | tuple[dict[str, int], ...]) -> dict[str, int]:
    return {key: sum(int(row.get(key, 0)) for row in rows) for key in STATS_KEYS}


def stats_delta(left: dict[str, int], right: dict[str, int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {key: int(left.get(key, 0)) - int(right.get(key, 0)) for key in STATS_KEYS}
    out["score"] = score(left) - score(right)
    return out


def score(stats: dict[str, int]) -> float:
    return (
        float(stats.get("rotations", 0)) * 8.0
        + float(stats.get("conjugations", 0)) * 8.0
        + float(stats.get("ct_pt_mults", 0))
        + float(stats.get("adds", 0)) * 0.05
    )


def summary(selected: dict[str, int], orion: dict[str, int]) -> dict[str, object]:
    delta = stats_delta(selected, orion)
    return {
        "ours": dict(selected),
        "orion": dict(orion),
        "delta_ours_minus_orion": dict(delta),
        "reduction_percent": {
            key: None if int(orion.get(key, 0)) == 0 else float(-int(delta[key]) * 100.0 / int(orion[key]))
            for key in ("rotations", "ct_pt_mults", "adds")
        },
        "score": {
            "ours": score(selected),
            "orion": score(orion),
            "delta": float(delta["score"]),
        },
    }
