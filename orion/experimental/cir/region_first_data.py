from __future__ import annotations

from dataclasses import dataclass


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

