from __future__ import annotations

import json

from orion.core.region_cir_replay import (
    build_r18_stage4_gs256_prototype_lattigo_microbench,
    build_r18_stage4_lattigo_microbench,
)


def main() -> int:
    baseline = build_r18_stage4_lattigo_microbench()
    prototype = build_r18_stage4_gs256_prototype_lattigo_microbench()
    payload = {
        "status": "ok" if baseline["status"] == "ok" and prototype["status"] == "ok" else "failed",
        "baseline": baseline,
        "prototype": prototype,
        "delta": {
            "rotations": int(prototype["stats_from_execution"]["rotations"]) - int(baseline["stats_from_execution"]["rotations"]),
            "ct_pt_mults": int(prototype["stats_from_execution"]["ct_pt_mults"]) - int(baseline["stats_from_execution"]["ct_pt_mults"]),
            "adds": int(prototype["stats_from_execution"]["adds"]) - int(baseline["stats_from_execution"]["adds"]),
            "compile_unified_s": float(prototype["timing_s"]["compile_unified_s"]) - float(baseline["timing_s"]["compile_unified_s"]),
            "evaluate_unified_s": float(prototype["timing_s"]["evaluate_unified_s"]) - float(baseline["timing_s"]["evaluate_unified_s"]),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
