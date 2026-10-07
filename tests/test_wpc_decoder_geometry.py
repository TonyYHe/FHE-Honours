from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion.experimental.wpc_decoder_geometry import (
    UNASSESSED_SECURITY_SCOPE, archive_config_input, decoder_geometry, resolve_decoder_config,
    validate_decoder_config, validate_parameter_manifest,
)
from orion.experimental.wpc_cips_trained_benchmark import CONFIGURABLE_WORKER_GATES, WORKER_GATES
from orion.experimental.wpc_online_encode_benchmark import compare_block, MODES
from tests.test_wpc_online_encode_benchmark import inputs
from tools.run_wpc_cips_trained_decoder import _config


def manifest(config, slots):
    ckks = config["ckks_params"]
    distribution = {"type": "Ternary", "parameters": {"P": 0, "H": ckks["H"]}}
    def params(q, p):
        return {"logn": ckks["LogN"], "ring_type": "Standard", "q": q, "p": p,
                "logqp": sum(math.log2(x) for x in q+p), "log_default_scale": ckks["LogScale"],
                "secret_distribution": copy.deepcopy(distribution),
                "error_distribution": {"type": "DiscreteGaussian", "parameters": {"Sigma": 3.2, "Bound": 19.2}}}
    q = [2**bits + 1 for bits in ckks["LogQ"]]
    p = [2**bits + 1 for bits in ckks["LogP"]]
    boot = params(q + [2**45+1], [2**bits+1 for bits in config["boot_params"]["LogP"]])
    return {"schema_version": 1, "security_assessed": False, "residual": params(q, p),
            "bootstrappers": [{"slots": slots, "parameters": boot, "ephemeral_secret_weight": 32,
                               "encapsulation": {"q": boot["q"][:1], "p": boot["p"][:1], "secret_hamming_weight": 32}}]}


def test_geometry_changes_transform_bootstrap_groups_and_level_schedule():
    old = decoder_geometry(10, 8, 8)
    assert old["learned_transform_count"] == 56
    assert old["bootstrap_ciphertext_group_count"] == 4
    new = decoder_geometry(11, 8, 8, 9)
    assert new["learned_transform_count_by_layer"] == {"checkpoint_up1": 2, "checkpoint_dec1a": 8, "checkpoint_dec1b": 4}
    assert new["learned_transform_count"] == 14
    assert new["bootstrap_ciphertext_group_count"] == 2
    assert new["levels"]["dec1a"] == 7
    assert new["levels"]["skip1"] == 8
    assert decoder_geometry(12, 16, 16)["learned_transform_count"] == 56
    assert decoder_geometry(11, 8, 16)["low_shape"] == [1, 64, 4, 8]


@pytest.mark.parametrize("logn,height,width,level", [(8,8,8,8), (10,3,8,8), (10,8,12,8), (10,32,32,8), (10,8,8,7)])
def test_geometry_rejects_unsupported_shapes(logn, height, width, level):
    with pytest.raises(ValueError): decoder_geometry(logn, height, width, level)


@pytest.mark.parametrize("problem", ["ring", "short", "unknown_boot", "h", "io", "scale", "nan", "empty"])
def test_configuration_rejects_ignored_or_unsupported_parameters(problem):
    config = _config(11)
    if problem == "ring": config["ckks_params"]["RingType"] = "ConjugateInvariant"
    elif problem == "short": config["ckks_params"]["LogQ"].pop()
    elif problem == "unknown_boot": config["boot_params"]["LogN"] = 16
    elif problem == "h": config["ckks_params"]["H"] = 2049
    elif problem == "io": config["orion"]["io_mode"] = "save"
    elif problem == "scale": config["ckks_params"]["LogScale"] = 50
    elif problem == "nan": config["ckks_params"]["H"] = float("nan")
    else: config["ckks_params"]["LogP"] = []
    with pytest.raises(ValueError): validate_decoder_config(config)


def test_config_source_is_hashed_and_conflicting_logn_rejected(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(_config(11)))
    args = SimpleNamespace(ckks_config=path, logn=None)
    config, source = resolve_decoder_config(args)
    assert args.logn == 11 and config == _config(11)
    assert len(source["sha256"]) == 64
    with pytest.raises(ValueError): resolve_decoder_config(SimpleNamespace(ckks_config=path, logn=10))
    archive = tmp_path / "archived.json"
    frozen = archive_config_input(config, source, archive)
    assert archive.read_bytes() == path.read_bytes()
    assert frozen["sha256"] == source["sha256"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        archive_config_input(config, source, archive)
    path.write_text(json.dumps(_config(12)))
    with pytest.raises(ValueError, match="changed after"):
        archive_config_input(config, source, tmp_path / "never_created.json")
    assert not (tmp_path / "never_created.json").exists()


@pytest.mark.parametrize("problem", ["claim", "ring", "h", "logqp", "q", "slots", "encapsulation", "noise"])
def test_actual_manifest_is_validated_not_trusted(problem):
    config = _config(11)
    data = manifest(config, 1024)
    validate_parameter_manifest(data, config, 1024)
    if problem == "claim": data["security_assessed"] = True
    elif problem == "ring": data["bootstrappers"][0]["parameters"]["logn"] = 10
    elif problem == "h": data["residual"]["secret_distribution"]["parameters"]["H"] = 128
    elif problem == "logqp": data["residual"]["logqp"] += 1
    elif problem == "q": data["residual"]["q"].pop()
    elif problem == "slots": data["bootstrappers"][0]["slots"] = 512
    elif problem == "encapsulation": data["bootstrappers"][0]["encapsulation"] = None
    else: data["residual"]["error_distribution"]["parameters"]["Sigma"] = float("nan")
    with pytest.raises(ValueError): validate_parameter_manifest(data, config, 1024)


def configurable_inputs():
    workers, rss = inputs()
    config = _config(11)
    geometry = decoder_geometry(11, 8, 8)
    for mode, worker in workers.items():
        online, compressed = mode == "online_encode", mode == "compressed"
        worker["schema_version"] = 3
        worker["experiment"].update({key: geometry[key] for key in ("logn", "slots", "low_shape", "high_shape")})
        worker["experiment"].update(geometry=geometry, ckks_config=config,
            security_scope=UNASSESSED_SECURITY_SCOPE, verify_exact_qp=True)
        worker["acceptance"].pop("bootstrap_uses_four_ciphertext_groups")
        worker["acceptance"].update(dict.fromkeys(CONFIGURABLE_WORKER_GATES - WORKER_GATES, True))
        worker["runtime_parameter_manifest"] = manifest(config, geometry["slots"])
        worker["bootstrap_compile"] = {"ciphertext_group_count": 2}
        worker["storage"]["learned_transform_count"] = 14
        row = worker["storage"]["by_layer"].pop("layer")
        worker["storage"]["by_layer"] = {
            name: {key: value * (1 if name == "checkpoint_up1" else 2) // 5 for key, value in row.items()}
            for name in geometry["learned_transform_count_by_layer"]}
        for name, count in geometry["learned_transform_count_by_layer"].items():
            worker["storage"]["by_layer"][name]["transform_count"] = count
        worker["storage"]["concat"]["transform_count"] = 4
        b = worker["backend"]
        b["compile_transform_encode_invocations"] = {"ordinary": 18 if mode in ("full", "compressed") else 4,
            "compressed": 14 if compressed else 0, "online_recipe": 0}
        b["measured_transform_encode_invocations"]["online_recipe"] = 28 if online else 0
        b["weight_plaintext_offline_encode_calls"] = 0 if online else 14
        b["weight_plaintext_online_encode_calls"] = 28 if online else 0
        b["online_recipe_global_stats"]["registered_transform_count"] = 14 if online else 0
        b["compressed_global_stats"]["registered_transform_count"] = 14 if compressed else 0
        worker["measurements"]["transform_encode_invocations"] = [14 if online else 0] * 2
        if compressed: worker["correctness"]["untimed_lifecycle_trace"][0] = worker["correctness"]["untimed_lifecycle_trace"][0][:14]
        worker["qp_verification"] = {"requested": True, "applicable": compressed, "retained_full_control_count": 0 if compressed else 14 if mode=="full" else 0,
            "rows": [{"layer": name, "transform": str(i), "exact_qp_match": True, "diagonal_count": 2,
                      "reconstructed_diagonal_count": 2} for name,count in geometry["learned_transform_count_by_layer"].items() for i in range(count)] if compressed else []}
    return workers, rss


def test_configurable_comparison_derives_counts_and_accepts_exact_qp_audit():
    workers, rss = configurable_inputs()
    block = compare_block(workers, rss, order=list(MODES), atol=.002)
    assert block["valid"] is True
    assert block["experiment"]["geometry"]["learned_transform_count"] == 14


@pytest.mark.parametrize("problem", ["geometry", "counts", "bootstrap", "qp", "duplicate_qp", "retained", "literal56"])
def test_configurable_comparison_rejects_wrong_contracts(problem):
    workers, rss = configurable_inputs()
    w = workers["compressed"]
    if problem == "geometry": w["experiment"]["geometry"] = copy.deepcopy(w["experiment"]["geometry"]); w["experiment"]["geometry"]["learned_transform_count"] = 56
    elif problem == "counts": w["storage"]["by_layer"]["checkpoint_dec1a"]["transform_count"] = 32
    elif problem == "bootstrap": w["bootstrap_compile"]["ciphertext_group_count"] = 4
    elif problem == "qp": w["qp_verification"]["rows"][0]["exact_qp_match"] = False
    elif problem == "duplicate_qp": w["qp_verification"]["rows"][1] = w["qp_verification"]["rows"][0]
    elif problem == "retained": w["qp_verification"]["retained_full_control_count"] = 14
    else: workers["online_encode"]["measurements"]["transform_encode_invocations"][0] = 56
    with pytest.raises(ValueError): compare_block(workers, rss, order=list(MODES), atol=.002)


def test_plan_only_does_not_need_checkpoint_or_shared_library(monkeypatch, capsys):
    from tools.run_wpc_decoder_feasibility import main
    monkeypatch.setattr("sys.argv", ["feasibility", "--plan-only", "--checkpoint", "/does/not/exist"])
    assert main() == 0
    request = json.loads(capsys.readouterr().out)
    assert request["geometry"]["high_shape"] == [1, 32, 16, 16]
    assert request["security_assessed"] is False


@pytest.mark.parametrize("guard", ["rss", "timeout"])
def test_watchdog_stops_worker_and_preserves_failure_sampling(monkeypatch, tmp_path, guard):
    from tools import run_wpc_cips_trained_isolated_benchmark as module
    args = module._parser().parse_args([])
    args.logn = 10
    args.max_worker_rss_mib = 1 if guard == "rss" else 0
    args.worker_timeout_s = 1 if guard == "timeout" else 0
    class Process:
        pid = 100
        returncode = None
        def poll(self): return self.returncode
        def wait(self): return self.returncode
    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(module, "_stop_worker", lambda p: setattr(p, "returncode", -15))
    monkeypatch.setattr(module, "_read_process_memory", lambda pid: ({"rss_bytes": 2 * 2**20}, "test"))
    ticks = iter(range(20))
    monkeypatch.setattr(module.time, "monotonic", lambda: float(next(ticks)))
    with pytest.raises(RuntimeError, match="exceeded"):
        module._sample_worker(mode="full", checkpoint=Path("unused"), out_dir=tmp_path, args=args)
    sampling = json.loads((tmp_path / "full.rss.json").read_text())
    assert sampling["return_code"] == -15
    assert sampling["termination_reason"] is not None
    assert sampling["total_sample_count"] > 0


def test_feasibility_refuses_existing_evidence_without_writing(monkeypatch, tmp_path):
    from tools.run_wpc_decoder_feasibility import main
    (tmp_path / "requested_run.json").write_text("preserved")
    monkeypatch.setattr("sys.argv", ["feasibility", "--out-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="refusing to overwrite"): main()
    assert (tmp_path / "requested_run.json").read_text() == "preserved"
    assert not (tmp_path / "feasibility.json").exists()


def test_watchdog_refuses_to_launch_when_rss_measurement_is_unavailable(monkeypatch, tmp_path):
    from tools import run_wpc_cips_trained_isolated_benchmark as module
    args = module._parser().parse_args([])
    args.logn, args.max_worker_rss_mib = 10, 1
    monkeypatch.setattr(module, "_read_process_memory", lambda pid: (None, "unavailable"))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not start an unguarded worker"))
    with pytest.raises(RuntimeError, match="refusing to launch"):
        module._sample_worker(mode="full", checkpoint=Path("unused"), out_dir=tmp_path, args=args)
