"""Configuration and independent geometry contracts for decoder experiments.

These checks establish executable shape/level contracts, not RLWE security.
Actual residual/bootstrap moduli are recorded by the backend for later review.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from orion.experimental.wpc_evidence_validation import (
    EvidenceValidationError, close, finite, integer,
)

UNASSESSED_SECURITY_SCOPE = "unassessed_experimental_parameters_not_secure_deployment"


def decoder_geometry(logn: int, height: int, width: int, max_level: int = 8) -> dict[str, Any]:
    logn = integer(logn, name="LogN", minimum=9)
    if logn > 20:
        raise EvidenceValidationError("LogN must be at most 20")
    height = integer(height, name="height", minimum=4)
    width = integer(width, name="width", minimum=4)
    if any(value & (value - 1) for value in (height, width)):
        raise EvidenceValidationError("decoder height/width must be powers of two")
    slots = 1 << (logn - 1)
    area = height * width
    if area > slots:
        raise EvidenceValidationError("spatial plane must fit in one ciphertext; spatial tiling is not implemented")
    level = integer(max_level, name="maximum residual level", minimum=8)
    high_capacity, low_capacity = slots // area, slots // (area // 4)
    groups = lambda channels, capacity: (channels + capacity - 1) // capacity
    high32, high64, low64 = groups(32, high_capacity), groups(64, high_capacity), groups(64, low_capacity)
    counts = {"checkpoint_up1": low64 * high32,
              "checkpoint_dec1a": high64 * high32,
              "checkpoint_dec1b": high32 * high32}
    # Two contiguous 32-channel branches. Power-of-two capacity either divides
    # each branch or contains both, so each source group intersects one output.
    return {"logn": logn, "slots": slots, "low_shape": [1, 64, height // 2, width // 2],
            "high_shape": [1, 32, height, width], "concat_shape": [1, 64, height, width],
            "low_channel_capacity": low_capacity, "high_channel_capacity": high_capacity,
            "learned_transform_count_by_layer": counts, "learned_transform_count": sum(counts.values()),
            "concat_transform_count": 2 * high32, "bootstrap_ciphertext_group_count": high32,
            "levels": {"up1": level, "skip1": level - 1, "concat": level - 1,
                       "dec1a": level - 2, "bridge_input": level - 3,
                       "dec1b": level, "output": level - 1}}


def validate_decoder_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject ignored/unsupported keys instead of silently changing a request."""
    config = copy.deepcopy(config)
    if set(config) != {"ckks_params", "boot_params", "orion"}:
        raise EvidenceValidationError("config requires exactly ckks_params, boot_params, and orion")
    ckks, boot, runtime = config["ckks_params"], config["boot_params"], config["orion"]
    if set(ckks) != {"LogN", "LogQ", "LogP", "LogScale", "H", "RingType"} or set(boot) != {"LogP"}:
        raise EvidenceValidationError("unsupported CKKS/bootstrap config keys")
    logn = ckks["LogN"] = integer(ckks["LogN"], name="LogN", minimum=9)
    if logn > 20 or ckks["RingType"] != "Standard":
        raise EvidenceValidationError("decoder supports LogN<=20 and Standard ring only")
    ckks["H"] = integer(ckks["H"], name="secret Hamming weight", minimum=1)
    if ckks["H"] > 1 << logn:
        raise EvidenceValidationError("secret Hamming weight exceeds ring degree")
    for name, values, maximum in (("LogQ", ckks["LogQ"], 60), ("LogP", ckks["LogP"], 61),
                                  ("bootstrap LogP", boot["LogP"], 61)):
        if not isinstance(values, list) or not values:
            raise EvidenceValidationError(f"{name} requires a nonempty list")
        for index, value in enumerate(values):
            value = values[index] = integer(value, name=name, minimum=2)
            if value <= logn + 1 or value > maximum:
                raise EvidenceValidationError(f"{name} prime sizes must lie in [{logn + 2}, {maximum}]")
    if len(ckks["LogQ"]) < 9 or len(ckks["LogP"]) > len(ckks["LogQ"]):
        raise EvidenceValidationError("decoder needs at least nine Q primes and no more P than Q primes")
    scale = ckks["LogScale"] = integer(ckks["LogScale"], name="LogScale", minimum=1)
    if scale > min(ckks["LogQ"][1:]):
        raise EvidenceValidationError("LogScale exceeds a residual rescale prime")
    expected = {"margin": 1, "backend": "lattigo", "embedding_method": "square", "io_mode": "none", "debug": False}
    if runtime != expected:
        raise EvidenceValidationError("decoder config must use the documented non-streaming in-memory runtime")
    return config


def resolve_decoder_config(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.run_wpc_cips_trained_decoder import _config

    path = getattr(args, "ckks_config", None)
    if path is None:
        config = _config(10 if args.logn is None else args.logn)
        source = {"kind": "built_in_functional_config", "path": None, "sha256": None}
    else:
        path = Path(path).expanduser().resolve()
        content = path.read_bytes()
        config = json.loads(content)
        source = {"kind": "json_file", "path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
        if args.logn is not None and args.logn != config["ckks_params"]["LogN"]:
            raise EvidenceValidationError("--logn conflicts with --ckks-config")
    config = validate_decoder_config(config)
    args.logn = config["ckks_params"]["LogN"]
    return config, source


def archive_config_input(config: dict[str, Any], source: dict[str, Any], path: Path) -> dict[str, Any]:
    """Keep exact file bytes so later transfer/review does not need its old path."""
    if path.exists():
        raise EvidenceValidationError(f"refusing to overwrite archived configuration: {path}")
    if source["kind"] == "json_file":
        content = Path(source["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise EvidenceValidationError("configuration file changed after it was loaded")
    else:
        content = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(content)
    return {"path": path.name, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def expected_worker_geometry(worker: dict[str, Any]) -> dict[str, Any] | None:
    """Legacy schemas retain their original fixed contract."""
    if worker.get("schema_version", 1) < 3:
        return None
    experiment = worker["experiment"]
    config = validate_decoder_config(experiment["ckks_config"])
    high = experiment["high_shape"]
    geometry = decoder_geometry(config["ckks_params"]["LogN"], high[2], high[3], len(config["ckks_params"]["LogQ"]) - 1)
    if experiment["geometry"] != geometry or any(experiment[key] != geometry[key] for key in ("logn", "slots", "low_shape", "high_shape")):
        raise EvidenceValidationError("recorded decoder geometry does not match independent derivation")
    if experiment.get("security_scope") != UNASSESSED_SECURITY_SCOPE:
        raise EvidenceValidationError("unassessed security scope must be explicit; configuration is not a security proof")
    return geometry


def validate_parameter_manifest(manifest: dict[str, Any], config: dict[str, Any], slots: int) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("security_assessed") is not False:
        raise EvidenceValidationError("runtime parameter manifest must not claim a security assessment")
    residual = manifest["residual"]
    bootstrappers = manifest["bootstrappers"]
    if len(bootstrappers) != 1 or bootstrappers[0]["slots"] != slots:
        raise EvidenceValidationError("runtime manifest must identify the decoder bootstrapper")
    boot = bootstrappers[0]
    for name, params in (("residual", residual), ("bootstrap", boot["parameters"])):
        if params["logn"] != config["ckks_params"]["LogN"] or params["ring_type"] != "Standard":
            raise EvidenceValidationError(f"{name} ring differs from requested configuration")
        if params["secret_distribution"] != {"type": "Ternary", "parameters": {"P": 0, "H": config["ckks_params"]["H"]}}:
            raise EvidenceValidationError(f"{name} secret distribution differs from request")
        error = params["error_distribution"]
        if error["type"] != "DiscreteGaussian" or set(error["parameters"]) != {"Sigma", "Bound"}:
            raise EvidenceValidationError("missing actual Gaussian error distribution")
        for value in error["parameters"].values():
            if finite(value, name="error distribution", minimum=0) <= 0:
                raise EvidenceValidationError("invalid error distribution")
        for chain in ("q", "p"):
            if not isinstance(params[chain], list) or not params[chain]:
                raise EvidenceValidationError("empty actual modulus chain")
            for prime in params[chain]:
                if isinstance(prime, bool) or not isinstance(prime, int):
                    raise EvidenceValidationError("actual modulus primes must be exact JSON integers")
                integer(prime, name="actual modulus prime", minimum=3)
        close(params["logqp"], sum(math.log2(prime) for prime in params["q"] + params["p"]), name="actual LogQP")
    for params, chain, requested in ((residual, "q", config["ckks_params"]["LogQ"]),
                                    (residual, "p", config["ckks_params"]["LogP"]),
                                    (boot["parameters"], "p", config["boot_params"]["LogP"])):
        if len(params[chain]) != len(requested) or any(abs(math.log2(prime) - bits) > .01 for prime, bits in zip(params[chain], requested)):
            raise EvidenceValidationError("actual prime sizes differ from the requested chain")
    if boot["parameters"]["q"][:len(residual["q"])] != residual["q"]:
        raise EvidenceValidationError("bootstrap residual Q prefix does not match")
    if residual["log_default_scale"] != config["ckks_params"]["LogScale"]:
        raise EvidenceValidationError("runtime default scale differs from request")
    weight = integer(boot["ephemeral_secret_weight"], name="ephemeral Hamming weight")
    encapsulation = boot["encapsulation"]
    expected = None if weight == 0 else {"q": boot["parameters"]["q"][:1], "p": boot["parameters"]["p"][:1], "secret_hamming_weight": weight}
    if encapsulation != expected:
        raise EvidenceValidationError("encapsulation parameters do not match the bootstrap key-generation contract")
