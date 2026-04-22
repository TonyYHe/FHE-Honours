from __future__ import annotations

import torch

from orion.experimental import (
    build_r34_phase1_report,
    build_r34_phase1_transition_bridge_plan,
    imported_layout_contract_by_haloed_node,
    imported_layout_contracts,
    kernel_binding_for_family,
    kernel_bindings,
    materializer_attrs_from_contract,
)


def test_r34_phase1_imported_layout_contracts_cover_selected_representatives() -> None:
    contracts = {contract.haloed_node: contract for contract in imported_layout_contracts()}

    assert set(contracts) == {
        "layer1_0_conv1_torch",
        "layer2_1_conv1_torch",
        "layer3_1_conv1_torch",
        "layer4_1_conv1_torch",
        "layer3_0_conv1_torch",
        "layer3_0_downsample_conv_torch",
    }
    assert contracts["layer1_0_conv1_torch"].orion_node == "layers_0_0_conv1"
    assert contracts["layer1_0_conv1_torch"].input_layout["stride"] == 4
    assert contracts["layer2_1_conv1_torch"].output_layout["stride"] == 8
    assert contracts["layer3_1_conv1_torch"].input_layout["alpha"] == 7
    assert contracts["layer4_1_conv1_torch"].input_layout["stride"] == 32
    assert contracts["layer3_0_downsample_conv_torch"].padding == (0, 0)
    assert contracts["layer3_0_conv1_torch"].stride == (2, 2)


def test_r34_phase1_materializer_attrs_keep_layout_contract_explicit() -> None:
    contract = imported_layout_contract_by_haloed_node("layer2_1_conv1_torch")
    attrs = materializer_attrs_from_contract(
        contract,
        weight=torch.zeros(contract.weight_shape, dtype=torch.float32),
    )

    assert attrs["input_layout"] == contract.input_layout
    assert attrs["layout"] == contract.output_layout
    assert attrs["module_target"] == contract.orion_node
    assert attrs["_node_name"] == contract.haloed_node
    assert attrs["stride"] == contract.stride
    assert attrs["padding"] == contract.padding


def test_r34_phase1_transition_bridge_plan_is_relu_safe_real_imag_pair() -> None:
    payload = build_r34_phase1_transition_bridge_plan()

    assert payload["status"] == "ok"
    assert payload["binding"]["provider_kind"] == "tile_local_transition_bridge"
    assert payload["binding"]["phase1_status"] == "implemented"
    assert payload["shared_layout_contract"]["input_layout"]["stride"] == 8
    assert payload["shared_layout_contract"]["output_layout"]["stride"] == 16
    assert payload["region"]["output_node_ids"] == ["layers_2_0_conv1", "layers_2_0_shortcut_0"]
    assert payload["lowered"]["relu_safe_boundary"] is True
    assert payload["lowered"]["boundary_actions"] == ["insert_extract", "validate_relu_safe"]
    assert payload["lowered"]["output_bank_count"] == payload["lowered"]["target_tile_count"]
    assert all(bank["kind"] == "real_imag_pair" for bank in payload["lowered"]["output_banks"])


def test_r34_phase1_report_surfaces_bindings_and_selected_results() -> None:
    payload = build_r34_phase1_report()

    assert payload["status"] == "ok"
    assert payload["layout_registry"]["selected_node_count"] == 6
    assert len(payload["kernel_bindings"]) == len(kernel_bindings())
    assert {binding["family_label"] for binding in payload["kernel_bindings"]} == {
        "stage1_same",
        "stage2_same",
        "stage3_same",
        "stage4_same",
        "stage3_transition",
    }
    assert len(payload["same_shape_surface"]) == 4
    assert payload["selected_results"]["same_shape_region"]["network"] == "R34"
    assert payload["selected_results"]["same_shape_region"]["candidate"]["rotations"] == 1928
    assert payload["selected_results"]["transition_region"]["publishable_executor_fact"] is False
    assert len(payload["selected_results"]["stage_materializer_references"]) == 4
    assert kernel_binding_for_family("stage1_same").materializer == "policy_inter_group_hybrid"
    assert kernel_binding_for_family("stage3_same").materializer == "policy_intra_group_pack2"
