import pytest
import torch

from orion.models.unet import (
    UNet22,
    UNet22KvasirPolyp256,
    UNet22MontgomeryLung64,
    get_unet22_medical_model,
    get_unet22_medical_spec,
    list_unet22_medical_specs,
)


def test_unet22_accepts_base_dim_alias() -> None:
    model = UNet22(dataset="tiny", base_dim=8)

    assert model.base_channels == 8
    assert model.enc1a.out_channels == 8


def test_unet22_rejects_conflicting_base_names() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        UNet22(dataset="tiny", base_channels=4, base_dim=8)


def test_medical_specs_describe_two_public_segmentation_models() -> None:
    specs = {spec.name: spec for spec in list_unet22_medical_specs()}

    assert set(specs) == {"kvasir_polyp_256", "montgomery_lung_64"}
    assert specs["montgomery_lung_64"].image_size == 64
    assert specs["montgomery_lung_64"].in_channels == 1
    assert specs["montgomery_lung_64"].out_channels == 1
    assert specs["montgomery_lung_64"].base_dim == 32
    assert specs["kvasir_polyp_256"].image_size == 256
    assert specs["kvasir_polyp_256"].in_channels == 3
    assert specs["kvasir_polyp_256"].out_channels == 1
    assert specs["kvasir_polyp_256"].base_dim == 32


def test_medical_model_factory_uses_aliases_and_base_dim_32() -> None:
    model64 = get_unet22_medical_model("medical64")
    model256 = get_unet22_medical_model("medical256")

    assert model64.dataset == "montgomery_lung_64"
    assert model64.base_channels == 32
    assert model64.enc1a.in_channels == 1
    assert model64.dec1b.out_channels == 1
    assert model256.dataset == "kvasir_polyp_256"
    assert model256.base_channels == 32
    assert model256.enc1a.in_channels == 3
    assert model256.dec1b.out_channels == 1


def test_medical_wrappers_default_to_required_shapes_without_heavy_forward() -> None:
    model64 = UNet22MontgomeryLung64()
    model256 = UNet22KvasirPolyp256()

    assert model64.base_channels == 32
    assert model64.enc1a.in_channels == 1
    assert model64.dec1b.out_channels == 1
    assert model256.base_channels == 32
    assert model256.enc1a.in_channels == 3
    assert model256.dec1b.out_channels == 1


def test_montgomery_lung64_forward_shape_on_small_base() -> None:
    model = UNet22MontgomeryLung64(base_dim=4)
    model.eval()

    out = model(torch.randn(1, 1, 64, 64))

    assert tuple(out.shape) == (1, 1, 64, 64)


def test_unknown_medical_spec_errors_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown UNet22 medical model"):
        get_unet22_medical_spec("not_a_dataset")
