import pytest
import torch

from orion.models.unet import (
    UNet22,
    UNet22PlusOutput,
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


def test_unet22_uses_concat_decoder_skips() -> None:
    model = UNet22(dataset="tiny", base_dim=4)

    assert type(model.cat4).__name__ == "Concat"
    assert model.dec4a.in_channels == 2 * model.up4.out_channels
    assert model.dec3a.in_channels == 2 * model.up3.out_channels
    assert model.dec2a.in_channels == 2 * model.up2.out_channels
    assert model.dec1a.in_channels == 2 * model.up1.out_channels


def test_unet22_plus_output_keeps_explicit_output_head() -> None:
    model = UNet22PlusOutput(in_channels=1, out_channels=4, base_channels=8, activation="silu", silu_degree=7)

    assert model.dec1b.in_channels == 8
    assert model.dec1b.out_channels == 8
    assert type(model.dec1b_act).__name__ == "SiLU"
    assert model.output.in_channels == 8
    assert model.output.out_channels == 4
    assert model.output.kernel_size == (1, 1)


def test_unet22_plus_output_forward_shape_on_small_base() -> None:
    model = UNet22PlusOutput(dataset="tiny", in_channels=1, out_channels=4, base_dim=4, activation=None)
    model.eval()

    out = model(torch.randn(1, 1, 32, 32))

    assert tuple(out.shape) == (1, 4, 32, 32)


@pytest.mark.parametrize(
    ("name", "shape", "expected_channels"),
    [
        ("medical64", (1, 1, 64, 64), 1),
        ("medical256", (1, 3, 256, 256), 1),
    ],
)
def test_medical_unet22_forward_shapes_on_small_base(name: str, shape: tuple[int, ...], expected_channels: int) -> None:
    model = get_unet22_medical_model(name, base_dim=4)
    model.eval()

    out = model(torch.randn(*shape))

    assert tuple(out.shape) == (shape[0], expected_channels, shape[2], shape[3])


def test_unknown_medical_spec_errors_clearly() -> None:
    with pytest.raises(ValueError, match="Unknown UNet22 medical model"):
        get_unet22_medical_spec("not_a_dataset")
