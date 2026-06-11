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
from tools.train_fhelipe_medseg_staged_unet import (
    ScaledChebyshevSiLU,
    clamp_cheb_prescales,
    clamp_cheb_postscales,
    fit_scaled_domain_chebyshev_silu,
    parse_poly_degree_overrides,
    parse_poly_prescale_caps,
    parse_poly_postscale_caps,
    replace_plain_silu_with_range_poly,
    set_poly_blend_alpha,
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


def test_range_poly_replacement_handles_orion_silu_modules() -> None:
    model = UNet22PlusOutput(dataset="tiny", in_channels=1, out_channels=1, base_dim=4, activation="silu", silu_degree=7)
    ranges = {
        name: {"observed_min": -2.0, "observed_max": 3.0}
        for name, module in model.named_modules()
        if type(module).__name__ == "SiLU"
    }

    metadata = replace_plain_silu_with_range_poly(model, ranges=ranges, degree=7, scale_margin=1.0)

    assert metadata
    assert all(type(module).__name__ != "SiLU" for module in model.modules())
    assert any(isinstance(module, ScaledChebyshevSiLU) for module in model.modules())


def test_range_poly_replacement_accepts_per_layer_degree_overrides() -> None:
    model = UNet22PlusOutput(dataset="tiny", in_channels=1, out_channels=1, base_dim=4, activation="silu", silu_degree=7)
    ranges = {
        name: {"observed_min": -2.0, "observed_max": 3.0}
        for name, module in model.named_modules()
        if type(module).__name__ == "SiLU"
    }

    metadata = replace_plain_silu_with_range_poly(
        model,
        ranges=ranges,
        degree=7,
        degree_overrides=parse_poly_degree_overrides("enc3b_act=15,dec1a_act=15"),
        scale_margin=1.0,
    )

    assert metadata["enc3b_act"]["degree"] == 15
    assert metadata["dec1a_act"]["degree"] == 15
    assert metadata["enc1a_act"]["degree"] == 7


def test_parse_poly_postscale_caps_and_clamp_trainable_scale() -> None:
    caps = parse_poly_postscale_caps("dec4a_act=3072, dec4b_act=6144")
    module = torch.nn.Module()
    module.dec4a_act = ScaledChebyshevSiLU([0.0, 0.5], postscale=8192.0, trainable_scale=True)

    changed = clamp_cheb_postscales(module, {"dec4a_act": caps["dec4a_act"]})

    assert changed["dec4a_act"]["clamped"] is True
    assert torch.exp(module.dec4a_act.log_postscale).item() == pytest.approx(3072.0)


def test_independent_prescale_survives_postscale_cap() -> None:
    module = torch.nn.Module()
    module.dec4a_act = ScaledChebyshevSiLU(
        [0.0, 0.5],
        postscale=8192.0,
        prescale=1.0 / 8192.0,
        trainable_scale=True,
        trainable_prescale=True,
    )
    before_prescale = torch.exp(module.dec4a_act.log_prescale).item()

    clamp_cheb_postscales(module, {"dec4a_act": 2048.0})

    assert torch.exp(module.dec4a_act.log_postscale).item() == pytest.approx(2048.0)
    assert torch.exp(module.dec4a_act.log_prescale).item() == pytest.approx(before_prescale)


def test_parse_poly_prescale_caps_and_clamp_trainable_prescale() -> None:
    caps = parse_poly_prescale_caps("dec4a_act=0.001")
    module = torch.nn.Module()
    module.dec4a_act = ScaledChebyshevSiLU(
        [0.0, 0.5],
        postscale=2048.0,
        prescale=0.01,
        trainable_prescale=True,
    )

    changed = clamp_cheb_prescales(module, {"dec4a_act": caps["dec4a_act"]})

    assert changed["dec4a_act"]["clamped"] is True
    assert torch.exp(module.dec4a_act.log_prescale).item() == pytest.approx(0.001)


def test_scaled_chebyshev_silu_coefficients_match_runtime_basis() -> None:
    coeffs = fit_scaled_domain_chebyshev_silu(degree=7, postscale=2.0)
    approx = ScaledChebyshevSiLU(coeffs, postscale=2.0)
    x = torch.linspace(-2.0, 2.0, 101)

    max_error = (approx(x) - torch.nn.functional.silu(x)).abs().max().item()

    assert max_error < 5.0e-4


def test_scaled_chebyshev_silu_homotopy_can_fall_back_to_exact_silu() -> None:
    coeffs = fit_scaled_domain_chebyshev_silu(degree=7, postscale=2.0)
    approx = ScaledChebyshevSiLU(coeffs, postscale=2.0, trainable_scale=True)
    x = torch.linspace(-4.0, 4.0, 33)

    set_poly_blend_alpha(approx, alpha=0.0)
    exact_error = (approx(x) - torch.nn.functional.silu(x)).abs().max().item()
    set_poly_blend_alpha(approx, alpha=1.0)
    pure_poly_error = (approx(x) - torch.nn.functional.silu(x)).abs().max().item()

    assert exact_error < 1.0e-6
    assert pure_poly_error > exact_error
    assert any(parameter.requires_grad for parameter in approx.parameters())


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
