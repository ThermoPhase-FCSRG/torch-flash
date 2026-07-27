"""Liquid n-paraffin diffusion correlation."""

from __future__ import annotations

import torch
from torch import Tensor

from torch_flash.exceptions import InvalidStateError


def hayduk_minhas_n_paraffin_diffusion_coefficient(
    temperature: Tensor,
    dynamic_viscosity: Tensor,
    solute_molar_volume: Tensor,
) -> Tensor:
    """Estimate infinite-dilution n-paraffin diffusion in an n-paraffin liquid.

    Parameters
    ----------
    temperature
        Liquid temperature in K.
    dynamic_viscosity
        Solvent or bulk-phase dynamic viscosity in Pa s.
    solute_molar_volume
        Diffusing solute molar volume at its normal boiling point in m3/mol.

    Returns
    -------
    Tensor
        Diffusion coefficient in m2/s with the broadcast input shape.

    Raises
    ------
    InvalidStateError
        If any input is nonfinite or nonpositive.

    Notes
    -----
    Implements Hayduk and Minhas, "Correlations for prediction of molecular
    diffusivities in liquids," *Canadian Journal of Chemical Engineering* 60
    (1982), 295-299, doi:10.1002/cjce.5450600213, normal-paraffin
    correlation, as reproduced in Pedersen et al. (2024), Eq. 10.100. The
    defining equation uses solvent viscosity in cP and solute normal-boiling
    molar volume in cm3/mol; this SI API performs both conversions. It applies
    to infinite-dilution n-paraffin solutes in n-paraffin solvents, not
    concentrated multicomponent diffusion.
    """
    temperature, viscosity, volume = torch.broadcast_tensors(
        temperature,
        dynamic_viscosity,
        solute_molar_volume,
    )
    if bool(
        (
            (~torch.isfinite(temperature))
            | (~torch.isfinite(viscosity))
            | (~torch.isfinite(volume))
            | (temperature <= 0.0)
            | (viscosity <= 0.0)
            | (volume <= 0.0)
        ).any()
    ):
        raise InvalidStateError("diffusion correlation requires positive finite inputs")
    viscosity_cp = 1000.0 * viscosity
    volume_cm3_mol = 1.0e6 * volume
    viscosity_exponent = 10.2 / volume_cm3_mol - 0.791
    result: Tensor = (
        13.3e-12
        * temperature.pow(1.47)
        * viscosity_cp.pow(viscosity_exponent)
        / volume_cm3_mol.pow(0.71)
    )
    return result
