"""Leachman normal-hydrogen Helmholtz equation of state.

This module isolates the normal-hydrogen thermodynamic equation defined by
Leachman et al. (2009) from the multiparameter mixture models that reuse it. The same
native PyTorch Helmholtz kernel evaluates the one-component equation and the
hydrogen contribution inside H2-tailored GERG mixtures.
"""

from __future__ import annotations

import torch

from torch_flash.config import resolve_tensor_options
from torch_flash.database import ParameterSource, load_model_parameters
from torch_flash.exceptions import ParameterDatabaseError

from .helmholtz import PureFluidHelmholtzEOS, PureFluidHelmholtzMetadata
from .named import _eoscg_eos

DEFAULT_LEACHMAN_NORMAL_HYDROGEN = "pure-helmholtz.leachman-2009-normal-hydrogen"
"""Bundled parameter identifier for the Leachman normal-hydrogen EOS."""


def leachman_normal_hydrogen(
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    trainable: bool = False,
    parameter_set: ParameterSource = DEFAULT_LEACHMAN_NORMAL_HYDROGEN,
) -> PureFluidHelmholtzEOS:
    """Construct the Leachman et al. normal-hydrogen Helmholtz EOS.

    Parameters
    ----------
    dtype, device
        Tensor placement, defaulting to the configured runtime policy.
    trainable
        Register supported Helmholtz coefficients as trainable parameters.
    parameter_set
        Compatible standalone normal-hydrogen parameter source.

    Returns
    -------
    PureFluidHelmholtzEOS
        Native PyTorch pure-fluid Helmholtz model for normal hydrogen.

    Raises
    ------
    ParameterDatabaseError
        If ``parameter_set`` does not identify the Leachman normal-hydrogen
        thermodynamic equation.

    Notes
    -----
    This constructor covers the thermodynamic Helmholtz equation. It does not
    add saturation ancillaries, melting or sublimation curves, or transport
    correlations.

    References
    ----------
    J. W. Leachman et al., *J. Phys. Chem. Ref. Data* 38 (2009) 721--748,
    doi:10.1063/1.3160306.
    """
    dtype, device = resolve_tensor_options(dtype, device)
    loaded = load_model_parameters(parameter_set)
    normalized = loaded.model.strip().lower().replace("_", "-")
    if loaded.model_kind != "pure_helmholtz" or not normalized.startswith(
        "leachman normal hydrogen"
    ):
        raise ParameterDatabaseError(
            "leachman_normal_hydrogen requires the Leachman normal-hydrogen "
            f"parameter set, got {loaded.model!r}"
        )
    kernel = _eoscg_eos(
        loaded,
        ("hydrogen",),
        dtype=dtype,
        device=device,
        trainable=trainable,
    )
    return PureFluidHelmholtzEOS(
        kernel,
        PureFluidHelmholtzMetadata(
            model=loaded.model,
            reference="doi:10.1063/1.3160306",
            version=loaded.version,
            fluid="hydrogen",
        ),
    )


__all__ = [
    "DEFAULT_LEACHMAN_NORMAL_HYDROGEN",
    "leachman_normal_hydrogen",
]
