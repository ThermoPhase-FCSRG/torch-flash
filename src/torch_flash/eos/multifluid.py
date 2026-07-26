"""Compatibility aliases for the former multiparameter EOS names.

New code should import :class:`~torch_flash.eos.MultiparameterEOS` and
:class:`~torch_flash.eos.MultiparameterMetadata`. The module remains importable
so existing callers do not fail immediately after the terminology migration.
"""

from .multiparameter import (
    GaoBTerms,
    HelmholtzTerms,
    IdealHelmholtzTerms,
    MultiparameterEOS,
    MultiparameterMetadata,
    NonAnalyticTerms,
)

MultiFluidEOS = MultiparameterEOS
"""Deprecated compatibility alias for :class:`MultiparameterEOS`."""

MultifluidMetadata = MultiparameterMetadata
"""Deprecated compatibility alias for :class:`MultiparameterMetadata`."""

__all__ = [
    "GaoBTerms",
    "HelmholtzTerms",
    "IdealHelmholtzTerms",
    "MultiFluidEOS",
    "MultifluidMetadata",
    "NonAnalyticTerms",
]
