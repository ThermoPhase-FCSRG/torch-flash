"""Package-specific exceptions and warnings."""

from __future__ import annotations


class TorchFlashError(Exception):
    """Base exception raised by torch-flash."""


class InvalidStateError(TorchFlashError, ValueError):
    """Raised when a thermodynamic state is outside a model's domain."""


class ConvergenceError(TorchFlashError, RuntimeError):
    """Raised when an iterative thermodynamic calculation does not converge."""


class ModelCapabilityError(TorchFlashError, NotImplementedError):
    """Raised when a backend cannot provide a requested model capability."""


class ParameterDatabaseError(TorchFlashError, ValueError):
    """Raised when a component or model parameter document is invalid."""


class ConvergenceWarning(RuntimeWarning):
    """Warn that an iterative result did not meet its convergence tolerance."""


class ExperimentalModelWarning(UserWarning):
    """Warn that a model is implemented but not yet broadly validated."""
