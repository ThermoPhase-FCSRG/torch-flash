"""Package-specific exceptions and warnings."""

from __future__ import annotations


class TorchFlashError(Exception):
    """Base class for torch-flash domain-specific errors.

    Catch this type when an application wants to handle all package-defined
    calculation and parameter errors without intercepting unrelated Python
    exceptions.
    """


class InvalidStateError(TorchFlashError, ValueError):
    """Indicate that a thermodynamic state is outside a model's domain.

    Typical causes include nonpositive temperature or pressure, inadmissible
    volume, and mechanically unstable response functions.
    """


class ConvergenceError(TorchFlashError, RuntimeError):
    """Indicate that an iterative thermodynamic calculation failed.

    The originating API should include a physically meaningful residual or
    iteration context in the exception message.
    """


class ModelCapabilityError(TorchFlashError, NotImplementedError):
    """Indicate that a model or optional backend lacks a requested operation.

    This is distinct from numerical non-convergence of an implemented
    capability.
    """


class ParameterDatabaseError(TorchFlashError, ValueError):
    """Indicate an invalid component or model parameter document.

    Examples include schema, unit, model-identity, provenance, and coefficient
    consistency failures.
    """


class ConvergenceWarning(RuntimeWarning):
    """Warn that a returned iterative result did not meet its tolerance.

    APIs emitting this warning also mark the associated result non-converged;
    callers must not silently treat it as a valid solution.
    """


class ExperimentalModelWarning(UserWarning):
    """Warn that an implemented model path has limited validation scope.

    The warning concerns scientific maturity or automated phase discovery,
    not necessarily numerical failure at the requested state.
    """


__all__ = [
    "ConvergenceError",
    "ConvergenceWarning",
    "ExperimentalModelWarning",
    "InvalidStateError",
    "ModelCapabilityError",
    "ParameterDatabaseError",
    "TorchFlashError",
]
