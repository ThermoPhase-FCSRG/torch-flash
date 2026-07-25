"""Process-wide runtime configuration for native torch-flash models.

The configuration is intentionally read when model or data tensors are
constructed, not inside thermodynamic kernels. Existing model instances are
therefore unaffected by later configuration changes and hot-path calls do not
pay for repeated policy lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_SUPPORTED_DEVICE_TYPES = ("cpu", "cuda", "xpu", "mps")
_ACCELERATOR_DEVICE_TYPES = ("cuda", "xpu", "mps")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable snapshot of torch-flash's tensor and execution policy.

    Parameters
    ----------
    device:
        Device used by named model, component, standard-state, and
        characterization factories when no explicit device is supplied.
    dtype:
        Floating-point dtype used by those factories. Float64 is the
        scientific default; float32 is available for explicit lower-precision
        studies.
    num_threads:
        Current PyTorch CPU intra-operation thread count.
    num_interop_threads:
        Current PyTorch CPU inter-operation thread count.
    deterministic:
        Whether PyTorch deterministic algorithms are enabled.
    deterministic_warn_only:
        Whether unavailable deterministic algorithms warn instead of raising.
    """

    device: torch.device
    dtype: torch.dtype
    num_threads: int
    num_interop_threads: int
    deterministic: bool
    deterministic_warn_only: bool

    @property
    def accelerated(self) -> bool:
        """Report whether the configured default device is an accelerator.

        Returns
        -------
        bool
            ``True`` for CUDA, XPU, or MPS devices and ``False`` for CPU.
        """
        return bool(self.device.type != "cpu")

    @property
    def tensor_options(self) -> dict[str, torch.dtype | torch.device]:
        """Return configured tensor-factory keyword arguments.

        Returns
        -------
        dict
            Mapping containing ``dtype`` and ``device``.
        """
        return {"dtype": self.dtype, "device": self.device}

    def tensor(
        self,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Construct a copied tensor using this runtime policy.

        Parameters
        ----------
        data
            Any value accepted by :func:`torch.tensor`.
        dtype, device
            Per-call overrides for configured defaults.
        requires_grad
            Whether autograd records operations on the new tensor.

        Returns
        -------
        Tensor
            Newly allocated tensor.
        """
        return torch.tensor(
            data,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            requires_grad=requires_grad,
        )

    def as_tensor(
        self,
        data: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Convert data using this runtime policy without unnecessary copies.

        Parameters
        ----------
        data
            Any value accepted by :func:`torch.as_tensor`.
        dtype, device
            Per-call overrides for configured defaults.

        Returns
        -------
        Tensor
            Converted tensor, sharing storage when PyTorch permits.
        """
        return torch.as_tensor(
            data,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )


_CONFIG_LOCK = RLock()
_CONFIG = RuntimeConfig(
    device=torch.device("cpu"),
    dtype=torch.float64,
    num_threads=torch.get_num_threads(),
    num_interop_threads=torch.get_num_interop_threads(),
    deterministic=torch.are_deterministic_algorithms_enabled(),
    deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
)


def _positive_thread_count(value: int | None, name: str, current: int) -> int:
    if value is None:
        return current
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_device(device: torch.device | str) -> torch.device:
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid PyTorch device {device!r}") from exc
    if parsed.type not in _SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(_SUPPORTED_DEVICE_TYPES)
        raise ValueError(
            f"unsupported torch-flash device type {parsed.type!r}; choose one of {supported}"
        )
    return parsed


def _device_error(device: torch.device, dtype: torch.dtype) -> BaseException | None:
    try:
        torch.empty(0, dtype=dtype, device=device)
    except (AssertionError, NotImplementedError, RuntimeError) as exc:
        return exc
    return None


def _resolve_device(
    requested: torch.device | str | None,
    dtype: torch.dtype,
    current: torch.device,
) -> torch.device:
    if requested is None:
        selected = current
    elif requested == "auto":
        for kind in (*_ACCELERATOR_DEVICE_TYPES, "cpu"):
            candidate = torch.device(kind)
            if _device_error(candidate, dtype) is None:
                return candidate
        raise RuntimeError(f"no PyTorch device supports configured dtype {dtype}")
    elif requested == "gpu":
        for kind in _ACCELERATOR_DEVICE_TYPES:
            candidate = torch.device(kind)
            if _device_error(candidate, dtype) is None:
                return candidate
        raise RuntimeError(f"no PyTorch GPU supports configured dtype {dtype}")
    else:
        selected = _parse_device(requested)

    error = _device_error(selected, dtype)
    if error is not None:
        raise RuntimeError(
            f"PyTorch device {selected} is unavailable or does not support {dtype}"
        ) from error
    return selected


def get_config() -> RuntimeConfig:
    """Return a current immutable runtime-policy snapshot.

    Returns
    -------
    RuntimeConfig
        Configured device/dtype plus live PyTorch thread and deterministic
        settings. Mutating process-wide PyTorch settings outside torch-flash
        is reflected in the returned snapshot.
    """
    with _CONFIG_LOCK:
        return replace(
            _CONFIG,
            num_threads=torch.get_num_threads(),
            num_interop_threads=torch.get_num_interop_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )


def configure(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    num_threads: int | None = None,
    num_interop_threads: int | None = None,
    deterministic: bool | None = None,
    deterministic_warn_only: bool | None = None,
) -> RuntimeConfig:
    """Set the process-wide torch-flash runtime policy before model creation.

    ``device="auto"`` selects the first available CUDA, XPU, or MPS device
    that supports the requested dtype, falling back to CPU. ``device="gpu"``
    performs the same search but raises if no accelerator supports the dtype.

    The dtype becomes PyTorch's default floating dtype. The selected device is
    applied only by torch-flash factories and :class:`RuntimeConfig` tensor
    helpers; this function deliberately does not call
    :func:`torch.set_default_device`, which adds overhead to every Python-level
    PyTorch API call.

    PyTorch thread counts and deterministic-algorithm flags are inherently
    process-wide. In particular, ``num_interop_threads`` can be changed only
    once and before inter-operation parallel work starts; PyTorch's failure is
    reported with an actionable error.

    Parameters
    ----------
    device:
        Explicit PyTorch device, ``"auto"`` for accelerator-then-CPU
        selection, ``"gpu"`` to require an accelerator, or ``None`` to retain
        the current setting.
    dtype:
        ``torch.float64`` or ``torch.float32``. ``None`` retains the current
        dtype.
    num_threads:
        Positive CPU intra-operation thread count.
    num_interop_threads:
        Positive CPU inter-operation thread count. PyTorch permits changing
        it only once before parallel work begins.
    deterministic:
        Enable or disable PyTorch deterministic algorithms.
    deterministic_warn_only:
        Warn instead of raising when a deterministic implementation is
        unavailable. Requires ``deterministic=True``.

    Returns
    -------
    RuntimeConfig
        New immutable runtime-policy snapshot.

    Raises
    ------
    ValueError
        If a dtype, device type, thread count, or deterministic option is
        invalid.
    RuntimeError
        If the requested device/dtype is unavailable or PyTorch can no longer
        change inter-operation threads.
    """
    global _CONFIG

    with _CONFIG_LOCK:
        current = get_config()
        selected_dtype = current.dtype if dtype is None else dtype
        if selected_dtype not in _SUPPORTED_DTYPES:
            raise ValueError("torch-flash runtime dtype must be torch.float32 or torch.float64")
        selected_device = _resolve_device(device, selected_dtype, current.device)
        selected_threads = _positive_thread_count(
            num_threads,
            "num_threads",
            current.num_threads,
        )
        selected_interop_threads = _positive_thread_count(
            num_interop_threads,
            "num_interop_threads",
            current.num_interop_threads,
        )
        selected_deterministic = current.deterministic if deterministic is None else deterministic
        if not isinstance(selected_deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if deterministic is False and deterministic_warn_only is None:
            selected_warn_only = False
        else:
            selected_warn_only = (
                current.deterministic_warn_only
                if deterministic_warn_only is None
                else deterministic_warn_only
            )
        if not isinstance(selected_warn_only, bool):
            raise ValueError("deterministic_warn_only must be a boolean")
        if selected_warn_only and not selected_deterministic:
            raise ValueError("deterministic_warn_only requires deterministic=True")

        if selected_interop_threads != current.num_interop_threads:
            try:
                torch.set_num_interop_threads(selected_interop_threads)
            except RuntimeError as exc:
                raise RuntimeError(
                    "PyTorch inter-operation threads must be configured once, before "
                    "inter-operation parallel work starts"
                ) from exc
        if selected_threads != current.num_threads:
            torch.set_num_threads(selected_threads)
        if (
            selected_deterministic != current.deterministic
            or selected_warn_only != current.deterministic_warn_only
        ):
            torch.use_deterministic_algorithms(
                selected_deterministic,
                warn_only=selected_warn_only,
            )
        if selected_dtype != torch.get_default_dtype():
            torch.set_default_dtype(selected_dtype)

        _CONFIG = RuntimeConfig(
            selected_device,
            selected_dtype,
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        return _CONFIG


def resolve_tensor_options(
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> tuple[torch.dtype, torch.device | str]:
    """Resolve optional tensor options against current runtime configuration.

    Parameters
    ----------
    dtype, device
        Explicit options or ``None`` to use configured defaults.

    Returns
    -------
    tuple
        Resolved dtype and device suitable for PyTorch tensor factories.
    """
    current = get_config()
    return (
        current.dtype if dtype is None else dtype,
        current.device if device is None else device,
    )


__all__ = ["RuntimeConfig", "configure", "get_config"]
