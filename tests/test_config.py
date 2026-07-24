from __future__ import annotations

import importlib

import pytest
import torch

from torch_flash import (
    PseudoComponentCut,
    RuntimeConfig,
    activity_model,
    component_set,
    configure,
    cpa_components_from_cuts,
    cpa_folas_2005,
    eoscg2021,
    gerg2008,
    get_config,
    pedersen_logarithmic_split,
    peng_robinson_1978,
    poling_ideal_gas,
    whitson_gamma_split,
)

runtime_module = importlib.import_module("torch_flash.config")


@pytest.fixture(autouse=True)
def _restore_pytorch_runtime():
    saved_config = runtime_module._CONFIG
    saved_dtype = torch.get_default_dtype()
    saved_threads = torch.get_num_threads()
    saved_deterministic = torch.are_deterministic_algorithms_enabled()
    saved_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.set_default_dtype(saved_dtype)
    torch.set_num_threads(saved_threads)
    torch.use_deterministic_algorithms(
        saved_deterministic,
        warn_only=saved_warn_only,
    )
    runtime_module._CONFIG = saved_config


def test_runtime_snapshot_and_tensor_helpers():
    runtime = configure(device="cpu", dtype=torch.float32)

    assert isinstance(runtime, RuntimeConfig)
    assert runtime == get_config()
    assert runtime.device == torch.device("cpu")
    assert runtime.dtype == torch.float32
    assert not runtime.accelerated
    assert runtime.tensor_options == {
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    assert torch.get_default_dtype() == torch.float32

    copied = runtime.tensor([1.0, 2.0], requires_grad=True)
    converted = runtime.as_tensor(copied)
    overridden = runtime.tensor([1.0], dtype=torch.float64)
    assert copied.dtype == torch.float32
    assert copied.device.type == "cpu"
    assert copied.requires_grad
    assert converted is copied
    assert overridden.dtype == torch.float64

    accelerator_snapshot = RuntimeConfig(
        torch.device("cuda"),
        torch.float64,
        1,
        1,
        False,
        False,
    )
    assert accelerator_snapshot.accelerated


def test_configured_factories_and_explicit_overrides():
    configure(device="cpu", dtype=torch.float32)

    components = component_set(("methane", "n_butane"))
    cubic = peng_robinson_1978(components)
    activity = activity_model(
        "pedersen-hv-propane-water",
        ("water", "propane"),
        covolumes=torch.tensor([2.0e-5, 8.0e-5], dtype=torch.float64),
    )
    cpa = cpa_folas_2005(("water",))
    gerg = gerg2008(("hydrogen", "methane"))
    eoscg = eoscg2021(("carbon_dioxide", "hydrogen"))
    standard_state = poling_ideal_gas(("methane",))
    pedersen = pedersen_logarithmic_split(0.1, 0.2, max_carbon_number=30)
    whitson = whitson_gamma_split(0.041, 0.227, max_carbon_number=20)
    cuts = cpa_components_from_cuts(
        (
            PseudoComponentCut(
                "C7-C10",
                1.0,
                430.0,
                0.76,
                0.12,
            ),
        )
    )

    tensors = (
        components.critical_temperature,
        cubic.critical_temperature,
        activity.energy_over_r,
        activity.covolumes,
        cpa.critical_temperature,
        gerg.critical_temperature,
        eoscg.critical_temperature,
        standard_state.heat_capacity_coefficients,
        pedersen.mole_fractions,
        whitson.mole_fractions,
        cuts.mole_fractions,
    )
    assert all(tensor.dtype == torch.float32 for tensor in tensors)
    assert all(tensor.device.type == "cpu" for tensor in tensors)

    overridden_components = component_set(("methane",), dtype=torch.float64)
    overridden_cpa = cpa_folas_2005(("water",), dtype=torch.float64)
    overridden_gerg = gerg2008(("methane",), dtype=torch.float64)
    tensor_selected_cpa = cpa_folas_2005(
        ("water",),
        kij=torch.zeros((1, 1), dtype=torch.float64),
    )
    assert overridden_components.critical_temperature.dtype == torch.float64
    assert overridden_cpa.critical_temperature.dtype == torch.float64
    assert overridden_gerg.critical_temperature.dtype == torch.float64
    assert tensor_selected_cpa.critical_temperature.dtype == torch.float64


def test_cpu_thread_and_deterministic_controls():
    old_threads = torch.get_num_threads()
    selected_threads = 1 if old_threads != 1 else 2
    runtime = configure(
        device="cpu",
        num_threads=selected_threads,
        deterministic=True,
        deterministic_warn_only=True,
    )

    assert runtime.num_threads == selected_threads
    assert torch.get_num_threads() == selected_threads
    assert runtime.deterministic
    assert runtime.deterministic_warn_only

    runtime = configure(deterministic=False)
    assert not runtime.deterministic
    assert not runtime.deterministic_warn_only


def test_interop_threads_are_applied_once_and_fail_clearly(monkeypatch):
    state = {"threads": torch.get_num_interop_threads()}

    def get_threads():
        return state["threads"]

    def set_threads(value):
        state["threads"] = value

    monkeypatch.setattr(torch, "get_num_interop_threads", get_threads)
    monkeypatch.setattr(torch, "set_num_interop_threads", set_threads)
    configured = configure(num_interop_threads=state["threads"] + 1)
    assert configured.num_interop_threads == state["threads"]

    def fail(_value):
        raise RuntimeError("parallel work already started")

    monkeypatch.setattr(torch, "set_num_interop_threads", fail)
    with pytest.raises(RuntimeError, match="must be configured once"):
        configure(num_interop_threads=state["threads"] + 1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"dtype": torch.float16}, "float32 or torch.float64"),
        ({"num_threads": 0}, "positive integer"),
        ({"num_threads": True}, "positive integer"),
        ({"num_interop_threads": -1}, "positive integer"),
        ({"deterministic": "yes"}, "must be a boolean"),
        ({"deterministic_warn_only": "yes"}, "must be a boolean"),
        (
            {"deterministic": False, "deterministic_warn_only": True},
            "requires deterministic=True",
        ),
        ({"device": "meta"}, "unsupported"),
        ({"device": "not-a-device"}, "invalid PyTorch device"),
    ],
)
def test_invalid_runtime_configuration(arguments, message):
    with pytest.raises(ValueError, match=message):
        configure(**arguments)


def test_device_selection_and_unavailable_device_errors(monkeypatch):
    unavailable = RuntimeError("unavailable")

    def fail_allocation(*_args, **_kwargs):
        raise unavailable

    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", fail_allocation)
    assert runtime_module._device_error(torch.device("cpu"), torch.float64) is unavailable
    monkeypatch.setattr(torch, "empty", original_empty)

    monkeypatch.setattr(
        runtime_module,
        "_device_error",
        lambda device, _dtype: None if device.type == "cuda" else unavailable,
    )
    assert configure(device="gpu").device.type == "cuda"

    def cpu_only(device, _dtype):
        return None if device.type == "cpu" else unavailable

    monkeypatch.setattr(runtime_module, "_device_error", cpu_only)
    assert configure(device="auto").device.type == "cpu"
    with pytest.raises(RuntimeError, match="no PyTorch GPU"):
        configure(device="gpu")
    with pytest.raises(RuntimeError, match="unavailable or does not support"):
        configure(device="cuda")

    monkeypatch.setattr(runtime_module, "_device_error", lambda _device, _dtype: unavailable)
    with pytest.raises(RuntimeError, match="no PyTorch device"):
        configure(device="auto")
