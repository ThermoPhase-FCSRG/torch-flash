"""Check that every callable in the documented public API has detailed docs."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from types import ModuleType

DOCUMENTED_MODULES = (
    "torch_flash",
    "torch_flash.constants",
    "torch_flash.config",
    "torch_flash.components",
    "torch_flash.database",
    "torch_flash.eos",
    "torch_flash.activity",
    "torch_flash.mixing",
    "torch_flash.initialization",
    "torch_flash.flash",
    "torch_flash.material_balance",
    "torch_flash.properties",
    "torch_flash.standard_state",
    "torch_flash.envelope",
    "torch_flash.characterization",
    "torch_flash.parameters",
    "torch_flash.fitting",
    "torch_flash.transport",
    "torch_flash.backends",
    "torch_flash.solvers",
    "torch_flash.types",
    "torch_flash.exceptions",
)


def _public_names(module: ModuleType) -> Iterable[str]:
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return exported
    return (
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and getattr(value, "__module__", None) == module.__name__
    )


def main() -> int:
    """Return nonzero when a documented public callable lacks API sections."""
    failures: list[str] = []
    audited: set[tuple[str, str]] = set()
    for module_name in DOCUMENTED_MODULES:
        module = importlib.import_module(module_name)
        for name in _public_names(module):
            value = getattr(module, name)
            identity = (
                getattr(value, "__module__", module_name),
                getattr(value, "__qualname__", name),
            )
            if identity in audited or not (inspect.isfunction(value) or inspect.isclass(value)):
                continue
            audited.add(identity)
            docstring = inspect.getdoc(value)
            if not docstring:
                failures.append(f"{module_name}.{name}: missing docstring")
                continue
            if inspect.isfunction(value):
                has_parameters = bool(inspect.signature(value).parameters)
                if has_parameters and "Parameters\n----------" not in docstring:
                    failures.append(f"{module_name}.{name}: missing Parameters section")
                if "Returns\n-------" not in docstring:
                    failures.append(f"{module_name}.{name}: missing Returns section")
            elif not issubclass(value, Exception | Warning) and not any(
                section in docstring
                for section in (
                    "Parameters\n----------",
                    "Attributes\n----------",
                    "Methods\n-------",
                )
            ):
                failures.append(f"{module_name}.{name}: missing Parameters or Attributes section")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"API documentation check passed for {len(audited)} public callables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
