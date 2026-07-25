"""Run ordinary tests with xdist and resource-intensive tests exclusively."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

NO_TESTS_COLLECTED = 5


def _split_marker_expression(arguments: Sequence[str]) -> tuple[list[str], str | None]:
    """Remove pytest marker options and return their combined expression."""
    remaining: list[str] = []
    expressions: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "-m":
            if index + 1 >= len(arguments):
                raise ValueError("-m requires a marker expression")
            expressions.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("-m="):
            expressions.append(argument.removeprefix("-m="))
            index += 1
            continue
        remaining.append(argument)
        index += 1
    if not expressions:
        return remaining, None
    return remaining, " and ".join(f"({expression})" for expression in expressions)


def _phase_marker(selected: str | None, *, serial: bool) -> str:
    isolation = "serial" if serial else "not serial"
    if selected is None:
        return isolation
    return f"({selected}) and ({isolation})"


def _run_pytest_phase(
    arguments: Sequence[str],
    selected: str | None,
    *,
    serial: bool,
    coverage: bool,
) -> int:
    label = "serial tests (exclusive)" if serial else "ordinary tests (xdist)"
    print(f"\n=== {label} ===", flush=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *arguments,
        "-m",
        _phase_marker(selected, serial=serial),
        "-n",
        "0" if serial else "auto",
    ]
    if coverage:
        command.extend(
            (
                "--cov=torch_flash",
                "--cov-branch",
                "--cov-fail-under=0",
                "--cov-report=",
            )
        )
        if serial:
            command.append("--cov-append")
    return subprocess.run(command, check=False).returncode


def _tests_succeeded(return_codes: Sequence[int]) -> bool:
    return any(code == 0 for code in return_codes) and all(
        code in (0, NO_TESTS_COLLECTED) for code in return_codes
    )


def _coverage_reports() -> int:
    xml = subprocess.run(
        [sys.executable, "-m", "coverage", "xml", "-o", "coverage.xml"],
        check=False,
    )
    report = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--fail-under=99"],
        check=False,
    )
    return xml.returncode or report.returncode


def main(arguments: Sequence[str] | None = None) -> int:
    """Run xdist-safe and serial tests in separate pytest sessions."""
    parser = argparse.ArgumentParser(
        description=(
            "Run tests not marked serial with xdist, then stop all workers and "
            "run serial-marked tests exclusively."
        )
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="collect combined branch coverage and enforce the configured threshold",
    )
    options, pytest_arguments = parser.parse_known_args(arguments)
    try:
        pytest_arguments, selected = _split_marker_expression(pytest_arguments)
    except ValueError as error:
        parser.error(str(error))

    parallel_code = _run_pytest_phase(
        pytest_arguments,
        selected,
        serial=False,
        coverage=options.coverage,
    )
    serial_code = _run_pytest_phase(
        pytest_arguments,
        selected,
        serial=True,
        coverage=options.coverage,
    )
    return_codes = (parallel_code, serial_code)
    if not _tests_succeeded(return_codes):
        for return_code in return_codes:
            if return_code not in (0, NO_TESTS_COLLECTED):
                return return_code
        return NO_TESTS_COLLECTED
    if options.coverage:
        return _coverage_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
