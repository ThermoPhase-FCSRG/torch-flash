"""Reject documentation math that escaped its MathJax container."""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path

_TEX_COMMAND = re.compile(r"\\(?:[A-Za-z]+|[()[\]])")
_IGNORED_TAGS = frozenset({"code", "pre", "script", "style"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _VisibleTextMathAudit(HTMLParser):
    """Find raw TeX in visible text outside math and code containers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, bool]] = []
        self.findings: list[tuple[int, str]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Track whether the current element suppresses visible-TeX checks."""
        classes = (dict(attrs).get("class") or "").split()
        inherited = self._stack[-1][1] if self._stack else False
        ignored = inherited or tag in _IGNORED_TAGS or "arithmatex" in classes
        if tag not in _VOID_TAGS:
            self._stack.append((tag, ignored))

    def handle_endtag(self, tag: str) -> None:
        """Discard the matching element and any implicitly closed children."""
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        """Record visible TeX commands that MathJax will not process."""
        ignored = self._stack[-1][1] if self._stack else False
        if not ignored and _TEX_COMMAND.search(data):
            excerpt = " ".join(data.split())[:180]
            self.findings.append((self.getpos()[0], excerpt))


def _markdown_delimiter_failures(docs_root: Path) -> list[str]:
    failures: list[str] = []
    for path in sorted(docs_root.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            line_number = index + 1
            if stripped == r"\[" and index > 0 and lines[index - 1].strip():
                failures.append(
                    f"{path}:{line_number}: display-math opening delimiter "
                    "requires a blank line before it"
                )
            if stripped == r"\]" and index + 1 < len(lines) and lines[index + 1].strip():
                failures.append(
                    f"{path}:{line_number}: display-math closing delimiter "
                    "requires a blank line after it"
                )
    return failures


def _generated_html_failures(site_root: Path) -> tuple[list[str], int]:
    failures: list[str] = []
    pages = sorted(site_root.rglob("*.html"))
    for path in pages:
        parser = _VisibleTextMathAudit()
        parser.feed(path.read_text(encoding="utf-8"))
        failures.extend(
            f"{path}:{line}: raw TeX outside a math/code container: {excerpt}"
            for line, excerpt in parser.findings
        )
    return failures, len(pages)


def main() -> int:
    """Audit Markdown delimiters and generated HTML for unprocessed TeX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=Path("docs"))
    parser.add_argument("--site-root", type=Path, default=Path("site"))
    arguments = parser.parse_args()

    failures = _markdown_delimiter_failures(arguments.docs_root)
    html_failures, page_count = _generated_html_failures(arguments.site_root)
    failures.extend(html_failures)
    if failures:
        print("\n".join(failures))
        return 1
    print(
        "documentation math audit passed: display delimiters are isolated and "
        f"{page_count} generated HTML pages contain no unprocessed TeX"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
