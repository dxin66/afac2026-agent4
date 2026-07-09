from __future__ import annotations

import html
import re


_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    lines = []
    for raw_line in text.splitlines():
        line = _SPACE_RE.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return _BLANK_LINE_RE.sub("\n\n", "\n".join(lines)).strip()


def compact_for_query(text: str) -> str:
    return _SPACE_RE.sub(" ", normalize_text(text)).replace("\n", " ").strip()


def first_nonempty_line(text: str, default: str) -> str:
    for line in normalize_text(text).splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return default

