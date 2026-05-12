#!/usr/bin/env python3
from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

FileLike = Union[str, Path]

_CONFIG_LINE_RE = re.compile(r"^(?P<key>[^\s=]+)(?:(?:\s*=\s*)|\s+)(?P<val>.*)$")


def read_config(path: FileLike) -> dict[str, str]:
    """
    Read a simple key/value config file.

    Accepted formats:
        key value
        key = value
        key=value

    Blank lines and lines starting with '#' are ignored. Keys are lowercased.
    """
    cfg_path = Path(path)
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    cfg: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        match = _CONFIG_LINE_RE.match(line)
        if match is None:
            key = line
            val = ""
        else:
            key = match.group("key").strip()
            val = match.group("val").strip()

        if key:
            cfg[key.lower()] = val

    return cfg


def parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid bool {s!r}")


def split_values(s: str) -> list[str]:
    """
    Tokenize a config value into a list.

    Shell-like quoting is supported.
    """
    return shlex.split(s)


def format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple)):
        return " ".join(format_value(x) for x in v)
    return str(v)


_ORDER = [
    "setup",
    "equi",
    "pdb_in",
    "refpdb",
    "capdb",
    "cadcd",
    "components",
    "component_types",
    "interactions",
    "refsel",
    "othersel",
    "bias",
    "biaslist",
    "biasdir",
    "k",
    "kinit",
    "kbias",
    "kdist",
    "kdistx",
    "kdisty",
    "kdistz",
    "kcent",
    "box",
    "surf",
    "temperature",
    "gamma",
    "tstep",
    "posk",
    "minsteps",
    "equisteps",
    "equiout",
    "initsteps",
    "initout",
    "prodsteps",
    "prodout",
    "seed",
    "device",
    "resources",
]


def write_config(path: FileLike, data: Mapping[str, str]) -> None:
    cfg_path = Path(path)
    keys = list(data.keys())

    ordered: list[str] = []
    seen: set[str] = set()

    for key in _ORDER:
        if key in data:
            ordered.append(key)
            seen.add(key)

    for key in sorted(keys):
        if key not in seen:
            ordered.append(key)

    lines = []
    for key in ordered:
        value = data.get(key, "")
        if value == "":
            continue
        lines.append(f"{key} {value}")

    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
