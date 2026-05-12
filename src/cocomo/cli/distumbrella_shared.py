#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

try:
    from .umbrella_config import format_value, parse_bool, split_values, write_config
except ImportError:
    from umbrella_config import format_value, parse_bool, split_values, write_config

FileLike = Union[str, Path]

_PERSISTENT_DESTS = (
    "setup",
    "equi",
    "pdb",
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
)

_K_INDIVIDUAL_DESTS = (
    "kinit",
    "kbias",
    "kdist",
    "kdistx",
    "kdisty",
    "kdistz",
    "kcent",
)

_PATH_DESTS = {
    "setup",
    "equi",
    "refpdb",
}

_FLOAT_DESTS = {
    "surf",
    "kinit",
    "kbias",
    "kdist",
    "kdistx",
    "kdisty",
    "kdistz",
    "kcent",
    "temperature",
    "gamma",
    "tstep",
    "posk",
}

_INT_DESTS = {
    "device",
    "minsteps",
    "equisteps",
    "equiout",
    "initsteps",
    "initout",
    "prodsteps",
    "prodout",
    "seed",
}

_CONFIG_KEY_BY_DEST = {
    "pdb": "pdb_in",
}


def _config_key(dest: str) -> str:
    return _CONFIG_KEY_BY_DEST.get(dest, dest)


def _config_default(dest: str, value: str) -> object:
    if dest in _PATH_DESTS:
        return Path(value)
    if dest in _FLOAT_DESTS:
        return float(value)
    if dest in _INT_DESTS:
        return int(value)
    if dest in {"orient", "flip"}:
        return parse_bool(value)
    if dest == "ff":
        return split_values(value)
    return value


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _normalize_config_value(dest: str, value: object) -> object:
    if value is None:
        return value
    if dest == "biaslist":
        return normalize_bias_list_arg(str(value))
    return value


def _resolve_config_value(
    dest: str,
    args: argparse.Namespace,
    value: object,
) -> object:
    if value is not None:
        return value

    if dest == "bias":
        if getattr(args, "biaslist", None) is None:
            return "6.0:9.0:0.1"
        return None

    if dest == "k":
        has_individual = any(getattr(args, name, None) is not None for name in _K_INDIVIDUAL_DESTS)
        if not has_individual:
            return "500:200"
        return None

    return value


def _normalize_visible(visible: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for item in visible:
        name = item.strip()
        if not name:
            continue
        names.add(name)
        names.add(name.lstrip("-").replace("-", "_"))
    return names


def _help_text(visible: set[str], dest: str, text: str) -> str:
    return text if dest in visible else argparse.SUPPRESS


def apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    for dest in _PERSISTENT_DESTS:
        key = _config_key(dest)
        if key not in cfg:
            continue
        defaults[dest] = _config_default(dest, cfg[key])

    if defaults:
        p.set_defaults(**defaults)


def _add_all_arguments(
    p: argparse.ArgumentParser,
    visible: set[str],
) -> None:
    p.add_argument(
        "--setup",
        type=Path,
        default=Path("setup"),
        help=_help_text(visible, "setup", "Setup directory"),
    )
    p.add_argument(
        "--equi",
        type=Path,
        default=Path("equi"),
        help=_help_text(visible, "equi", "Equilibration output directory"),
    )
    p.add_argument(
        "--pdb",
        type=str,
        default=None,
        help=_help_text(
            visible,
            "pdb",
            "Input PDB file. Searched relative to --setup, parents, then CWD.",
        ),
    )
    p.add_argument(
        "--refpdb",
        type=Path,
        default=None,
        help=_help_text(visible, "refpdb", "Reference PDB for selections/restraints"),
    )
    p.add_argument(
        "--capdb",
        type=str,
        default="CA.pdb",
        help=_help_text(visible, "capdb", "Reference trajectory (CA only)"),
    )
    p.add_argument(
        "--cadcd",
        type=str,
        default="ca.dcd",
        help=_help_text(visible, "cadcd", "Reference PDB (CA only)"),
    )
    p.add_argument(
        "--components",
        type=str,
        default="dimer.components",
        help=_help_text(visible, "components", "Assembly component list"),
    )
    p.add_argument(
        "--component-types",
        dest="component_types",
        type=str,
        default="component_types_files",
        help=_help_text(visible, "component_types", "Assembly component-type list"),
    )
    p.add_argument(
        "--interactions",
        type=str,
        default="interactions",
        help=_help_text(visible, "interactions", "Assembly interaction file"),
    )
    p.add_argument(
        "--refsel",
        type=str,
        default="A:B:C:D:E:F.2-91",
        help=_help_text(visible, "refsel", "Reference component selection"),
    )
    p.add_argument(
        "--othersel",
        type=str,
        default="G:H:I:J:K:L.2-91",
        help=_help_text(visible, "othersel", "Other component selection"),
    )
    p.add_argument(
        "--bias",
        type=str,
        default=None,
        help=_help_text(visible, "bias", "Bias range as 'min:max:delta' in nm"),
    )
    p.add_argument(
        "--biaslist",
        type=str,
        default=None,
        help=_help_text(visible, "biaslist", "Explicit bias values or value file"),
    )
    p.add_argument(
        "--biasdir",
        type=str,
        default="x",
        help=_help_text(visible, "biasdir", "Bias direction: x, y, or z"),
    )
    p.add_argument(
        "--k",
        type=str,
        default=None,
        help=_help_text(visible, "k", "Force constants: kinit:kbias[:kdist[:kcent]]"),
    )
    p.add_argument(
        "--kinit",
        type=float,
        default=None,
        help=_help_text(visible, "kinit", "Initial force constant"),
    )
    p.add_argument(
        "--kbias",
        type=float,
        default=None,
        help=_help_text(visible, "kbias", "Production force constant for bias axis"),
    )
    p.add_argument(
        "--kdist",
        type=float,
        default=None,
        help=_help_text(visible, "kdist", "Default orthogonal distance force constant"),
    )
    p.add_argument(
        "--kdistx",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdistx",
            "Force constant for distance x, if not bias",
        ),
    )
    p.add_argument(
        "--kdisty",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdisty",
            "Force constant for distance y, if not bias",
        ),
    )
    p.add_argument(
        "--kdistz",
        type=float,
        default=None,
        help=_help_text(
            visible,
            "kdistz",
            "Force constant for distance z, if not bias",
        ),
    )
    p.add_argument(
        "--kcent",
        type=float,
        default=None,
        help=_help_text(visible, "kcent", "Force constant for central force"),
    )
    p.add_argument(
        "--box",
        type=str,
        default=None,
        help=_help_text(visible, "box", "Box size in nm: x, x:y, or x:y:z"),
    )
    p.add_argument(
        "--surf",
        type=float,
        default=0.7,
        help=_help_text(visible, "surf", "COCOMO surface scaling parameter"),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help=_help_text(visible, "temperature", "Temperature in K"),
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help=_help_text(visible, "gamma", "Langevin friction in 1/ps"),
    )
    p.add_argument(
        "--tstep",
        type=float,
        default=0.03,
        help=_help_text(visible, "tstep", "Integrator timestep in ps"),
    )
    p.add_argument(
        "--posk",
        type=float,
        default=10.0,
        help=_help_text(visible, "posk", "Equilibration CA restraint in kJ/mol/nm^2"),
    )
    p.add_argument(
        "--minsteps",
        type=int,
        default=1000,
        help=_help_text(visible, "minsteps", "Energy minimization iterations"),
    )
    p.add_argument(
        "--equisteps",
        type=int,
        default=1000,
        help=_help_text(visible, "equisteps", "Equilibration MD steps"),
    )
    p.add_argument(
        "--equiout",
        type=int,
        default=1000,
        help=_help_text(visible, "equiout", "Equilibration report interval"),
    )
    p.add_argument(
        "--initsteps",
        type=int,
        default=1000,
        help=_help_text(visible, "initsteps", "Initial pulling steps per window"),
    )
    p.add_argument(
        "--initout",
        type=int,
        default=200,
        help=_help_text(visible, "initout", "Initial pulling report interval"),
    )
    p.add_argument(
        "--prodsteps",
        type=int,
        default=200,
        help=_help_text(visible, "prodsteps", "Short production steps per window"),
    )
    p.add_argument(
        "--prodout",
        type=int,
        default=100,
        help=_help_text(visible, "prodout", "Short production report interval"),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help=_help_text(visible, "seed", "Base random seed"),
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help=_help_text(visible, "device", "OpenMM device index"),
    )
    p.add_argument(
        "--resources",
        type=str,
        default="auto",
        help=_help_text(visible, "resources", "OpenMM resources: auto, CUDA, or CPU"),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help=_help_text(visible, "config", "Config file to read/write"),
    )
    p.add_argument(
        "--no-write-config",
        dest="write_config",
        action="store_false",
        help=_help_text(visible, "write_config", "Disable writing updated config"),
    )
    p.set_defaults(write_config=True)


def _explicit_dests(
    p: argparse.ArgumentParser,
    argv: Optional[Sequence[str]],
) -> set[str]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    option_map: dict[str, str] = {}

    for action in p._actions:
        for option in action.option_strings:
            option_map[option] = action.dest

    explicit: set[str] = set()
    for token in tokens:
        if token == "--":
            break
        if not token.startswith("-"):
            continue

        option = token.split("=", 1)[0]
        dest = option_map.get(option)
        if dest not in {None, "help"}:
            explicit.add(dest)

    return explicit


def parse_args(
    cfg: dict[str, str],
    visible: Sequence[str],
    argv: Optional[Sequence[str]] = None,
    *,
    prog: str,
) -> argparse.Namespace:
    visible_names = _normalize_visible(visible)
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_all_arguments(parser, visible_names)
    apply_config_defaults(parser, cfg)

    args = parser.parse_args(argv)
    setattr(args, "_distumbrella_visible", frozenset(visible_names))
    setattr(args, "_distumbrella_explicit", frozenset(_explicit_dests(parser, argv)))
    return args


def write_args_config(
    cfg_path: FileLike,
    cfg: dict[str, str],
    args: argparse.Namespace,
    *,
    overrides: Optional[dict[str, object]] = None,
) -> dict[str, str]:
    out = dict(cfg)
    visible = set(getattr(args, "_distumbrella_visible", ()))
    explicit = set(getattr(args, "_distumbrella_explicit", ()))
    override_map: dict[str, object] = {}

    if overrides is not None:
        for name, value in overrides.items():
            key = _config_key(name) if name in _PERSISTENT_DESTS else name
            override_map[key] = value

    for dest in _PERSISTENT_DESTS:
        key = _config_key(dest)
        include = dest in visible or dest in explicit or key in out or key in override_map
        if not include or not hasattr(args, dest):
            continue

        if key in override_map:
            value = override_map.pop(key)
        else:
            value = getattr(args, dest)

        resolved = _resolve_config_value(dest, args, value)
        out[key] = format_value(_normalize_config_value(dest, resolved))

    for key, value in override_map.items():
        out[key] = format_value(value)

    write_config(cfg_path, out)
    return out


def parse_config_path(argv: Optional[Sequence[str]] = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=Path("config"))
    ns, _ = parser.parse_known_args(argv)
    return Path(ns.config)


def parse_floats(spec: str, defaults: Sequence[float], n_out: int) -> list[float]:
    values = [float(item) for item in spec.split(":") if item.strip()]
    out: list[float] = []
    last: Optional[float] = None

    for idx in range(n_out):
        if idx < len(values):
            last = values[idx]
            out.append(last)
            continue
        if idx < len(defaults):
            last = float(defaults[idx])
            out.append(last)
            continue
        out.append(last if last is not None else 0.0)

    return out


def find_input_file(start_dir: FileLike, filename: FileLike) -> Path:
    path = Path(filename).expanduser()
    if path.is_absolute():
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Could not find '{path}'")

    root = Path(start_dir).expanduser().resolve()
    for directory in (root, *root.parents):
        candidate = directory / path
        if candidate.is_file():
            return candidate.resolve()

    candidate = Path.cwd() / path
    if candidate.is_file():
        return candidate.resolve()

    msg = f"Could not find '{filename}' in {root}, its parents, or CWD"
    raise FileNotFoundError(msg)


def output_path(base_dir: FileLike, filename: FileLike) -> Path:
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir).expanduser().resolve() / path


def split_selection(selection: str) -> list[str]:
    base, dot, suffix = selection.partition(".")
    parts = [part for part in base.split(":") if part]
    if dot:
        return [f"{part}.{suffix}" for part in parts]
    return parts


def ca_selection(selection: str) -> str:
    cleaned = selection.strip()
    if cleaned.endswith(".CA"):
        return cleaned
    return f"{cleaned}.CA"


def format_bias_tag(bias: float) -> str:
    return f"{bias:.2f}"


def float_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise SystemExit(f"ERROR: range step must be > 0, got {step!r}")

    values: list[float] = []
    index = 0
    limit = stop + 1.0e-8

    while True:
        value = start + index * step
        if value > limit:
            break
        values.append(round(value, 12))
        index += 1

    return values


def _clean_value_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def _maybe_biaslist_file(spec: str, base_dir: Optional[FileLike]) -> Optional[Path]:
    raw = spec.strip()
    if not raw:
        return None

    candidates = [Path(raw).expanduser()]
    if base_dir is not None:
        candidates.append(Path(base_dir).expanduser().resolve() / raw)
    candidates.append(Path.cwd() / raw)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def parse_bias_list_arg(
    spec: str,
    *,
    base_dir: Optional[FileLike] = None,
) -> list[float]:
    path = _maybe_biaslist_file(spec, base_dir)
    if path is not None:
        text = path.read_text(encoding="utf-8")
    else:
        text = spec

    cleaned = _clean_value_text(text).replace(",", " ").replace(";", " ")
    tokens = shlex.split(cleaned)
    if not tokens:
        raise SystemExit("ERROR: bias list is empty")

    values: list[float] = []
    for token in tokens:
        if token.count(":") == 2:
            bmin, bmax, bdel = parse_floats(token, [6.0, 9.0, 0.1], n_out=3)
            values.extend(float_range(bmin, bmax, bdel))
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid bias value {token!r}") from exc

    return values


def normalize_bias_list_arg(spec: str) -> str:
    if _maybe_biaslist_file(spec, None) is not None:
        return spec
    try:
        values = parse_bias_list_arg(spec)
    except SystemExit:
        return spec
    return " ".join(_format_float(value) for value in values)


def build_bias_list(
    bias: Optional[str],
    biaslist: Optional[str] = None,
    *,
    base_dir: Optional[FileLike] = None,
) -> list[float]:
    if biaslist is not None:
        return parse_bias_list_arg(biaslist, base_dir=base_dir)

    bias_spec = "6.0:9.0:0.1" if bias is None else bias
    bmin, bmax, bdel = parse_floats(bias_spec, [6.0, 9.0, 0.1], n_out=3)
    return float_range(bmin, bmax, bdel)


def force_constants(args: argparse.Namespace) -> tuple[float, float, float, float, float, float]:
    base_spec = "500:200:0:0" if args.k is None else str(args.k)
    kinit0, kbias0, kdist0, kcent0 = parse_floats(
        base_spec,
        [500.0, 200.0, 0.0, 0.0],
        n_out=4,
    )

    kinit = kinit0 if args.kinit is None else float(args.kinit)
    kbias = kbias0 if args.kbias is None else float(args.kbias)
    kdist = kdist0 if args.kdist is None else float(args.kdist)
    kcent = kcent0 if args.kcent is None else float(args.kcent)

    kdistx = kdist if args.kdistx is None else float(args.kdistx)
    kdisty = kdist if args.kdisty is None else float(args.kdisty)
    kdistz = kdist if args.kdistz is None else float(args.kdistz)

    return kinit, kbias, kdistx, kdisty, kdistz, kcent


def normalize_resources(resources: str) -> str:
    requested = resources.strip()
    if requested.lower() != "auto":
        return requested

    try:
        from openmm import OpenMMException, Platform
    except ImportError:
        return "CPU"

    for name in ("CUDA", "CPU"):
        try:
            Platform.getPlatformByName(name)
        except OpenMMException:
            continue
        return name

    return "CPU"


def parse_bias_target(spec: str) -> tuple[float, Optional[float]]:
    value = spec.strip()
    if not value:
        raise SystemExit("ERROR: empty bias value")

    if _count_pair_separators(value) == 0:
        try:
            return float(value), None
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid bias value {spec!r}") from exc

    pairs = parse_bias_pairs_arg(value)
    if len(pairs) != 1:
        msg = "ERROR: --bias must define exactly one pair, got expanded input "
        raise SystemExit(msg + f"{spec!r}")
    bias, biasangle = pairs[0]
    return bias, biasangle


def parse_bias_pairs_arg(spec: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []

    for item in _iter_bias_pair_items(spec):
        bias_spec, angle_spec = _split_bias_item(item)
        bias_values = _parse_value_spec(bias_spec, name="bias")
        angle_values = _parse_value_spec(angle_spec, name="biasangle")

        for bias in bias_values:
            for angle in angle_values:
                pairs.append((bias, angle))

    if not pairs:
        raise SystemExit("ERROR: biaspairs must define at least one pair")

    return pairs


def normalize_bias_pairs_arg(spec: str) -> str:
    return "=".join(_iter_bias_pair_items(spec))


def build_bias_pairs(
    bias: Optional[str],
    biasangle: Optional[str] = None,
    biaspairs: Optional[str] = None,
) -> list[tuple[float, Optional[float]]]:
    if biaspairs is not None:
        return [(biasval, angleval) for biasval, angleval in parse_bias_pairs_arg(biaspairs)]

    bias_values = build_bias_list(bias)

    if biasangle is None:
        return [(biasval, None) for biasval in bias_values]

    amin, amax, adel = parse_floats(biasangle, [90.0, 180.0, 15.0], n_out=3)
    angle_values = float_range(amin, amax, adel)
    return [(biasval, angleval) for biasval in bias_values for angleval in angle_values]


def _iter_bias_pair_items(spec: str) -> list[str]:
    items: list[str] = []

    for raw_item in _split_top_level(spec, "="):
        item = raw_item.strip()
        if not item:
            continue

        try:
            _split_bias_item(item)
        except SystemExit:
            subitems = _split_top_level_whitespace(item)
            if len(subitems) <= 1:
                raise

            for subitem in subitems:
                cleaned = subitem.strip()
                if not cleaned:
                    continue
                _split_bias_item(cleaned)
                items.append(cleaned)
        else:
            items.append(item)

    return items


def _count_pair_separators(spec: str) -> int:
    return len(_pair_separator_positions(spec))


def _split_bias_item(item: str) -> tuple[str, str]:
    positions = _pair_separator_positions(item)
    if len(positions) != 1:
        msg = (
            "ERROR: each biaspairs entry must be 'bias:biasangle', "
            "'bias_biasangle', or use braces such as "
            "'5.0:{90,120}={5.4,5.6}:{90,120}'"
        )
        raise SystemExit(msg)

    idx = positions[0]
    left = item[:idx].strip()
    right = item[idx + 1 :].strip()
    if not left or not right:
        raise SystemExit(f"ERROR: invalid biaspairs entry {item!r}")

    return left, right


def _pair_separator_positions(spec: str) -> list[int]:
    positions: list[int] = []
    depth = 0

    for idx, char in enumerate(spec):
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")
            continue
        if depth == 0 and char in {":", "_"}:
            positions.append(idx)

    if depth != 0:
        raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")

    return positions


def _parse_value_spec(spec: str, name: str) -> list[float]:
    token = spec.strip()
    if not token:
        raise SystemExit(f"ERROR: empty {name} in biaspairs")

    if token.startswith("{") or token.endswith("}"):
        if not (token.startswith("{") and token.endswith("}")):
            raise SystemExit(f"ERROR: invalid {name} list {spec!r}")
        inner = token[1:-1].strip()
        if not inner:
            raise SystemExit(f"ERROR: empty {name} list in {spec!r}")
        items = [part.strip() for part in _split_top_level(inner, ",")]
    else:
        items = [token]

    values: list[float] = []
    for item in items:
        if not item:
            raise SystemExit(f"ERROR: empty {name} entry in {spec!r}")
        try:
            values.append(float(item))
        except ValueError as exc:
            raise SystemExit(f"ERROR: invalid {name} value {item!r}") from exc

    return values


def _split_top_level(spec: str, sep: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0

    for idx, char in enumerate(spec):
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")
            continue
        if char == sep and depth == 0:
            parts.append(spec[start:idx])
            start = idx + 1

    if depth != 0:
        raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")

    parts.append(spec[start:])
    return parts


def _split_top_level_whitespace(spec: str) -> list[str]:
    parts: list[str] = []
    token: list[str] = []
    depth = 0

    for char in spec:
        if char == "{":
            depth += 1
            token.append(char)
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")
            token.append(char)
            continue
        if depth == 0 and char.isspace():
            if token:
                parts.append("".join(token))
                token = []
            continue
        token.append(char)

    if depth != 0:
        raise SystemExit(f"ERROR: unbalanced braces in {spec!r}")

    if token:
        parts.append("".join(token))
    return parts
