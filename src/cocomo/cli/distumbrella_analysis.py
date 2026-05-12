#!/usr/bin/env python3
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Optional

from mdsim import PDBReader, StructureSelector, harmonic_energy_xyz, load_dcd
from openmm.unit import kilojoule, mole, nanometer

try:
    from . import distumbrella_shared as shared
    from .umbrella_config import read_config
except ImportError:
    import distumbrella_shared as shared
    from umbrella_config import read_config

_HELP_OPTIONS = (
    "setup",
    "refpdb",
    "capdb",
    "cadcd",
    "refsel",
    "othersel",
    "bias",
    "biasdir",
    "k",
    "kbias",
    "biaslist",
    "kdist",
    "kdistx",
    "kdisty",
    "kdistz",
    "config",
    "write_config",
)

_AXES = ("x", "y", "z")


def _ca_selection(selection: str) -> str:
    if hasattr(shared, "ca_selection"):
        return shared.ca_selection(selection)

    cleaned = selection.strip()
    if cleaned.endswith(".CA"):
        return cleaned
    return f"{cleaned}.CA"


def _selected_ca_indices(structure, selection: str) -> list[int]:
    indices = StructureSelector(_ca_selection(selection)).atom_indices(structure)
    out = [int(i) for i in indices]
    if len(out) == 0:
        raise SystemExit(f"ERROR: selection matched no CA atoms: {selection!r}")
    return out


def _parse_bias_values(spec: str, base_dir: Optional[Path]) -> list[float]:
    path = _maybe_file(spec, base_dir)
    text = path.read_text(encoding="utf-8") if path is not None else spec
    cleaned = _strip_comments(text).replace(",", " ").replace(";", " ")
    values = [float(token) for token in shlex.split(cleaned)]
    if len(values) == 0:
        raise SystemExit("ERROR: bias list is empty")
    return values


def _strip_comments(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def _maybe_file(spec: str, base_dir: Optional[Path]) -> Optional[Path]:
    candidates = [Path(spec).expanduser()]
    if base_dir is not None:
        candidates.append(base_dir / spec)
    candidates.append(Path.cwd() / spec)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _build_bias_list(
    bias: Optional[str],
    biaslist: Optional[str],
    base_dir: Optional[Path],
) -> list[float]:
    if hasattr(shared, "build_bias_list"):
        return shared.build_bias_list(bias=bias, biaslist=biaslist, base_dir=base_dir)

    if biaslist is not None:
        return _parse_bias_values(biaslist, base_dir)

    pairs = shared.build_bias_pairs(bias)
    return [float(bias_value) for bias_value, _ in pairs]


def _force_constants(args) -> tuple[float, float, float, float, float, float]:
    if hasattr(shared, "force_constants"):
        return shared.force_constants(args)

    base_spec = "500:200:0:0" if args.k is None else str(args.k)
    kinit0, kbias0, kdist0, kcent0 = shared.parse_floats(
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


def _format_window_name(name: str, tag: str) -> str:
    try:
        return name.format(tag=tag, bias=tag)
    except (IndexError, KeyError, ValueError):
        return name


def _find_window_input(
    run_dir: Path,
    filename: str,
    tag: str,
    *,
    fallback: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    names = [_format_window_name(filename, tag)]
    if fallback is not None:
        names.append(_format_window_name(fallback, tag))

    start_dirs = [run_dir]
    if base_dir is not None:
        start_dirs.append(base_dir)

    errors = []
    for start_dir in start_dirs:
        for name in names:
            try:
                return shared.find_input_file(start_dir, name)
            except FileNotFoundError as exc:
                errors.append(str(exc))

    raise FileNotFoundError("\n".join(errors))


def _load_distance_vector(
    pdb_path: Path,
    dcd_path: Path,
    refsel: str,
    othersel: str,
):
    structure = PDBReader(str(pdb_path)).select_CA()
    group_a = _selected_ca_indices(structure, refsel)
    group_b = _selected_ca_indices(structure, othersel)
    traj = load_dcd(str(dcd_path), structure)
    return traj.distance_vector(group_a, group_b)


def _harmonic_axis(distance_vector, k_value: float, target: float, axis: str):
    return harmonic_energy_xyz(
        distance_vector,
        k_value * kilojoule / mole / nanometer**2,
        target * nanometer,
        axis=axis,
    )


def _value_in_unit(value, unit) -> float:
    if hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(unit))
    return float(value)


def _write_window_outputs(
    run_dir: Path,
    distance_vector,
    *,
    biasdir: str,
    biasval: float,
    kbias: float,
    kdistx: float,
    kdisty: float,
    kdistz: float,
) -> None:
    biasx = _harmonic_axis(distance_vector, kdistx, 0.0, "x")
    biasy = _harmonic_axis(distance_vector, kdisty, 0.0, "y")
    biasz = _harmonic_axis(distance_vector, kdistz, 0.0, "z")

    if biasdir == "x":
        biasx = _harmonic_axis(distance_vector, kbias, biasval, "x")
    elif biasdir == "y":
        biasy = _harmonic_axis(distance_vector, kbias, biasval, "y")
    elif biasdir == "z":
        biasz = _harmonic_axis(distance_vector, kbias, biasval, "z")
    else:
        raise SystemExit(f"ERROR: invalid biasdir {biasdir!r}")

    with (run_dir / "bias.dat").open("w", encoding="utf-8") as handle:
        handle.write("Step\tx_dist_bias[kJ/mol]\ty_dist_bias[kJ/mol]\t" "z_dist_bias[kJ/mol]\n")

        for index in range(len(distance_vector)):
            bx = _value_in_unit(biasx[index], kilojoule / mole)
            by = _value_in_unit(biasy[index], kilojoule / mole)
            bz = _value_in_unit(biasz[index], kilojoule / mole)
            handle.write(f"{index}\t{bx}\t{by}\t{bz}\n")

    with (run_dir / "geometry.dat").open("w", encoding="utf-8") as handle:
        handle.write("Step\tX_Distance[nm]\tY_Distance[nm]\tZ_Distance[nm]\n")

        for index, vector in enumerate(distance_vector):
            gx = _value_in_unit(vector[0], nanometer)
            gy = _value_in_unit(vector[1], nanometer)
            gz = _value_in_unit(vector[2], nanometer)
            handle.write(f"{index}\t{gx}\t{gy}\t{gz}\n")


def main() -> None:
    cfg_path = shared.parse_config_path()
    cfg = read_config(cfg_path)

    args = shared.parse_args(cfg, _HELP_OPTIONS, prog="distumbrella_analysis.py")

    if bool(args.write_config):
        shared.write_args_config(cfg_path, cfg, args)

    setup_dir = Path(args.setup).expanduser().resolve()
    biasdir = str(args.biasdir).lower()
    if biasdir not in _AXES:
        raise SystemExit(f"ERROR: invalid biasdir {biasdir!r}; expected x, y, or z")

    bias_list = _build_bias_list(
        bias=None if args.bias is None else str(args.bias),
        biaslist=None if args.biaslist is None else str(args.biaslist),
        base_dir=setup_dir,
    )

    force_values = _force_constants(args)
    kbias = force_values[1]
    kdistx = force_values[2]
    kdisty = force_values[3]
    kdistz = force_values[4]

    for biasval in bias_list:
        tag = shared.format_bias_tag(biasval)
        run_dir = Path(f"run_{tag}")
        run_dir.mkdir(parents=True, exist_ok=True)

        refpdb = "dimer.protein.pdb" if args.refpdb is None else str(args.refpdb)
        pdb_path = _find_window_input(
            run_dir,
            str(args.capdb),
            tag,
            fallback=refpdb,
            base_dir=setup_dir,
        )
        dcd_path = _find_window_input(
            run_dir,
            str(args.cadcd),
            tag,
            fallback=f"biasinit_{tag}.dcd",
        )

        distance_vector = _load_distance_vector(
            pdb_path,
            dcd_path,
            str(args.refsel),
            str(args.othersel),
        )
        _write_window_outputs(
            run_dir,
            distance_vector,
            biasdir=biasdir,
            biasval=float(biasval),
            kbias=kbias,
            kdistx=kdistx,
            kdisty=kdisty,
            kdistz=kdistz,
        )

        print(f"finished {tag}")


if __name__ == "__main__":
    main()
