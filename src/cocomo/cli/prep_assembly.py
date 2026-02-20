#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mdsim import (
    PDBReader,
)
from mdsim.mdsim_config import format_value, read_config, write_config
from openmm.unit import nanometer

from cocomo import COCOMO, Assembly


def _as_float(name: str, s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        raise SystemExit(f"ERROR: {name} must be a float, got {s!r}") from e


def _find(tdir: Path, filename: str) -> Path:
    tdir = Path(tdir).expanduser().resolve()

    # try relative to tdir then parents
    for d in (tdir, *tdir.parents):
        candidate = d / filename
        if candidate.is_file():
            return candidate.resolve()

    # try CWD
    candidate = Path.cwd() / filename
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find '{filename}' in {tdir} or its parent directories or CWD"
    )


def _parse_config_args(argv: Sequence[str] | None = None) -> Path:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    ns, _ = p.parse_known_args(argv)
    return Path(ns.config)


@dataclass(frozen=True)
class BoxNM:
    x: float
    y: float
    z: float

    def as_units(self) -> tuple:
        return (self.x * nanometer, self.y * nanometer, self.z * nanometer)


def _parse_box_nm(s: str) -> BoxNM:
    parts = [p.strip() for p in s.split(":") if p.strip() != ""]
    if len(parts) == 1:
        x = _as_float("box", parts[0])
        return BoxNM(x, x, x)
    if len(parts) == 2:
        x = _as_float("boxx", parts[0])
        y = _as_float("boxy", parts[1])
        return BoxNM(x, y, y)
    if len(parts) == 3:
        x = _as_float("boxx", parts[0])
        y = _as_float("boxy", parts[1])
        z = _as_float("boxz", parts[2])
        return BoxNM(x, y, z)
    raise SystemExit("ERROR: --box must be 'x', 'x:y', or 'x:y:z' in nm (e.g. 22:11:9)")


def _apply_config_defaults(
    p: argparse.ArgumentParser,
    cfg: dict[str, str],
) -> None:
    defaults: dict[str, object] = {}

    if "pdb_in" in cfg:
        defaults["pdb"] = cfg["pdb_in"]
    if "box" in cfg:
        defaults["box"] = cfg["box"]
    if "surf" in cfg:
        defaults["surf"] = float(cfg["surf"])
    if "repulsion" in cfg:
        defaults["repulsion"] = float(cfg["repulsion"])

    if defaults:
        p.set_defaults(**defaults)


def _parse_args(
    cfg: dict[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prep_assembly.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--pdb",
        type=str,
        default=None,
        help="Input PDB file",
    )
    p.add_argument(
        "--box",
        type=str,
        default=None,
        help="Box size in nm: x, x:y, or x:y:z (e.g. 22:11:9).",
    )
    p.add_argument(
        "--surf",
        type=float,
        default=0.7,
        help="surface scaling parameter",
    )

    p.add_argument(
        "--repulsion",
        type=float,
        default=None,
        help="repulsive inter component term",
    )

    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device index (OpenMM platform device id)",
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help="OpenMM platform/resources string",
    )

    p.add_argument(
        "--config",
        type=Path,
        default=Path("config"),
        help="Config file (key/value) to read/write",
    )
    p.add_argument(
        "--no-write-config",
        dest="write_config",
        action="store_false",
        help="Disable writing updated config values",
    )
    p.set_defaults(write_config=True)

    _apply_config_defaults(p, cfg)
    return p.parse_args(argv)


def _split_reftile(refsel: str) -> list[str]:
    base, dot, suffix = refsel.partition(".")
    tiles = base.split(":")
    if dot:
        return [f"{t}.{suffix}" for t in tiles]
    return tiles


def main() -> None:
    tdir = Path(".")

    cfg_path = _parse_config_args()
    cfg = read_config(cfg_path)

    args = _parse_args(cfg)

    resources = str(args.resources)
    device = int(args.device)

    if args.pdb is None:
        pdb_arg = "ca.pdb"
    else:
        pdb_arg = str(args.pdb)

    if args.box is None:
        box_str = "100"
    else:
        box_str = str(args.box)

    box_nm = _parse_box_nm(box_str)
    boxx, boxy, boxz = box_nm.as_units()

    cfg_path = Path(args.config)
    if bool(args.write_config):
        cfg["pdb_in"] = format_value(pdb_arg)
        cfg["box"] = format_value(box_str)
        cfg["surf"] = format_value(args.surf)
        if args.repulsion is not None:
            cfg["repulsion"] = format_value(args.repulsion)
        write_config(cfg_path, cfg)

    pdb_path = _find(tdir, pdb_arg)
    s = PDBReader(str(pdb_path))

    surf = float(args.surf)

    components = _find(tdir, "components")

    try:
        component_types = _find(tdir, "component_types_files")
    except FileNotFoundError:
        try:
            component_types = _find(tdir, "component_types")
        except FileNotFoundError as e:
            raise SystemExit(
                "ERROR: could not find 'component_types_files' or 'component_types' "
                "in . or parent directories."
            ) from e

    try:
        interactions = _find(tdir, "interactions")
    except FileNotFoundError:
        interactions = None

    if interactions is None:
        asm = Assembly(components, component_types, structure=s)
    else:
        asm = Assembly(components, component_types, structure=s, interactions=interactions)

    if args.repulsion is not None:
        sim = COCOMO(
            asm,
            box=(boxx, boxy, boxz),
            version=2,
            surfscale=surf,
            intercomp_repulsion=float(args.repulsion),
        )
    else:
        sim = COCOMO(asm, box=(boxx, boxy, boxz), version=2, surfscale=surf)

    sim.setup_simulation(resources=resources, device=device, tstep=0.01)
    sim.write_system("system.xml")

    print(f"openmm energy: {sim.get_potentialEnergy()}")

    sim.minimize(nstep=1000)
    print(f"openmm energy: {sim.get_potentialEnergy()}")
    sim.write_state("restart_0.xml")
    sim.write_pdb("min.pdb")


if __name__ == "__main__":
    main()
