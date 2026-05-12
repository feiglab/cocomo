#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from mdsim import PDBReader, StructureSelector

try:
    from cocomo import COCOMO
except ImportError:
    try:
        from .cocomo_model import COCOMO
    except ImportError:
        from cocomo_model import COCOMO

try:
    from .distumbrella_shared import (
        build_bias_list,
        ca_selection,
        find_input_file,
        force_constants,
        format_bias_tag,
        normalize_resources,
        output_path,
        parse_args,
        parse_config_path,
        split_selection,
        write_args_config,
    )
    from .umbrella_config import read_config
except ImportError:
    from distumbrella_shared import (
        build_bias_list,
        ca_selection,
        find_input_file,
        force_constants,
        format_bias_tag,
        normalize_resources,
        output_path,
        parse_args,
        parse_config_path,
        split_selection,
        write_args_config,
    )
    from umbrella_config import read_config


_HELP_OPTIONS = (
    "setup",
    "equi",
    "refpdb",
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
    "temperature",
    "gamma",
    "tstep",
    "initsteps",
    "initout",
    "prodsteps",
    "prodout",
    "seed",
    "device",
    "resources",
    "config",
    "write_config",
)

_AXES = ("x", "y", "z")
_EPS = 1.0e-8


def _selected_ca_indices(structure, selection: str) -> list[int]:
    indices = StructureSelector(ca_selection(selection)).atom_indices(structure)
    out = [int(i) for i in indices]
    if len(out) == 0:
        raise SystemExit(f"ERROR: selection matched no CA atoms: {selection!r}")
    return out


def _active_axes(
    biasdir: str,
    kdistx: float,
    kdisty: float,
    kdistz: float,
) -> dict[str, bool]:
    return {
        "x": biasdir == "x" or kdistx > _EPS,
        "y": biasdir == "y" or kdisty > _EPS,
        "z": biasdir == "z" or kdistz > _EPS,
    }


def _add_initial_distance_biases(
    sim,
    group_a: list[int],
    group_b: list[int],
    *,
    biasdir: str,
    biasval: float,
    kinit: float,
    active: dict[str, bool],
) -> None:
    targets = {"x": 0.0, "y": 0.0, "z": 0.0}
    targets[biasdir] = float(biasval)

    for axis in _AXES:
        if not active[axis]:
            continue
        sim.set_umbrella_xyz_distance(
            group_a,
            group_b,
            direction=axis,
            target=targets[axis],
            k=kinit,
        )


def _update_production_distance_biases(
    sim,
    *,
    biasdir: str,
    kbias: float,
    kdistx: float,
    kdisty: float,
    kdistz: float,
    active: dict[str, bool],
) -> None:
    constants = {
        "x": kdistx,
        "y": kdisty,
        "z": kdistz,
    }
    constants[biasdir] = kbias

    for axis in _AXES:
        if not active[axis]:
            continue
        sim.update_umbrella_xyz_distance(axis, constants[axis])


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="initbias_distumbrella.py")

    setup_dir = Path(args.setup).expanduser().resolve()
    equi_dir = Path(args.equi).expanduser().resolve()

    refpdb = "dimer.protein.pdb" if args.refpdb is None else str(args.refpdb)
    system_xml = "system.xml"

    if bool(args.write_config):
        write_args_config(
            cfg_path,
            cfg,
            args,
            overrides={
                "refpdb": refpdb,
            },
        )

    biasdir = str(args.biasdir).lower()
    if biasdir not in _AXES:
        raise SystemExit(f"ERROR: invalid biasdir {args.biasdir!r}; expected x, y, or z")

    kinit, kbias, kdistx, kdisty, kdistz, kcent = force_constants(args)
    active = _active_axes(biasdir, kdistx, kdisty, kdistz)
    bias_list = build_bias_list(
        bias=None if args.bias is None else str(args.bias),
        biaslist=None if args.biaslist is None else str(args.biaslist),
        base_dir=setup_dir,
    )

    pdb_path = find_input_file(setup_dir, refpdb)
    system_path = find_input_file(setup_dir, system_xml)
    restart = output_path(equi_dir, "equi_final.xml")
    if not restart.is_file():
        raise FileNotFoundError(f"Could not find equilibration state: {restart}")

    structure = PDBReader(str(pdb_path))
    group_a = _selected_ca_indices(structure, str(args.refsel))
    group_b = _selected_ca_indices(structure, str(args.othersel))
    ref_parts = split_selection(str(args.refsel))
    center_groups = [_selected_ca_indices(structure, part) for part in ref_parts]
    center_targets = [structure.center(group)[0] for group in center_groups]

    resources = normalize_resources(str(args.resources))
    print(
        "umbrella groups: "
        f"ref={len(group_a)} CA, other={len(group_b)} CA, windows={len(bias_list)}"
    )
    print(
        "force constants: "
        f"kinit={kinit:g}, kbias={kbias:g}, "
        f"kdist=({kdistx:g}, {kdisty:g}, {kdistz:g}), kcent={kcent:g}"
    )

    for window_index, biasval in enumerate(bias_list):
        tag = format_bias_tag(biasval)
        run_dir = Path(f"run_{tag}")
        run_dir.mkdir(parents=True, exist_ok=True)

        sim = COCOMO(
            structure.topology(),
            xml=str(system_path),
            restart=str(restart),
            version=2,
        )

        _add_initial_distance_biases(
            sim,
            group_a,
            group_b,
            biasdir=biasdir,
            biasval=float(biasval),
            kinit=kinit,
            active=active,
        )

        if kcent > _EPS:
            sim.set_umbrella_center(center_groups, k=kinit, target=center_targets)

        sim.set_force_groups()
        sim.write_system(str(run_dir / f"bias_system_{tag}.xml"))

        sim.setup_simulation(
            resources=resources,
            device=int(args.device),
            temperature=float(args.temperature),
            gamma=float(args.gamma),
            tstep=float(args.tstep),
        )
        sim.set_velocities(seed=int(args.seed) + window_index)

        sim.simulate(
            nstep=int(args.initsteps),
            nout=int(args.initout),
            logfile=str(run_dir / f"biasinit_{tag}.log"),
            dcdfile=str(run_dir / f"biasinit_{tag}.dcd"),
        )

        biasinit_xml = run_dir / f"biasinit_{tag}.xml"
        sim.write_state(str(biasinit_xml))

        _update_production_distance_biases(
            sim,
            biasdir=biasdir,
            kbias=kbias,
            kdistx=kdistx,
            kdisty=kdisty,
            kdistz=kdistz,
            active=active,
        )

        if kcent > _EPS:
            sim.update_umbrella_center(kcent)

        sim.simulate(
            nstep=int(args.prodsteps),
            nout=int(args.prodout),
            logfile=str(run_dir / "biasprod_0.log"),
        )

        sim.write_state(str(run_dir / "biasprod_0.xml"))
        restart = biasinit_xml
        print(f"finished {tag}")


if __name__ == "__main__":
    main()
