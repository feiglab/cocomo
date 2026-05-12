#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from mdsim import PDBReader

try:
    from cocomo import COCOMO
except ImportError:
    try:
        from .cocomo_model import COCOMO
    except ImportError:
        from cocomo_model import COCOMO

try:
    from .distumbrella_shared import (
        find_input_file,
        normalize_resources,
        output_path,
        parse_args,
        parse_config_path,
        write_args_config,
    )
    from .umbrella_config import read_config
except ImportError:
    from distumbrella_shared import (
        find_input_file,
        normalize_resources,
        output_path,
        parse_args,
        parse_config_path,
        write_args_config,
    )
    from umbrella_config import read_config


_HELP_OPTIONS = (
    "setup",
    "equi",
    "refpdb",
    "temperature",
    "gamma",
    "tstep",
    "posk",
    "minsteps",
    "equisteps",
    "equiout",
    "seed",
    "device",
    "resources",
    "config",
    "write_config",
)


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="equi_distumbrella.py")

    setup_dir = Path(args.setup).expanduser().resolve()
    equi_dir = Path(args.equi).expanduser().resolve()
    equi_dir.mkdir(parents=True, exist_ok=True)

    refpdb = "dimer.protein.pdb" if args.refpdb is None else str(args.refpdb)
    system_xml = "system.xml"
    initial_xml = "initial.xml"

    if bool(args.write_config):
        write_args_config(
            cfg_path,
            cfg,
            args,
            overrides={
                "refpdb": refpdb,
            },
        )

    refpdb_path = find_input_file(setup_dir, refpdb)
    system_path = find_input_file(setup_dir, system_xml)
    initial_path = find_input_file(setup_dir, initial_xml)

    sim = COCOMO(
        PDBReader(str(refpdb_path)).topology(),
        xml=str(system_path),
        restart=str(initial_path),
    )

    sim.set_position_restraint(selection="name CA", k=float(args.posk))
    sim.setup_simulation(
        resources=normalize_resources(str(args.resources)),
        device=int(args.device),
        temperature=float(args.temperature),
        gamma=float(args.gamma),
        tstep=float(args.tstep),
    )

    sim.minimize(nstep=int(args.minsteps))
    print(f"minimized energy: {sim.get_potentialEnergy()}")

    sim.set_velocities(seed=int(args.seed))
    sim.simulate(
        nstep=int(args.equisteps),
        nout=int(args.equiout),
        logfile=str(output_path(equi_dir, "equi.log")),
    )
    print(f"energy after equilibration: {sim.get_potentialEnergy()}")

    sim.write_state(str(output_path(equi_dir, "equi_final.xml")))


if __name__ == "__main__":
    main()
