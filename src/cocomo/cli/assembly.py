#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from cocomo import COCOMO


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="assembly.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--run",
        dest="nrun",
        type=int,
        default=1,
        help="Production run index to write (expects restart from run-1)",
    )
    p.add_argument(
        "--nstep",
        type=int,
        default=100000,
        help="Number of MD steps",
    )
    p.add_argument(
        "--tstep",
        type=float,
        default=0.01,
        help="Timestep",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Langevin friction (1/ps)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Temperature (K)",
    )
    p.add_argument(
        "--nout",
        type=int,
        default=10000,
        help="Output/report interval (steps)",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="OpenMM resource device index",
    )
    p.add_argument(
        "--resources",
        type=str,
        default="CUDA",
        help="OpenMM platform/resources string",
    )

    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.nrun < 0:
        raise SystemExit("ERROR: --run must be >= 0")

    last = args.nrun - 1
    restart = Path(f"restart_{last}.xml")
    if not restart.is_file():
        raise SystemExit(f"ERROR: restart file does not exist: {restart}")

    sysxml = Path("system.xml")
    if not sysxml.is_file():
        raise SystemExit(f"ERROR: system xml does not exist: {sysxml}")

    sim = COCOMO(xml=str(sysxml), restart=str(restart))

    sim.setup_simulation(
        resources=str(args.resources),
        device=int(args.device),
        temperature=float(args.temperature),
        tstep=float(args.tstep),
        gamma=float(args.gamma),
    )

    nrun = int(args.nrun)
    sim.simulate(
        nstep=int(args.nstep),
        nout=int(args.nout),
        logfile=str(f"run_{nrun}.log"),
        dcdfile=str(f"run_{nrun}.dcd"),
    )
    sim.write_state(f"restart_{nrun}.xml")


if __name__ == "__main__":
    main()
