#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdsim import PDBReader, StructureSelector
from openmm.unit import nanometer

try:
    from cocomo import COCOMO, Assembly
except ImportError:
    try:
        from .cocomo_model import COCOMO
        from .system_handling import Assembly
    except ImportError:
        from cocomo_model import COCOMO
        from system_handling import Assembly

try:
    from .distumbrella_shared import (
        ca_selection,
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
        ca_selection,
        find_input_file,
        normalize_resources,
        output_path,
        parse_args,
        parse_config_path,
        write_args_config,
    )
    from umbrella_config import read_config


@dataclass(frozen=True)
class BoxNM:
    x: float
    y: float
    z: float

    def as_units(self) -> tuple:
        return (
            self.x * nanometer,
            self.y * nanometer,
            self.z * nanometer,
        )


_HELP_OPTIONS = (
    "setup",
    "pdb",
    "components",
    "component_types",
    "interactions",
    "refsel",
    "othersel",
    "box",
    "biasdir",
    "surf",
    "temperature",
    "gamma",
    "tstep",
    "device",
    "resources",
    "config",
    "write_config",
)


def _as_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be a float, got {value!r}") from exc


def _parse_box_nm(spec: str) -> BoxNM:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
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

    raise SystemExit("ERROR: --box must be 'x', 'x:y', or 'x:y:z' in nm")


def _center_target(box: BoxNM, biasdir: str) -> tuple:
    boxx, boxy, boxz = box.as_units()
    direction = biasdir.lower()

    if direction == "x":
        return (boxx / 4.0, boxy / 2.0, boxz / 2.0)
    if direction == "y":
        return (boxx / 2.0, boxy / 4.0, boxz / 2.0)
    if direction == "z":
        return (boxx / 2.0, boxy / 2.0, boxz / 4.0)

    raise SystemExit(f"ERROR: invalid biasdir {biasdir!r}; expected x, y, or z")


def main() -> None:
    cfg_path = parse_config_path()
    cfg = read_config(cfg_path)

    args = parse_args(cfg, _HELP_OPTIONS, prog="prep_distumbrella.py")

    setup_dir = Path(args.setup).expanduser().resolve()
    setup_dir.mkdir(parents=True, exist_ok=True)

    pdb_in = "dimer.ca.pdb" if args.pdb is None else str(args.pdb)
    box_spec = "100" if args.box is None else str(args.box)
    box_nm = _parse_box_nm(box_spec)
    box_units = box_nm.as_units()

    pdb_out = "dimer.protein.pdb"
    system_xml = "system.xml"
    initial_xml = "initial.xml"

    if bool(args.write_config):
        write_args_config(
            cfg_path,
            cfg,
            args,
            overrides={
                "pdb": pdb_in,
                "refpdb": pdb_out,
                "box": box_spec,
                "components": str(args.components),
                "component_types": str(args.component_types),
                "interactions": str(args.interactions),
            },
        )

    pdb_path = find_input_file(setup_dir, pdb_in)
    structure = PDBReader(str(pdb_path))

    ref_indices = StructureSelector(ca_selection(str(args.refsel))).atom_indices(structure)
    if len(ref_indices) == 0:
        raise SystemExit(f"ERROR: refsel matched no CA atoms: {args.refsel!r}")

    ref_center = structure.center(ref_indices)[0]
    target = _center_target(box_nm, str(args.biasdir))
    translate = [target[0] - ref_center[0], target[1] - ref_center[1]]
    translate.append(target[2] - ref_center[2])
    structure.translate(translate)

    shift_nm = [float(x.value_in_unit(nanometer)) for x in translate]
    print(
        "translated reference center by "
        f"({shift_nm[0]:.3f}, {shift_nm[1]:.3f}, {shift_nm[2]:.3f}) nm"
    )

    pdb_out_path = output_path(setup_dir, pdb_out)
    structure.write_pdb(str(pdb_out_path))

    components = find_input_file(setup_dir, str(args.components))
    component_types = find_input_file(setup_dir, str(args.component_types))
    interactions = find_input_file(setup_dir, str(args.interactions))

    assembly = Assembly(
        components,
        component_types,
        structure=structure,
        interactions=interactions,
    )
    sim = COCOMO(
        assembly,
        box=box_units,
        version=2,
        surfscale=float(args.surf),
    )

    sim.setup_simulation(
        resources=normalize_resources(str(args.resources)),
        device=int(args.device),
        temperature=float(args.temperature),
        gamma=float(args.gamma),
        tstep=float(args.tstep),
    )
    print(f"openmm energy: {sim.get_potentialEnergy()}")

    sim.write_system(str(output_path(setup_dir, system_xml)))
    sim.write_state(str(output_path(setup_dir, initial_xml)))


if __name__ == "__main__":
    main()
