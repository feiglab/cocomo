from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import copysign, sqrt
from types import MappingProxyType

import mdtraj as md
import numpy as np
from openmm import (
    CMMotionRemover,
    CustomBondForce,
    CustomCentroidBondForce,
    CustomExternalForce,
    CustomNonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    LangevinIntegrator,
    Platform,
    State,
    System,
    Vec3,
    XmlSerializer,
)
from openmm.app import (
    DCDReporter,
    PDBFile,
    Simulation,
    StateDataReporter,
    Topology,
    element,
)
from openmm.unit import (
    Quantity,
    amu,
    degrees,
    kelvin,
    kilojoule,
    mole,
    nanometer,
    norm,
    picoseconds,
    radian,
)

from .__version__ import __version__

# --- Data containers ---------------------------------------------------------


@dataclass(frozen=True)
class ResPar:
    mass: float
    charge: float
    radius: float
    epsilon: float
    azero: float
    surface: float

    def surface_scale(self, area: float, surfscale: float = 1.0) -> float:
        if self.surface > 0.0001 and surfscale > 0.0001:
            return min(area / self.surface / surfscale, 1.0)
        else:
            return 1.0


@dataclass(frozen=True)
class ResidueParameters(Mapping[str, ResPar]):
    residues: Mapping[str, ResPar]

    def __post_init__(self) -> None:
        # Deep-freeze the mapping to prevent mutation after construction
        object.__setattr__(self, "residues", MappingProxyType(dict(self.residues)))

    # Mapping interface so params[resname] works
    def __getitem__(self, key: str) -> ResPar:
        return self.residues[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.residues)

    def __len__(self) -> int:
        return len(self.residues)

    def get(self, key: str, default=None):
        return self.residues.get(key, default)


# --- Parameters ---------------------------------------------------------

epsilon_pol_v1 = 0.40
epsilon_hp_v1 = 0.41
epsilon_na_v1 = 0.41

azero_pol_v1 = 0.05
azero_hp_v1 = 0.00
azero_na_v1 = 0.05

RESIDUE_PARAMS_V1 = ResidueParameters(
    residues={
        "ALA": ResPar(71.079, 0.0, 0.2845, epsilon_hp_v1, azero_hp_v1, 0.00),
        "ARG": ResPar(157.197, 1.0, 0.3567, epsilon_pol_v1, azero_pol_v1, 0.00),
        "ASN": ResPar(114.104, 0.0, 0.3150, epsilon_pol_v1, azero_pol_v1, 0.00),
        "ASP": ResPar(114.080, -1.0, 0.3114, epsilon_pol_v1, azero_pol_v1, 0.00),
        "CYS": ResPar(103.139, 0.0, 0.3024, epsilon_pol_v1, azero_pol_v1, 0.00),
        "GLN": ResPar(128.131, 0.0, 0.3311, epsilon_pol_v1, azero_pol_v1, 0.00),
        "GLU": ResPar(128.107, -1.0, 0.3279, epsilon_pol_v1, azero_pol_v1, 0.00),
        "GLY": ResPar(57.052, 0.0, 0.2617, epsilon_hp_v1, azero_hp_v1, 0.00),
        "HIS": ResPar(137.142, 0.0, 0.3338, epsilon_pol_v1, azero_pol_v1, 0.00),
        "HSD": ResPar(137.142, 0.0, 0.3338, epsilon_pol_v1, azero_pol_v1, 0.00),
        "HSE": ResPar(137.142, 0.0, 0.3338, epsilon_pol_v1, azero_pol_v1, 0.00),
        "ILE": ResPar(113.160, 0.0, 0.3360, epsilon_hp_v1, azero_hp_v1, 0.00),
        "LEU": ResPar(113.160, 0.0, 0.3363, epsilon_hp_v1, azero_hp_v1, 0.00),
        "LYS": ResPar(129.183, 1.0, 0.3439, epsilon_pol_v1, azero_pol_v1, 0.00),
        "MET": ResPar(131.193, 0.0, 0.3381, epsilon_hp_v1, azero_hp_v1, 0.00),
        "PHE": ResPar(147.177, 0.0, 0.3556, epsilon_hp_v1, azero_hp_v1, 0.00),
        "PRO": ResPar(98.125, 0.0, 0.3187, epsilon_hp_v1, azero_hp_v1, 0.00),
        "SER": ResPar(87.078, 0.0, 0.2927, epsilon_pol_v1, azero_pol_v1, 0.00),
        "THR": ResPar(101.105, 0.0, 0.3108, epsilon_pol_v1, azero_pol_v1, 0.00),
        "TRP": ResPar(186.214, 0.0, 0.3754, epsilon_hp_v1, azero_hp_v1, 0.00),
        "TYR": ResPar(163.176, 0.0, 0.3611, epsilon_hp_v1, azero_hp_v1, 0.00),
        "VAL": ResPar(99.133, 0.0, 0.3205, epsilon_hp_v1, azero_hp_v1, 0.00),
        "ADE": ResPar(315.697, -1.0, 0.4220, epsilon_na_v1, azero_na_v1, 0.00),
        "CYT": ResPar(305.200, -1.0, 0.4110, epsilon_na_v1, azero_na_v1, 0.00),
        "GUA": ResPar(345.200, -1.0, 0.4255, epsilon_na_v1, azero_na_v1, 0.00),
        "URA": ResPar(305.162, -1.0, 0.4090, epsilon_na_v1, azero_na_v1, 0.00),
    }
)

epsilon_pol_v2 = 0.17596101
epsilon_hp_v2 = 0.29519131

azero_pol_v2 = 0.00
azero_hp_v2 = 0.000245894

RESIDUE_PARAMS_V2 = ResidueParameters(
    residues={
        "ALA": ResPar(71.079, 0.0, 0.2845, epsilon_hp_v2, azero_hp_v2, 0.796),
        "ARG": ResPar(157.197, 1.0, 0.3567, epsilon_pol_v2, azero_pol_v2, 1.921),
        "ASN": ResPar(114.104, 0.0, 0.3150, epsilon_pol_v2, azero_pol_v2, 1.281),
        "ASP": ResPar(114.080, -1.0, 0.3114, epsilon_pol_v2, azero_pol_v2, 1.162),
        "CYS": ResPar(103.139, 0.0, 0.3024, epsilon_hp_v2, azero_hp_v2, 1.074),
        "GLN": ResPar(128.131, 0.0, 0.3311, epsilon_pol_v2, azero_pol_v2, 1.575),
        "GLU": ResPar(128.107, -1.0, 0.3279, epsilon_pol_v2, azero_pol_v2, 1.462),
        "GLY": ResPar(57.052, 0.0, 0.2617, epsilon_hp_v2, azero_hp_v2, 0.544),
        "HIS": ResPar(137.142, 0.0, 0.3338, epsilon_pol_v2, azero_pol_v2, 1.634),
        "HSD": ResPar(137.142, 0.0, 0.3338, epsilon_pol_v2, azero_pol_v2, 1.634),
        "HSE": ResPar(137.142, 0.0, 0.3338, epsilon_pol_v2, azero_pol_v2, 1.634),
        "ILE": ResPar(113.160, 0.0, 0.3360, epsilon_hp_v2, azero_hp_v2, 1.410),
        "LEU": ResPar(113.160, 0.0, 0.3363, epsilon_hp_v2, azero_hp_v2, 1.519),
        "LYS": ResPar(129.183, 1.0, 0.3439, epsilon_pol_v2, azero_pol_v2, 1.923),
        "MET": ResPar(131.193, 0.0, 0.3381, epsilon_hp_v2, azero_hp_v2, 1.620),
        "PHE": ResPar(147.177, 0.0, 0.3556, epsilon_hp_v2, azero_hp_v2, 1.869),
        "PRO": ResPar(98.125, 0.0, 0.3187, epsilon_hp_v2, azero_hp_v2, 0.974),
        "SER": ResPar(87.078, 0.0, 0.2927, epsilon_pol_v2, azero_pol_v2, 0.933),
        "THR": ResPar(101.105, 0.0, 0.3108, epsilon_pol_v2, azero_pol_v2, 1.128),
        "TRP": ResPar(186.214, 0.0, 0.3754, epsilon_hp_v2, azero_hp_v2, 2.227),
        "TYR": ResPar(163.176, 0.0, 0.3611, epsilon_hp_v2, azero_hp_v2, 2.018),
        "VAL": ResPar(99.133, 0.0, 0.3205, epsilon_hp_v2, azero_hp_v2, 1.232),
        "ADE": ResPar(315.697, -1.0, 0.4220, epsilon_na_v1, azero_na_v1, 0.00),
        "CYT": ResPar(305.200, -1.0, 0.4110, epsilon_na_v1, azero_na_v1, 0.00),
        "GUA": ResPar(345.200, -1.0, 0.4255, epsilon_na_v1, azero_na_v1, 0.00),
        "URA": ResPar(305.162, -1.0, 0.4090, epsilon_na_v1, azero_na_v1, 0.00),
    }
)

bond_l0_protein = 0.38  # nm
bond_l0_na = 0.5  # nm
bond_k_protein = 4184  # kJ/mol/nm^2
bond_k_na = 4184  # kJ/mol/nm^2

angle_t0_protein = 180.0  # degrees
angle_t0_na = 180.0  # degrees
angle_k_protein = 4.184  # kJ/mol/rad^2
angle_k_na = 5.021  # kJ/mol/rad^2

eps_catpi_propro_v1 = 0.30
eps_catpi_prona_v1 = 0.20
eps_pipi_propro_v1 = 0.00

eps_catpi_propro_v2 = 0.30
eps_catpi_prona_v2 = 0.20
eps_pipi_propro_v2 = 0.10

EPS_V1 = {
    "catpi_propro": eps_catpi_propro_v1,
    "catpi_prona": eps_catpi_prona_v1,
    "pipi_propro": eps_pipi_propro_v1,
}

EPS_V2 = {
    "catpi_propro": eps_catpi_propro_v2,
    "catpi_prona": eps_catpi_prona_v2,
    "pipi_propro": eps_pipi_propro_v2,
}

surf_scale_v2 = 0.7

# --- Other useful data -------------------------------------------------------

aminoacids = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HSD",
    "HSE",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

nucleicacids = ["ADE", "CYT", "GUA", "URA", "THY"]

# --- Main class for COCOMO system --------------------------------------------


class COCOMO:
    def __init__(
        self,
        topology=None,
        *,
        version=None,  # version: 1 or 2
        params=None,  # default set from version
        eps=None,  # default set from version
        surfscale=None,  # default set from version
        box=100,  # 100 or (50,20,40), nm
        cuton=2.9,  # nm
        cutoff=3.1,  # nm
        switching="original",  # 'original', 'openmm', or None
        kappa=1.0,  # nm
        sasa=None,  # for surface scaling
        enmpairs=None,  # list of ENM pairs
        domains=None,  # turns on ENM
        positions=None,  # needed for ENM
        restart=None,
        enmforce=500.0,  # 1/nm**2
        enmcutoff=0.9,  # nm
        interactions=None,  # list of interactions
        removecmmotion=True,
        xml=None,
    ):

        self.simulation = None
        self.topology = None
        self.positions = None
        self.velocities = None
        self.box = None
        self.box_vectors = None

        self.params = params
        self.eps = eps
        self.surfscale = surfscale

        self.version = 2 if version is None else version
        self.set_params(self.version)

        self.removecmmotion = removecmmotion
        self.k0 = kappa * nanometer
        self.cuton = cuton * nanometer
        self.cutoff = cutoff * nanometer
        self.switching = switching

        self.enmforce = enmforce / nanometer**2
        self.enmcutoff = enmcutoff * nanometer

        if topology.__class__.__name__ == "Assembly":
            assembly = topology
            self.topology = assembly.model.topology()
            self.sasa = assembly.get_sasa()
            self.set_positions(assembly.model.positions())
            self.enmpairs = assembly.get_enmpairs()
            self.interactions = assembly.get_interactions()
        else:
            self.topology = topology
            self.set_sasa(sasa)
            self.set_positions(positions)
            self.set_domains(domains)
            if enmpairs is None:
                self.enmpairs = self._findENMPairs()
            else:
                self.enmpairs = enmpairs
            self.interactions = interactions

        if xml is not None:
            self.read_system(xml)
            self.read_restart(restart)
            return

        self.system = System()
        self.setup_particles()

        if box:
            self.set_box(box * nanometer)

        self.setup_forces()
        self.read_restart(restart)

    def set_dummy_topology(self):
        if not self.topology and self.system:
            n_atoms = self.system.getNumParticles()
            top = Topology()
            chain = top.addChain()
            res = top.addResidue("DUM", chain)
            for i in range(n_atoms):
                top.addAtom("C", element.carbon, res)
            self.topology = top

    def describe(self) -> str:
        return f"This is COCOMO CG model simulator version {__version__}"

    def write_system(self, fname="system.xml"):
        with open(fname, "w") as file:
            file.write(XmlSerializer.serialize(self.system))

    def read_system(self, fname="system.xml"):
        with open(fname) as file:
            self.system = XmlSerializer.deserialize(file.read())

    def write_state(self, fname="state.xml"):
        if self.simulation is not None:
            self.simulation.saveState(fname)

    def read_state(self, fname="state.xml"):
        if self.simulation is not None:
            self.simulation.loadState(fname)

    def read_restart(self, fname):
        if _is_readable_file(fname):
            with open(fname) as f:
                state = XmlSerializer.deserialize(f.read())
            self.positions = state.getPositions()
            self.velocities = state.getVelocities()
            self.box_vectors = state.getPeriodicBoxVectors()

    def write_pdb(self, fname="state.pdb"):
        positions = self.simulation.context.getState(getPositions=True).getPositions()
        with open(fname, "w") as f:
            PDBFile.writeFile(self.simulation.topology, positions, f)

    def get_potentialEnergy(self):
        if self.simulation is not None:
            return self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        else:
            return 0.0

    def get_energies(self):
        # Desired order by class name
        priority = {
            "HarmonicBondForce": 0,
            "HarmonicAngleForce": 1,
            "CustomNonbondedForce": 2,
            "CustomBondForce": 3,
            "CustomCentroidForce": 4,
        }

        forces = list(self.system.getForces())

        indexed_forces = []
        for idx, frc in enumerate(forces):
            cls_name = frc.__class__.__name__
            order = priority.get(cls_name, 100)
            indexed_forces.append((order, idx, frc))

        indexed_forces.sort(key=lambda x: (x[0], x[1]))

        energies = {}
        for _, _, frc in indexed_forces:
            g = frc.getForceGroup()
            e = self._group_energy(self.simulation.context, g)
            name = frc.getName() or frc.__class__.__name__
            key = f"{name}({g})"
            energies[key] = e

        return energies

    def set_velocities(self, *, seed=None, newtemp=None):
        if self.simulation is not None:
            if newtemp is not None:
                temperature = newtemp * kelvin
            else:
                temperature = self.temperature
            if seed is None:
                seed = np.random.SeedSequence().entropy
            seed = int(seed) & 0x7FFFFFFF
            self.simulation.context.setVelocitiesToTemperature(temperature, seed)

    def minimize(self, *, nstep=1000, tol=0.001):
        if self.simulation is not None:
            tolerance = tol * kilojoule / (nanometer * mole)
            self.simulation.minimizeEnergy(tolerance=tolerance, maxIterations=nstep)

    class EnergyReporter:
        def __init__(self, file, reportInterval, bias_force_names_to_groups):
            """
            file: output filename
            reportInterval: reporting interval in steps
            bias_force_names_to_groups: dict like {"angle": 1, "torsion": 2}
            """
            self.file = open(file, "w")
            self.interval = reportInterval
            self.bias_force_names_to_groups = bias_force_names_to_groups
            header = f"{'Step':<20s}" + "".join(
                f"{name:<20s}" for name in bias_force_names_to_groups
            )
            self.file.write(header + "\n")

        def describeNextReport(self, simulation):
            return (self.interval, False, False, False, False)

        def report(self, simulation, state):
            # First column: step, left-aligned in width 20
            line = f"{simulation.currentStep:<20d}"
            for name, group in self.bias_force_names_to_groups.items():
                bias_state = simulation.context.getState(getEnergy=True, groups={group})
                energy = bias_state.getPotentialEnergy().value_in_unit(kilojoule / mole)
                line += f"{energy:<20.8f}"

            self.file.write(line + "\n")
            self.file.flush()

        def __del__(self):
            self.file.close()

    def simulate(
        self, *, nstep=1000, nout=1000, logfile=None, dcdfile=None, elogfile=None, forcelist=None
    ):
        if self.simulation is not None:
            if dcdfile:
                dcd = DCDReporter(dcdfile, nout)
                self.simulation.reporters.append(dcd)
            if logfile:
                log = StateDataReporter(
                    logfile,
                    nout,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    kineticEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    volume=True,
                    progress=True,
                    remainingTime=True,
                    speed=True,
                    totalSteps=nstep,
                    separator=" ",
                )
                self.simulation.reporters.append(log)
            if elogfile and forcelist:
                log = self.EnergyReporter(elogfile, nout, self.get_force_groups(forcelist))
                self.simulation.reporters.append(log)
            self.simulation.step(nstep)

    def set_sasa(self, sasa=None):
        if self.topology is not None:
            natoms = self.topology.getNumAtoms()
            if natoms > 0:
                self.sasa = np.ones(natoms) * 9999.0
                if sasa is not None:
                    if len(sasa) != natoms:
                        raise ValueError("Length of SASA array does not match atoms")
                    else:
                        for i in range(len(sasa)):
                            self.sasa[i] = sasa[i]

    def set_domains(self, domains):
        if domains is not None:
            self.domains = [row.copy() for row in domains]
        else:
            self.domains = None

    def setup_simulation(
        self,
        *,
        temperature=298,
        gamma=0.01,
        tstep=0.01,
        resources="CPU",
        device=0,
        restart=None,
        positions=None,
        velocities=None,
        resetvelocities=False,
        box=None,
    ) -> None:
        # temperature: in K
        # tstep: in ps
        # gamma: in 1/ps
        # resources: 'CPU' or 'CUDA'

        assert self.system is not None, "need openMM system object to be defined"

        self.resources = resources

        if restart is not None:
            self.read_restart(restart)
        if positions:
            self.positions = positions
        if velocities:
            self.velocities = velocities
        if box:
            self.set_box(box * nanometer)

        if temperature:
            self.temperature = temperature * kelvin
        if tstep:
            self.tstep = tstep * picoseconds
        if gamma:
            self.gamma = gamma / picoseconds

        if not self.topology:
            self.set_dummy_topology()

        self.integrator = LangevinIntegrator(self.temperature, self.gamma, self.tstep)
        self.platform = Platform.getPlatformByName(self.resources)
        self.simulation = None
        if self.resources == "CUDA":
            prop = dict(CudaPrecision="mixed", CudaDeviceIndex=str(device))
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, self.platform, prop
            )
        if self.resources == "CPU":
            self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)

        if self.positions:
            self.simulation.context.setPositions(self.positions)

        if resetvelocities:
            self.set_velocities()
        else:
            if self.velocities:
                self.simulation.context.setVelocities(self.velocities)
            else:
                self.set_velocities()

        if self.box_vectors:
            a, b, c = self.box_vectors
            self.simulation.context.setPeriodicBoxVectors(a, b, c)

    def set_positions(self, positions) -> None:
        self.positions = positions
        if self.simulation is not None:
            self.simulation.context.setPositions(positions)

    def get_positions(self):
        if self.simulation:
            return self.simulation.context.getState(getPositions=True).getPositions()
        else:
            return None

    def get_velocities(self):
        if self.simulation:
            return self.simulation.context.getState(getVelocities=True).getVelocities()
        else:
            return None

    def get_masses(self):
        if self.system:
            nparticles = self.system.getNumParticles()
            return [self.system.getParticleMass(i) for i in range(nparticles)]
        else:
            return None

    def setup_particles(self) -> None:
        if self.topology is None:
            return
        if self.params is None:
            self.set_params(self.version)
        for atm in self.topology.atoms():
            self.system.addParticle(self.params[atm.residue.name].mass * amu)

    def set_box(self, box) -> None:
        ax_nm, by_nm, cz_nm = self._normalize_box(box)

        a_sys = Vec3(ax_nm, 0.0, 0.0) * nanometer
        b_sys = Vec3(0.0, by_nm, 0.0) * nanometer
        c_sys = Vec3(0.0, 0.0, cz_nm) * nanometer

        # record on the class
        self.box: tuple[float, float, float] = (ax_nm, by_nm, cz_nm)
        self.box_vectors = (a_sys, b_sys, c_sys)

        # set on topology and system
        a_top = Vec3(ax_nm, 0.0, 0.0)
        b_top = Vec3(0.0, by_nm, 0.0)
        c_top = Vec3(0.0, 0.0, cz_nm)

        if self.topology is not None:
            self.topology.setPeriodicBoxVectors((a_top, b_top, c_top))
        self.system.setDefaultPeriodicBoxVectors(a_sys, b_sys, c_sys)

    def set_force_groups(self):
        if self.system:
            for i, force in enumerate(self.system.getForces()):
                force.setForceGroup(i)

    def get_force_groups(self, names: Sequence[str]) -> dict[str, int]:
        """
        Return a mapping {force_key: force_group} for forces whose names
        match any of the provided strings.

        If a force name appears multiple times, additional entries are created
        with keys 'name', 'name_2', 'name_3', ...

        Parameters
        ----------
        names : sequence of str
            Name fragments to match against Force.getName() (or class name).

        Returns
        -------
        dict[str, int]
            Keys are disambiguated force labels, values are their force group indices.
        """
        if self.system is None:
            raise RuntimeError("System has not been created yet.")

        # Normalize input
        if isinstance(names, str):
            patterns = [names]
        else:
            patterns = list(names)

        patterns = [p for p in patterns if p]
        if not patterns:
            return {}

        forcelist: dict[str, int] = {}

        for force in self.system.getForces():
            fname = force.getName() or force.__class__.__name__
            # check if any pattern matches this force name
            if not any(pat in fname for pat in patterns):
                continue

            # determine dict key, handling duplicates: name, name_2, name_3, ...
            key = fname
            if key in forcelist:
                i = 2
                while f"{fname}_{i}" in forcelist:
                    i += 1
                key = f"{fname}_{i}"

            forcelist[key] = force.getForceGroup()

        return forcelist

    def setup_forces(self) -> None:
        self.forces = {}
        if self.topology is not None:
            self.setupBondForce()
            self.setupAngleForce()
            self.setupLongRangeForce()
            self.setupShortRangeForce()
            self.setupENMForce()
            self.setupInteractionForce()
            if self.removecmmotion:
                self.setupCMMotionRemover()
            self.set_force_groups()

    def setupBondForce(
        self, *, l0protein=bond_l0_protein, l0na=bond_l0_na, kprotein=bond_k_protein, kna=bond_k_na
    ) -> None:
        if self.topology is not None:
            if self.topology.getNumBonds() == 0:
                self.set_bonds()
            force = HarmonicBondForce()
            for bond in self.topology.bonds():
                if bond[0].residue.name in aminoacids:
                    force.addBond(
                        bond[0].index,
                        bond[1].index,
                        l0protein * nanometer,
                        kprotein * (kilojoule / mole) / (nanometer**2),
                    )
                elif bond[0].residue.name in nucleicacids:
                    force.addBond(
                        bond[0].index,
                        bond[1].index,
                        l0na * nanometer,
                        kna * (kilojoule / mole) / (nanometer**2),
                    )
                else:
                    errstr = f"Invalid residue name {bond[0].residue.name} in setupBondForce"
                    raise ValueError(errstr)

            force.setName("bond")
            self.forces["bond"] = force
            self.system.addForce(force)

    def setupAngleForce(
        self,
        *,
        t0protein=angle_t0_protein,
        t0na=angle_t0_na,
        kprotein=angle_k_protein,
        kna=angle_k_na,
    ) -> None:
        if self.topology is not None:
            force = HarmonicAngleForce()
            for chain in self.topology.chains():
                chatm = []
                for atom in chain.atoms():
                    chatm.append(atom)
                for i in range(len(chatm) - 2):
                    if chatm[i].residue.name in aminoacids:
                        force.addAngle(
                            chatm[i].index,
                            chatm[i + 1].index,
                            chatm[i + 2].index,
                            t0protein * degrees,
                            kprotein * (kilojoule / mole) / (radian**2),
                        )
                    elif chatm[i].residue.name in nucleicacids:
                        force.addAngle(
                            chatm[i].index,
                            chatm[i + 1].index,
                            chatm[i + 2].index,
                            t0na * degrees,
                            kna * (kilojoule / mole) / (radian**2),
                        )
                    else:
                        errstr = f"Invalid residue name {chatm[i].residue.name} in setupAngleForce"
                        raise ValueError(errstr)

            force.setName("angle")
            self.forces["angle"] = force
            self.system.addForce(force)

    def setupLongRangeForce(self) -> None:
        if self.topology is not None:
            if self.params is None:
                self.set_params(self.version)
            if self.switching == "original":
                equation = "select( step(r_on-r), longrange+sdel, switch ); "
                equation += "longrange = (AZ)*exp(-r/K0)/r; "
                equation += "sdel = sk*((1/r_on)^3-1/(r_off)^3)^2 - (AZ)/r_on*exp(-r_on/K0); "
                equation += "switch = sk*((1/r)^3-1/(r_off)^3)^2; "
                equation += "sk = -longrange_deriv_Ron/switch_deriv_Ron; "
                equation += "longrange_deriv_Ron = -1*(AZ)*exp(-r_on/K0)/r_on*(1/K0+1/r_on); "
                equation += "switch_deriv_Ron = 6*(1/r_on^3-1/r_off^3)*1/r_on^4; "
            else:
                equation = "(AZ)/r*exp(-r/K0); "

            if self.surfscale is not None:
                equation += "AZ=S*(A+Z); "
                equation += "S=(S1*S2)^0.5; "
            else:
                equation += "AZ=A+Z; "

            equation += "A=A1*A2; "
            equation += "Z=Z1+Z2 "

            force = CustomNonbondedForce(equation)
            force.addGlobalParameter("K0", self.k0)
            force.addPerParticleParameter("A")
            force.addPerParticleParameter("Z")
            if self.surfscale is not None:
                force.addPerParticleParameter("S")
            force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(self.cutoff)
            if self.switching == "original":
                force.addGlobalParameter("r_on", self.cuton)
                force.addGlobalParameter("r_off", self.cutoff)
            elif self.switching == "openmm":
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(self.cuton)

            for i, atom in enumerate(self.topology.atoms()):
                chg = self.params[atom.residue.name].charge
                a = copysign(sqrt(0.75 * abs(chg)), chg) * nanometer * kilojoule / mole
                a0 = self.params[atom.residue.name].azero * (nanometer * kilojoule / mole) ** 0.5
                if self.surfscale is not None:
                    s = self.params[atom.residue.name].surface_scale(self.sasa[i], self.surfscale)
                    force.addParticle([a, a0, s])
                else:
                    force.addParticle([a, a0])

            if self.topology.getNumBonds() == 0:
                self.set_bonds()
            for bond in self.topology.bonds():
                force.createExclusionsFromBonds([(bond[0].index, bond[1].index)], 1)

            force.setName("longrange")
            self.forces["longrange"] = force
            self.system.addForce(force)

    def setupShortRangeForce(self) -> None:
        if self.topology is not None:
            if self.params is None:
                self.set_params(self.version)

            if self.switching == "original":
                equation = "select( step(r_on-r), shortrange+sdel, switch ); "
                equation += "shortrange = 4*eps*((sigma/r)^10-(sigma/r)^5); "
                equation += "sdel = sk*((1/r_on)^3-1/(r_off)^3)^2 "
                equation += " - 4*eps*((sigma/r_on)^10-(sigma/r_on)^5); "
                equation += "switch = sk*((1/r)^3-1/(r_off)^3)^2; "
                equation += "sk = -shortrange_deriv_Ron/switch_deriv_Ron; "
                equation += "shortrange_deriv_Ron = 4*eps*(-10*(sigma/r_on)^11"
                equation += "*1/r_on+5*(sigma/r_on)^5*1/r_on ); "
                equation += "switch_deriv_Ron = 6*(1/r_on^3-1/r_off^3)*1/r_on^4;"
            else:
                equation = "4*eps*((sigma/r)^10-(sigma/r)^5);"

            equation += "sigma=0.5*(sigma1+sigma2);"
            if self.surfscale is not None:
                equation += "eps = S*(eps_mix"
                equation += "+ eps_catpi_pro*min(1,catpipropair) "
                equation += "+ eps_catpi_na*min(1,catpinapair) "
                equation += "+ eps_pipi*min(1,catpipipair)); "
                equation += "S=(S1*S2)^0.5; "
            else:
                equation += "eps = eps_mix"
                equation += "+ eps_catpi_pro*min(1,catpipropair) "
                equation += "+ eps_catpi_na*min(1,catpinapair) "
                equation += "+ eps_pipi*min(1,catpipipair); "
            equation += "catpipropair=isCation1*isAromatic2 + isCation2*isAromatic1;"
            equation += "catpinapair=isCation1*isNucleic2 + isCation2*isNucleic1;"
            equation += "catpipipair=isAromatic1*isAromatic2 + isAromatic2*isAromatic1;"
            equation += "eps_mix = sqrt(epsilon1*epsilon2);"

            force = CustomNonbondedForce(equation)
            force.addPerParticleParameter("sigma")
            force.addPerParticleParameter("epsilon")
            force.addPerParticleParameter("isCation")
            force.addPerParticleParameter("isAromatic")
            force.addPerParticleParameter("isNucleic")
            if self.surfscale is not None:
                force.addPerParticleParameter("S")

            force.addGlobalParameter("eps_catpi_pro", self.eps["catpi_propro"] * kilojoule / mole)
            force.addGlobalParameter("eps_catpi_na", self.eps["catpi_prona"] * kilojoule / mole)
            force.addGlobalParameter("eps_pipi", self.eps["pipi_propro"] * kilojoule / mole)

            force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(self.cutoff)
            if self.switching == "original":
                force.addGlobalParameter("r_on", self.cuton)
                force.addGlobalParameter("r_off", self.cutoff)
            elif self.switching == "openmm":
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(self.cuton)

            for i, atom in enumerate(self.topology.atoms()):
                sigma = self.params[atom.residue.name].radius * 2 * 2 ** (-1 / 6) * nanometer
                epsilon = self.params[atom.residue.name].epsilon * kilojoule / mole
                isCation = 1.0 if atom.residue.name in ["ARG", "LYS"] else 0.0
                isAromatic = 1.0 if atom.residue.name in ["PHE", "TRP", "TYR"] else 0.0
                isNucleic = 1.0 if atom.residue.name in nucleicacids else 0.0
                if self.surfscale is not None:
                    s = self.params[atom.residue.name].surface_scale(self.sasa[i], self.surfscale)
                    force.addParticle([sigma, epsilon, isCation, isAromatic, isNucleic, s])
                else:
                    force.addParticle([sigma, epsilon, isCation, isAromatic, isNucleic])

            if self.topology.getNumBonds() == 0:
                self.set_bonds()
            for bond in self.topology.bonds():
                force.createExclusionsFromBonds([(bond[0].index, bond[1].index)], 1)

            force.setName("shortrange")
            self.forces["shortrange"] = force
            self.system.addForce(force)

    def _findENMPairs(self) -> list[int, int, float]:
        if self.topology is None:
            return None

        if self.domains is None or self.positions is None:
            return None

        pairs = []
        atm = list(self.topology.atoms())
        res = np.fromiter((a.residue.index for a in atm), dtype=np.int32)
        chain = np.fromiter((id(a.residue.chain) for a in atm), dtype=np.int64)

        for d in self.domains:
            idx = np.array(sorted(set(d)), dtype=np.int32)
            if idx.size < 2:
                continue
            for i, j in combinations(idx, 2):
                if (abs(res[i] - res[j]) <= 2) and (chain[i] == chain[j]):
                    continue
                distance = norm(self.positions[i] - self.positions[j])
                if distance < self.enmcutoff:
                    pairs.append([i, j, distance])
        return pairs

    def setupENMForce(self) -> None:
        if self.topology is not None:
            if self.enmpairs is None:
                self.enmpairs = self._findENMPairs()

            if self.enmpairs is not None and len(self.enmpairs) > 0:
                equation = "0.5*kenm*(r-r0)^2"
                force = CustomBondForce(equation)
                force.addGlobalParameter("kenm", self.enmforce)
                force.addPerBondParameter("r0")

                for enm in self.enmpairs:
                    force.addBond(enm[0], enm[1], [enm[2]])

                force.setName("enm")
                self.forces["enm"] = force
                self.system.addForce(force)

    def setupInteractionForce(self) -> None:
        """
        Build CustomBondForce objects for pairwise interactions specified in
        self.interactions (a list[Interaction] from system_handling).

        Implemented forms:

        - 'switch':
              V(r) = -eps / (1 + exp(alpha*(r - r0)))
          where:
              eps   ← Interaction.strength (energy scale)
              r0    ← Interaction.distance [nm]
              alpha ← Interaction.parameter if non-zero, else a default

        - 'Go':
              V(r) = eps * ((r0/r)**12 - 2*(r0/r)**6)
          minimum at r = r0 with depth -eps.

        - 'harmonic':
              V(r) = 0.5 * k * (r - r0)**2
          where k ← Interaction.strength (force constant).

        All varying quantities are per-bond parameters; we use at most one
        CustomBondForce per functional form to avoid the global-parameter
        conflict that triggered the 'eps' error.
        """
        if self.topology is None or not self.interactions:
            return

        # Default alpha (1/nm) for switched potential if parameter == 0.0
        def_alpha = 10.0

        # Lazily created forces, one per functional form
        switch_force = None
        go_force = None
        harmonic_force = None

        for intr in self.interactions:
            if not intr.pairs:
                continue

            func = (intr.function or "switch").lower()
            strength = float(intr.strength)
            r0 = float(intr.distance)

            if func == "switch":
                # Create the force on first use
                if switch_force is None:
                    equation = "-eps/(1+exp(y)); "
                    equation += "y = min(max(x, -50), 50); "
                    equation += "x = alpha*(r-r0);"
                    f = CustomBondForce(equation)
                    f.addPerBondParameter("eps")  # energy
                    f.addPerBondParameter("r0")  # nm
                    f.addPerBondParameter("alpha")  # 1/nm
                    f.setUsesPeriodicBoundaryConditions(True)
                    f.setName("interaction_switch")
                    switch_force = f

                # Choose alpha: interaction.parameter if non-zero, else default
                alpha = float(intr.parameter) if intr.parameter not in (None, 0.0) else def_alpha

                for i, j in intr.pairs:
                    if i == j:
                        continue
                    ia, jb = (int(i), int(j))
                    # consistent ordering is not strictly needed for bonds but doesn’t hurt
                    if ia > jb:
                        ia, jb = jb, ia

                    switch_force.addBond(
                        ia,
                        jb,
                        [
                            strength * kilojoule / mole,
                            r0 * nanometer,
                            alpha / nanometer,
                        ],
                    )

            elif func == "go":
                if go_force is None:
                    equation = "eps*((r0/r)^12 - 2*(r0/r)^6)"
                    f = CustomBondForce(equation)
                    f.addPerBondParameter("eps")  # energy
                    f.addPerBondParameter("r0")  # nm
                    f.setUsesPeriodicBoundaryConditions(True)
                    f.setName("interaction_go")
                    go_force = f

                for i, j in intr.pairs:
                    if i == j:
                        continue
                    ia, jb = (int(i), int(j))
                    if ia > jb:
                        ia, jb = jb, ia

                    go_force.addBond(
                        ia,
                        jb,
                        [
                            strength * kilojoule / mole,
                            r0 * nanometer,
                        ],
                    )

            elif func == "harmonic":
                if harmonic_force is None:
                    equation = "0.5*k*(r-r0)^2"
                    f = CustomBondForce(equation)
                    f.addPerBondParameter("k")  # energy / nm^2
                    f.addPerBondParameter("r0")  # nm
                    f.setUsesPeriodicBoundaryConditions(True)
                    f.setName("interaction_harmonic")
                    harmonic_force = f

                for i, j in intr.pairs:
                    if i == j:
                        continue
                    ia, jb = (int(i), int(j))
                    if ia > jb:
                        ia, jb = jb, ia

                    harmonic_force.addBond(
                        ia,
                        jb,
                        [
                            strength * kilojoule / (mole * nanometer**2),
                            r0 * nanometer,
                        ],
                    )

            else:
                warnings.warn(f"Unknown interaction function {intr.function!r}; skipping.")
                continue

        # Register forces with the System and self.forces dict
        if switch_force is not None:
            self.system.addForce(switch_force)
            self.forces["interaction_switch"] = switch_force

        if go_force is not None:
            self.system.addForce(go_force)
            self.forces["interaction_go"] = go_force

        if harmonic_force is not None:
            self.system.addForce(harmonic_force)
            self.forces["interaction_harmonic"] = harmonic_force

    def set_position_restraint(
        self, *, selection="name CA", atomlist=None, k=100.0, positions=None
    ):
        """
        Apply PBC-aware positional restraints to a set of atoms.

        Parameters
        ----------
        selection : str
            MDTraj selection string (ignored if `atomlist` is provided).
        atomlist : sequence of int or openmm.app.Atom
            Atoms to restrain, given as indices or Atom objects.
        k : float
            Force constant in kJ/mol/nm^2.
        positions : reference positions
        """
        if self.system is None:
            raise RuntimeError("System has not been created yet.")

        # Determine atom indices
        if atomlist is not None:
            # Accept list of ints or list of Atom objects
            if len(atomlist) == 0:
                return
            first = atomlist[0]
            if isinstance(first, int):
                indices = list(atomlist)
            else:
                # assume OpenMM Atom objects
                indices = [a.index for a in atomlist]
        else:
            if self.topology is None:
                raise RuntimeError("Topology is required for selection-based restraints.")
            md_top = md.Topology.from_openmm(self.topology)
            indices = md_top.select(selection).tolist()

        if not indices:
            return  # nothing to restrain

        # Get reference positions (in nm)
        if positions is not None:
            pos = positions
        elif self.positions is not None:
            pos = self.positions
        elif self.simulation is not None:
            pos = self.simulation.context.getState(getPositions=True).getPositions()
        else:
            raise RuntimeError("No positions available to define restraint reference points.")

        force = CustomExternalForce("0.5 * posk * periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("posk", k * kilojoule / (nanometer**2 * mole))
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        # Convert positions to nm if they are a Quantity
        if hasattr(pos, "value_in_unit"):
            # pos is a Quantity (e.g. from State.getPositions())
            pos_nm = pos.value_in_unit(nanometer)
        else:
            # pos is already a list/array of Vec3 or xyz triples in nm
            pos_nm = pos

        for idx in indices:
            p = pos_nm[idx]
            # p can be a Vec3 or a length-3 iterable of floats
            if hasattr(p, "x"):
                x0, y0, z0 = p.x, p.y, p.z
            else:
                x0, y0, z0 = p[0], p[1], p[2]
            force.addParticle(idx, [x0, y0, z0])

        force.setName("PositionalRestraints")
        self.system.addForce(force)

    def set_umbrella_xyz_distance(
        self, groupa, groupb, *, direction="x", target=0.0, k=10.0, center="cog"
    ):
        if self.system:
            bias = f"0.5 * uk_{direction} * ((abs({direction}2 - {direction}1) - target)^2)"
            force = CustomCentroidBondForce(2, bias)
            force.addPerBondParameter("target")  # target distance (nm)
            force.addGlobalParameter(f"uk_{direction}", k * kilojoule / mole / nanometer**2)
            if center.lower() == "cog":
                force.addGroup(groupa, [1.0] * len(groupa))
                force.addGroup(groupb, [1.0] * len(groupb))
            elif center.lower() == "com":
                force.addGroup(groupa)
                force.addGroup(groupb)
            else:
                raise ValueError(f"Center option {center} is not valid.")

            force.addBond([0, 1], [target * nanometer])
            force.setName(f"Umbrella_{direction}")
            self.system.addForce(force)

    def update_umbrella_xyz_distance(self, direction="x", k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter(
                f"uk_{direction}", k * kilojoule / mole / nanometer**2
            )

    def set_umbrella_distance(
        self, groupa, groupb, *, target=0.0, k=10.0, periodic=False, center="cog"
    ):
        if self.system:
            bias = "0.5 * uk_dist * ((distance(g1,g2) - target)^2)"
            force = CustomCentroidBondForce(2, bias)
            force.addPerBondParameter("target")  # target distance (nm)
            force.addGlobalParameter("uk_dist", k * kilojoule / mole / nanometer**2)
            if center.lower() == "cog":
                force.addGroup(groupa, [1.0] * len(groupa))
                force.addGroup(groupb, [1.0] * len(groupb))
            elif center.lower() == "com":
                force.addGroup(groupa)
                force.addGroup(groupb)
            else:
                raise ValueError(f"Center option {center} is not valid.")

            force.addBond([0, 1], [target * nanometer])
            if self.box_vectors and periodic:
                force.setUsesPeriodicBoundaryConditions(True)
            force.setName("Umbrella_distance")
            self.system.addForce(force)

    def update_umbrella_distance(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_dist", k * kilojoule / mole / nanometer**2)

    def _compute_center(self, group, positions_nm, *, center="cog"):
        """Compute the center (COG or COM) of `group` using positions in nm.

        Parameters
        ----------
        group : sequence[int]
            Atom indices.
        positions_nm : sequence
            Positions in nm, either a list of Vec3 or an (N,3)-like array.
        center : {"cog", "com"}
            - "cog": center of geometry (equal weights)
            - "com": center of mass (particle masses from `self.system`)

        Returns
        -------
        np.ndarray
            Shape (3,) array (x,y,z) in nm.
        """
        if group is None or len(group) == 0:
            raise ValueError("group must contain at least one atom index")

        mode = (center or "cog").lower()
        if mode not in {"cog", "com"}:
            raise ValueError(f"Center option {center} is not valid. Use 'cog' or 'com'.")

        # Coordinates
        coords = np.empty((len(group), 3), dtype=float)
        for i, idx in enumerate(group):
            p = positions_nm[int(idx)]
            if hasattr(p, "x"):
                coords[i, 0] = float(p.x)
                coords[i, 1] = float(p.y)
                coords[i, 2] = float(p.z)
            else:
                coords[i, 0] = float(p[0])
                coords[i, 1] = float(p[1])
                coords[i, 2] = float(p[2])

        if mode == "cog":
            return coords.mean(axis=0)

        # COM
        if self.system is None:
            raise RuntimeError("System has not been created yet; cannot compute COM.")

        masses = np.empty((len(group),), dtype=float)
        for i, idx in enumerate(group):
            m = self.system.getParticleMass(int(idx))
            # Quantity-like in dalton; we only need a unitless weight.
            if hasattr(m, "value_in_unit") and hasattr(m, "unit"):
                masses[i] = float(m.value_in_unit(m.unit))
            else:
                masses[i] = float(getattr(m, "_value", m))

        m_tot = float(masses.sum())
        if m_tot == 0.0:
            raise ValueError("Total mass of group is zero; cannot compute COM.")
        return (masses[:, None] * coords).sum(axis=0) / m_tot

    def set_umbrella_center(
        self,
        group,
        *,
        k=10.0,
        target=None,  # single or per-group; see docstring
        periodic=False,
        center="cog",
    ):
        """
        Restrain the center (COM or COG) of one or more groups to fixed points.

        Parameters
        ----------
        group
            Either:
              - sequence[int]: one group of atom indices, or
              - sequence[sequence[int]]: multiple groups, each a sequence of atom indices.
        k
            Force constant in kJ/mol/nm^2 (shared for all groups).
        target
            - None:
                For each group, target is taken as its current center.
            - Single (x,y,z) in nm (tuple/list) or 3-element Quantity:
                Same target used for all groups.
            - Sequence of length 1 or n_groups:
                Per-group targets; each element may be:
                  * None  -> use current center of that group
                  * (x,y,z) in nm (tuple/list)
                  * 3-element Quantity with length units
        periodic
            If True and a periodic box is defined, enable PBC on the bias.
        center
            - cog:
                Center of geometry (default)
            - com
                Center of mass
        """
        from collections.abc import Sequence

        if self.system is None:
            raise RuntimeError("System has not been created yet.")

        # --- normalize groups to a list of lists of ints --------------------
        def _is_int_like(x):
            return isinstance(x, int)

        if not isinstance(group, Sequence) or len(group) == 0:
            return

        if _is_int_like(group[0]):
            # single group: [i,j,k,...]
            groups = [list(group)]
        else:
            # multiple groups: [[...], [...], ...]
            groups = [list(g) for g in group]

        if not groups:
            return

        n_groups = len(groups)

        # --- positions in nm (computed lazily, only if needed) -------------
        pos_nm = None

        def _get_pos_nm():
            nonlocal pos_nm
            if pos_nm is not None:
                return pos_nm

            if self.positions is not None:
                pos = self.positions
            elif self.simulation is not None:
                pos = self.simulation.context.getState(getPositions=True).getPositions()
            else:
                raise RuntimeError("No positions available to define center restraint reference.")

            if hasattr(pos, "value_in_unit"):
                pos_nm_local = pos.value_in_unit(nanometer)
            else:
                pos_nm_local = pos  # assume already in nm

            pos_nm = pos_nm_local
            return pos_nm

        # --- helpers --------------------------------------------------------
        def _norm_xyz(t):
            """Return (x,y,z) in nm as floats from Quantity or 3-sequence."""
            if hasattr(t, "value_in_unit"):
                arr = t.value_in_unit(nanometer)
                return float(arr[0]), float(arr[1]), float(arr[2])
            # assume 3-sequence of numbers
            return float(t[0]), float(t[1]), float(t[2])

        def _is_scalar_xyz(seq):
            """Heuristic: 3 non-sequence elements -> treat as single xyz."""
            if not isinstance(seq, Sequence):
                return False
            if len(seq) != 3:
                return False
            for v in seq:
                if isinstance(v, Sequence) and not hasattr(v, "value_in_unit"):
                    return False
            return True

        # --- build per-group reference coordinates --------------------------
        xyz_list = []

        if target is None:
            # All targets from current centers
            pos_nm = _get_pos_nm()
            for g in groups:
                xyz_list.append(self._compute_center(g, pos_nm, center=center))
        else:
            # target provided; could be:
            # - Quantity -> same for all groups
            # - (x,y,z) -> same for all groups
            # - sequence of per-group entries
            if hasattr(target, "value_in_unit") or _is_scalar_xyz(target):
                base = _norm_xyz(target)
                xyz_list = [base for _ in range(n_groups)]
            else:
                # Treat as per-group target list
                if not isinstance(target, Sequence):
                    raise TypeError(
                        "target must be None, a single (x,y,z)/Quantity, "
                        "or a sequence of per-group targets."
                    )

                # allow broadcasting: length 1 -> repeat for all groups
                if len(target) == 1 and n_groups > 1:
                    per_group = list(target) * n_groups
                else:
                    if len(target) != n_groups:
                        raise ValueError(
                            "Per-group target sequence length must be 1 or match "
                            "the number of groups."
                        )
                    per_group = list(target)

                for g, t in zip(groups, per_group):
                    if t is None:
                        pos_nm = _get_pos_nm()
                        xyz_list.append(self._compute_center(g, pos_nm, center=center))
                    else:
                        xyz_list.append(_norm_xyz(t))

        # --- define the CustomCentroidBondForce -----------------------------
        bias = "0.5 * uk_center * ((x1 - x0)^2 + (y1 - y0)^2 + (z1 - z0)^2)"
        force = CustomCentroidBondForce(1, bias)
        force.addGlobalParameter("uk_center", k * kilojoule / (mole * nanometer**2))
        force.addPerBondParameter("x0")
        force.addPerBondParameter("y0")
        force.addPerBondParameter("z0")

        # add groups
        group_ids = []
        if center.lower() == "cog":
            for g in groups:
                gid = force.addGroup(g, [1.0] * len(g))
                group_ids.append(gid)
        elif center.lower() == "com":
            for g in groups:
                gid = force.addGroup(g)
                group_ids.append(gid)
        else:
            raise ValueError(f"Center option {center} is not valid.")

        # add one bond per group
        for gid, (x0, y0, z0) in zip(group_ids, xyz_list):
            force.addBond([gid], [x0, y0, z0])

        if self.box_vectors and periodic:
            force.setUsesPeriodicBoundaryConditions(True)

        force.setName("Umbrella_center")
        self.system.addForce(force)

    def update_umbrella_center(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_center", k * kilojoule / (mole * nanometer**2))

    def set_umbrella_angle_norm(
        self,
        groupa,
        groupa1,
        groupa2,
        groupb,
        groupb1,
        groupb2,
        *,
        target=np.radians(0),
        k=10.0,
        center="cog",
    ):
        """
        Harmonic umbrella on the angle between two plane normals (groups A and B).

        Angle is in radians; target is in radians; k is in kJ/mol/rad^2.
        """
        if not self.system:
            return

        bias = (
            # Harmonic in the angle between plane normals
            "0.5 * uk_angle_norm * (theta - target)^2;"
            # angle in [0, pi]: y >= 0 ensures atan2 ∈ [0, pi]
            "theta = atan2(sinang, cosang);"
            "sinang = magCross / denom;"
            "cosang = dotAB / denom;"
            # denom ~ |nA||nB| via dot/cross identity, with small epsilon to avoid 0
            "denom = sqrt(dotAB*dotAB + magCross*magCross) + 1e-8;"
            "magCross = sqrt(cx*cx + cy*cy + cz*cz);"
            # nA × nB
            "cx = nyA_tmp*nzB_tmp - nzA_tmp*nyB_tmp;"
            "cy = nzA_tmp*nxB_tmp - nxA_tmp*nzB_tmp;"
            "cz = nxA_tmp*nyB_tmp - nyA_tmp*nxB_tmp;"
            # nA · nB
            "dotAB = nxA_tmp*nxB_tmp + nyA_tmp*nyB_tmp + nzA_tmp*nzB_tmp;"
            # plane A normal nA = vA1 × vA2
            "nxA_tmp = vA1y*vA2z - vA1z*vA2y;"
            "nyA_tmp = vA1z*vA2x - vA1x*vA2z;"
            "nzA_tmp = vA1x*vA2y - vA1y*vA2x;"
            # plane B normal nB = vB1 × vB2
            "nxB_tmp = vB1y*vB2z - vB1z*vB2y;"
            "nyB_tmp = vB1z*vB2x - vB1x*vB2z;"
            "nzB_tmp = vB1x*vB2y - vB1y*vB2x;"
            # in-plane vectors for plane A: (A1-A0) and (A2-A0)
            "vA1x = x2 - x1;"
            "vA1y = y2 - y1;"
            "vA1z = z2 - z1;"
            "vA2x = x3 - x1;"
            "vA2y = y3 - y1;"
            "vA2z = z3 - z1;"
            # in-plane vectors for plane B: (B1-B0) and (B2-B0)
            "vB1x = x5 - x4;"
            "vB1y = y5 - y4;"
            "vB1z = z5 - z4;"
            "vB2x = x6 - x4;"
            "vB2y = y6 - y4;"
            "vB2z = z6 - z4;"
        )

        force = CustomCentroidBondForce(6, bias)
        force.addPerBondParameter("target")  # radians
        force.addGlobalParameter("uk_angle_norm", k * kilojoule / (mole * radian**2))

        # group order: (A0, A1, A2, B0, B1, B2)
        if center.lower() == "cog":
            force.addGroup(groupa, [1.0] * len(groupa))
            force.addGroup(groupa1, [1.0] * len(groupa1))
            force.addGroup(groupa2, [1.0] * len(groupa2))
            force.addGroup(groupb, [1.0] * len(groupb))
            force.addGroup(groupb1, [1.0] * len(groupb1))
            force.addGroup(groupb2, [1.0] * len(groupb2))
        elif center.lower() == "com":
            force.addGroup(groupa)
            force.addGroup(groupa1)
            force.addGroup(groupa2)
            force.addGroup(groupb)
            force.addGroup(groupb1)
            force.addGroup(groupb2)
        else:
            raise ValueError(f"Center option {center} is not valid.")

        force.addBond([0, 1, 2, 3, 4, 5], [target * radian])

        force.setName("Umbrella_angle_norm")
        self.system.addForce(force)

    def update_umbrella_angle_norm(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter(
                "uk_angle_norm", k * kilojoule / (mole * radian**2)
            )

    def set_umbrella_dihedral(
        self,
        groupa,
        groupb,
        groupc,
        groupd,
        *,
        target=0.0,
        k=10.0,
        center="cog",
    ):
        """
        Harmonic umbrella on the dihedral angle between four centroids.

        The minimum is at the dihedral = target (in radians), using a 2π-periodic
        quadratic distance: we choose the smallest of (Δ, Δ+2π, Δ-2π).

        Parameters
        ----------
        groupa, groupb, groupc, groupd : sequence[int]
            Atom indices for the four centroid groups.
        target : float
            Target dihedral angle in radians.
        k : float
            Force constant in kJ/mol/rad^2. Only used on first creation;
            later calls just add more bonds. Adjust at runtime with
            `update_umbrella_dihedral`.
        """
        if not self.system:
            return

        # Try to find an existing Umbrella_dihedral force
        force = None
        for f in self.system.getForces():
            if isinstance(f, CustomCentroidBondForce) and f.getName() == "Umbrella_dihedral":
                force = f
                break

        if force is None:
            # Create the force the first time
            bias = (
                "0.5 * uk_dihedral * delta^2; "
                "delta = delta - 2*pi*floor((delta + pi)/(2*pi));"
                "pi = acos(-1);"
                "delta = d - target;"
                "d = dihedral(g1, g2, g3, g4);"
            )

            force = CustomCentroidBondForce(4, bias)
            force.addPerBondParameter("target")  # radians
            force.addGlobalParameter(
                "uk_dihedral",
                k * kilojoule / (mole * radian**2),
            )
            force.setName("Umbrella_dihedral")
            self.system.addForce(force)

        # For each call, add new centroid groups + a new bond
        if center.lower() == "cog":
            idx_a = force.addGroup(groupa, [1.0] * len(groupa))
            idx_b = force.addGroup(groupb, [1.0] * len(groupb))
            idx_c = force.addGroup(groupc, [1.0] * len(groupc))
            idx_d = force.addGroup(groupd, [1.0] * len(groupd))
        elif center.lower() == "com":
            idx_a = force.addGroup(groupa)
            idx_b = force.addGroup(groupb)
            idx_c = force.addGroup(groupc)
            idx_d = force.addGroup(groupd)
        else:
            raise ValueError(f"Center option {center} is not valid.")

        force.addBond([idx_a, idx_b, idx_c, idx_d], [target * radian])

    def update_umbrella_dihedral(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_dihedral", k * kilojoule / mole / radian**2)

    def set_umbrella_angle(
        self,
        groupa,
        groupb,
        groupc,
        *,
        target=np.pi / 2.0,
        k=10.0,
        center="cog",
    ):
        """
        Harmonic umbrellas on an angle defined by centroid triplets.

        Angle is in radians; restrained to `target`.

        The angle is:
          - angle(groupa, groupb, groupc)

        All angle restraints share the same global force constant `uk_angle`.
        """
        if not self.system:
            return

        # Try to find an existing Umbrella_angle force
        force = None
        for f in self.system.getForces():
            if isinstance(f, CustomCentroidBondForce) and f.getName() == "Umbrella_angle":
                force = f
                break

        if force is None:
            # Create the force the first time
            bias = "0.5 * uk_angle * (angle(g1, g2, g3) - target)^2"
            force = CustomCentroidBondForce(3, bias)
            force.addPerBondParameter("target")  # radians
            force.addGlobalParameter("uk_angle", k * kilojoule / (mole * radian**2))
            force.setName("Umbrella_angle")

            self.system.addForce(force)

        # For each call, add new centroid groups + a new bond
        if center.lower() == "cog":
            idx_a = force.addGroup(groupa, [1.0] * len(groupa))
            idx_b = force.addGroup(groupb, [1.0] * len(groupb))
            idx_c = force.addGroup(groupc, [1.0] * len(groupc))
        elif center.lower() == "com":
            idx_a = force.addGroup(groupa)
            idx_b = force.addGroup(groupb)
            idx_c = force.addGroup(groupc)
        else:
            raise ValueError(f"Center option {center} is not valid.")

        force.addBond([idx_a, idx_b, idx_c], [target * radian])

    def update_umbrella_angle(self, k=10.0):
        if self.system and self.simulation:
            self.simulation.context.setParameter("uk_angle", k * kilojoule / mole / radian**2)

    def setupCMMotionRemover(self) -> None:
        if self.topology is not None and self.removecmmotion:
            force = CMMotionRemover()
            force.setName("cmmotion")
            self.forces["cmmotion"] = force
            self.system.addForce(force)

    @staticmethod
    def _group_energy(context, group: int):
        mask = 1 << group
        st: State = context.getState(getEnergy=True, groups=mask)
        return st.getPotentialEnergy()

    def set_params(self, version=None):
        if version is not None and version == 1:
            if self.params is None:
                self.params = RESIDUE_PARAMS_V1
            if self.eps is None:
                self.eps = EPS_V1
            if self.surfscale is None:
                self.surfscale = None
        else:
            # default is version 2
            if self.params is None:
                self.params = RESIDUE_PARAMS_V2
            if self.eps is None:
                self.eps = EPS_V2
            if self.surfscale is None:
                self.surfscale = surf_scale_v2

    def set_bonds(self):
        if self.topology is not None:
            for c in self.topology.chains():
                atm = []
                for r in c.residues():
                    for a in r.atoms():
                        atm.append(a)
                for i in range(len(atm) - 1):
                    self.topology.addBond(atm[i], atm[i + 1])

    @staticmethod
    def _normalize_box(box) -> tuple[float, float, float]:
        """
        Normalize user input to a 3-tuple of floats in nanometers.
        Accepts:
          - scalar number (int/float)
          - 3-sequence of numbers
          - openmm.unit.Quantity scalar/3-sequence with length units
        """
        # Quantity support (optional but handy)
        if isinstance(box, Quantity):
            # convert to nm and pull magnitude
            box_in_nm = box.value_in_unit(nanometer)
            if isinstance(box_in_nm, (int, float)):
                val = float(box_in_nm)
                return (val, val, val)
            # sequence quantity
            if isinstance(box_in_nm, Sequence) and len(box_in_nm) == 3:
                ax, by, cz = map(float, box_in_nm)
                return (ax, by, cz)
            raise TypeError("Quantity box must be scalar or length-3.")

        # Plain numeric assumed to be in angstroms
        if isinstance(box, (int, float)):
            val = float(box) / 10.0
            return (val, val, val)

        # Plain sequence assumed to be in angstroms
        if isinstance(box, Sequence) and len(box) == 3:
            ax, by, cz = box
            if not all(isinstance(v, (int, float)) for v in (ax, by, cz)):
                raise TypeError("Box tuple must contain numbers.")
            return (float(ax) / 10.0, float(by) / 10.0, float(cz) / 10.0)

        raise TypeError("box must be a number, a length-3 tuple, or a Quantity.")


def _is_readable_file(path):
    """Return True if `path` is a readable file; False for None or invalid types."""
    if not isinstance(path, str):
        return False
    return os.path.isfile(path) and os.access(path, os.R_OK)
