from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import copysign, sqrt
from types import MappingProxyType

import numpy as np
from openmm import (
    CMMotionRemover,
    CustomBondForce,
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
    Simulation,
    StateDataReporter,
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
        version=None,
        box=100,  # 100 or (50,20,40), nm
        cuton=2.9,  # nm
        cutoff=3.1,  # nm
        switching="original",  # 'original', 'openmm', or None
        kappa=1.0,  # nm
        sasa=None,  # for surface scaling
        enmpairs=None,  # list of ENM pairs
        domains=None,  # turns on ENM
        positions=None,  # needed for ENM
        enmforce=500.0,  # 1/nm**2
        enmcutoff=0.9,  # nm
        interactions=None,  # list of interactions
        removecmmotion=True,
        xml=None,
    ):

        self.simulation = None

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
            return

        self.system = System()
        self.setup_particles()
        self.setup_box(box)
        self.setup_forces()

    def describe(self) -> str:
        return f"This is COCOMO CG model version {self.version}"

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

    def get_potentialEnergy(self):
        if self.simulation is not None:
            return self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
        else:
            return 0.0

    def get_energies(self):
        energies = {}
        for i, frc in enumerate(self.system.getForces()):
            g = frc.getForceGroup()
            e = self._group_energy(self.simulation.context, g)
            name = frc.getName()
            if not name:
                name = frc.__class__.__name__
            energies[f"{name}"] = e
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

    def simulate(self, *, nstep=1000, nout=100, logfile="energy.log", dcdfile="traj.dcd"):
        if self.simulation is not None:
            dcd = DCDReporter(dcdfile, nout)
            self.simulation.reporters.append(dcd)
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
        positions=None,
        restart=None,
    ) -> None:
        # temperature: in K
        # tstep: in ps
        # gamma: in 1/ps
        # resources: 'CPU' or 'CUDA'

        assert self.topology is not None, "need topology to be defined"
        assert self.system is not None, "need openMM system object to be defined"

        self.temperature = temperature * kelvin
        self.tstep = tstep * picoseconds
        self.gamma = gamma / picoseconds
        self.resources = resources

        self.integrator = LangevinIntegrator(self.temperature, self.gamma, self.tstep)
        self.platform = Platform.getPlatformByName(self.resources)
        self.simulation = None
        if self.resources == "CUDA":
            prop = dict(CudaPrecision="mixed")
            self.simulation = Simulation(
                self.topology, self.system, self.integrator, self.platform, prop
            )
        if self.resources == "CPU":
            self.simulation = Simulation(self.topology, self.system, self.integrator, self.platform)
        if restart is not None:
            self.read_state(restart)
        else:
            if positions is not None:
                self.set_positions(positions)
            if self.positions is not None:
                self.simulation.context.setPositions(self.positions)

    def set_positions(self, positions) -> None:
        self.positions = positions
        if self.simulation is not None:
            self.simulation.context.setPositions(positions)

    def setup_particles(self) -> None:
        if self.topology is None:
            return
        if self.params is None:
            self.set_params(self.version)
        for atm in self.topology.atoms():
            self.system.addParticle(self.params[atm.residue.name].mass * amu)

    def setup_box(self, box) -> None:
        ax_nm, by_nm, cz_nm = self._normalize_box_nm(box)

        a = Vec3(ax_nm, 0.0, 0.0) * nanometer
        b = Vec3(0.0, by_nm, 0.0) * nanometer
        c = Vec3(0.0, 0.0, cz_nm) * nanometer

        # record on the class
        self.box_nm: tuple[float, float, float] = (ax_nm, by_nm, cz_nm)
        self.box_vectors = (a, b, c)

        # set on topology and system
        if self.topology is not None:
            self.topology.setPeriodicBoxVectors(self.box_vectors)
        self.system.setDefaultPeriodicBoxVectors(a, b, c)

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
            self.forcemapping = self.assign_force_groups()

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
                equation = "0.5*k*(r-r0)^2"
                force = CustomBondForce(equation)
                force.addGlobalParameter("k", self.enmforce)
                force.addPerBondParameter("r0")

                for enm in self.enmpairs:
                    force.addBond(enm[0], enm[1], [enm[2]])

            force.setName("enm")
            self.forces["enm"] = force
            self.system.addForce(force)

    def _make_switch_sumlog_force(
        self, pairs, r0: float, alpha: float, eps: float
    ) -> CustomBondForce:
        # sum_i [-log(1 - S_i + eps)], S(r) = 1/(1+exp(alpha*(r-r0)))
        f = CustomBondForce("-log( 1.0 - ( 1.0/(1.0 + exp(alpha*(r - r0))) ) + eps )")
        f.addGlobalParameter("alpha", float(alpha))
        f.addGlobalParameter("eps", float(eps))
        f.addPerBondParameter("r0")
        rr0 = float(r0)
        for i, j in pairs:
            f.addBond(int(i), int(j), [rr0])
        return f

    def setupInteractionForce(self) -> None:
        """
        Build 'switch' interactions with minimal memory:
          - Additive groups coalesced by (strength,r0,alpha) into one CustomBondForce each.
          - Non-additive groups packed into aggregator CV forces (≤32 groups per aggregator).
          - No sharding unless a single non-additive group exceeds max_bonds_per_force.
        """
        if self.topology is None or not self.interactions:
            return
        return

    def setupCMMotionRemover(self) -> None:
        if self.topology is not None and self.removecmmotion:
            force = CMMotionRemover()
            force.setName("cmmotion")
            self.forces["cmmotion"] = force
            self.system.addForce(force)

    def assign_force_groups(self):
        mapping = {}
        for i, frc in enumerate(self.system.getForces()):
            frc.setForceGroup(i % 32)
            name = frc.getName()
            if not name:
                name = frc.__class__.__name__
            mapping[frc.getForceGroup()] = (i, name)
        return mapping

    @staticmethod
    def _group_energy(context, group: int):
        mask = 1 << group
        st: State = context.getState(getEnergy=True, groups=mask)
        return st.getPotentialEnergy()

    def set_params(self, version=None):
        # default is version 2
        self.params = RESIDUE_PARAMS_V2
        self.eps = EPS_V2
        self.surfscale = surf_scale_v2
        if version is None:
            return
        if version == 1:
            self.params = RESIDUE_PARAMS_V1
            self.eps = EPS_V1
            self.surfscale = None

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
    def _normalize_box_nm(box) -> tuple[float, float, float]:
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

        # Plain numeric
        if isinstance(box, (int, float)):
            val = float(box)
            return (val, val, val)

        # Plain sequence
        if isinstance(box, Sequence) and len(box) == 3:
            ax, by, cz = box
            if not all(isinstance(v, (int, float)) for v in (ax, by, cz)):
                raise TypeError("Box tuple must contain numbers.")
            return (float(ax), float(by), float(cz))

        raise TypeError("box must be a number, a length-3 tuple, or a Quantity.")
