from __future__ import print_function
import numpy as np
from openmm.unit import *
from openmm import *
from openmm.app import *
from parmed import load_file, unit as u
from parmed.openmm.reporters import NetCDFReporter
import parmed as pmd
from parmed.openmm import topsystem
from biopandas.pdb import PandasPdb
import argparse
import pickle
import os

## Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='path', help='Input and output file path', required=True)
parser.add_argument('-n', dest='count', help='Number of monomers', required=True)
parser.add_argument('-surf', dest='surf', help='Surface scaling', required=False)
parser.add_argument('-d', dest='domain', nargs='+', type=int, help='Domain boundaries, if non are provided, system is treated as IDP')
parser.add_argument('-b', dest='box', help='Box length', required=True)
parser.add_argument('-r', dest='resources', help='Compute resource (CUDA or CPU)', default='CUDA')
parser.add_argument('-f', dest='pdb_file', help='PDB file name', required=False)
parser.add_argument('-s', dest='surf_file', help='SASA file name', required=False)
args = parser.parse_args()

resources = args.resources.upper()
if resources not in ['CUDA', 'CPU']:
    raise ValueError("Invalid resource specified. Use 'CUDA' or 'CPU'.")

path = args.path + '/'
count = int(args.count)
surf = float(args.surf) if args.surf else 0.7
BOX = float(args.box)

if args.pdb_file:
    pdb_file = path + args.pdb_file
else:
    pdb_file = path + "input.pdb"

def surface_calc(a,surf):
    if surf > 0:
        return np.min([a,surf])*1/surf
    else:
        return 1

## COCOMO Force Field parameters
# Force Field Parameters
kbond = 4184
theta0 = 180
l0_pro, l0_rna = 0.38, 0.5
kangle_pro, kangle_rna = 4.184, 5.021
cationpi_propro,cationpi_prorna,pipi_propro = 0.30, 0.20, 0.10
eps_polar, eps_nopol = 0.176, 0.295
azero_polar, azero_hydro = 0, 0.0002
kappa = 1

## Define elastic network parameters
force_constant = 500.0
cutoff_distance = 0.9 * nanometer

## Amino acids paramters (+RNA parameters from COCOMO)
ff_param = {'ALA': {'mass':  71.079, 'charge':  0.0, 'radius': 0.2845, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 0.796},
            'ARG': {'mass': 157.197, 'charge':  1.0, 'radius': 0.3567, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.921},
            'ASN': {'mass': 114.104, 'charge':  0.0, 'radius': 0.3150, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.281},
            'ASP': {'mass': 114.080, 'charge': -1.0, 'radius': 0.3114, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.162},
            'CYS': {'mass': 103.139, 'charge':  0.0, 'radius': 0.3024, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 1.074},
            'GLN': {'mass': 128.131, 'charge':  0.0, 'radius': 0.3311, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.575},
            'GLU': {'mass': 128.107, 'charge': -1.0, 'radius': 0.3279, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.462},
            'GLY': {'mass':  57.052, 'charge':  0.0, 'radius': 0.2617, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 0.544},
            'HIS': {'mass': 137.142, 'charge':  0.0, 'radius': 0.3338, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.634},
            'ILE': {'mass': 113.160, 'charge':  0.0, 'radius': 0.3360, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 1.410},
            'LEU': {'mass': 113.160, 'charge':  0.0, 'radius': 0.3363, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 1.519},
            'LYS': {'mass': 129.183, 'charge':  1.0, 'radius': 0.3439, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.923},
            'MET': {'mass': 131.193, 'charge':  0.0, 'radius': 0.3381, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 1.620},
            'PHE': {'mass': 147.177, 'charge':  0.0, 'radius': 0.3556, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 1.869},
            'PRO': {'mass':  98.125, 'charge':  0.0, 'radius': 0.3187, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 0.974},
            'SER': {'mass':  87.078, 'charge':  0.0, 'radius': 0.2927, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 0.933},
            'THR': {'mass': 101.105, 'charge':  0.0, 'radius': 0.3108, 'epsilon': eps_polar, 'azero': azero_polar, 'surface': 1.128},
            'TRP': {'mass': 186.214, 'charge':  0.0, 'radius': 0.3754, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 2.227},
            'TYR': {'mass': 163.176, 'charge':  0.0, 'radius': 0.3611, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 2.018},
            'VAL': {'mass':  99.133, 'charge':  0.0, 'radius': 0.3205, 'epsilon': eps_nopol, 'azero': azero_hydro, 'surface': 1.232}}
            #'ADE': {'mass': 315.697, 'charge': -1.0, 'radius': 0.4220, 'epsilon': 0.41,      'azero': 0.05}, 
            #'CYT': {'mass': 305.200, 'charge': -1.0, 'radius': 0.4110, 'epsilon': 0.41,      'azero': 0.05}, 
            #'GUA': {'mass': 345.200, 'charge': -1.0, 'radius': 0.4255, 'epsilon': 0.41,      'azero': 0.05}, 
            #'URA': {'mass': 305.162, 'charge': -1.0, 'radius': 0.4090, 'epsilon': 0.41,      'azero': 0.05}} 


## build chain list (this part can be modified in case of oligomers, multiple systems, etc.)
chains_comp = []
for i in range(count+1):
    chains_comp += [i]
    
## Read atom list and coordinates from PDB file
def read_pdb(filename):
    pdb_file = PandasPdb().read_pdb(filename)
    atoms = pdb_file.df['ATOM'][['atom_name','residue_name','segment_id']].values
    return atoms

positions = PDBFile(pdb_file).positions
atom_list = read_pdb(pdb_file)

segnames = [atom_list[i,2] != atom_list[i+1,2] for i in range(-1,atom_list.shape[0]-1)]
segnames = atom_list[segnames,2]

## in case if the simulation is only run with one chain
if len(segnames) == 0:
    segnames = ['AAA']

## Set domain definition + read in and adjust per-residue SASA information. 
if args.domain:
    domain = args.domain
    if args.surf_file:
        surface_vector = np.loadtxt(path + args.surf_file).tolist()
    else:
        surface_vector = np.loadtxt(path + 'surface').tolist()
    surface_vector = np.array(surface_vector)
    
    if domain[-1] > len(surface_vector):
        print("Domain range larger than the size of the surface vector! Use all atoms for elastic network.")
        domain = [1,len(surface_vector)]

    for j in range(len(domain)):
        if j == 0 and domain[j] > 1:
            surface_vector[:domain[j] - 1] = 5
        elif j % 2 == 0:
            surface_vector[domain[j - 1] - 1:domain[j] - 1] = 5
        elif j == len(domain) - 1 and domain[j] < len(surface_vector):
            surface_vector[domain[j] - 1:] = 5

    surface_vector = surface_vector.tolist() * count
    surface_vector = np.array(surface_vector)
else:
    domain = [1,len(atom_list)]
    surface_vector = [5] * len(atom_list) * count


## Generate topolgy
top = topology.Topology()
for seg in segnames:
    chain = top.addChain(seg)
    for atm in atom_list:
        if atm[2] == seg:
            residue = top.addResidue(atm[1], chain)
            top.addAtom(atm[0], element=element.carbon, residue=residue)
            #
    atom = [i for i in chain.atoms()]
    for i in range(len(atom)-1):
        top.addBond(atom[i],atom[i+1])

a = Quantity(np.zeros([3]), nanometers)
a[0] = BOX * nanometers
b = unit.Quantity(np.zeros([3]), nanometers)
b[1] = BOX * nanometers
c = unit.Quantity(np.zeros([3]), nanometers)
c[2] = BOX * nanometers
box = (a,b,c)
top.setPeriodicBoxVectors(box)

# Build system
system = openmm.System()

# add Particles to the system
for i in atom_list[:,1]: system.addParticle(ff_param[i]['mass']*unit.amu)

# set BOX vectors to the system
system.setDefaultPeriodicBoxVectors(a, b, c)


# Set energy switching function on or use cutoff only
# Switching implementation following Steinbach & Brooks paper
# equation (10), https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.540150702
# this switching implementation slows simulation by ~20%
SWITCHING = False #True

if SWITCHING == True:
    R_ON  = 2.9
    R_OFF = 3.1
else:
    R_OFF = 3.0

## Add COCOMO forces

# bond force
f_bond = openmm.HarmonicBondForce()
for bond in top.bonds():
    if bond[0].residue.name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE','LEU', 'LYS', 'MET',
                                'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
        f_bond.addBond(bond[0].index, bond[1].index, l0_pro*nanometer, kbond*kilojoules_per_mole/(nanometer**2))
    elif bond[0].residue.name in ['ADE', 'CYT', 'GUA', 'URA']:
        f_bond.addBond(bond[0].index, bond[1].index, l0_rna*nanometer, kbond*kilojoules_per_mole/(nanometer**2))

system.addForce(f_bond)

# angle force
f_angle = openmm.HarmonicAngleForce()
for atoms in [[i for i in top.atoms() if i.residue.chain.id == seg] for seg in segnames]:
    if atoms[0].residue.name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                                 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
        for i in range(len(atoms)-2):
            f_angle.addAngle(atoms[i].index, atoms[i+1].index, atoms[i+2].index, theta0*degrees, kangle_pro*kilojoule_per_mole/(radian**2))
    elif atoms[0].residue.name in ['ADE', 'CYT', 'GUA', 'URA']:
        for i in range(len(atoms)-2):
            f_angle.addAngle(atoms[i].index, atoms[i+1].index, atoms[i+2].index, theta0*degrees, kangle_rna*kilojoule_per_mole/(radian**2))

system.addForce(f_angle)

# openMM Center of Mass Motion Remover
system.addForce(openmm.CMMotionRemover())

# electrostactic
k0 = kappa*nanometer

if SWITCHING == True:
    equation1  = "select( step(r_on-r), longrange+sdel, switch ); "
    equation1 += "longrange = (A+Z)*exp(-r/K0)/r; "
    equation1 += "sdel = sk*((1/r_on)^3-1/(r_off)^3)^2 - (A+Z)/r_on*exp(-r_on/K0); "
    equation1 += "switch = sk*((1/r)^3-1/(r_off)^3)^2; "
    equation1 += "sk = -longrange_deriv_Ron/switch_deriv_Ron; "
    equation1 += "longrange_deriv_Ron = -1*(A+Z)*exp(-r_on/K0)/r_on*(1/K0+1/r_on); "
    equation1 += "switch_deriv_Ron = 6*(1/r_on^3-1/r_off^3)*1/r_on^4; "
    equation1 += "A=A1*A2; "
    equation1 += "Z=Z1+Z2 "
else:
    equation1  = "S*(A+Z)/r*exp(-r/K0); "   
    equation1 += "A=A1*A2; "
    equation1 += "Z=Z1+Z2; "
    equation1 += "S=(S1*S2)^(1/2)"

force1 = CustomNonbondedForce(equation1)
force1.addGlobalParameter("K0", k0)
force1.addPerParticleParameter("A")
force1.addPerParticleParameter("Z")
force1.addPerParticleParameter("S")
force1.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
force1.setCutoffDistance(R_OFF*nanometer)
if SWITCHING == True:
    force1.addGlobalParameter("r_on", R_ON)
    force1.addGlobalParameter("r_off", R_OFF)

for atom in top.atoms():
    force1.addParticle(
            [(np.sqrt(0.75*np.abs(ff_param[atom.residue.name]['charge']))*np.sign(ff_param[atom.residue.name]['charge']))*nanometer*kilojoule/mole,
                ff_param[atom.residue.name]['azero'] *(nanometer*kilojoule/mole)**(1/2),
                surface_calc(surface_vector[atom.index]/ff_param[atom.residue.name]['surface'],surf)])

force1.createExclusionsFromBonds([(i[0].index, i[1].index) for i in top.bonds()], 1)
system.addForce(force1)

# short range
if SWITCHING == True:
    equation2  = "select( step(r_on-r), shortrange+sdel, switch ); "
    equation2 += "shortrange = 4*epsilon*((sigma/r)^10-(sigma/r)^5); "
    equation2 += "sdel = sk*((1/r_on)^3-1/(r_off)^3)^2 - 4*epsilon*((sigma/r_on)^10-(sigma/r_on)^5); "
    equation2 += "switch = sk*((1/r)^3-1/(r_off)^3)^2; "
    equation2 += "sk = -shortrange_deriv_Ron/switch_deriv_Ron; "
    equation2 += "shortrange_deriv_Ron = 4*epsilon*(-10*(sigma/r_on)^11*1/r_on+5*(sigma/r_on)^5*1/r_on ); "
    equation2 += "switch_deriv_Ron = 6*(1/r_on^3-1/r_off^3)*1/r_on^4; "
    equation2 += "sigma=0.5*(sigma1+sigma2); "
    equation2 += "epsilon=sqrt(epsilon1*epsilon2)"
else:
    equation2  = "S*4*epsilon*((sigma/r)^10-(sigma/r)^5); "  ## add scaling value
    equation2 += "sigma=0.5*(sigma1+sigma2); "
    equation2 += "epsilon=sqrt(epsilon1*epsilon2); "
    equation2 += "S=(S1*S2)^(1/2)"

force2 = CustomNonbondedForce(equation2)
force2.addPerParticleParameter("sigma")
force2.addPerParticleParameter("epsilon")
force2.addPerParticleParameter("S")

force2.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
force2.setCutoffDistance(R_OFF*nanometer)
if SWITCHING == True:
    force2.addGlobalParameter("r_on", R_ON)
    force2.addGlobalParameter("r_off", R_OFF)


for atom in top.atoms():
    force2.addParticle([ff_param[atom.residue.name]['radius']*2*2**(-1/6)*nanometer, 
                        ff_param[atom.residue.name]['epsilon']*kilojoule/mole,
                        surface_calc(surface_vector[atom.index]/ff_param[atom.residue.name]['surface'],surf)])

force2.createExclusionsFromBonds([(i[0].index, i[1].index) for i in top.bonds()], 1)
system.addForce(force2)
    
# protein - protein cation pi 
if any(True for i in ['ARG', 'LYS'] if i in [atom.residue.name for atom in top.atoms()]) and any(True for i in ['PHE', 'TRP', 'TYR'] if i in [atom.residue.name for atom in top.atoms()]) :
    force3 = CustomNonbondedForce(equation2)
    force3.addPerParticleParameter("sigma")
    force3.addPerParticleParameter("epsilon")
    force3.addPerParticleParameter("S")
    force3.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force3.setCutoffDistance(R_OFF*nanometer)
    if SWITCHING == True:
        force3.addGlobalParameter("r_on", R_ON)
        force3.addGlobalParameter("r_off", R_OFF)

    for atom in top.atoms():
        force3.addParticle([ff_param[atom.residue.name]['radius']*2*2**(-1/6)*nanometer, 
            cationpi_propro*kilojoule/mole,
            surface_calc(surface_vector[atom.index]/ff_param[atom.residue.name]['surface'],surf)])
        #
    force3.createExclusionsFromBonds([(i[0].index, i[1].index) for i in top.bonds()], 1)
    arg_lys  = [atom.index for atom in top.atoms() if atom.residue.name in ['ARG', 'LYS']]
    aromatic = [atom.index for atom in top.atoms() if atom.residue.name in ['PHE', 'TRP', 'TYR']]
    force3.addInteractionGroup(arg_lys, aromatic)
    system.addForce(force3)

## protein - rna cation pi
#if any(True for i in ['ARG', 'LYS'] if i in [atom.residue.name for atom in top.atoms()]) and any(True for i in ['ADE', 'CYT', 'GUA', 'URA'] if i in [atom.residue.name for atom in top.atoms() ]) :
#    force4 = CustomNonbondedForce(equation2)
#    force4.addPerParticleParameter("sigma")
#    force4.addPerParticleParameter("epsilon")
#    force4.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
#    force4.setCutoffDistance(R_OFF*nanometer)
#    if SWITCHING == True:
#        force4.addGlobalParameter("r_on", R_ON)
#        force4.addGlobalParameter("r_off", R_OFF)
#    
#    for atom in top.atoms():
#        force4.addParticle([ff_param[atom.residue.name]['radius']*2*2**(-1/6)*nanometer,
#            cationpi_prorna*kilojoule/mole,
#            surface_calc(surface_vector[atom.index]/ff_param[atom.residue.name]['surface'],surf)]) # added S value
#    
#    force4.createExclusionsFromBonds([(i[0].index, i[1].index) for i in top.bonds()], 1)
#    arg_lys  = [atom.index for atom in top.atoms() if atom.residue.name in ['ARG', 'LYS']]
#    nucleic = [atom.index for atom in top.atoms() if atom.residue.name in ['ADE', 'CYT', 'GUA', 'URA']]
#    force4.addInteractionGroup(arg_lys, nucleic)
#    system.addForce(force4)

# Aro - Aro interaction
if any(True for i in ['PHE', 'TRP', 'TYR'] if i in [atom.residue.name for atom in top.atoms()]) :
    force5 = CustomNonbondedForce(equation2)
    force5.addPerParticleParameter("sigma")
    force5.addPerParticleParameter("epsilon")
    force5.addPerParticleParameter("S")
    force5.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force5.setCutoffDistance(R_OFF*nanometer)
    if SWITCHING == True:
        force5.addGlobalParameter("r_on", R_ON)
        force5.addGlobalParameter("r_off", R_OFF)

    for atom in top.atoms():
        force5.addParticle([ff_param[atom.residue.name]['radius']*2*2**(-1/6)*nanometer,
            pipi_propro*kilojoule/mole,
            surface_calc(surface_vector[atom.index]/ff_param[atom.residue.name]['surface'],surf)])
    force5.createExclusionsFromBonds([(i[0].index, i[1].index) for i in top.bonds()], 1)
    aromatic = [atom.index for atom in top.atoms() if atom.residue.name in ['PHE', 'TRP', 'TYR']]
    force5.addInteractionGroup(aromatic, aromatic)
    system.addForce(force5)


## Building elastic network if domain information are provided
if args.domain:
    elastic_force = CustomBondForce("0.5*k*(r-r0)^2")
    elastic_force.addGlobalParameter("k", force_constant / nanometer ** 2)
    elastic_force.addPerBondParameter("r0")

    ## iterate through chain
    for a in range(1,len(chains_comp)):
        atoms = []
        for b in range(chains_comp[a-1],chains_comp[a]):
            atoms += list(list(top.chains())[b].atoms())
    
        for k in range(0,len(domain),2):
            for i in range(domain[k],domain[k+1]-3):
                for j in range(i + 3, domain[k+1]):
                    atom1, atom2 = atoms[i], atoms[j]
                    distance = unit.norm(positions[atom1.index] - positions[atom2.index])
                    if distance < cutoff_distance:
                        elastic_force.addBond(atom1.index, atom2.index, [distance])

        print(f"Building ENM: {a}/{count}")
            
    # Add elastic network force to the system
    system.addForce(elastic_force)

## Save system files
with open(path + 'system.xml', 'w') as output:
    output.write(XmlSerializer.serialize(system))

import pickle
with open(path + 'top.pkl', 'wb') as file:
    pickle.dump(top, file)

with open(path + 'positions.pkl', 'wb') as file:
    pickle.dump(positions, file)


structure = topsystem.load_topology(top)
psf_filename = path + "output.psf"
structure.save(psf_filename, overwrite=True)

# Build simulation context
integrator = LangevinIntegrator(298*kelvin, 0.01/picosecond, 0.01*picoseconds)

# Set platform
platform = Platform.getPlatformByName(resources)
if resources == 'CUDA' :
    prop = dict(CudaPrecision='mixed')

if resources == 'CUDA' : simulation = Simulation(top, system, integrator, platform, prop)
if resources == 'CPU'  : simulation = Simulation(top, system, integrator, platform)

simulation.context.setPositions(positions)

print("\nInitial system energy")
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

## minimization
mini_nstep = 5000
print("\nEnergy minimization: %s steps" % mini_nstep)
simulation.minimizeEnergy(tolerance=100.0*kilojoule/(nanometer*mole), maxIterations=mini_nstep)
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

## generate velocities
simulation.context.setVelocitiesToTemperature(298*kelvin) # 870516298)

## run short simulation
nstep  = 100 
nstout = 10 
nstdcd = 10 
odcd = path + 'trajectory.dcd'
olog = path + 'trajectory.log'
simulation.reporters.append(DCDReporter(odcd, nstdcd))
simulation.reporters.append(StateDataReporter(olog, nstout, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, progress=True, remainingTime=True, speed=True, totalSteps=nstep, separator='\t'))

print("\nMD run: %s steps" % nstep)
simulation.step(nstep)
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

simulation.saveState(path + f'checkpoint_0.xml')

