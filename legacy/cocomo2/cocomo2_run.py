## Load libraries

from __future__ import print_function
import numpy as np

from openmm.unit import *
from openmm import *
from openmm.app import *

import pickle

import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='path', help='Input and output file path', required=True)
parser.add_argument('-n', dest='count', help='checkpoint counter', required=True)
args = parser.parse_args()



#define_file_name
resources = 'CUDA'
path = args.path
def get_output_filename(counter):
    checkpoint = path + '/' + f'checkpoint_{counter}.xml'
    odcd = path + '/' + f'output_{counter}.dcd'
    olog = path + '/' + f'output_{counter}.log'
    return checkpoint, odcd, olog 

restart_counter = int(args.count)
checkpoint, odcd, olog = get_output_filename(restart_counter)

#load files
with open(path + '/' + 'system.xml', 'r') as file:
    serialized_system = file.read()
system = XmlSerializer.deserialize(serialized_system)

with open(path + '/' + 'top.pkl', 'rb') as file:
    top = pickle.load(file)

with open(path + '/' + 'positions.pkl', 'rb') as file:
    positions = pickle.load(file)


# Build simulation context
integrator = LangevinIntegrator(298*kelvin, 0.01/picosecond, 0.01*picoseconds)

# Set platform
platform = Platform.getPlatformByName(resources)
if resources == 'CUDA' :
    prop = dict(CudaPrecision='mixed')

if resources == 'CUDA' : simulation = Simulation(top, system, integrator, platform, prop)
if resources == 'CPU'  : simulation = Simulation(top, system, integrator, platform)

# Path and filename of the restart file
state_file = path + '/' + f'checkpoint_{restart_counter-1}.xml'
simulation.loadState(state_file)

# run simulation
nstep  = 5000000    
nstout = 100000    
nstdcd = 100000   

print("\nInitial system energy")
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

simulation.reporters.append(DCDReporter(odcd, nstdcd))
simulation.reporters.append(StateDataReporter(olog, nstout, 
                                              step=True, time=True, 
                                              potentialEnergy=True, kineticEnergy=True, 
                                              totalEnergy=True, temperature=True, volume=True, 
                                              density=True, progress=True, remainingTime=True, 
                                              speed=True, totalSteps=nstep, separator='\t'))

checkpoint_file = path + '/' + 'checkpoint.chk'
simulation.reporters.append(CheckpointReporter(checkpoint_file, reportInterval=1000000))

print("\nMD run: %s steps" % nstep)
simulation.step(nstep)
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

simulation.saveState(checkpoint)
