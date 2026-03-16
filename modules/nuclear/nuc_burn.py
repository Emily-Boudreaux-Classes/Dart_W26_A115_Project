from fourdst.composition import Composition
from gridfire.policy import MainSequencePolicy 
from gridfire.solver import PointSolver, GridSolver, GridSolverContext
from gridfire.type import NetIn
import numpy as np
import matplotlib.pyplot as plt 

def init_composition() -> Composition:
    Y = [7.0262E-01, 9.7479E-06, 6.8955E-02, 2.5000E-04, 7.8554E-05, 6.0144E-04, 8.1031E-05, 2.1513E-05] # Note these are molar abundances
    S = ["H-1", "He-3", "He-4", "C-12", "N-14", "O-16", "Ne-20", "Mg-24"]
    return Composition(S, Y) # returns mass abundances converted to mass fraction



def init_netIn(temp: float, rho: float, time: float, comp: Composition) -> NetIn:
    netIn = NetIn()
    netIn.temperature = temp
    netIn.density = rho
    netIn.tMax = time
    netIn.dt0 = 1e-12
    netIn.composition = comp
    return netIn



# 'temp', 'rho', and 'comp' should be arrays. 'time' is your time step in seconds. 
# burn() outputs the delta energy, mean molecular mass, and mass fraction results for your given time step.
# for the very first time step, leave 'comp' as none (it calls a function with an initial composition)
# for the following time steps, update 'comp' with the previous 'mass_frac' burn() output
# -----------------------------------------------------------
# Example usage: 
# - For initial run
# temps = np.linspace(1.5e7, 2e7, 100)
# rhos = np.linspace(1.5e2, 1.5e2, 100)
# e, mu, mf = nuc_burn.burn(temps, rhos, 1000)
# - For following runs
# e2, mu2, mf2 = nuc_burn.burn(newtemp, newrho, 1000, mf)
def burn(temp: float, rho: float, time: float, comp=None):
    if comp is None:
        C = init_composition() 
    else:
        C = comp
    netIns = []
    for T, R in zip(temp, rho):
        netIns.append(init_netIn(T, R, time, C))
    policy = MainSequencePolicy(C) # ensures it burns like a MS star
    construct = policy.construct()
    local_solver = PointSolver(construct.engine)
    grid_solver = GridSolver(construct.engine, local_solver)
    solver_ctx = GridSolverContext(construct.scratch_blob)
    results = grid_solver.evaluate(solver_ctx, netIns)
 
    epsilon = []
    mu = []
    mass_frac = results[0].composition

    for i,j in enumerate(results):
        epsilon.append(j.energy)
        mu.append(j.composition.getMeanParticleMass())

    return epsilon, mu, mass_frac
    

