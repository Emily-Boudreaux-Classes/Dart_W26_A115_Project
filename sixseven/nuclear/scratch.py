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


def burn(temp: float, rho: float, time: float):
    
    C = init_composition()
    netIns = []
    for T, R in zip(temp, rho):
        netIns.append(init_netIn(T, R, time, C))
    policy = MainSequencePolicy(C)
    construct = policy.construct()
    local_solver = PointSolver(construct.engine)
    grid_solver = GridSolver(construct.engine, local_solver)
    solver_ctx = GridSolverContext(construct.scratch_blob)
    results = grid_solver.evaluate(solver_ctx, netIns)
    #print(type(results[0]))
    # output specific energy (ergs/g/s) pick mass to multiply to get total internal energy and mean molecular mass
    
    #print(results[0].composition.getMassFraction())
    #print(results[0].composition.getMeanParticleMass())
    print(results[0].__dir__())
    #return results

    
    #plt.plot(temps,energies)
    #plt.xlabel('Temperature [K]')
    #plt.ylabel('Energy [erg/g]')
    #plt.savefig('../../output/plots/temp-vs-energy.png')
    #plt.clf()
    #plt.plot(rhos,energies)
    #plt.xlabel('Density [g/cm^3]')
    #plt.ylabel('Energy [erg/g]')
    #plt.savefig('../../output/plots/density-vs-energy.png')

    # take results, energy will be one of them, specific neutrino loss will be another, will need to extract these, multiply by total mass of shell that's burning, then take the difference
    # epsilon might be specific energy
    # Calculate mean molecular mass, give to cassie b/c this changes the EOS
    # maybe graph of energy generation, burning rate

    # make a graph that shows energy generation as a function of temperature
   
temps = np.linspace(1.5e7, 2e7, 100)
rhos = np.linspace(1.5e2, 1.5e2, 100) 
print(burn(temps,rhos,1000))

