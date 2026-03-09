# this is for testing the Strang Splitting approach to stepping - will be used in final version of full project
# -----
# run this file with the command "python -m sixseven.timestep.staggered_step" in the project directory
# -----

import sys
import numpy as np
import matplotlib.pyplot as plt

from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import burn
from sixseven.transport.transport_simple import transport_step

def main():
    print("Working ... ")

    parr = [] # stores values most changed of any variable at each iteration
    dparr = [] # stores largest change of any variable at each iteration
    Tarr = [] # stores temperatures per mass element per iteration
    rhoarr = [] # stores densities per mass element per iteration
    epsarr = [] # stores energy generated per mass element per iteration
    muarr = [] # stores mean mol weights per mass element per iteration
    sarr = [] # stores step sizes for each iteraetion
    tarr = [] # simulation time at each step (seconds)

    step = 1e14 # initial step 
    max_t = 1e16 # total time

    t = 0 # initial time
    n = 0 # initial step counter

    N = 100 # number of mass elements tracked 
    P = np.ones(N) * 1e17 # fixed value for now, roughly that of solar core, whatever

    u = np.empty((4,N)) # array of initial T, rho, eps, mu for each dM - shape: (4,N)
    T = np.ones(N) * 1e7 # burn requires arrays
    rho = np.ones(N) * 1e2
    dM = np.linspace(1e-5,1,N) * 1e32 # mass enclosed!

    # specific heats for ideal gas - set in stone
    add_index = ad_index(CONST.Cp_ideal,CONST.Cv_ideal)

    results = burn(temps=T,rhos=rho,time=1.,comps=None)

    eps = []
    mu = []
    mol_abund = []
    for i,j in enumerate(results):
        eps.append(j.energy)
        mu.append(j.composition.getMeanParticleMass()) 
        mol_abund.append(j.composition)

    eps = np.array(eps)
    mu = np.array(mu)
    mol_abund = np.array(mol_abund)

    Hp = (1.36e-16 * T) / (mu * 1.67e-24 * 2.74e4)
    structure = {"m":dM, 
                 "Hp":np.ones(N) * Hp, 
                 "v_mlt":np.ones(N)*1e5, # guess
                 "is_convective":np.full(N, True, dtype=bool), 
                 "grad_rad":np.ones(N) * nabla_rad(P=P,T=T,dT_dP=np.gradient(T,P)), 
                 "grad_ad":np.ones(N) * nabla_ad(add_index), 
                 "grad_mu":np.ones(N) * 0.01, # guess
                 "K":np.ones(N) * 1e7, # guess
                 "Cp":np.ones(N) * CONST.Cp_ideal,
                 "rho":rho,
                 "T":T}
    # we ball ?

    # intital diffusion from aretemis
    diff_results = transport_step(comps=results,structure=structure,dt=1.)
    
    u[0],u[1],u[2],u[3] = T,rho,eps,mu # initial conditions, after 1 sec

    U = init_U(mu=u[3],dM=dM,T=u[0])

    print("Running ... ")

    while t < max_t:
        if (n % 100) == 1:
            print("Iteration: ", n)
            print("log10(Step / s): ", np.log10(step))
            print("Dens: ", rho)
            print("Temp: ", T)
            print("Eps: ", eps)
            print("Mu: ", mu)


        # i have my structure/state at a given time - good 
        # i solve the micro stuff (burning, diffusion, eos, opacity, etc) at that time over a interval of step/2
        # i then solve the structure over the full step based on the half step micro results 
        # calc new FULL step size here 
        # evaluate micro over step/2 again 
        # rinse and repeat - micro and macro are step/2 apart, with new step ever time i evaluate macro
        

if __name__ == "__main__":
    main()