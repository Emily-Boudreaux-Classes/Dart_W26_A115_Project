# this is the file where integration has been happening - copy, edit, and/or add your own work in your version of this to test integration!
# -----
# run this file with the command "python -m sixseven.timestep.testing" in the project directory
# -----

import sys
import numpy as np
import matplotlib.pyplot as plt

from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import burn
from sixseven.transport.transport_simple import transport_step

# mass cord is grams, mass enclosed 
# Hp is 1e9 ish 
# v conv is 1e3-5 in cgs

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

        T,rho,eps,mu = u[0],u[1],u[2],u[3] # setting variables from array for readability 

        results = burn(temps=T,rhos=rho,time=step,comps=mol_abund) # sasha - nuc burning 

        # only need to update these, the rest are set for now (guesses, constants, and ideal gas stuff)
        Hp = (1.36e-16 * T) / (mu * 1.67e-24 * 2.74e4)
        structure["Hp"] = Hp
        structure["grad_rad"] = nabla_rad(P=P,T=T,dT_dP=np.gradient(T,P))
        structure["rho"] = rho
        structure["T"] = T
        
        # diffusion of species after burning
        diff_results = transport_step(comps=results,structure=structure, dt=step) # artemis - simple diffusion
        
        eps = []
        mu = [] 
        mol_abund = []
        for i,j in enumerate(diff_results):
            eps.append(j.energy)
            mu.append(j.composition.getMeanParticleMass()) 
            mol_abund.append(j.composition)

        eps = np.asarray(eps) # setting the lists as arrays
        mu = np.asarray(mu)
        mol_abund = np.array(mol_abund)

        U = update_U(U,eps) # cassie - updates internal energy 
        T = temperature_solver(dM=dM,mu=mu,U=U) # cassie - solves temperature
        rho = simple_eos(P=P,mu=mu,T=T) # cassie - gets dens from ideal gas eos

        # make this derivatives instead of simple differences?
        du = np.array([T - u[0], rho - u[1], eps - u[2], mu - u[3]]) # change between steps
        step, p, dp = dyn_timestep(u, du, step, hfactor=1e15, min_step=1e8) # calc timestep, p, dp

        sarr.append(step)
        parr.append(p) 
        dparr.append(dp)
        Tarr.append(T[-1]) # take single values for now to plot, just checking things 
        rhoarr.append(rho[-1])
        epsarr.append(eps[-1])
        muarr.append(mu[-1])
        tarr.append(t)

        u[0],u[1],u[2],u[3] = T,rho,eps,mu # updating array w/ new values 
        t += step # update sim time
        n += 1 # update iteration number
    
    ### --- plotting --- ###
    parr = np.array(parr)
    dparr = np.array(dparr)
    sarr = np.array(sarr)
    Tarr = np.array(Tarr)
    rhoarr = np.array(rhoarr)
    epsarr = np.array(epsarr)
    muarr = np.array(muarr)
    tarr = np.array(tarr)

    plt.figure(figsize=(8,8))
    plt.loglog(range(n),dparr/parr)
    plt.xlabel("Iteration (n)")
    plt.ylabel("Max Change in Step / Value (dp/p)")
    # plt.savefig("./modules/timestep/n-frac-change.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.loglog(tarr,sarr)
    plt.xlabel("Sim Time")
    plt.ylabel("Step Size")
    # plt.savefig("./modules/timestep/n-frac-change.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(sarr, dparr)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Step Size (s)")
    plt.ylabel("Max. Change per Itereation")
    # plt.savefig("./modules/timestep/step-change.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(tarr / 3.16e13,Tarr)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Sim. TIme (Myr)")
    plt.ylabel("Core Temperature (K)")
    # plt.savefig("./modules/timestep/time-temp.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(tarr / 3.16e13,rhoarr)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Sim. Time (Myr)")
    plt.ylabel("Core Density (g cm^-3)")
    # plt.savefig("./modules/timestep/time-density.png",bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()