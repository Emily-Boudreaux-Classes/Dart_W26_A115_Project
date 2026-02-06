# -----
# run this file with the command "python -m modules.timestep.testing" in the project directory
# 
# -----
import numpy as np
import matplotlib.pyplot as plt

from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import *

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

    step = 1e15 # initial step 
    max_t = 1e17 # total time

    # single mass element dM - representative of the core of the star
    # this N can be changed to track more dMs, doing one for now
    N = 1
    u = np.empty((4,N)) # array of initial T, rho, eps, mu for each dM - shape: (4,N)
    T = np.array([1e7]) # burn() requires array inputs, this is the quick and dirty fix, will not be an issue later because each dM will have an associate initial T stored in an array
    rho = np.array([1e2])
    eps,mu,mass_frac = burn(temp=T,rho=rho,time=1,comp=None)
    u[0],u[1],u[2],u[3] = T,rho,eps,mu # initial conditions, after 1 sec

    P = 1e17 # fixed value for now, roughly that of slar core
    dM = 1e32 # grams  - 1e32 is sun core mass, mass of the one element 
    U = init_U(mu=u[3],dM=dM,T=u[0])

    print(" ----- Initial Values ----- ")
    print("Initial Internal Energy:", U)
    print("Mass: ", dM)
    print("Pressure: ", P)
    print("Temp: ", u[0])
    print("Dens: ", u[1])
    print("Eps: ", u[2])
    print("Mu: ", u[3])
    print(" ----- ")

    t = 0 # initial time
    n = 0 # initial step counter

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

        eps, mu, mass_frac = burn(temp=T,rho=rho,time=step,comp=mass_frac) # sasha function
        eps = np.asarray(eps) # setting the lists as arrays
        mu = np.asarray(mu)

        U = update_U(U,eps) # cassie - updates internal energy 
        T = temperature_solver(dM=dM,mu=mu,U=U) # cassie - solves temperature
        rho = simple_eos(P=P,mu=mu,T=T) # cassie - gets dens from ideal gas eos

        du = np.array([T - u[0], rho - u[1], eps - u[2], mu - u[3]]) # change between steps
        step, p, dp = dyn_timestep(u, du, step, hfactor=1e13, min_step=1e8) # calc timestep, p, dp

        sarr.append(step)
        parr.append(p) 
        dparr.append(dp)
        Tarr.append(T[0]) # take single values for now, doesn't matter when N = 1
        rhoarr.append(rho[0])
        epsarr.append(eps[0])
        muarr.append(mu[0])
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
    plt.savefig("./modules/timestep/n-frac-change.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(sarr, dparr)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Step Size (s)")
    plt.ylabel("Max. Change per Itereation")
    plt.savefig("./modules/timestep/step-change.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(tarr / 3.16e13,Tarr)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Sim. TIme (Myr)")
    plt.ylabel("Core Temperature (K)")
    plt.savefig("./modules/timestep/time-temp.png",bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8,8))
    plt.scatter(tarr / 3.16e13,rhoarr)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Sim. Time (Myr)")
    plt.ylabel("Core Density (g cm^-3)")
    plt.savefig("./modules/timestep/time-density.png",bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()