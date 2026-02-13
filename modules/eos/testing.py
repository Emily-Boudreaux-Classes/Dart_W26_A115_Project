from modules.nuclear import nuc_burn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import os 
from modules.timestep.timestep import dyn_timestep
from modules.eos.eos_functions import *
from modules.nuclear.nuc_burn import *

def main():
    S_arr = [] # entropy at each timestep 
    t_arr = []
    u_arr = []
    T_arr = []
    rho_arr = []
    nad_grid=[]
    nrad_grid = []
    del_nrad = []
    step = 1e15 # initial step 
    max_t = 1e16 # total time
    N = 10 # number of mass elements to track
    u = np.empty((4,N))
    temps = np.linspace(1.5e7, 2e7, N)
    rhos = np.linspace(1.5e2, 1.e2, N)
    results = nuc_burn.burn(temps, rhos, N**2, comp=None)
    epsilon= []
    mu= []
    mass_frac = results[0].composition
    for i,j in enumerate(results):
            epsilon.append(j.energy)
            mu.append(j.composition.getMeanParticleMass())
    u[0],u[1],u[2],u[3] = temps,rhos,epsilon,mu # initial conditions of each mass element after 1 sec 
    P = np.linspace(1e17, 1e15, N) #1e17 is roughly stellar core pressure 
    #Press, Temp = np.meshgrid(P, temps)
    #rho_grid = simple_eos(Press, mu, Temp)
    dM = np.linspace(1e32, 1e30, N) # grams  - 1e32 is sun core mass, mass of the one element
    U = init_U(mu=u[3],dM=dM,T=u[0]) # initial internal energy of each mass element after 1 sec
    Rho, Temp = np.meshgrid(rhos, temps, indexing='xy')
    dTdP= dT_dP(Rho, dM)
    nrad = nabla_rad(P, Temp, dTdP)
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
    while t < max_t:
        if (n % 100) == 1:
            print("Iteration:", n)
            print("log10(Step / s): ", np.log10(step))
        T,rho,eps,mu = u[0],u[1],u[2],u[3]
        U = update_U(U,eps)
        T = temperature_solver(dM=dM,mu=mu,U=U)
        rho = simple_eos(P=P,mu=mu,T=T)
        results = burn(temp=T,rho=rho,time=step,comp=mass_frac)
        mass_frac = results[0].composition
        for i,j in enumerate(results):
            eps[i]= j.energy
            mu[i]= j.composition.getMeanParticleMass()
        S= entropy(T, U, P, 0)
        S_arr.append(S)
        t_arr.append(t)
        u_arr.append(U)
        T_arr.append(T)
        rho_arr.append(rho)
        Rho, Temp =np.meshgrid(rho, T,indexing='xy')
        init_nrad = nrad 
        dTdP=dT_dP(Rho, dM)
        nrad = nabla_rad(P, Temp, dTdP)
        nrad_grid.append(nrad)
        del_nrad.append(nrad-init_nrad)
        du = np.array([T - u[0], rho - u[1], eps - u[2], mu - u[3]])
        step, p, dp = dyn_timestep(u, du, step, hfactor=1e13, min_step=1e8)
        u[0],u[1],u[2],u[3] = T,rho,eps,mu
        t += step
        n += 1 
    ## --plotting -- ##
    gamma = ad_index(CONST.Cp_ideal, CONST.Cv_ideal)
    nad = nabla_ad(gamma)
    idx = 5
    T= T_arr[idx]
    rho = rho_arr[idx]
    nrad = nrad_grid[idx]
    conv_grid = nrad- nad 
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    im = ax.pcolormesh(T/np.max(T), rho/np.max(rho), conv_grid.T/np.max(conv_grid))
    plt.colorbar(im, ax=ax, label = r'$\nabla_{rad}$ - $\nabla_{ad}$ (normalized)')
    ax.set_xlabel('Temperature (normalized)')
    ax.set_ylabel(r'$\rho$ (normalized)')
    plt.title(f"t={t_arr[idx]:.2e}")
    plt.savefig('output/plots/schwarzschild_criterion.png', dpi = 300)
    file = np.linspace(0, len(t_arr)-1, len(t_arr))
    file = [int(f) for f in file]
    gamma = ad_index(CONST.Cp_ideal, CONST.Cv_ideal)
    nad = nabla_ad(gamma)
    for i in range(len(t_arr)): 
        idx = i
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        T = T_arr[idx]
        rho = rho_arr[idx]
        im = ax.pcolormesh(T/np.max(T), rho/np.max(rho), del_nrad[idx].T, vmin = -0.1, vmax=0)
        plt.colorbar(im, ax=ax, label = r'Change in $\nabla_{rad}$')
        ax.set_xlabel('Temperature (normalized)')
        ax.set_ylabel(r'$\rho$ (normalized)')
        plt.suptitle(f"t={t_arr[idx]:.2e}")
        t=t_arr[idx]
        plt.savefig(os.path.join('output/plots', 'frame%02d.png' % (file[i]))) 

if __name__ == "__main__":
    main()