# -----
# run this file with the command "python -m sixseven.timestep.testing" in the project directory
# -----

import sys
import os 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import burn
from sixseven.transport.transport_simple import transport_step

from dataclasses import dataclass 
@dataclass(frozen=True)
class _INIT: 
        nad: float = nabla_ad(ad_index(CONST.Cp_ideal, CONST.Cv_ideal)) # adiabtic index, constant for ideal gas
        step: float =  1e14 # initial step 
        max_t: float = 1e16 # total time
        N: int = 5 
        rmax: int = 12 
        r = np.logspace(1,rmax, N)
        dM = np.linspace(1e-5,1,N) * 1e32
        P = np.linspace(1e17, 1e1, N)
        min_temp = 2558585.88691 # K 
        H1 = [np.nan]*N
        He3 = [np.nan]*N
        C12 = [np.nan]*N
        O16 = [np.nan]*N
        N14 = [np.nan]*N 
        Mg24 = [np.nan]*N
        comp_list = [
                     H1, 
                     He3, 
                     C12, 
                     O16, 
                     N14, 
                     Mg24, 
                     ]
        comp_names = [
                "H-1", 
                "He-3", 
                "C-12", 
                "O-16", 
                "N-14", 
                "Mg-24", 
                ]
INIT = _INIT()
def temp_func(r):
       return 1e7*r**(-0.05)
def T_to_r(T): 
        return (T/1e7)**(-1/0.05)
def density_func(r): 
       return 150*r**(-0.5)
def Hp_func(T, mu): 
        return (1.36e-16 * T) / (mu * 1.67e-24 * 2.74e4)
def retrieve_comp(results): 
        eps = [np.nan]*INIT.N
        mu = [np.nan]*INIT.N
        mol_abund = [np.nan]*INIT.N
        for i,j in enumerate(results):
            eps[i] = j.energy
            mu[i] = j.composition.getMeanParticleMass()
            mol_abund[i]= j.composition
        eps = np.array(eps)
        mu=np.array(mu)
        mol_abund=np.array(mol_abund)
        return eps, mu, mol_abund
def tempgrad_plot(nrad):
       r = INIT.r
       nad = INIT.nad
       fig, ax = plt.subplots(1,1,figsize=(8,6)) 
       ax.plot(r, nrad, color = 'k', label = r'$\nabla_{rad}$')
       ax.axhline(y = nad, color = 'r', linestyle = '--', label = r'$\nabla_{ad}$')
       ax.legend()
       ax.set_xscale('log')
       ax.set_yscale('log')
       ax.set_xlabel('Radius (cm)')
       ax.set_ylabel('Temperature gradients (dimensionless)')
       secx_x = ax.secondary_xaxis("top", functions=(temp_func, T_to_r))
       secx_x.set_xlabel("Temperature (K)")
       ax.axvspan(xmin = T_to_r(2558585.88691), xmax = np.max(r)*5, color = 'gray', alpha = 0.3)
       x_coord = ((T_to_r(2558585.88691) +  np.max(r)*3.8)/2)/(np.max(r)*2.3)
       y_coord = np.log10(np.mean(np.arange(np.min(nrad), np.max(nrad))))
       ax.text(x_coord,                 
               0.5,                
               "No burning occurs",  
               transform=ax.transAxes,   
                 ha='center',       
                 va='center',       
                 fontsize=12,
                 rotation = 90,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
       ax.set_xlim(np.min(r), np.max(r)*5)
       plt.show()
       plt.close()
def retrieve_abund(mol_abund): 
    comp_list = INIT.comp_list
    for i, el in enumerate(comp_list):
        for j, arr in enumerate(mol_abund): 
                el[j] = mol_abund[j].getMolarAbundance(INIT.comp_names[i])
    return comp_list
       
def plot_composition(mol_abund): 
        comp_list = retrieve_abund(mol_abund)
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        for el, name in zip(comp_list, INIT.comp_names): 
                ax.plot(INIT.r,el,label = name)
        ax.axvspan(xmin = T_to_r(INIT.min_temp), xmax = np.max(INIT.r)*5, color = 'gray', alpha = 0.3)
        x_coord = ((T_to_r(INIT.min_temp) +  np.max(INIT.r)*3.8)/2)/(np.max(INIT.r)*2.3)
        ax.text(x_coord,                 
               0.5,                
               "No burning occurs",  
               transform=ax.transAxes,   
                 ha='center',       
                 va='center',       
                 fontsize=12,
                 rotation = 90,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Radius (cm)')
        ax.set_ylabel("Molar abundance")
        ax.legend()
        ax.set_xlim(np.min(INIT.r), np.max(INIT.r)*5)
        plt.show()
        plt.close()
def main():
        print("Working ... ")
        
        # --- outputs --- # 
        
        parr = [] # stores values most changed of any variable at each iteration
        dparr = [] # stores largest change of any variable at each iteration
        temparr = [] # stores temperatures per mass element per iteration
        rhoarr = [] # stores densities per mass element per iteration
        epsarr = [] # stores energy generated per mass element per iteration
        muarr = [] # stores mean mol weights per mass element per iteration
        sarr = [] # stores step sizes for each iteraetion
        tarr = [] # simulation time at each step (seconds)
        nradarr= []
        comparr = []
        eps = []
        mu = []
        mol_abund = []
        T = temp_func(INIT.r)
        rho = density_func(INIT.r)
        m = [False if t < INIT.min_temp else True for t in T]
        i = 0
        print("Running nuclear burning...")
        while True: 
                try: 
                        results = burn(temps=T[m],rhos=rho[m],time=1.,comps=None)
                        break
                except RuntimeError as e: 
                        i +=1
                        m[-i] = False      
        u = np.empty((4,INIT.N))
        Nm = len(results)
        m= np.asarray(m)
        structure = {"m":INIT.dM[m], 
                        "Hp":np.ones(Nm) * 1e9, 
                        "v_mlt":np.ones(Nm)*1e5, # guess
                        "is_convective":np.full(Nm, False, dtype=bool), 
                        "grad_rad":np.ones(Nm) * 0.4, # guess
                        "grad_ad":np.ones(Nm) * 0.3, # guess
                        "grad_mu":np.ones(Nm) * 0.01, # guess
                        "K":np.ones(Nm) * 1e7, # guess
                        "Cp":np.ones(Nm) * CONST.Cp_ideal, # guess
                        "rho":rho[m],
                        "T":T[m]}
        diff_results = transport_step(comps=results,structure=structure,dt=1.)
        eps, mu, mol_abund = retrieve_comp(diff_results)
        u[0],u[1],u[2],u[3] = T,rho,eps,mu # initial conditions, after 1 sec
        U = init_U(mu=u[3],dM=INIT.dM,T=u[0]) # initial energy 
        dTdP = dT_dP(u[1], INIT.dM)
        nrad = nabla_rad(INIT.P, u[0], dTdP)
        is_conv = np.array([True if val > INIT.nad else False for val in nrad])
        #tempgrad_plot(nrad)
        #plot_composition(mol_abund[m])
        print("Begining evolution...")
        t = 0 # initial time
        n = 0 # initial step counter
        step = INIT.step
        while n < 10: 
                print(step)
                if (n % 100) == 1:
                        print("Iteration: ", n)
                        print("log10(Step / s): ", np.log10(step))
                        print("Dens: ", rho)
                        print("Temp: ", T)
                        print("Eps: ", eps)
                        print("Mu: ", mu)
                T,rho,eps,mu = u[0],u[1],u[2],u[3]
                half_step = step / 2 # run this twice over the loop
                for i in range(2):
                        m = [False if (t < INIT.min_temp) or (np.isnan(t)) else True for t in T]
                        i = 1
                        if(len(T[m]) > 0):
                            while True: 
                                    try: 
                                            results = burn(temps=T[m],rhos=rho[m],time=half_step,comps=None)
                                            break
                                    except RuntimeError as e: 
                                            m[-i] = False
                                            i +=1
                            results=np.asarray(results)
                            mu_notransport = [np.nan]*INIT.N 
                            m2 = [True]*len(results)
                            for i,j in enumerate(m): 
                                   if(j):
                                        mu_notransport[i] = results[i].composition.getMeanParticleMass()
                                        if(np.isnan(mu_notransport[i])): 
                                                m2[i] = False 
                                                m[i] = False   
                            m= np.asarray(m)
                            m2 = np.asarray(m2)
                            results=results[m2]
                            Nm = len(results)
                            Hp = Hp_func(T, mu)
                            u = np.empty((4,INIT.N))
                            structure = {"m":INIT.dM[m], 
                                            "Hp":Hp[m], 
                                            "v_mlt":np.ones(Nm)*1e5, # guess
                                            "is_convective":is_conv[m], 
                                            "grad_rad":nrad[m], # guess
                                            "grad_ad":np.ones(Nm) * INIT.nad, # guess
                                            "grad_mu":np.ones(Nm) * 0.01, # guess
                                            "K":np.ones(Nm) * 1e7, # guess
                                            "Cp":np.ones(Nm) * 1e8, # guess
                                            "rho":rho[m],
                                            "T":T[m]}
                            diff_results = transport_step(comps=results,structure=structure, dt= half_step)
                            eps, mu, mol_abund= retrieve_comp(diff_results)
    
                        else: 
                               m= np.asarray(m)
                               eps = [np.nan]*INIT.N
                               mu=[np.nan]*INIT.N
                               mol_abund=[np.nan]*INIT.N
                               eps = np.array(eps)
                               mu=np.array(mu)
                               mol_abund=np.array(mol_abund)
                        comp_list = retrieve_abund(mol_abund[m])
                        U = update_U(U,eps)
                        T = temperature_solver(dM=INIT.dM,mu=mu,U=U) 
                        print(T)
                        rho = [np.nan]*INIT.N
                        rho = np.array(rho)
                        rho[m] = simple_eos(P=INIT.P[m],mu=mu[m],T=T[m])
                        du = np.array([T - u[0], rho - u[1], eps - u[2], mu - u[3]])
                        step, p, dp = dyn_timestep(u, du, step, hfactor=1e15, min_step=1e8)
                        sarr.append(step)
                        parr.append(p) 
                        dparr.append(dp)
                        temparr.append(T)
                        rhoarr.append(rho)
                        epsarr.append(eps)
                        muarr.append(mu)
                        tarr.append(t)
                        nradarr.append(nrad)
                        comparr.append(comp_list)
                        dTdP= dT_dP(rho, INIT.dM)
                        nrad= nabla_rad(INIT.P, T, dTdP) 
                        u[0],u[1],u[2],u[3] = T,rho,eps,mu 
                        t += step # update sim time
                        n += 1 # update iteration number
        parr = np.array(parr)
        dparr = np.array(dparr)
        sarr = np.array(sarr)
        temparr = np.array(temparr)
        rhoarr = np.array(rhoarr)
        epsarr = np.array(epsarr)
        muarr = np.array(muarr)
        tarr = np.array(tarr)
        nradarr=np.array(nradarr)
        comparr=np.array(comparr)
        output_arr = [sarr, 
                        parr, 
                        dparr, 
                        temparr, 
                        rhoarr,
                        epsarr, 
                        muarr, 
                        tarr, 
                        nradarr, 
                        comp_list
                    ]
        file_path = './run/'
        file_name = [
               'sarr', 
                'parr', 
                'dparr', 
                'temparr', 
                'rhoarr', 
                'epsarr', 
                'muarr', 
                'tarr' 
                'nradarr',
                'comparr'
        ]
        for i, name in enumerate(file_name): 
            file = os.path.join(file_path, f"{name}.pkl")
            with open(file, 'wb') as f: 
                   print(file_name[i])
                   print(output_arr[i])
                   pickle.dump(output_arr[i], f)
if __name__ == "__main__":
        main()