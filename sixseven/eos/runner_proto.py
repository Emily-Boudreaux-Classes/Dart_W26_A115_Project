# -----
# run this file with the command "python -m sixseven.eos.runner_proto" in the project directory
# -----

import sys
import os 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import burn
from sixseven.transport.transport_simple import transport_step

from dataclasses import dataclass 
@dataclass(frozen=True)
class _INIT: 
        """
        Defines constant variables that do not change throughout simulation
        """
        nad: float = nabla_ad(ad_index(CONST.Cp_ideal, CONST.Cv_ideal)) # adiabtic index, constant for ideal gas
        min_temp = 2558585.88691 # K 
INIT = _INIT()
def radius_func(dM): 
       """
       Uses the MS relation R ~ M^(0.8) to retrieve the radius based on the user-inputted mass
       """
       return dM**(0.8)
def temp_func(r):
       """
       Defines how the temperature changes as a function of radius. 
       There is no physicality behind my choice for this function. Feel free to change it 
       """
       return 1e8*r**(-0.05)
def density_func(r): 
       """
       Defines how density changes as a function of radius.
       Again, there is no physicality behind my choice for this function. Feel free to change it 
       """
       return 150*r**(-0.5)
def Hp_func(T, mu): 
        """
        Defines Hp. For more details see the timestep module 
        """
        return (1.36e-16 * T) / (mu * 1.67e-24 * 2.74e4)
def retrieve_comp(results, N): 
        """
        Retrieves the energy, mean molecular weight, and molar abundance from the array outputted by 
        the nuclear module. 
        """
        eps = [np.nan]*N
        mu = [np.nan]*N
        mol_abund = [np.nan]*N
        for i,j in enumerate(results):
            eps[i] = j.energy
            mu[i] = j.composition.getMeanParticleMass()
            mol_abund[i]= j.composition
        eps = np.array(eps)
        mu=np.array(mu)
        mol_abund=np.array(mol_abund)
        return eps, mu, mol_abund
def retrieve_abund(mol_abund, comp_list, comp_names): 
    """
    Retrieves the specific abundances for all elements defined in comp_names. 
    Currently, this code is set to retrieve the abundances for He1, He3, C12, O16, N14, and Mg24.
    """
    comp_list = comp_list
    for i, el in enumerate(comp_list):
        for j, arr in enumerate(mol_abund): 
                if(type(mol_abund[j]) is not float):
                        el[j] = mol_abund[j].getMolarAbundance(comp_names[i])
                else: 
                       el[j] = np.nan
    return comp_list
       
def run_code(N, M, Pcore, tmax, init_step):
       """
       Runs the simulation (does not control outputs or plotting). 
       """
       
       # --- initial conditions --- # 
       dM = np.linspace(1e-5,1,N) * M # mass elements, constant 
       r = radius_func(dM) # radius elements, constant 
       T = temp_func(r) # initial temperature array 
       rho = density_func(r) # initial density array 
       P = np.linspace(Pcore, 1e1, N)

       # --- initializing output arrays --- # 
       temparr = [] # stores temperatures per mass element per iteration
       rhoarr = [] # stores densities per mass element per iteration
       epsarr = []# stores energy generated per mass element per iteration
       muarr = []  # stores mean mol weights per mass element per iteration
       tarr = [] # stores step sizes for each iteration
       nradarr= [] # stores radiative temperature gradient per mass element per iteration 
       comparr = [] # stores arrays for H1, He3, C12, O16, N14, Mg24. Each array stores the composition per mass element per iteration
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
       m = [False if t < INIT.min_temp else True for t in T] # check whether temp is high enough for burning 
       i = 0
       print("Running nuclear burning...")
       while True: 
                try: 
                        results = burn(temps=T[m],rhos=rho[m],time=1.,comps=None)
                        break
                except RuntimeError as e: 
                        i +=1
                        m[-i] = False      
       
       u = np.empty((4,N))
       Nm = len(results)
       m = np.asarray(m) 
       # setting up input for diffusion 
       structure = {"m":dM[m], 
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
       diff_results = transport_step(comps=results,structure=structure,dt=1.) # diffusion 
       eps, mu, mol_abund = retrieve_comp(diff_results, N)
       u[0],u[1],u[2],u[3] = T,rho,eps,mu # initial conditions, after 1 sec
       U = init_U(mu=u[3],dM=dM,T=u[0]) # initial energy 
       dTdP = dT_dP(u[1], dM)
       nrad = nabla_rad(P, u[0], dTdP) # radiative temp gradient 
       is_conv = np.array([True if val > INIT.nad else False for val in nrad]) # checks whether region is convective 
       print("Begining evolution...")
       t = 0 # initial time
       n = 0 # initial step counter
       step = init_step
       while t< tmax:
                if (n % 100) == 1:
                        print("Iteration: ", n)
                        print("log10(Step / s): ", np.log10(step))
                        print("Dens: ", rho)
                        print("Temp: ", T)
                        print("Eps: ", eps)
                        print("Mu: ", mu)
                
                T,rho,eps,mu = u[0],u[1],u[2],u[3] # retrieve conditions from prev step 
                half_step = step / 2 # run this twice over the loop
                for i in range(2):
                        m = [False if (t < INIT.min_temp) or (np.isnan(t)) else True for t in T] # check whether temp is high enough for burning 
                        if(len(T[m]) > 0):
                            while True: 
                                    try: 
                                            results = burn(temps=T[m],rhos=rho[m],time=half_step,comps=mol_abund)
                                            break
                                    except RuntimeError as e: 
                                            m[-i] = False
                                            i +=1
                            results=np.asarray(results)
                            mu_notransport = [np.nan]*N
                            m2 = [True]*len(results)
                            # completes extra check that catches cases where a shell has been burned through
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
                            u = np.empty((4,N))
                            structure = {"m":dM[m], 
                                            "Hp":Hp[m], 
                                            "v_mlt":np.ones(Nm)*1e5, # guess
                                            "is_convective":is_conv[m], 
                                            "grad_rad":nrad[m], # guess
                                            "grad_ad":np.ones(Nm) * INIT.nad, # guess
                                            "grad_mu":np.ones(Nm) * 0.01, # guess
                                            "K":np.ones(Nm) * 1e7, # guess
                                            "Cp":np.ones(Nm)*CONST.Cp_ideal, # guess
                                            "rho":rho[m],
                                            "T":T[m]}
                            diff_results = transport_step(comps=results,structure=structure, dt= half_step)
                            eps, mu, mol_abund= retrieve_comp(diff_results,N)
                        else: 
                               # this happens in the case where ALL temperatures are below where burning occurs 
                               m= np.asarray(m)
                               eps = [np.nan]*N
                               mu=[np.nan]*N
                               mol_abund=[np.nan]*N
                               eps = np.array(eps)
                               mu=np.array(mu)
                               mol_abund=np.array(mol_abund)
                        # update 
                        comp_list = retrieve_abund(mol_abund[m], comp_list, comp_names)
                        U = update_U(U,eps)
                        T = temperature_solver(dM=dM,mu=mu,U=U) 
                        rho = [np.nan]*N
                        rho = np.array(rho)
                        rho[m] = simple_eos(P=P[m],mu=mu[m],T=T[m])
                        du = np.array([T - u[0], rho - u[1], eps - u[2], mu - u[3]])
                        step, p, dp = dyn_timestep(u, du, step, hfactor=1e15, min_step=1e8)

                        # --- appending output arrays --- # 
                        temparr.append(T)
                        rhoarr.append(rho)
                        epsarr.append(eps)
                        muarr.append(mu)
                        tarr.append(t)
                        nradarr.append(nrad)
                        comparr.append(comp_list)
                        
                        # --- updates for next timestep --- # 
                        dTdP= dT_dP(rho, dM)
                        nrad= nabla_rad(P, T, dTdP) 
                        u[0],u[1],u[2],u[3] = T,rho,eps,mu 
                        t += step # update sim time
                        n += 1 # update iteration number
       
       temparr = np.array(temparr)
       rhoarr = np.array(rhoarr)
       epsarr = np.array(epsarr)
       muarr = np.array(muarr)
       tarr = np.array(tarr)
       nradarr=np.array(nradarr)
       comparr=np.array(comparr)
       output_arr = [
              temparr, 
              rhoarr, 
              epsarr, 
              muarr, 
              tarr,
              nradarr, 
              comparr
       ]
       dM = np.asarray(dM)
       r = np.asarray(r)
       const_vals = [dM,r]
       return output_arr, const_vals

def write_outputs(output_arr, const_vals, output_temp, output_rho, output_eps, output_mu, output_t, output_nrad, output_comp, output_var):
       """
       Writes outputs for all variables that are set to True 
       For example, if output_t is set to True, the tarr.txt file will be written. 
       All variables will be outputted into .txt files except for the elemental composition of various elements 
       which will be outputted in a .npz file. 
       """
       file_path = './output/models'
       file_name = [
                'temparr', 
                'rhoarr', 
                'epsarr', 
                'muarr', 
                'tarr', 
                'nradarr',
                'comparr'
        ]
       comments = [
              '## Each column represents the temperature (K) at each mass/radius element along the star. Each row represents a different time step/iteration \n', 
              '## Each column represents the density (g/cm^3) at each mass/radius element along the star. Each row represents a different time step/iteration \n', 
              '## Each column represents the specific internal energy (erg) at each mass/radius element along the star. Each row represents a different time step/iteration \n',
              '## Each column represents the mean molecular weight (g/mol) at each mass/radius element along the star. Each row represents a different time step/iteratio \n', 
              '## This single row array represents the times (s) the simulation iterates across \n', 
              '## Each column represents the radiative temperature gradient (unitless) at each mass/radius element along the star. Each row represents a different time step/iteration \n', 
              '## You must select the element you wish to access after loading the .npz file in. For each element, the columns represent the composition at each mass/radius element along the star. Each row represents a different time step/iteration \n'
       ]
       output_bool = [output_temp, output_rho,output_eps, output_mu,  output_t, output_nrad, output_comp]
       if(output_var == "r"): 
              head = [f"{num:.2e}" for num in const_vals[1]]
       elif(output_var == "M"): 
              head = [f"{num:.2e}" for num in const_vals[0]]
       else: 
              N = np.linspace(0, len(output_arr[0][0]), len(output_arr[0][0]))
              head = [f"{int(num)}" for num in N]
       header=""
       for val in head: 
              header = header + "Element" + str(val) + ","
       header = header[:-1]
       for i,f in enumerate(file_name[:-1]):
              if(output_bool[i]): 
                     path = os.path.join(file_path, f"{file_name[i]}.txt") 
                     np.savetxt(
                            fname=path,
                            X=output_arr[i],
                            header=header,
                            comments = comments[i], 
                            delimiter=','
                     )
                     print(f"Array successfully saved to {path}")
       if(output_bool[-1]): 
               path = os.path.join(file_path, f"{file_name[-1]}") 
               comps = output_arr[-1]
               Harr =[np.nan]*len(N)
               Hearr = [np.nan]*len(N)
               Carr=[np.nan]*len(N)
               Oarr=[np.nan]*len(N)
               Narr=[np.nan]*len(N)
               Mgarr=[np.nan]*len(N)
               for i in range(len(comps)): 
                      Harr = np.vstack((Harr, comps[i][0]))
                      Hearr=np.vstack((Hearr, comps[i][1]))
                      Carr=np.vstack((Carr, comps[i][2]))
                      Oarr=np.vstack((Oarr, comps[i][3]))
                      Narr=np.vstack((Narr, comps[i][4]))
                      Mgarr=np.vstack((Mgarr, comps[i][5]))
               np.savez(path, H1 = Harr[1:], He3= Hearr[1:], C12 = Carr[1:], O16 = Oarr[1:], N14=Narr[1:], Mg24=Mgarr[1:], metadata = comments[-1], headers = header)
               print(f"Array successfully saved to {path}")
       if(output_bool[4]): 
              path = os.path.join(file_path, f"{file_name[4]}.txt") 
              np.savetxt(
                            fname=path,
                            X=output_arr[4],
                            header="Time (s)",
                            comments = comments[4], 
                            delimiter=','
                     )
              print(f"Array successfully saved to {path}")
def set_up_plotting(config_font, darktheme): 
       """
       Purely aesthetic.
       If config_font = True, the font will be changed from the classic python font to the LaTeX font. 
       If darktheme= True, plots will be returned with a black background and white text. Otherwise, the inverse is True. 
       """
       if(config_font):
                os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH'] 
                plt.rcParams.update({
                        'mathtext.fontset': 'stix',
                        'font.family': 'serif',
                        'font.size': 11
                })
       if(darktheme): 
              palette = [
                "#4FC3F7",  
                "#FF8A65",  
                "#81C784",  
                "#FFD54F",  
                "#BA68C8",  
                "#4DB6AC",  
                "#FFB74D",  
                "#F06292",  
                "#AED581",  
                "#90CAF9"   
                ]
              plt.rcParams.update({
                "axes.prop_cycle": cycler(color=palette),
                "figure.facecolor": "black",
                "axes.facecolor": "black",
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "text.color": "white",
                })
       else: 
              palette= [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf"
                ]
              plt.rcParams.update({
                "axes.prop_cycle": cycler(color=palette),
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "text.color": "black",
                })
def energy_temp_plot(temp, energy, time, normalization, savefig):
       """
       Plots energy versus temperature for all times outputted by the simulation. 
       If normalization = True, energy and time will be normalized. 
       If savefig = True, the figure will be saved. Otherwise, the figure will be shown. 
       """
       fig, ax = plt.subplots(1,1, figsize = (5,5))
       if(normalization): 
              for i in range(len(temp)): 
                ax.plot(temp[i]/np.nanmax(temp[i]), energy[i]/np.nanmax(energy[i]), label = f"t={time[i]:.1e}")
                ax.set_xlabel('Temperature (K)')
                ax.set_ylabel('Energy (erg)')
                ax.legend()
                plt.tight_layout()
       else:
                for i in range(len(temp)): 
                        ax.plot(temp[i], energy[i], label = f"t={time[i]:.1e}")
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.set_xlabel('Temperature (K)')
                        ax.set_ylabel('Energy (erg)')
                        ax.legend()
                        plt.tight_layout()
       if(savefig):
              plt.savefig('./output/plots/energy_temp.png', dpi=300)
       else: 
              plt.show()
def temp_gradients_plot(radius, nrad, time, savefig=True):
       """
       Plots the radiative temperature gradient over the radius for all times outputted by the simulation. 
       If savefig = True, the figure will be saved. Otherwise, the figure will be shown. 
       """
       fig, ax = plt.subplots(1,1, figsize = (5,5))
       for i in range(len(nrad)): 
                ax.plot(radius, nrad[i], label = f"t={time[i]:.1e}")
       ax.set_xlabel('Radius (cm)')
       ax.set_ylabel('Temperature gradients')
       ax.set_xscale('log')
       ax.set_yscale('log')  
       ax.axhline(y = INIT.nad, linestyle = '--', color = 'red')
       ax.text(0.05, 0.85, r'$\nabla_{rad}$', transform=ax.transAxes, 
                fontsize=16, verticalalignment='top', horizontalalignment='left')
       ax.text(radius[0], 0.35, r'$\nabla_{ad}$', 
                fontsize=16, verticalalignment='top', horizontalalignment='left', color = 'red')
       ax.set_ylim(0.2, np.max(nrad[0])*2)
       ax.legend()
       plt.tight_layout()
       if(savefig): 
              plt.savefig('./output/plots/temp_gradients.png', dpi=300)
       else: 
              plt.show()
def run(N, M=1e32, Pcore = 1e17, tmax=1e16, init_step=1e14, output_temp = True, output_rho = True, output_eps = True, output_mu = True, output_t = True, output_nrad = True, output_comp=True, output_var = None, run_sim = True, make_plots = True, config_font=True, darktheme = False, savefigs= False): 
      """
      Controls the inputs of all other functions defined in this file. 
      Runs the simulation with run_sim
      Makes plots with make_plots
      Control what variables are outputted into arrays with output_rho = True, output_eps=True, etc. 
      """
      if(run_sim):
             outputs, const = run_code(N, M, Pcore, tmax, init_step) 
             write_outputs(outputs, const, output_temp = output_temp, output_rho = output_rho, output_eps = output_eps, output_mu = output_mu, output_t = output_t, output_nrad = output_nrad, output_comp=output_comp, output_var = output_var)
      if(make_plots): 
             set_up_plotting(config_font=config_font, darktheme = darktheme)
             try: 
                    energy= np.genfromtxt('output/models/epsarr.txt', comments='##', delimiter=',', dtype= None, skip_header =2)
                    temp= np.genfromtxt('output/models/temparr.txt', comments='##', delimiter=',', dtype= None, skip_header =2)
                    time= np.genfromtxt('output/models/tarr.txt', comments='##', delimiter=',', dtype= None, skip_header =2)
                    energy_temp_plot(temp, energy, time, normalization=False, savefig=savefigs)
             except Exception as e: 
                    print("Energy, temperature, and time must be saved as .txt files in order to make this plot.")
             try: 
                    nrad=np.genfromtxt('output/models/nradarr.txt', comments='##', delimiter=',', dtype= None, skip_header =2)
                    if(run_sim): 
                           radius = const[1]
                    else: 
                        dM = np.linspace(1e-5,1,N) * M
                        radius = dM**(0.8)   
                    temp_gradients_plot(radius, nrad, time, savefig=savefigs)
             except Exception as e: 
                    print("The radiative temperature gradient must be saved as a .txt file to make this plot.")