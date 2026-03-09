import numpy as np
import matplotlib.pyplot as plt

from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import *

# run with python -m sixseven.eos.testing 

N=10
r = np.logspace(1,15, N)
temps = 1.5E8 * r**(-0.01)
rho = 150*r**(-0.01)
results = burn([1.5e8], [150], N**2, comp=None)
epsilon= []
mu= []
mass_frac = results[0].composition
for i,j in enumerate(results):
        epsilon.append(j.energy)
        mu.append(j.composition.getMeanParticleMass())
dM = np.linspace(1e32, 1e30, N)
P = np.linspace(1e17, 1e15, N)
dTdP= dT_dP(rho, dM)
nrad = nabla_rad(P, temps, dTdP)
gamma = ad_index(CONST.Cp_ideal, CONST.Cv_ideal)
nad = nabla_ad(gamma)
fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(r, nrad, color = 'k')
ax.axhline(y=nad, linestyle='--', color = 'r')
ax.set_xlabel('Radius')
ax.set_ylabel('Temperature gradients')
ax.set_xscale('log')
ax.set_yscale('log')
ax.text(5,0.43, r'$\nabla_{ad}$',color = 'red', fontsize=12)
ax.text(60, 8, r'$\nabla_{rad}$', color = 'k', fontsize=12)
plt.savefig('output/plots/temp_gradients.png', dpi=300)