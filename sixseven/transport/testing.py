import numpy as np
import matplotlib.pyplot as plt

from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.nuclear.nuc_burn import *
from sixseven.transport.transport_simple import transport_step


def main():

    n_shells = 100
    dt = 1e5  # seconds 

    temps = np.linspace(1.5e7, 4e6, n_shells)
    density = np.linspace(160, 100, n_shells)
    m = np.linspace(0, 1, n_shells)
    
    dM = np.diff(m)[0] 
    
    radius = np.linspace(0, 7e10, n_shells) 

    structure = {
        "m": m,
        "dM": np.ones(n_shells) * dM,          
        "r": radius,                          
        "Hp": np.ones(n_shells) * 1e8,
        "v_mlt": np.ones(n_shells) * 1e2,
        "is_convective": np.ones(n_shells, dtype=bool),
        "grad_rad": np.ones(n_shells) * 0.4,
        "grad_ad": np.ones(n_shells) * 0.3,
        "grad_mu": np.ones(n_shells) * 0.01,
        "K": np.ones(n_shells) * 1e7,
        "Cp": np.ones(n_shells) * 1e8,
        "rho": density,
        "T": temps,
    }
    
    results = burn(temps=temps, rhos=density, time=3e12, comps=None)

    comps = results 
    X_new = transport_step(comps, structure, dt)

    comps = results 
    
    old_H1 = comps[0].composition.getMolarAbundance("H-1")
    
    X_new = transport_step(comps, structure, dt)

    # Compare old vs new
    print(f"Before: {old_H1}")
    print(f"After:  {X_new[0].composition.getMolarAbundance('H-1')}")


main()
