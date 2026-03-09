import nuc_burn
import numpy as np

T = np.linspace(1.5e7, 2e7, 100)
Rho = np.linspace(1.5e2, 1.5e2, 100) 
Time=1000
results = nuc_burn.burn(T, Rho, Time)

epsilon = []
mu = []
composition = [] # Composition object per shell

for i,j in enumerate(results):
    epsilon.append(j.energy)
    mu.append(j.composition.getMeanParticleMass()) 
    composition.append(j.composition) 

print(results[0].composition.getMolarAbundance("H-1"))

#results2 = nuc_burn.burn(T, Rho, Time, composition)

