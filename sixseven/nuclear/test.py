from sixseven.nuclear import nuc_burn
import numpy as np

T = np.linspace(1.5e7, 2e7, 100)
Rho = np.linspace(1.5e2, 1.5e2, 100) 
Time=1000
results = nuc_burn.burn(T, Rho, Time)

print(results)
