import nuc_burn
import numpy as np

T=np.linspace(1.5e7,5.6e3,100)
Rho=np.linspace(150.0,1.0e-7,100)
Time=1000
e, mu = nuc_burn.burn(T, Rho, Time)

print(e)
print(mu)