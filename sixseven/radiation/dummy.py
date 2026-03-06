

import numpy as np
import os
import argparse
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sixseven.nuclear import nuc_burn


# Opacity functions
def kbf(rho, T, X, Y, Z):
    '''
    Compute bound-free opacity
    
    :param rho: density (g/cm3)
    :param T: temperature (K)
    :param X: canonical composition X
    :param Y: canonical composition Y
    :param Z: canonical composition Z
    '''
    return (4.34e25) * (1 + X) * Z * rho * (T**(-7/2))

def kff(rho, T, X, Y, Z):
    '''
    Compute free-free opacity
    
    :param rho: density (g/cm3)
    :param T: temperature (K)
    :param X: canonical composition X
    :param Y: canonical composition Y
    :param Z: canonical composition Z
    '''
    return (3.68e22) * (1 - Z) * (1 + X) * rho * (T**(-7/2))

def kts(rho, T, X, Y, Z):
    '''
    Compute Thomson opacity
    
    :param rho: density (g/cm3)
    :param T: temperature (K)
    :param X: canonical composition X
    :param Y: canonical composition Y
    :param Z: canonical composition Z
    '''
    return 0.2 * (1 + X)

def khion(rho, T, X, Y, Z):
    '''
    Compute ionized Hydrogen opacity
    
    :param rho: density (g/cm3)
    :param T: temperature (K)
    :param X: canonical composition X
    :param Y: canonical composition Y
    :param Z: canonical composition Z
    '''
    return (1.1e-25) * (Z**(1/2)) * (rho**(1/2)) * (T**(-7/2))


if __name__ == '__main__':

    # let's initialize our canonical metallicitiez
    X = 0.7381
    Y = 0.2485
    Z= 0.0134

    # create our densities and temps
    rhos = np.logspace(-10, 2, 100)
    temps = np.logspace(3.3, 8, 100)

    # create the meshgrid
    rr,tt = np.meshgrid(rhos,temps)

    # now lets compute all the opacities
    bf = kbf(rr,tt,X,Y,Z)
    ff = kff(rr,tt,X,Y,Z)
    ts = kts(rr,tt,X,Y,Z)
    hion = khion(rr,tt,X,Y,Z)

    _ = plt.figure()
    cbar = plt.pcolormesh(rr,tt,bf*rr,norm=LogNorm())
    plt.contourf(rr,tt,bf*rr,norm=LogNorm())
    plt.xlabel('Density (g/cm2)')
    plt.ylabel('Temperature (K)')
    plt.yscale('log')
    plt.xscale('log')
    plt.colorbar(cbar,label=r'Optical Depth, $\tau$')
    plt.show()


    

