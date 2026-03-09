'''
Call the main radiation module

Author: Guinevere Herron
'''

import numpy as np
import os
import argparse
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from sixseven.nuclear import nuc_burn
from scipy.integrate import quad


# Global Variables
REPO_DIR = str(Path(__file__).resolve().parent.parent.parent)
CONSTANTS = {'G': 6.674e-8}   

# Opacities
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


def kramer_opacity(rho, T, X, Y, Z):
    '''
    Implementation of the Kramer opacity law. 
    
    :param rho: density (units: grams cm^-3)
    :param T: temperature (units: Kelvin)

    '''

    # compute the three opacities
    bf = kbf(rho,T,X,Y,Z)
    ff = kff(rho,T,X,Y,Z)
    ts = kts(rho,T,X,Y,Z)
    hion = khion(rho,T,X,Y,Z)

    # sum all opacity sources together and average
    avg = (bf + ff + ts + hion)


    return avg

def plot_kramer_sun(filename,delRho=100,delT=100, **kwargs):
    '''
    Plot the Kramer opacity for the Sun at various densities and temperatures
    
    :param filename: filename for the plot
    :param delRho: step size for densitiy
    :param delT: step size for temperature
    '''

    # let's go ahead and run tests for the sun
    # all of this info is coming from wikipedia
    rhos = np.logspace(-10, 3, delRho)
    temps = np.logspace(3.3, 8, delT)
    X = 0.7381
    Y = 0.2485
    Z= 0.0134

    # make a mesh grid
    xx, yy = np.meshgrid(temps,rhos)
    
    _ = plt.figure(figsize=(8,7),dpi=500)

    taus = kramer_opacity(rho=yy, T=xx, X=X, Y=Y, Z=Z)

    plt.pcolormesh(yy, xx, taus*yy, cmap='plasma',norm=LogNorm(vmax=100))
    
    plt.xscale('log')
    plt.yscale('log')

    plt.ylabel('Temperature (K)')
    plt.xlabel('Density (g cm$^{-3}$)')
    plt.colorbar(label='Opacity, $\\tau$')

    plt.savefig(filename)

    return

def solar_density(r)
    '''
    compute density for sun at some radius
    r: float, radius
    ''' 
    return (519 * r**4) + (1630 * r**3) + (1844 * r**2) + (889 * r) + 155

def boundary_conditions(R, T, L, X, P, M):
    '''
    compute the next pressure fit and temperature fit for our boundary conditions
    
    :param R: array-like, radii at different dm
    :param T: array-like, temperatures at different dm
    :param L: array-like, luminosities at different dm
    :param X: gridfire.Composition, compositions at different dm
    :param P: array-like, pressures at different dm
    :param M: array-like, array of masses
    '''

    # calculate g and Teff

    g = ( CONSTANTS['G'] * M**2 ) /  R**2

    Teff = (( L ) / (4 * np.pi * CONSTANTS['stefbolt'] * R**2) )**(1/4)
    
    # here we will get our taus
    X,Y,Z = nuc_burn.getCanonicalComposition(massFrac=X)

    kmean = kramer_opacity(rho,T,X,Y,Z)
    taus = kmean * rho

    sel = taus >= 2/3

    def integrand(g,P,T):
        '''
        integrand for the pressure fit

        g: any, surface gravity
        P: any, pressure
        T: any, temperature
        '''
        return ( g / kramer_opacity(P,T) )
    
    Pfit = quad(integrand(g,P,T), 0, kramer_opacity(P,T)*rhos)

    Tfit = Teff * ( (3/4)*taus[sel])
    

    return Pfit, Tfit


def boundaries_sun(filename):
    '''
    test boundary conditions for a Sun-like star

    filename: string, where to output plots

    '''

    # let's initialize all of our solar variables
    X = 0.7381
    Y = 0.2485
    Z = 0.0134
    R = 6.957e10
    L = 3.828e33
    M = 1.988e33

    rho = solar_density(R)
    P, T = boundary_conditions(R, T, L, [X, Y,Z], M)
if __name__ == '__main__':

    # here we will create our argument parser
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument('inputs', help='input parameters from other modules')
    parser.add_argument('-v','--verbose',action='store_true',
                       help='output verbosity')
    parser.add_argument('-f','--force',action='store_true',
                       help='force overwrite')
    parser.add_argument('-S', '--solar', action='store_true',
                        help = 'Run radiation module for solar inputs')

    # parse arguments
    args = parser.parse_args()

    #lets set the logging level
    #level = logging.DEBUG if args.verbose else logging.INFO
    #logging.getLogger().setLevel(level)

    if args.solar:
        plot_kramer_sun(REPO_DIR+'/output/plots/opacity_sun.png')

    