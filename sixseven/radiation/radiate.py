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
from pathlib import Path
from sixseven.nuclear import nuc_burn


# Global Variables
REPO_DIR = str(Path(__file__).resolve().parent.parent.parent)
CONSTANTS = {'G': 6.674e-8}   

# opacities
def kbf(rho, T, X, Y, Z):
    # Bound-free (Kramers): assumes ionized gas, invalid below ~10^4 K
    if T < 1e4:
        return 0.0
    return (4.34e23) * (1 + X) * Z * rho * (T**(-7/2))
def kff(rho, T, X, Y, Z):
    # Free-free (Kramers): assumes ionized gas, invalid below ~10^4 K
    if T < 1e4:
        return 0.0
    return (3.68e22) * (1 - Z) * (1 + X) * rho * (T**(-7/2))
def kts(rho, T, X, Y, Z):
    # Electron scattering (Thomson)
    return 0.2 * (1 + X)
def khion(rho, T, X, Y, Z):
    # H⁻ bound-free opacity: dominant in cool stellar envelopes (T ~ 3000-10000 K)
    # Fitting formula from Hansen, Kawaler & Trimble (2004): κ ∝ Z ρ^{1/2} T^9
    # H⁻ is destroyed by ionization above ~1.6e4 K
    if np.ndim(T) == 0:
        if T > 1.6e4:
            return 0.0
        return 2.5e-31 * (Z / 0.02) * rho**0.5 * T**9
    # array path
    T_arr = np.asarray(T, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    result = np.where(T_arr < 1.6e4,
                      2.5e-31 * (Z / 0.02) * rho_arr**0.5 * T_arr**9,
                      0.0)
    return result


def kramer_opacity(rho, T, X, Y, Z):
    '''
    Implementation of the Kramer opacity law. 
    
    :param rho: density (units: grams cm^-3)
    :param T: temperature (units: Kelvin)

    '''

    # compute all opacity sources and sum (Rosseland mean approximation)
    bf   = kbf(rho, T, X, Y, Z)
    ff   = kff(rho, T, X, Y, Z)
    ts   = kts(rho, T, X, Y, Z)   # electron scattering floor
    hion = khion(rho, T, X, Y, Z)

    # Rosseland mean: sum contributions; electron scattering sets the floor
    total = bf + ff + ts + hion

    # Cap at 300 cm^2/g to prevent divergence at cool/dense surfaces
    # (Kramers formulas break down below ~10^4 K; electron scattering dominates in the deep interior)
    total = float(np.clip(total, ts, 300.0))

    return total

def plot_kramer_sun(filename,delRho=100,delT=100, **kwargs):
    '''
    Plot the Kramer opacity for the Sun at various densities and temperatures
    
    :param filename: filename for the plot
    :param delRho: step size for densitiy
    :param delT: step size for temperature
    '''

    # let's go ahead and run tests for the sun
    # all of this info is coming from wikipedia
    rhos = np.linspace(0.001, 150, delRho)
    temps = np.linspace(5800, 15.7e6, delT)
    X = 0.7381
    Y = 0.2485
    Z= 0.0134

    # make a mesh grid
    xx, yy = np.meshgrid(temps,rhos)
    
    _ = plt.figure(figsize=(8,7),dpi=500)

    taus = kramer_opacity(rho=yy, T=xx, X=X, Y=Y, Z=Z)

    plt.pcolormesh(yy, xx, taus, cmap='plasma')
    
    plt.xscale('log')
    plt.yscale('log')

    plt.ylabel('Temperature (K)')
    plt.xlabel('Density (g cm$^{-3}$)')
    plt.colorbar(label='Opacity, $\\tau$')

    print(taus)
    print(taus[0,0])
    print(xx[0,0], yy[0,0])


    print(kbf(yy[0,0],xx[0,0],X,Y,Z), 'computed:', taus[0,0])
    print('break')

    plt.savefig(filename)

    return
    
def Pfit(params_0):

    # initial values for module

    r = params_0['R']
    T = params_0['T']
    L = params_0['L']
    X = params_0['X']
    Y = params_0['Y']
    Z = params_0['Z']
    mu = params_0['mu']
    P = params_0['P']
    M_total = params_0['M_total']

    # calculate Pfit

    g = ( CONSTANTS['G'] * M_total ) /  r**2

    Teff = (( L ) / (4 * np.pi * CONSTANTS['stefbolt'] * r**2) )**(1/4)

    # Use EOS function to get density
    rho = ef.simple_eos(P, mu, T)

    # here we will get our taus
    kmean = kramer_opacity(rho, T, X, Y, Z)

    # Surface pressure approximation (Eddington)
    # P_surf ~ (2/3) g / kappa
    P_surf = (2/3) * g / kmean
    
    return P_surf



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

    