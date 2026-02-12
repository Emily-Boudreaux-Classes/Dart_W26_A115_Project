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


def kramer_opacity(rho, T, t, massFrac):
    '''
    Implementation of the Kramer opacity law. 
    
    :param rho: density (units: grams cm^-3)
    :param T: temperature (units: Kelvin)

    '''
    # compute compositions
    X, Y, Z = nuc_burn.getCanonicalComposition(massFrac)

    # compute the three opacities
    bf = (4.34e25) * (1 + X) * Z * rho * (T**(-7/2))
    ff = (3.68e22) * (1 - Z) * (1 + X) * rho * (T**(-7/2))
    ts = 0.3 * (1 + X)

    # sum all opacity sources together
    sum = bf + ff + ts
    return sum

def plot_kramer_sun(filename,delRho=10,delT=100, **kwargs):
    '''
    Plot the Kramer opacity for the Sun at various densities and temperatures
    
    :param filename: filename for the plot
    :param delRho: step size for densitiy
    :param delT: step size for temperature
    '''
    # let's go ahead and run tests for the sun
    # all of this info is coming from wikipedia
    temps = np.linspace(1.5e7, 2e7, 100)
    rhos = np.linspace(1.5e2, 1.5e2, 100)
    xx, yy = np.meshgrid(temps,rhos)
    ar_shape = xx.shape
    Time = 1000
    e, mu, mf = nuc_burn.burn(xx.flatten(), yy.flatten(), Time)

    _ = plt.figure(figsize=(8,7),dpi=500)
    taus = []
    #for rho in rhos:
    #    tau = kramer_opacity(rho,temps,Time,mf)
    #    taus.append(tau)
    taus = kramer_opacity(xx.flatten(),yy.flatten(),Time,mf)

    plt.contourf(xx.reshape(ar_shape), yy.reshape(ar_shape),taus.reshape(ar_shape))

    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Density (g cm$^{-3}$)')
    plt.colorbar(label='Opacity, $\\tau$')

    plt.savefig(filename)

    
def Pfit(params_0):

    # initial values for module

    r = params_0['R']
    T = params_0['T']
    L = params_0['L']
    X = params_0['X']
    P = params_0['P']

    # calculate Pfit

    #g = CONSTANTS['G'] *
    



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

    