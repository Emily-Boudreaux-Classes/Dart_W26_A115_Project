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

REPO_DIR = str(Path(__file__).resolve().parent.parent.parent)
    

def kramer_opacity(rho, T):
    '''
    Implementation of the Kramer opacity law. 
    
    :param rho: density (units: grams cm^-3)
    :param T: temperature (units: Kelvin)

    '''

    return rho * T**(-7/2)

def plot_kramer_sun(filename,delRho=10,delT=100, **kwargs):
    '''
    Plot the Kramer opacity for the Sun at various densities and temperatures
    
    :param rho: density in g cm^-3
    :param T: temperature in K
    :param filename: filename for the plot
    :param delRho: step size for densitiy
    :param delT: step size for temperature
    '''
    # let's go ahead and run tests for the sun
    # all of this info is coming from wikipedia
    rhos = np.linspace(0.001, 150, delRho)
    temps = np.linspace(5800, 15.7e6, delT)

    _ = plt.figure(figsize=(10,10),dpi=500)

    taus = []
    for rho in rhos:
        tau = kramer_opacity(rho,temps)
        taus.append(tau)
        plt.plot(temps, tau, label = f'$\\rho =${rho}')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Opacity, $\tau$')

    plt.savefig(filename)

    




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

    