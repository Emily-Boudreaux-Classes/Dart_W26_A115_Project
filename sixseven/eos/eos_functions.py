import numpy as np


def constants():
    """
    Returns constants 
    [0] : mass of proton 
    [1] : Boltzmann constant 
    [2] : Radiation constant (4 sigmaSB/c_s)
    """ 
    mh = 1.67262192E-24 # mass of proton in g 
    sigma_sb = 	5.670374419E-5 # SB constant: erg⋅cm−2⋅s−1⋅K−4
    kB= 1.380649E-16 # Boltzmann constant: erg/K 
    c_s = 3E10 # speed of light: cm/s 
    a = 4*sigma_sb/c_s 
    R = 8.31446261815324E7 # ideal gas constant : erg/K/mol
    return mh, kB, a, R
def init_U(mu, dM, T): 
    """
    Defines an initial specific internal energy 
    
    :param mu: Description
    :param M: Description
    :param T: Description
    """
    mh, kB, a, R = constants()
    dM = np.asarray(dM, dtype=float)
    T = np.asarray(T, dtype=float)
    n = dM / mu 
    Cv = (3/2)*R # specific heat at constant volume for a monatomic ideal gas
    U_init = n*Cv*T # specific internal energy in erg
    U_init = np.asarray(U_init, dtype=float)
    return U_init

def update_U(U_init, dU):
    new_U = U_init + dU
    new_U = np.asarray(new_U, dtype=float)
    return new_U

def temperature_solver(dM, mu, U): 
    """
    Takes in mass, mean molecular weight, and specific internal energy after 
    recieving mu and U from the nuclear module. Solves for the specific heat at a constant 
    volume for an ideal monatomic gas and solves for T. 
    :param M: Mass (g)
    :param mu: Mean molecular weight (g/mole or amu/number of particles)
    :param U: Specific internal energy (erg)
    """
    mh, kB, a, R = constants()
    Cv = (3/2)*R # specific heat at constant volume for a monatomic ideal gas 
    n = dM / mu  

    T = U/(n*Cv) # in Kelvin 
    T = np.asarray(T, dtype=float)
    return T 
def simple_eos(P, mu, T): 
    """
    Solves the polytropic form of the PV-nKbT equation of state. 
    This function does not handle cases where electron degeneracy 
    or ultra relativistic gas are present. It will best model stars 
    that fall between 0.7Msun <= M 5Msun. 
    :param P: Pressure (g cm^-1 s^-2)
    :param mu: molar mass (g/moles or amu/number of particles) 
    :param M: enclosed mass (g) 
    :param U: specific internal energy (erg)
    """
    mh, kB, a, R = constants()
    P = np.asarray(P, dtype=float)
    P_rad = (1/3)*a*T**4
    rho = (P - P_rad)*(mu*mh)/(kB*T)
    rho = np.asarray(rho, dtype=float)
    return rho