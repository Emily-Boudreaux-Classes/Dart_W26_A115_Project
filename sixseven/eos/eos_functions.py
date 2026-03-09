import numpy as np
from dataclasses import dataclass 
@dataclass(frozen=True)
class _CONST: 
    """
    Dataclass defining the constants necessary for the EoS module
    """
    mh: float = 1.67262192E-24 # mass of proton in g 
    sigma_sb: float = 5.670374419E-5 # SB constant: erg⋅cm−2⋅s−1⋅K−4
    kB: float = 1.380649E-16 # Boltzmann constant: erg/K 
    c_s: float = 3E10 # speed of light: cm/s 
    Na = 6.02214076e23 # Avogadro's number 
    @property
    def Cp_ideal(self) -> float:              
        """Specific heat at a constant pressure for a monatomic ideal gas"""
        return (5/2)*self.Na * self.kB
    @property
    def Cv_ideal(self) -> float: 
        """Specific heat at a constant volume for a monatomic ideal gas"""
        return (3/2)*self.Na*self.kB
CONST = _CONST()
def ad_index(Cp, Cv): 
    """
    Retrieves the adiabatic index given Cp and Cv. This will always be 5/3 for a monatomic ideal gas.
    :param Cp: The specific heat at constant pressure (=20.8E7 erg/mol/K for a monatomic ideal gas)
    :param Cv: The specific heat at constant volume (=1.25E8 erg/mol/K for a monatomic ideal gas)
    """
    return Cp/Cv
def init_U(mu, dM, T): 
    """
    Defines an initial specific internal energy for an ideal gas 
    by solving U = Cv*N*T over an array of compositions, 
    mass elements, and temperatures.
    
    :param mu: Mean molecular weight (g/mole or amu/number of particles)
    :param dM: Mass element (g)
    :param T: Temperature (K)
    """
    dM = np.asarray(dM, dtype=float)
    T = np.asarray(T, dtype=float)
    n = dM / mu 
    U_init = n*CONST.Cv_ideal*T 
    U_init = np.asarray(U_init, dtype=float)
    return U_init

def update_U(U_init, dU):
    """
    Updates the specific internal energy according to a given
    change in energy dU
    :param U_init: Initial specific internal energy (erg)
    :param dU: Change in specific internal energy (erg)
    """
    new_U = U_init + dU
    new_U = np.asarray(new_U, dtype=float)
    return new_U

def temperature_solver(dM, mu, U): 
    """
    Takes in mass, mean molecular weight, specific internal energy after 
    recieving mu and U from the nuclear module before solving for T. 
    :param dM: Mass element (g)
    :param mu: Mean molecular weight (g/mole or amu/number of particles)
    :param U: Specific internal energy (erg)
    """
    n = dM / mu  
    T = U/(n*CONST.Cv_ideal) # in Kelvin 
    T = np.asarray(T, dtype=float)
    return T 
def simple_eos(P, mu, T): 
    """
    Solves the ideal gas law to retrieve a polytropic equation of state.
    This function does not handle cases where electron degeneracy 
    or ultra relativistic gas are present. It will best model stars 
    that fall between 0.7Msun <= M 5Msun. 
    :param P: Pressure (g cm^-1 s^-2)
    :param mu: molar mass (g/moles or amu/number of particles) 
    :param T: Temperature (K)
    """
    P = np.asarray(P, dtype=float)
    T=np.asarray(T, dtype=float)
    a= 4*(CONST.sigma_sb/CONST.c_s)
    P_rad = (1/3)*a*T**4
    rho = (P - P_rad)*(mu*CONST.mh)/(CONST.kB*T)
    rho = np.asarray(rho, dtype=float)
    return rho
def entropy(T, U, P, S0): 
    """
    Uses the fundamental thermodynamic relation to solve for entropy given a temperature,
    pressure, and energy. 
    :param T: Temperature (K)
    :param U: Specific internal energy (erg) 
    :param P: Pressure (g cm^-1 s^-2)
    :param S0: Some arbitrary initial entropy 
    """
    S = (1/T)*(U-(CONST.Na*CONST.kB*T*np.log10(P))+S0)
    return S
def nabla_ad(gamma): 
    """
    Uses the adiabatic index to obtain the adiabatic temperature 
    gradient 
    :param gamma: The adiabatic temperature gradient (Cp/Cv)
    """
    n_ad = (gamma-1)/gamma
    return n_ad 
def dT_dP(rho, dM): 
    """
    The deriative of temperature with respect to pressure over constant entropy. 
    Useful in determining the radiative temperature gradient 
    :param rho: Density (g/cm^3)
    :param dM: Mass element (g)
    """
    rho = np.asarray(rho, dtype=float)
    dM = np.asarray(rho, dtype=float)
    dV = dM/rho
    result = dV/(CONST.Na * CONST.kB)
    result=np.asarray(result, dtype=float)
    return result
def nabla_rad(P, T, dT_dP): 
    """
    The radiative temperature gradient. 
    :param P: Pressure (g cm^-1 s^-2)
    :param T: Temperature (K)
    :param dT_dP: The deriative of temperature with respect to pressure over constant entropy. 
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    n_rad = (P/T) *dT_dP
    n_rad = np.asarray(n_rad, dtype=float)
    return n_rad
