#Code written by George with Copilot assistance for troubleshooting


import math
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
from modules.nuclear.nuc_burn import *
from modules.timestep.timestep import dyn_timestep
from modules.eos.eos_functions import *

#        eps, mu, mass_frac = burn(temp=T,rho=rho,time=step,comp=mass_frac) # sasha function
#        eps = np.asarray(eps) # setting the lists as arrays
#        mu = np.asarray(mu)

#        U = update_U(U,eps) # cassie - updates internal energy 
#        T = temperature_solver(dM=dM,mu=mu,U=U) # cassie - solves temperature
#        rho = simple_eos(P=P,mu=mu,T=T) # cassie - gets dens from ideal gas eos


# ----------------------------
# Physical constants (cgs)
# ----------------------------
G = 6.67430e-8                 # gravitational constant [cm^3 g^-1 s^-2]
a_rad = 7.5657e-15             # radiation density constant [erg cm^-3 K^-4]
c_light = 2.99792458e10        # speed of light [cm s^-1]
k_B = 1.380649e-16             # Boltzmann [erg K^-1]
m_u = 1.66053906660e-24        # atomic mass unit [g]
m_H = 1.6735575e-24            # hydrogen atom mass [g]
sigma_SB = 5.670374419e-5      # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
pi = math.pi
M_sun = 1.98847e33
# Nominal IAU 2015 "solar units" as exact conversion factors (converted to cgs)
R_SUN = 6.957e10          # cm
L_SUN = 3.828e33          # erg / s
T_EFF_SUN = 5772.0        # K

# Typical present-day solar center values (model-dependent, but widely used ballpark)
P_C_SUN = 2.453e17        # dyn / cm^2
#P_C_SUN = 2.453e25      # Test
T_C_SUN = 1.559e7         # K

# Photosphere ("surface") is definition-dependent; this is a rough characteristic value
T_SURF_SUN = T_EFF_SUN    # K
P_SURF_SUN = 3.0e4        # dyn / cm^2

@dataclass(frozen=True)
class Composition:
    X: float = 1.00 #Hydrogen mass fraction
    Y: float = 0.00 #Helium mass fraction
    Z: float = 0.00   #Metal mass fraction

#Microphysics placeholder more will have to be added
def mean_molecular_weight(comp: Composition) -> float:
    """
    mu(comp). Mean molecular weight mu for a fully ionized gas."""
    return 1.0 / (2.0 * comp.X + 0.75 * comp.Y + 0.5 * comp.Z)

def density(lnP, lnT): # Note: Get from Cassie
    P, T = np.exp(lnP), np.exp(lnT)
    P_rad = (1/3) * a_rad * T**4
    P_gas = np.maximum(P - P_rad, 1e-10)
    return (P_gas * m_u * m_H)/(k_B * T)

def eos_rho(P, T, comp: Composition):
    """
    EOS: Ideal gas. Vectorized for array inputs.
    """
    mu = mean_molecular_weight(comp)
    P_rad = 0#(a_rad / 3.0) * T**4
    P_gas = np.maximum(P - P_rad, 1e-99)
    rho = (P_gas * mu * m_u) / (k_B * T)
    return np.maximum(rho, 1e-99)


def opacity_kappa(rho, T, comp: Composition):
    """
    Opacity kappa(rho,T,comp). Placeholder of 0.2. Vectorized.
    """
    return np.full_like(rho, 1e-4)


def energy_generation_eps(rho, T, comp: Composition):
    """
    Nuclear energy generation eps(rho,T,comp). Handles both scalar and array inputs.
    The burn() function expects arrays, so scalars are wrapped and unwrapped.
    """
    # Check if inputs are scalars
    rho_is_scalar = np.isscalar(rho)
    T_is_scalar = np.isscalar(T)
    
    # Convert scalars to 1-element arrays for burn()
    rho_arr = np.atleast_1d(rho)
    T_arr = np.atleast_1d(T)
    
    result = burn(temp=T_arr, rho=rho_arr, time=1, comp=None)  # sasha function
    
    # burn() returns (epsilon_list, mu_list, mass_frac)
    # but may return a single value on error — guard against that
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError(f"burn() returned unexpected result (type={type(result)}, "
                         f"value={result}). Expected (eps, mu, mass_frac) tuple.")
    
    eps, mu, mass_frac = result
    eps = np.asarray(eps)
    mu = np.asarray(mu)
    
    # If inputs were scalars, return scalar outputs
    if rho_is_scalar and T_is_scalar:
        eps = float(eps.flat[0])
        mu = float(mu.flat[0])

    return eps, mu, mass_frac


def kap_test(m, rho, T, comp: Composition):
    """
    Approx rho based on solar values.
    """
    m = np.asarray(m, dtype=float)
    q = np.clip(m / M_sun, 0.0, 1.0)

    kap_c = 2.17     # cm^2/g
    kap_s = 0.3    # cm^2/g

    # Shape control: larger n makes rho stay high until close to the surface
    n = 8.0

    # Smooth monotone drop from center (q=0) to surface (q=1)
    kap = kap_s + (kap_c - kap_s) * (1.0 - q)**n

    return np.maximum(kap, 1e-99)

def rho_test(m, rho_c=1.5e2):
    """
    Approx rho based on solar values.
    rho_c is a tunable parameter (central density in g/cm³)
    """
    m = np.asarray(m, dtype=float)
    q = np.clip(m / M_sun, 0.0, 1.0)

    #rho_c = 1.5e6    # g/cm^3 test
    # rho_c is now a parameter
    rho_s = 3.0e-7    # g/cm^3

    # Shape control: larger n makes rho stay high until close to the surface
    n = 8.0

    # Smooth monotone drop from center (q=0) to surface (q=1)
    rho = rho_s + (rho_c - rho_s) * (1.0 - q)**n

    return np.maximum(rho, 1e-99)



# ----------------------------
# Stellar structure ODEs (Lagrangian mass coordinate)
# Working in LOG SPACE: y = [ln(r), ln(P)]
# ----------------------------
def stellar_structure_rhs(m, y, comp: Composition, rho_c=1.5e2):
    """
    dy/dm = [d(ln r)/dm, d(ln P)/dm]
    For solve_ivp: signature is (m, y), comp passed via lambda.
    
    Using log transformation:
        u = ln(r), v = ln(P)
        du/dm = (1/r) * dr/dm
        dv/dm = (1/P) * dP/dm
    """
    ln_r, ln_P = y

    # Convert from log space
    r = np.exp(ln_r)
    P = np.exp(ln_P)

    rho = rho_test(m, rho_c)

    # 1) Mass conservation: dr/dm = 1 / (4π r² ρ)
    #    => d(ln r)/dm = (1/r) * dr/dm = 1 / (4π r³ ρ)
    dln_r_dm = 1.0 / (4.0 * pi * r**3 * rho)

    # 2) Hydrostatic equilibrium: dP/dm = -G m / (4π r⁴)
    #    => d(ln P)/dm = (1/P) * dP/dm = -G m / (4π r⁴ P)
    dln_P_dm = - G * m / (4.0 * pi * r**4 * P)

    return [dln_r_dm, dln_P_dm]


def get_initial_conditions(m_0, P_c, rho_c):
    """
    Get initial conditions at m = m_0 (near center).
    Returns [ln(r_0), ln(P_0)]
    P_c and rho_c are tunable parameters.
    """
    rho_center = rho_test(m_0, rho_c)
    
    # Taylor expansion near center: r ~ (3m / 4πρ_c)^(1/3)
    r_0 = (3 * m_0 / (4 * np.pi * rho_center))**(1/3)
    
    # Central pressure
    P_0 = P_c
    
    return [np.log(r_0), np.log(P_0)]


def integrate_star(P_c, rho_c, M, m_0, comp, return_full=False):
    """
    Integrate stellar structure from center to surface.
    Returns final radius (and optionally full solution).
    """
    y0 = get_initial_conditions(m_0, P_c, rho_c)
    
    def ode_wrapper(m, y):
        return stellar_structure_rhs(m, y, comp, rho_c)
    
    # Integrate
    sol = solve_ivp(ode_wrapper, [m_0, M], y0, method='RK45',
                    rtol=1e-8, atol=1e-10, dense_output=True)
    
    if not sol.success:
        if return_full:
            return np.inf, None
        return np.inf
    
    # Get final radius
    r_final = np.exp(sol.y[0, -1])
    
    if return_full:
        return r_final, sol
    return r_final


def find_Pc_for_target_radius(R_target, rho_c, M, m_0, comp, 
                               P_c_min=1e14, P_c_max=1e20, verbose=True):
    """
    Use root-finding (Brent's method) to find P_c that gives R_target.
    """
    def residual(log_P_c):
        P_c = 10**log_P_c
        r_final = integrate_star(P_c, rho_c, M, m_0, comp)
        if verbose:
            print(f"  P_c = {P_c:.3e}, R_final = {r_final/R_SUN:.4f} R_sun")
        return r_final - R_target
    
    try:
        log_P_c_solution = brentq(residual, np.log10(P_c_min), np.log10(P_c_max), 
                                   xtol=1e-6, maxiter=50)
        P_c_solution = 10**log_P_c_solution
        return P_c_solution
    except ValueError as e:
        print(f"Root finding failed: {e}")
        return None


def optimize_Pc_and_rhoc(R_target, M, m_0, comp, 
                          P_c_guess=2.5e17, rho_c_guess=1.5e2, verbose=True):
    """
    Optimize both P_c and rho_c to hit target radius.
    Uses Nelder-Mead on log parameters.
    """
    def objective(params):
        log_P_c, log_rho_c = params
        P_c = 10**log_P_c
        rho_c = 10**log_rho_c
        
        r_final = integrate_star(P_c, rho_c, M, m_0, comp)
        
        # Relative error in radius
        error = ((r_final - R_target) / R_target)**2
        
        if verbose:
            print(f"  P_c = {P_c:.3e}, rho_c = {rho_c:.3e}, "
                  f"R = {r_final/R_SUN:.4f} R_sun, error = {np.sqrt(error)*100:.2f}%")
        
        return error
    
    x0 = [np.log10(P_c_guess), np.log10(rho_c_guess)]
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 200})
    
    P_c_opt = 10**result.x[0]
    rho_c_opt = 10**result.x[1]
    
    return P_c_opt, rho_c_opt, result


# ----------------------------
# 4-Equation Stellar Structure (r, P, L, T)
# Variables: y = [ln(r), ln(P), L, ln(T)]
# ----------------------------
def stellar_structure_rhs_4eq(m, y, comp: Composition, kappa=0.4):
    """
    4-equation stellar structure in log coordinates.
    y = [ln(r), ln(P), L, ln(T)]
    """
    ln_r, ln_P, L, ln_T = y
    r = np.exp(ln_r)
    P = np.exp(ln_P)
    T = np.exp(ln_T)
    
    # Get density from EOS
    rho = eos_rho(P, T, comp)
    
    # Get nuclear energy generation rate
    eps_nuc = energy_generation_eps(rho, T, comp)
    
    # 1) Mass conservation: d(ln r)/dm = 1 / (4π r³ ρ)
    dln_r_dm = 1.0 / (4.0 * pi * r**3 * rho)
    
    # 2) Hydrostatic equilibrium: d(ln P)/dm = -G m / (4π r⁴ P)
    dln_P_dm = -G * m / (4.0 * pi * r**4 * P)
    
    # 3) Energy generation: dL/dm = eps_nuc from burn function
    dL_dm = eps_nuc
    
    # 4) Temperature gradient (radiative): d(ln T)/dm = -3 κ L / (64 π² a_c r⁴ T⁴)
    #    where a_c = a_rad * c_light / 4 = σ_SB / c
    dln_T_dm = -3.0 * kappa * L / (64.0 * pi**2 * a_rad * c_light * r**4 * T**4)
    
    return [dln_r_dm, dln_P_dm, dL_dm, dln_T_dm]


def integrate_star_4eq(P_c, T_c, M, m_0, comp, kappa=0.4, return_full=False):
    """
    Integrate 4-equation stellar structure from center to surface.
    Initial conditions: P_c, T_c at m_0.
    """
    # Get initial density from EOS
    rho_c = eos_rho(P_c, T_c, comp)
    
    # Initial radius from Taylor expansion
    r_0 = (3 * m_0 / (4 * np.pi * rho_c))**(1/3)
    
    # Initial luminosity from energy generation at center
    eps_0, mu_0, mass_frac_0 = energy_generation_eps(rho_c, T_c, comp)
    L_0 = eps_0 * m_0
    
    # Initial conditions in log space
    y0 = [np.log(r_0), np.log(P_c), L_0, np.log(T_c)]
    
    def ode_wrapper(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, kappa)
    
    # Integrate
    sol = solve_ivp(ode_wrapper, [m_0, M], y0, method='RK45',
                    rtol=1e-8, atol=1e-10, dense_output=True)
    
    if not sol.success:
        if return_full:
            return np.inf, None
        return np.inf
    
    # Get final radius
    r_final = np.exp(sol.y[0, -1])
    
    if return_full:
        return r_final, sol
    return r_final


def optimize_Pc_Tc_4eq(R_target, M, m_0, comp, 
                        P_c_guess=2.5e17, T_c_guess=1.5e7, 
                        kappa=0.4, verbose=True):
    """
    Optimize P_c and T_c to hit target radius using the 4-equation system.
    Uses Nelder-Mead on log parameters.
    """
    def objective(params):
        log_P_c, log_T_c = params
        P_c = 10**log_P_c
        T_c = 10**log_T_c
        
        r_final = integrate_star_4eq(P_c, T_c, M, m_0, comp, kappa)
        
        # Relative error in radius
        error = ((r_final - R_target) / R_target)**2
        
        if verbose:
            print(f"  P_c = {P_c:.3e}, T_c = {T_c:.3e}, "
                  f"R = {r_final/R_SUN:.4f} R_sun, error = {np.sqrt(error)*100:.2f}%")
        
        return error
    
    x0 = [np.log10(P_c_guess), np.log10(T_c_guess)]
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 200})
    
    P_c_opt = 10**result.x[0]
    T_c_opt = 10**result.x[1]
    
    return P_c_opt, T_c_opt, result


if __name__ == "__main__":
    comp = Composition()
    M = 1.0 * M_sun
    m_0 = 1e-8 * M
    R_TARGET = 5.0 * R_SUN
    print("=" * 80)
    print("SHOOTING METHOD: Optimizing Pc and Tc to match target radius (4eq system)")
    print("=" * 80)
    print(f"Target radius: {R_TARGET/R_SUN:.1f} R_sun")
    print(f"Total mass: {M/M_sun:.4f} M_sun")
    print("=" * 80 + "\n")
    # Optimize Pc and Tc
    Pc_opt, Tc_opt, result = optimize_Pc_Tc_4eq(R_TARGET, M, m_0, comp, P_c_guess=2.5e17, T_c_guess=1.5e7, kappa=0.4, verbose=True)
    print(f"\n>>> OPTIMAL: Pc = {Pc_opt:.4e} dyn/cm², Tc = {Tc_opt:.4e} K")
    # Integrate with optimal params
    r_final, sol = integrate_star_4eq(Pc_opt, Tc_opt, M, m_0, comp, kappa=0.4, return_full=True)
    m_eval = np.logspace(np.log10(m_0), np.log10(M), 400)
    m_eval[0] = m_0
    m_eval[-1] = M
    y_eval = sol.sol(m_eval)
    r_sol = np.exp(y_eval[0])
    P_sol = np.exp(y_eval[1])
    T_sol = np.exp(y_eval[3])
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Central pressure: {Pc_opt:.4e} dyn/cm²")
    print(f"Central temperature: {Tc_opt:.4e} K")
    print(f"Central radius: {r_sol[0]:.4e} cm = {r_sol[0]/R_SUN:.6f} R_sun")
    print(f"Surface radius: {r_sol[-1]:.4e} cm = {r_sol[-1]/R_SUN:.4f} R_sun")
    print(f"Surface pressure: {P_sol[-1]:.4e} dyn/cm²")
    print(f"Surface temperature: {T_sol[-1]:.4e} K")
    print(f"Target radius: {R_TARGET/R_SUN:.1f} R_sun")
    print(f"Radius error: {(r_sol[-1] - R_TARGET)/R_TARGET * 100:.4f}%")
    print("=" * 80 + "\n")
    
    # Compute density for each shell
    rho_sol = eos_rho(P_sol, T_sol, comp)
    L_sol = y_eval[2]
    
    # Print first 10 shells
    print("=" * 100)
    print("FIRST 10 SHELLS (near center)")
    print("=" * 100)
    print(f"{'Shell':>6} {'m/M_sun':>12} {'r/R_sun':>12} {'P [dyn/cm²]':>14} {'T [K]':>12} {'rho [g/cm³]':>14} {'L [erg/s]':>14}")
    print("-" * 100)
    for i in range(min(10, len(m_eval))):
        print(f"{i:>6} {m_eval[i]/M_sun:>12.6e} {r_sol[i]/R_SUN:>12.6e} {P_sol[i]:>14.4e} {T_sol[i]:>12.4e} {rho_sol[i]:>14.4e} {L_sol[i]:>14.4e}")
    print("=" * 100 + "\n")
    
    # Print last 10 shells
    print("=" * 100)
    print("LAST 10 SHELLS (near surface)")
    print("=" * 100)
    print(f"{'Shell':>6} {'m/M_sun':>12} {'r/R_sun':>12} {'P [dyn/cm²]':>14} {'T [K]':>12} {'rho [g/cm³]':>14} {'L [erg/s]':>14}")
    print("-" * 100)
    for i in range(max(0, len(m_eval)-10), len(m_eval)):
        print(f"{i:>6} {m_eval[i]/M_sun:>12.6e} {r_sol[i]/R_SUN:>12.6e} {P_sol[i]:>14.4e} {T_sol[i]:>12.4e} {rho_sol[i]:>14.4e} {L_sol[i]:>14.4e}")
    print("=" * 100 + "\n")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(r_sol / R_SUN, P_sol)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('r / R_sun')
    axes[0].set_ylabel('P [dyn/cm²]')
    axes[0].set_title('Pressure vs Radius')
    axes[1].plot(r_sol / R_SUN, T_sol)
    axes[1].set_xlabel('r / R_sun')
    axes[1].set_ylabel('T [K]')
    axes[1].set_title('Temperature vs Radius')
    plt.tight_layout()
    plt.savefig('shooting_method_solution_4eq.png', dpi=150)
    plt.show()