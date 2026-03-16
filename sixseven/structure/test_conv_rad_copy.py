#Code written by George with Copilot assistance for troubleshooting

import os
import sys


import math
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
from sixseven.nuclear.nuc_burn import *
from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos.eos_functions import *
from sixseven.eos import eos_functions as ef 

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
mu = 1.004
# Typical present-day solar center values (model-dependent, but widely used ballpark)
P_C_SUN = 2.453e17        # dyn / cm^2
#P_C_SUN = 2.453e25      # Test
T_C_SUN = 1.559e7         # K

# Photosphere ("surface") is definition-dependent; this is a rough characteristic value
T_SURF_SUN = T_EFF_SUN    # K
P_SURF_SUN = 3.0e4        # dyn / cm^2


def eos_rho(P, T, comp: Composition):
    """
    EOS: Ideal gas. Vectorized for array inputs.
    """
    rho = ef.simple_eos(P, mu, T)
    return np.maximum(rho, 1e-99)


def energy_generation_eps(rho, T, comp: Composition):
    """
    Nuclear energy generation eps(rho,T,comp). Handles both scalar and array inputs.
    The burn() function expects arrays, so scalars are wrapped and unwrapped.
    In the sixseven package, burn() returns a list of NetOut objects.
    """
    # Check if inputs are scalars
    rho_is_scalar = np.isscalar(rho)
    T_is_scalar = np.isscalar(T)
    
    # Convert scalars to 1-element arrays for burn()
    rho_arr = np.atleast_1d(rho)
    T_arr = np.atleast_1d(T)
    
    # Suppress noisy KINSol stderr output from burn()
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        results = burn(temps=T_arr, rhos=rho_arr, time=1e10, comps=None)  # returns list of NetOut objects
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
    
    # Extract energy, mu, and composition from the NetOut results
    eps = np.asarray([r.energy for r in results])
    mu_burn = np.asarray([r.composition.getMeanParticleMass() for r in results])
    mass_frac = results[0].composition
    
    


    # If inputs were scalars, return scalar outputs
    if rho_is_scalar and T_is_scalar:
        eps = float(eps[0])
        mu_burn = float(mu_burn[0])

    return eps, mu_burn, mass_frac

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




# ----------------------------
# 4-Equation Stellar Structure (r, P, L, T)
# Variables: y = [ln(r), ln(P), L, ln(T)]
# ----------------------------
def get_eps_at_m(m, m_shells, eps_shells):
    """
    Get the nuclear energy generation rate at mass coordinate m
    by interpolating between the burn() shell values.
    Returns 0 outside the burning region.
    """
    if m <= m_shells[0]:
        return eps_shells[0]
    if m >= m_shells[-1]:
        return 0.0  # outside burning region
    return float(np.interp(m, m_shells, eps_shells))


def stellar_structure_rhs_4eq(m, y, comp: Composition, kappa=0.4,
                               eps_nuc=0.0, m_core=None,
                               m_shells=None, eps_shells=None,
                               use_convection=True):
    """
    4-equation stellar structure in log coordinates.
    y = [ln(r), ln(P), L, ln(T)]
    
    Nuclear burning is only applied in the burning region (m <= m_core).
    If m_shells/eps_shells are provided, eps_nuc is interpolated from
    the multi-shell burn() results. Otherwise uses constant eps_nuc.
    
    use_convection=True  -> Schwarzschild criterion (convective if nabla_rad > nabla_ad)
    use_convection=False -> purely radiative temperature gradient everywhere
    """
    ln_r, ln_P, L, ln_T = y
    r = np.exp(ln_r)
    P = np.exp(ln_P)
    T = np.exp(ln_T)
    
    # Adiabatic gradient for monatomic ideal gas: nabla_ad = 2/5
    nabla_ad = 0.4
    
    # Get density from EOS
    rho = eos_rho(P, T, comp)
    
    # 1) Mass conservation: d(ln r)/dm = 1 / (4π r³ ρ)
    dln_r_dm = 1.0 / (4.0 * pi * r**3 * rho)
    
    # 2) Hydrostatic equilibrium: d(ln P)/dm = -G m / (4π r⁴ P)
    dln_P_dm = -G * m / (4.0 * pi * r**4 * P)
    
    # 3) Energy generation: interpolated from burn() shells or constant
    if m_core is not None and m > m_core:
        dL_dm = 0.0  # no burning outside the core
    elif m_shells is not None and eps_shells is not None:
        dL_dm = get_eps_at_m(m, m_shells, eps_shells)
    else:
        dL_dm = eps_nuc
    
    # 4) Temperature gradient with optional Schwarzschild criterion
    #    Radiative: d(ln T)/dm = -3 κ L / (64 π² a_rad c r⁴ T⁴)
    dln_T_dm_rad = -3.0 * kappa * L / (64.0 * pi**2 * a_rad * c_light * r**4 * T**4)
    
    if use_convection and m > 0:
        # Schwarzschild criterion: nabla_rad = d ln T / d ln P (radiative)
        # nabla_rad = 3 κ L P / (16 π a_rad c G m T⁴)
        nabla_rad = (3.0 * kappa * L * P) / (16.0 * pi * a_rad * c_light * G * m * T**4)
        if nabla_rad > nabla_ad:
            # Convective: d(ln T)/dm = nabla_ad * d(ln P)/dm
            dln_T_dm = nabla_ad * dln_P_dm
        else:
            dln_T_dm = dln_T_dm_rad
    else:
        dln_T_dm = dln_T_dm_rad
    
    return [dln_r_dm, dln_P_dm, dL_dm, dln_T_dm]


def integrate_star_4eq(P_c, T_c, M, m_0, comp, kappa=0.4,
                       n_burn_shells=5, use_convection=True, return_full=False):
    """
    Integrate 4-equation stellar structure from center to surface.
    Initial conditions: P_c, T_c at m_0.
    
    Nuclear burning (via burn()) is computed for n_burn_shells zones
    at the center. Each shell gets its own T and rho from a simple
    estimate, and burn() is called once with arrays of length n_burn_shells.
    Outside the last burning shell, dL/dm = 0.
    
    Parameters
    ----------
    n_burn_shells : int
        Number of mass shells to burn at center (default 5).
    use_convection : bool
        If True, apply Schwarzschild criterion (convective where nabla_rad > nabla_ad).
        If False, purely radiative temperature gradient everywhere.
    """
    # Get initial density from EOS
    rho_c = eos_rho(P_c, T_c, comp)
    
    # Initial radius from Taylor expansion
    r_0 = (3 * m_0 / (4 * np.pi * rho_c))**(1/3)
    
    # Define the burning shell mass coordinates (log-spaced from m_0 into the core)
    # Core boundary is at the outermost burning shell
    m_shells = np.logspace(np.log10(m_0), np.log10(m_0) + n_burn_shells * 0.5, n_burn_shells)
    m_core = m_shells[-1]
    
    # Estimate T and rho at each shell (simple linear drop from center)
    # T drops ~proportionally to pressure drop; rough estimate
    T_shells = np.linspace(T_c, T_c * 0.95, n_burn_shells)  # slight T gradient
    rho_shells = np.array([eos_rho(P_c, T_s, comp) for T_s in T_shells])
    
    # Call burn() once with all shell arrays
    try:
        # Suppress noisy KINSol stderr output from burn()
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            results = burn(temps=T_shells, rhos=rho_shells, time=1, comps=None)
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
        
        eps_shells = np.array([r.energy for r in results])
    except Exception:
        # burn() solver failed for these conditions — return inf so optimizer moves on
        if return_full:
            return np.inf, None
        return np.inf
    
    # Initial luminosity from first shell
    L_0 = eps_shells[0] * m_0
    
    # Initial conditions in log space
    y0 = [np.log(r_0), np.log(P_c), L_0, np.log(T_c)]
    
    def ode_wrapper(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, kappa,
                                         m_core=m_core,
                                         m_shells=m_shells,
                                         eps_shells=eps_shells,
                                         use_convection=use_convection)
    
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
                        kappa=0.4, n_burn_shells=5,
                        use_convection=True, verbose=True):
    """
    Optimize P_c and T_c to hit target radius using the 4-equation system.
    Uses Nelder-Mead on log parameters.
    """
    def objective(params):
        log_P_c, log_T_c = params
        P_c = 10**log_P_c
        T_c = 10**log_T_c
        
        r_final = integrate_star_4eq(P_c, T_c, M, m_0, comp, kappa,
                                      n_burn_shells, use_convection)
        
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
    R_TARGET = 1.0 * R_SUN
    N_BURN = 1           # number of shells to burn
    USE_CONV = True      # True = Schwarzschild criterion, False = purely radiative

    # ================================================================
    # DIAGNOSTIC: Test each step of the pipeline before optimizing
    # ================================================================
    P_c_test = 2.5e17
    T_c_test = 1.5e7
    print("=" * 80)
    print("DIAGNOSTICS: Testing pipeline with P_c={:.3e}, T_c={:.3e}".format(P_c_test, T_c_test))
    print(f"  use_convection = {USE_CONV},  n_burn_shells = {N_BURN}")
    print("=" * 80)

    # 1) EOS: get central density
    rho_c_test = eos_rho(P_c_test, T_c_test, comp)
    print(f"[1] EOS rho_c        = {rho_c_test:.6e} g/cm^3")
    print(f"    (global mu used   = {mu})")

    # 2) Initial radius
    r_0_test = (3 * m_0 / (4 * np.pi * rho_c_test))**(1/3)
    print(f"[2] r_0              = {r_0_test:.6e} cm  ({r_0_test/R_SUN:.6e} R_sun)")

    # 3) Multi-shell burn() call
    m_shells_test = np.logspace(np.log10(m_0), np.log10(m_0) + N_BURN * 0.5, N_BURN)
    T_shells_test = np.linspace(T_c_test, T_c_test * 0.95, N_BURN)
    rho_shells_test = np.array([eos_rho(P_c_test, Ts, comp) for Ts in T_shells_test])
    print(f"[3] Calling burn() with {N_BURN} shells ...")
    print(f"    m_shells / M_sun = {m_shells_test / M_sun}")
    print(f"    T_shells         = {T_shells_test}")
    print(f"    rho_shells       = {rho_shells_test}")
    try:
        results_test = burn(temps=T_shells_test, rhos=rho_shells_test, time=1e10, comps=None)
        eps_shells_test = np.array([r.energy for r in results_test])
        mu_shells_test = np.array([r.composition.getMeanParticleMass() for r in results_test])
        print(f"    burn() succeeded!")
        for i in range(N_BURN):
            print(f"    shell {i}: eps = {eps_shells_test[i]:.4e} erg/g/s, mu = {mu_shells_test[i]:.4f}")
    except Exception as e:
        print(f"    burn() FAILED: {e}")
        eps_shells_test = np.zeros(N_BURN)

    # 4) Initial luminosity
    L_0_test = eps_shells_test[0] * m_0
    print(f"[4] L_0 = eps[0]*m_0 = {L_0_test:.6e} erg/s  ({L_0_test/3.828e33:.4e} L_sun)")

    # 5) Test ODE RHS at initial point + Schwarzschild check
    y0_test = [np.log(r_0_test), np.log(P_c_test), L_0_test, np.log(T_c_test)]
    m_core_test = m_shells_test[-1]
    rhs_test = stellar_structure_rhs_4eq(m_0, y0_test, comp, kappa=0.4,
                                          m_core=m_core_test,
                                          m_shells=m_shells_test,
                                          eps_shells=eps_shells_test,
                                          use_convection=USE_CONV)
    # Compute nabla_rad for diagnostic
    if m_0 > 0:
        nabla_rad_0 = (3.0 * 0.4 * L_0_test * P_c_test) / (16.0 * pi * a_rad * c_light * G * m_0 * T_c_test**4)
    else:
        nabla_rad_0 = 0.0
    print(f"[5] ODE RHS at m_0:")
    print(f"    d(ln r)/dm       = {rhs_test[0]:.6e}")
    print(f"    d(ln P)/dm       = {rhs_test[1]:.6e}")
    print(f"    dL/dm            = {rhs_test[2]:.6e}")
    print(f"    d(ln T)/dm       = {rhs_test[3]:.6e}")
    print(f"    nabla_rad        = {nabla_rad_0:.4f}  (nabla_ad = 0.4)")
    print(f"    CONVECTIVE       = {nabla_rad_0 > 0.4}  (Schwarzschild criterion)")
    print(f"    m_core           = {m_core_test:.4e} g  ({m_core_test/M:.4e} M)")

    # 6) Try a short integration (just 10% of mass)
    print("[6] Test integration to 10% of M ...")
    def ode_test(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, 0.4,
                                          m_core=m_core_test,
                                          m_shells=m_shells_test,
                                          eps_shells=eps_shells_test,
                                          use_convection=USE_CONV)
    sol_test = solve_ivp(ode_test, [m_0, 0.1*M], y0_test, method='RK45',
                         rtol=1e-8, atol=1e-10, dense_output=True, max_step=M/100)
    print(f"    success          = {sol_test.success}")
    print(f"    message          = {sol_test.message}")
    print(f"    nsteps           = {len(sol_test.t)}")
    if sol_test.success:
        r_10 = np.exp(sol_test.y[0, -1])
        P_10 = np.exp(sol_test.y[1, -1])
        T_10 = np.exp(sol_test.y[3, -1])
        L_10 = sol_test.y[2, -1]
        print(f"    r(0.1M)          = {r_10:.6e} cm  ({r_10/R_SUN:.4f} R_sun)")
        print(f"    P(0.1M)          = {P_10:.6e}")
        print(f"    T(0.1M)          = {T_10:.6e}")
        print(f"    L(0.1M)          = {L_10:.6e} erg/s")
        rho_10 = eos_rho(P_10, T_10, comp)
        print(f"    rho(0.1M)        = {rho_10:.6e}")
    else:
        r_last = np.exp(sol_test.y[0, -1])
        P_last = np.exp(sol_test.y[1, -1])
        T_last = np.exp(sol_test.y[3, -1])
        print(f"    r(last)          = {r_last:.6e} cm  ({r_last/R_SUN:.4f} R_sun)")
        print(f"    P(last)          = {P_last:.6e}")
        print(f"    T(last)          = {T_last:.6e}")
        print(f"    m(last)          = {sol_test.t[-1]:.6e} ({sol_test.t[-1]/M:.4e} M)")

    # 7) Full integration
    print("[7] Full integration to M ...")
    r_final_diag = integrate_star_4eq(P_c_test, T_c_test, M, m_0, comp, kappa=0.4,
                                       n_burn_shells=N_BURN, use_convection=USE_CONV,
                                       return_full=False)
    print(f"    R_final          = {r_final_diag:.6e} cm  ({r_final_diag/R_SUN:.4f} R_sun)")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("SHOOTING METHOD: Optimizing Pc and Tc to match target radius (4eq system)")
    print(f"  use_convection = {USE_CONV},  n_burn_shells = {N_BURN}")
    print("=" * 80)
    print(f"Target radius: {R_TARGET/R_SUN:.1f} R_sun")
    print(f"Total mass: {M/M_sun:.4f} M_sun")
    print("=" * 80 + "\n")
    # Optimize Pc and Tc
    Pc_opt, Tc_opt, result = optimize_Pc_Tc_4eq(R_TARGET, M, m_0, comp,
                                                 P_c_guess=2.5e17, T_c_guess=1.5e7,
                                                 kappa=0.4, n_burn_shells=N_BURN,
                                                 use_convection=USE_CONV, verbose=True)
    print(f"\n>>> OPTIMAL: Pc = {Pc_opt:.4e} dyn/cm², Tc = {Tc_opt:.4e} K")
    # Integrate with optimal params
    r_final, sol = integrate_star_4eq(Pc_opt, Tc_opt, M, m_0, comp, kappa=0.4,
                                       n_burn_shells=N_BURN, use_convection=USE_CONV,
                                       return_full=True)
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
    
    # Plot vs m/M
    q = m_eval / M  # fractional mass coordinate
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(q, P_sol)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('m / M')
    axes[0, 0].set_ylabel('P [dyn/cm²]')
    axes[0, 0].set_title('Pressure')
    
    axes[0, 1].plot(q, np.log10(T_sol))
    axes[0, 1].set_xlabel('m / M')
    axes[0, 1].set_ylabel('log10(T [K])')
    axes[0, 1].set_title('Temperature')
    
    axes[1, 0].plot(q, r_sol / R_SUN)
    axes[1, 0].set_xlabel('m / M')
    axes[1, 0].set_ylabel('r / R_sun')
    axes[1, 0].set_title('Radius')
    
    axes[1, 1].plot(q, L_sol / L_SUN)
    axes[1, 1].set_xlabel('m / M')
    axes[1, 1].set_ylabel('L / L_sun')
    axes[1, 1].set_title('Luminosity')
    
    plt.tight_layout()
    plt.savefig('shooting_method_solution_4eq.png', dpi=150)
    plt.show()