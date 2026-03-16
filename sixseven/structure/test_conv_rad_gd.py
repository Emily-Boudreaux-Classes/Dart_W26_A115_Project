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

# Hardcoded mass grid parameters
N_SHELLS  = 100          # total mass shells (log-spaced, centre to surface)
N_BURN    = 88           # first N_BURN shells burn (~inner 10% by mass with log spacing)
BURN_TIME = 1e6  # 1e16        # burn() integration time [s]; output is erg/g over this interval


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
        results = burn(temps=T_arr, rhos=rho_arr, time=BURN_TIME, comps=None)
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)
    
    # Extract energy, mu, and composition from the NetOut results
    eps = np.asarray([r.energy for r in results]) / BURN_TIME   # erg/g -> erg/g/s
    mu_burn = np.asarray([r.composition.getMeanParticleMass() for r in results])
    mass_frac = results[0].composition
    
    


    # If inputs were scalars, return scalar outputs
    if rho_is_scalar and T_is_scalar:
        eps = float(eps[0])
        mu_burn = float(mu_burn[0])

    return eps, mu_burn, mass_frac



# ----------------------------
# 4-Equation Stellar Structure (r, P, L, T)
# Variables: y = [ln(r), ln(P), L, ln(T)]
# ----------------------------
def get_eps_at_m(m, m_shells, eps_shells):
    """
    Get the nuclear energy generation rate at mass coordinate m
    by interpolating between the shell values (erg/g/s).
    """
    return float(np.interp(m, m_shells, eps_shells))


def stellar_structure_rhs_4eq(m, y, comp: Composition, kappa=0.4,
                               eps_nuc=0.0,
                               m_shells=None, eps_shells=None,
                               use_convection=True):
    """
    4-equation stellar structure in log coordinates.
    y = [ln(r), ln(P), L, ln(T)]
    
    If m_shells/eps_shells are provided, eps_nuc is interpolated from
    the shell array. Otherwise uses constant eps_nuc.
    
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
    
    # 3) Energy generation: interpolated from shell values or constant
    if m_shells is not None and eps_shells is not None:
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
                       use_convection=True, return_full=False):
    """
    Integrate 4-equation stellar structure from center to surface.
    
    N_SHELLS log-spaced mass shells from m_0 to M (whole star).
    Only the first N_BURN shells undergo nuclear burning via burn();
    the remaining shells have eps = 0.
    
    burn() returns erg/g over BURN_TIME seconds, so we divide by
    BURN_TIME to get the rate eps [erg/g/s] needed for dL/dm.
    """
    # Central density and initial radius
    rho_c = eos_rho(P_c, T_c, comp)
    r_0 = (3 * m_0 / (4 * np.pi * rho_c))**(1/3)
    
    # 1000 log-spaced shells from centre to surface
    m_shells = np.logspace(np.log10(m_0), np.log10(M), N_SHELLS)
    
    # --- Preliminary integration (eps=0) to get T, P at the burn shells ---
    m_burn_edge = m_shells[N_BURN - 1]
    y0_pre = [np.log(r_0), np.log(P_c), 0.0, np.log(T_c)]
    
    def ode_pre(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, kappa,
                                          eps_nuc=0.0,
                                          use_convection=use_convection)
    
    sol_pre = solve_ivp(ode_pre, [m_0, m_burn_edge], y0_pre, method='RK45',
                        rtol=1e-8, atol=1e-10, dense_output=True)
    
    burn_m = m_shells[:N_BURN]
    if sol_pre.success:
        y_at = sol_pre.sol(burn_m)
        P_burn = np.exp(y_at[1])
        T_burn = np.exp(y_at[3])
    else:
        P_burn = np.full(N_BURN, P_c)
        T_burn = np.full(N_BURN, T_c)
    
    rho_burn = np.array([eos_rho(Pb, Tb, comp) for Pb, Tb in zip(P_burn, T_burn)])
    
    # --- Call burn() on the N_BURN innermost shells only ---
    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            results = burn(temps=T_burn, rhos=rho_burn, time=BURN_TIME, comps=None)
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
        
        # erg/g  ->  erg/g/s
        eps_burn = np.array([r.energy for r in results]) / BURN_TIME
    except Exception:
        if return_full:
            return np.inf, None
        return np.inf
    
    # Full eps array: first N_BURN from burn(), rest = 0
    eps_shells = np.zeros(N_SHELLS)
    eps_shells[:N_BURN] = eps_burn
    
    # Initial luminosity
    L_0 = eps_shells[0] * m_0
    
    # --- Full integration (centre -> surface) ---
    y0 = [np.log(r_0), np.log(P_c), L_0, np.log(T_c)]
    
    def ode_wrapper(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, kappa,
                                         m_shells=m_shells,
                                         eps_shells=eps_shells,
                                         use_convection=use_convection)
    
    sol = solve_ivp(ode_wrapper, [m_0, M], y0, method='RK45',
                    rtol=1e-8, atol=1e-10, dense_output=True)
    
    if not sol.success:
        if return_full:
            return np.inf, None
        return np.inf
    
    r_final = np.exp(sol.y[0, -1])
    
    if return_full:
        return r_final, sol
    return r_final


def optimize_Pc_Tc_4eq(T_surf_target, P_surf_target, M, m_0, comp, 
                        P_c_guess=2.5e17, T_c_guess=1.5e7, 
                        kappa=0.4,
                        use_convection=True, verbose=True):
    """
    Optimize P_c and T_c to match surface boundary conditions
    T_surf_target and P_surf_target using the 4-equation system.
    Uses Nelder-Mead on log parameters (2 unknowns, 2 BCs).
    """
    def objective(params):
        log_P_c, log_T_c = params
        P_c = 10**log_P_c
        T_c = 10**log_T_c
        
        r_final, sol = integrate_star_4eq(P_c, T_c, M, m_0, comp, kappa,
                                           use_convection, return_full=True)
        
        if sol is None:  # integration failed
            return 1e30
        
        # Surface values at m = M
        T_surf = np.exp(sol.y[3, -1])
        P_surf = np.exp(sol.y[1, -1])
        R_surf = np.exp(sol.y[0, -1])
        L_surf = sol.y[2, -1]
        
        # Relative errors in surface T and P
        err_T = ((T_surf - T_surf_target) / T_surf_target)**2
        err_P = ((P_surf - P_surf_target) / P_surf_target)**2
        error = err_T + err_P
        
        if verbose:
            print(f"  P_c = {P_c:.3e}, T_c = {T_c:.3e}, "
                  f"R = {R_surf/R_SUN:.4f} R_sun, "
                  f"T_surf = {T_surf:.1f} K, P_surf = {P_surf:.3e}, "
                  f"L = {L_surf/L_SUN:.4f} L_sun, "
                  f"err = {np.sqrt(error)*100:.2f}%")
        
        return error
    
    x0 = [np.log10(P_c_guess), np.log10(T_c_guess)]
    
    # Explicit initial simplex so Nelder-Mead explores both Pc AND Tc
    simplex = np.array([
        x0,
        [x0[0] + 0.3, x0[1]],        # vary Pc
        [x0[0],        x0[1] + 0.3],  # vary Tc
    ])
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'initial_simplex': simplex,
                               'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 500})
    
    P_c_opt = 10**result.x[0]
    T_c_opt = 10**result.x[1]
    
    return P_c_opt, T_c_opt, result


if __name__ == "__main__":
    comp = Composition()
    M = 1.0 * M_sun
    m_0 = 1e-8 * M
    R_TARGET = 1.0 * R_SUN
    USE_CONV = True      # True = Schwarzschild criterion, False = purely radiative

    # ================================================================
    # DIAGNOSTIC: Test each step of the pipeline before optimizing
    # ================================================================
    P_c_test = 2.5e17
    T_c_test = 1.5e7
    print("=" * 80)
    print("DIAGNOSTICS: Testing pipeline with P_c={:.3e}, T_c={:.3e}".format(P_c_test, T_c_test))
    print(f"  use_convection = {USE_CONV},  N_SHELLS = {N_SHELLS},  N_BURN = {N_BURN},  BURN_TIME = {BURN_TIME}")
    print("=" * 80)

    # 1) EOS: get central density
    rho_c_test = eos_rho(P_c_test, T_c_test, comp)
    print(f"[1] EOS rho_c        = {rho_c_test:.6e} g/cm^3")
    print(f"    (global mu used   = {mu})")

    # 2) Initial radius
    r_0_test = (3 * m_0 / (4 * np.pi * rho_c_test))**(1/3)
    print(f"[2] r_0              = {r_0_test:.6e} cm  ({r_0_test/R_SUN:.6e} R_sun)")

    # 3) Multi-shell burn() — 1000 shells centre to surface, first N_BURN burn
    m_shells_test = np.logspace(np.log10(m_0), np.log10(M), N_SHELLS)
    m_burn_edge = m_shells_test[N_BURN - 1]
    
    # Preliminary integration (eps=0) to get structure at the burn shells
    r_0_pre = (3 * m_0 / (4 * np.pi * rho_c_test))**(1/3)
    y0_pre = [np.log(r_0_pre), np.log(P_c_test), 0.0, np.log(T_c_test)]
    def ode_pre_diag(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, 0.4,
                                          eps_nuc=0.0,
                                          use_convection=USE_CONV)
    sol_pre = solve_ivp(ode_pre_diag, [m_0, m_burn_edge], y0_pre, method='RK45',
                        rtol=1e-8, atol=1e-10, dense_output=True)
    
    burn_m_test = m_shells_test[:N_BURN]
    if sol_pre.success:
        y_at = sol_pre.sol(burn_m_test)
        P_burn_test = np.exp(y_at[1])
        T_burn_test = np.exp(y_at[3])
        print(f"[3] Preliminary integration (eps=0) succeeded -> P, T at {N_BURN} burn shells")
    else:
        P_burn_test = np.full(N_BURN, P_c_test)
        T_burn_test = np.full(N_BURN, T_c_test)
        print(f"[3] Preliminary integration failed, falling back to P_c, T_c for burn shells")
    
    rho_burn_test = np.array([eos_rho(Pb, Tb, comp) for Pb, Tb in zip(P_burn_test, T_burn_test)])
    print(f"    m_shells: {N_SHELLS} total, first {N_BURN} burn  (m_0/M={m_shells_test[0]/M:.4e}, m_burn_edge/M={m_burn_edge/M:.4e})")
    for i in range(N_BURN):
        print(f"    burn shell {i}: m/M={burn_m_test[i]/M:.4e}, "
              f"P={P_burn_test[i]:.4e}, T={T_burn_test[i]:.4e}, rho={rho_burn_test[i]:.4e}")
    
    print(f"    Calling burn() with {N_BURN} shells, BURN_TIME={BURN_TIME:.0e} s ...")
    try:
        results_test = burn(temps=T_burn_test, rhos=rho_burn_test, time=BURN_TIME, comps=None)
        eps_burn_test = np.array([r.energy for r in results_test]) / BURN_TIME  # erg/g -> erg/g/s
        mu_burn_test = np.array([r.composition.getMeanParticleMass() for r in results_test])
        print(f"    burn() succeeded!  (raw erg/g divided by BURN_TIME -> erg/g/s)")
        for i in range(N_BURN):
            print(f"    burn shell {i}: eps = {eps_burn_test[i]:.4e} erg/g/s, mu = {mu_burn_test[i]:.4f}")
    except Exception as e:
        print(f"    burn() FAILED: {e}")
        eps_burn_test = np.zeros(N_BURN)

    # Build full eps array (first N_BURN from burn, rest = 0)
    eps_shells_test = np.zeros(N_SHELLS)
    eps_shells_test[:N_BURN] = eps_burn_test

    # 4) Initial luminosity
    L_0_test = eps_shells_test[0] * m_0
    print(f"[4] L_0 = eps[0]*m_0 = {L_0_test:.6e} erg/s  ({L_0_test/3.828e33:.4e} L_sun)")

    # 5) Test ODE RHS at initial point + Schwarzschild check
    y0_test = [np.log(r_0_test), np.log(P_c_test), L_0_test, np.log(T_c_test)]
    rhs_test = stellar_structure_rhs_4eq(m_0, y0_test, comp, kappa=0.4,
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

    # 6) Try a short integration (just 10% of mass)
    print("[6] Test integration to 10% of M ...")
    def ode_test(m, y):
        return stellar_structure_rhs_4eq(m, y, comp, 0.4,
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
                                       use_convection=USE_CONV,
                                       return_full=False)
    print(f"    R_final          = {r_final_diag:.6e} cm  ({r_final_diag/R_SUN:.4f} R_sun)")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("SHOOTING METHOD: Optimizing Pc and Tc to match surface BCs (4eq system)")
    print(f"  use_convection = {USE_CONV},  N_SHELLS = {N_SHELLS},  N_BURN = {N_BURN}")
    print(f"  T_surf_target = {T_SURF_SUN:.1f} K,  P_surf_target = {P_SURF_SUN:.3e} dyn/cm²")
    print("=" * 80)
    print(f"Total mass: {M/M_sun:.4f} M_sun")
    print("=" * 80 + "\n")
    # Optimize Pc and Tc to match surface T and P
    Pc_opt, Tc_opt, result = optimize_Pc_Tc_4eq(T_SURF_SUN, P_SURF_SUN, M, m_0, comp,
                                                 P_c_guess=2.5e17, T_c_guess=1.5e7,
                                                 kappa=0.4,
                                                 use_convection=USE_CONV, verbose=True)
    print(f"\n>>> OPTIMAL: Pc = {Pc_opt:.4e} dyn/cm², Tc = {Tc_opt:.4e} K")
    # Integrate with optimal params
    r_final, sol = integrate_star_4eq(Pc_opt, Tc_opt, M, m_0, comp, kappa=0.4,
                                       use_convection=USE_CONV,
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
    print(f"Surface temperature: {T_sol[-1]:.4e} K  (target: {T_SURF_SUN:.1f} K, err: {(T_sol[-1]-T_SURF_SUN)/T_SURF_SUN*100:.2f}%)")
    print(f"Surface luminosity: {y_eval[2,-1]/L_SUN:.4f} L_sun")
    print(f"Target T_surf: {T_SURF_SUN:.1f} K,  Target P_surf: {P_SURF_SUN:.3e} dyn/cm²")
    print(f"P_surf error: {(P_sol[-1]-P_SURF_SUN)/P_SURF_SUN*100:.2f}%")
    print(f"T_surf error: {(T_sol[-1]-T_SURF_SUN)/T_SURF_SUN*100:.2f}%")
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