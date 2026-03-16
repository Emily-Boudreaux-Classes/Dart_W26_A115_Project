#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:53:44 2026

@author: george
"""

#Code written by George with Copilot assistance for troubleshooting

import os
import sys
import time


import math
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
from sixseven.nuclear.nuc_burn import *
from sixseven.timestep.timestep import dyn_timestep
from sixseven.eos import eos_functions as ef 
from sixseven.radiation.radiate import kramer_opacity

# ----------------------------
# Physical constants (cgs)
# ----------------------------
G = 6.67430e-8                 # gravitational constant [cm^3 g^-1 s^-2]
a_rad = 4 * ef.CONST.sigma_sb / ef.CONST.c_s  # radiation density constant [erg cm^-3 K^-4]
c_light = ef.CONST.c_s         # speed of light [cm s^-1]
k_B = ef.CONST.kB              # Boltzmann [erg K^-1]
m_u = 1.66053906660e-24        # atomic mass unit [g]
m_H = ef.CONST.mh              # hydrogen atom mass [g]
sigma_SB = ef.CONST.sigma_sb   # Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
pi = math.pi
M_sun = 1.98847e33
# Nominal IAU 2015 "solar units" as exact conversion factors (converted to cgs)
R_SUN = 6.957e10          # cm
L_SUN = 3.828e33          # erg / s
T_EFF_SUN = 5772.0        # K
mu = 1.004
# Solar composition (char)
X_SOLAR = 0.7381
Y_SOLAR = 0.2485
Z_SOLAR = 0.0134

# Photosphere ("surface") is definition-dependent; this is a rough characteristic value
T_SURF_SUN = T_EFF_SUN    # K
P_SURF_SUN = 3.0e4        # dyn / cm^2

# Hardcoded mass grid parameters
N_SHELLS  = 200                # total mass shells (log-spaced, centre to surface)
N_BURN    = 170                # Reduced from 175 to ~inner 5% by mass to avoid luminosity overshoot
BURN_TIME = 1e6  # 1e16        # burn() integration time [s]; output is erg/g over this interval


def polytrope_guess(M_star, R_star, mu_val, n=3):
    """
    Estimate central P_c, T_c, rho_c from an n-index polytrope (Lane-Emden)
    given total mass M_star and target radius R_star.
    
    Uses tabulated Lane-Emden constants for n = 1, 1.5, 2, 3, 4.
    Returns (P_c, T_c, rho_c).
    
    n=3 is the Eddington standard model (radiation-dominated, good for
    solar-type stars).  n=1.5 is fully convective.
    """
    # Lane-Emden constants: (xi_1, -xi_1^2 * theta'(xi_1), D_n = rho_c / rho_bar)
    LE = {
        1:   (3.14159, 3.14159,   3.290),
        1.5: (3.65375, 2.71406,   5.991),
        2:   (4.35287, 2.41105,  11.40),
        3:   (6.89685, 2.01824,  54.18),
        4:   (14.9716, 1.79723, 622.4),
    }
    xi1, neg_xi2_dtheta, D_n = LE[n]

    # Mean density
    rho_bar = 3.0 * M_star / (4.0 * pi * R_star**3)
    rho_c = D_n * rho_bar

    # Central pressure from Lane-Emden:
    #   alpha = R / xi_1
    #   P_c = 4 pi G rho_c^2 alpha^2 / (n + 1)
    alpha = R_star / xi1
    P_c = 4.0 * pi * G * rho_c**2 * alpha**2 / (n + 1)

    # Central temperature from ideal gas:  P = rho k_B T / (mu m_H)
    T_c = P_c * mu_val * m_H / (rho_c * k_B)

    return P_c, T_c, rho_c


def eos_rho(P, T, comp: Composition = None):
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
    
    # Suppress noisy KINSol output from burn() (both stdout and stderr)
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        results = burn(temps=T_arr, rhos=rho_arr, time=BURN_TIME, comps=None)
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
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


def pp_cno_eps(rho, T, X, Z):
    """
    Analytical pp-chain + CNO cycle energy generation rate (erg/g/s).
    Gamow-peak formulas (Kippenhahn, Weigert & Weiss, eqs. 18.63, 18.65).
    """
    T9 = T / 1e9
    if T9 <= 1e-4:       # below ~100 kK, no nuclear burning
        return 0.0

    # pp-chain
    g11 = 1.0 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4
    eps_pp = 2.57e4 * g11 * rho * X**2 * T9**(-2./3.) * np.exp(-3.381 / T9**(1./3.))

    # CNO cycle  (X_CNO ≈ Z/2)
    X_CNO = Z / 2.0
    g141 = 1.0 - 2.00*T9 + 3.41*T9**2 - 2.43*T9**3
    eps_cno = 8.24e25 * g141 * X_CNO * X * rho * T9**(-2./3.) * np.exp(-15.231 / T9**(1./3.))

    return eps_pp + eps_cno


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


def stellar_structure_rhs_4eq(m, y, comp: Composition,
                               m_shells=None, eps_shells=None,
                               use_convection=True, force_adiabatic=False,
                               use_analytical_eps=False):
    """
    4-equation stellar structure in log coordinates.
    y = [ln(r), ln(P), L, ln(T)]
    
    If use_analytical_eps=True, eps is computed from the analytical
    pp+CNO formula at the local (rho, T).  Otherwise eps is
    interpolated from the m_shells/eps_shells table.
    
    use_convection=True  -> Schwarzschild criterion (convective if nabla_rad > nabla_ad)
    use_convection=False -> purely radiative temperature gradient everywhere
    """
    ln_r, ln_P, L, ln_T = y
    r = np.exp(ln_r)
    P = np.exp(ln_P)
    T = np.exp(ln_T)
    
    # Get density from EOS
    rho = eos_rho(P, T, comp)
    
    # Dynamic opacity and adiabatic gradient from user libraries
    kappa = kramer_opacity(rho, T, X_SOLAR, Y_SOLAR, Z_SOLAR)
    nabla_ad = ef.nabla_ad(5.0 / 3.0)   # ideal monatomic gas: (gamma-1)/gamma = 0.4
    
    # 1) Mass conservation: d(ln r)/dm = 1 / (4π r³ ρ)
    dln_r_dm = 1.0 / (4.0 * pi * r**3 * rho)
    
    # 2) Hydrostatic equilibrium: d(ln P)/dm = -G m / (4π r⁴ P)
    dln_P_dm = -G * m / (4.0 * pi * r**4 * P)
    
    # 3) Energy generation
    if use_analytical_eps:
        dL_dm = pp_cno_eps(rho, T, X_SOLAR, Z_SOLAR)
    elif m_shells is not None:
        dL_dm = get_eps_at_m(m, m_shells, eps_shells)
    else:
        dL_dm = 0.0
    
    # 4) Temperature gradient with optional Schwarzschild criterion
    #    Radiative: d(ln T)/dm = -3 κ L / (64 π² a_rad c r⁴ T⁴)
    dln_T_dm_rad = - 3.0 * kappa * L / (64.0 * pi**2 * a_rad * c_light * r**4 * T**4)
    
    if force_adiabatic:
        dln_T_dm = nabla_ad * dln_P_dm
    elif use_convection and m > 0:
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


def integrate_star_4eq(P_c, T_c, M, m_0, comp, 
                       use_convection=True, return_full=False,
                       use_analytical_eps=False):
    """
    Integrate 4-equation stellar structure from center to surface.
    
    If use_analytical_eps=True, eps is computed on the fly from the
    analytical pp+CNO formula — no burn() call needed.
    Otherwise uses burn() on the inner N_BURN shells.
    """
    # Central density and initial radius
    rho_c = eos_rho(P_c, T_c, comp)
    r_0 = (3 * m_0 / (4 * np.pi * rho_c))**(1/3)

    # ---- Analytical eps path (no burn()) ----
    if use_analytical_eps:
        eps_c = pp_cno_eps(rho_c, T_c, X_SOLAR, Z_SOLAR)
        L_0 = eps_c * m_0
        y0 = [np.log(r_0), np.log(P_c), L_0, np.log(T_c)]

        def ode_analytical(m, y):
            return stellar_structure_rhs_4eq(m, y, comp,
                                             use_convection=use_convection,
                                             use_analytical_eps=True)

        sol = solve_ivp(ode_analytical, [m_0, M], y0, method='RK45',
                        rtol=1e-8, atol=1e-10, dense_output=True)
        if not sol.success:
            if return_full:
                return np.inf, None
            return np.inf
        r_final = np.exp(sol.y[0, -1])
        if return_full:
            return r_final, sol
        return r_final

    # ---- Original burn()-based path ----
    # Get central eps to seed a realistic preliminary T-gradient
    eps_c, _, _ = energy_generation_eps(rho_c, T_c, comp)
    L_0_guess = eps_c * m_0

    # 1000 log-spaced shells from centre to surface
    m_shells = np.logspace(np.log10(m_0), np.log10(M), N_SHELLS)
    
    # --- Preliminary integration (using eps_c) to get T, P at the burn shells ---
    m_burn_edge = m_shells[N_BURN - 1]
    y0_pre = [np.log(r_0), np.log(P_c), L_0_guess, np.log(T_c)]
    
    def ode_pre_wrapper(m, y):
        # Using a constant eps_c for the seeding gradient calculation.
        # This will set L(m) = eps_c * m.
        # stellar_structure_rhs_4eq will then automatically use nabla_rad or nabla_ad
        # based on this seeding luminosity, allowing either radiative or convective cores.
        return stellar_structure_rhs_4eq(m, y, comp, 
                                          m_shells=np.array([m_0, M]), 
                                          eps_shells=np.array([eps_c, eps_c]),
                                          use_convection=use_convection,
                                          force_adiabatic=False)
    
    sol_pre = solve_ivp(ode_pre_wrapper, [m_0, m_burn_edge], y0_pre, method='RK45',
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
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        saved_stdout = os.dup(stdout_fd)
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            results = burn(temps=T_burn, rhos=rho_burn, time=BURN_TIME, comps=None)
        finally:
            os.dup2(saved_stdout, stdout_fd)
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stdout)
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
        return stellar_structure_rhs_4eq(m, y, comp,
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
                        use_convection=True, verbose=True,
                        use_analytical_eps=False):
    """
    Optimize P_c and T_c to match surface boundary conditions
    T_surf_target and P_surf_target using the 4-equation system.
    Uses Nelder-Mead on log parameters (2 unknowns, 2 BCs).
    
    Errors are computed in LOG SPACE so that values off by orders of
    magnitude still produce a smooth, navigable landscape for Nelder-Mead.
    """
    log_T_target = np.log10(T_surf_target)
    log_P_target = np.log10(P_surf_target)
    
    def objective(params):
        log_P_c, log_T_c = params
        P_c = 10**log_P_c
        T_c = 10**log_T_c
        
        r_final, sol = integrate_star_4eq(P_c, T_c, M, m_0, comp,
                                           use_convection=use_convection, return_full=True,
                                           use_analytical_eps=use_analytical_eps)
        
        if sol is None:  # integration failed
            return 1e30
        
        # Surface values at m = M
        T_surf = np.exp(sol.y[3, -1])
        P_surf = np.exp(sol.y[1, -1])
        R_surf = np.exp(sol.y[0, -1])
        L_surf = sol.y[2, -1]
        
        # Guard against non-physical surface values
        if T_surf <= 0 or P_surf <= 0:
            return 1e30
        
        # Log-space errors — smooth landscape even when off by orders of magnitude
        err_T = (np.log10(T_surf) - log_T_target)**2
        err_P = (np.log10(P_surf) - log_P_target)**2
        error = err_T + err_P
        
        if verbose:
            print(f"  P_c = {P_c:.3e}, T_c = {T_c:.3e}, "
                  f"R = {R_surf/R_SUN:.4f} R_sun, "
                  f"T_surf = {T_surf:.1f} K, P_surf = {P_surf:.3e}, "
                  f"L = {L_surf/L_SUN:.4f} L_sun, "
                  f"err(logT)={np.sqrt(err_T):.3f} dex, err(logP)={np.sqrt(err_P):.3f} dex")
        
        return error
    
    x0 = [np.log10(P_c_guess), np.log10(T_c_guess)]
    
    # Small simplex steps (0.1 dex) to keep optimizer near the guess
    simplex = np.array([
        x0,
        [x0[0] + 0.1, x0[1]],        # vary Pc
        [x0[0],        x0[1] + 0.1],  # vary Tc
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
    USE_ANALYTICAL_EPS = True  # True = analytical pp+CNO eps, False = burn()

    # ================================================================
    # POLYTROPE INITIAL GUESS (n=3 Eddington standard model)
    # ================================================================
    Pc_poly, Tc_poly, rhoc_poly = polytrope_guess(M, R_TARGET, mu, n=3)
    print("=" * 80)
    print("POLYTROPE n=3 INITIAL GUESS")
    print("=" * 80)
    print(f"  M = {M/M_sun:.4f} M_sun,  R_target = {R_TARGET/R_SUN:.4f} R_sun")
    print(f"  P_c  = {Pc_poly:.4e} dyn/cm²")
    print(f"  T_c  = {Tc_poly:.4e} K")
    print(f"  rho_c= {rhoc_poly:.4e} g/cm³")
    print("=" * 80 + "\n")

    # ================================================================
    # DIAGNOSTIC: Test each step of the pipeline before optimizing
    # ================================================================
    P_c_test = Pc_poly
    T_c_test = Tc_poly
    print("=" * 80)
    print("DIAGNOSTICS: Testing pipeline with P_c={:.3e}, T_c={:.3e}".format(P_c_test, T_c_test))
    print(f"  use_convection = {USE_CONV},  analytical_eps = {USE_ANALYTICAL_EPS}")
    if not USE_ANALYTICAL_EPS:
        print(f"  N_SHELLS = {N_SHELLS},  N_BURN = {N_BURN},  BURN_TIME = {BURN_TIME}")
    print("=" * 80)

    # 1) EOS: get central density
    rho_c_test = eos_rho(P_c_test, T_c_test, comp)
    print(f"[1] EOS rho_c        = {rho_c_test:.6e} g/cm^3")
    print(f"    (global mu used   = {mu})")

    # 2) Initial radius
    r_0_test = (3 * m_0 / (4 * np.pi * rho_c_test))**(1/3)
    print(f"[2] r_0              = {r_0_test:.6e} cm  ({r_0_test/R_SUN:.6e} R_sun)")

    # 3) Energy generation at center
    if USE_ANALYTICAL_EPS:
        eps_c_test = pp_cno_eps(rho_c_test, T_c_test, X_SOLAR, Z_SOLAR)
        print(f"[3] Analytical pp+CNO eps_c = {eps_c_test:.6e} erg/g/s")
    else:
        m_shells_test = np.logspace(np.log10(m_0), np.log10(M), N_SHELLS)
        m_burn_edge = m_shells_test[N_BURN - 1]
        eps_c_test, _, _ = energy_generation_eps(rho_c_test, T_c_test, comp)
        L_0_test = eps_c_test * m_0
        r_0_pre = (3 * m_0 / (4 * np.pi * rho_c_test))**(1/3)
        y0_pre = [np.log(r_0_pre), np.log(P_c_test), L_0_test, np.log(T_c_test)]
        def ode_pre_diag(m, y):
            return stellar_structure_rhs_4eq(m, y, comp,
                                              m_shells=np.array([m_0, M]),
                                              eps_shells=np.array([0.0, 0.0]),
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
        print(f"    Calling burn() with {N_BURN} shells, BURN_TIME={BURN_TIME:.0e} s ...")
        try:
            results_test = burn(temps=T_burn_test, rhos=rho_burn_test, time=BURN_TIME, comps=None)
            eps_burn_test = np.array([r.energy for r in results_test]) / BURN_TIME
            print(f"    burn() succeeded!")
        except Exception as e:
            print(f"    burn() FAILED: {e}")
            eps_burn_test = np.zeros(N_BURN)
        eps_shells_test = np.zeros(N_SHELLS)
        eps_shells_test[:N_BURN] = eps_burn_test

    # 4) Initial luminosity
    L_0_test = eps_c_test * m_0
    print(f"[4] L_0 = eps_c*m_0 = {L_0_test:.6e} erg/s  ({L_0_test/L_SUN:.4e} L_sun)")

    # 5) Test ODE RHS at initial point + Schwarzschild check
    y0_test = [np.log(r_0_test), np.log(P_c_test), L_0_test, np.log(T_c_test)]
    if USE_ANALYTICAL_EPS:
        rhs_test = stellar_structure_rhs_4eq(m_0, y0_test, comp,
                                              use_convection=USE_CONV,
                                              use_analytical_eps=True)
    else:
        rhs_test = stellar_structure_rhs_4eq(m_0, y0_test, comp,
                                              m_shells=m_shells_test,
                                              eps_shells=eps_shells_test,
                                              use_convection=USE_CONV)
    rho_0 = eos_rho(P_c_test, T_c_test, comp)
    kappa_0 = kramer_opacity(rho_0, T_c_test, X_SOLAR, Y_SOLAR, Z_SOLAR)
    if m_0 > 0:
        nabla_rad_0 = (3.0 * kappa_0 * L_0_test * P_c_test) / (16.0 * pi * a_rad * c_light * G * m_0 * T_c_test**4)
    else:
        nabla_rad_0 = 0.0
    print(f"[5] ODE RHS at m_0:")
    print(f"    d(ln r)/dm       = {rhs_test[0]:.6e}")
    print(f"    d(ln P)/dm       = {rhs_test[1]:.6e}")
    print(f"    dL/dm (eps)      = {rhs_test[2]:.6e}")
    print(f"    d(ln T)/dm       = {rhs_test[3]:.6e}")
    print(f"    kappa_0          = {kappa_0:.4e} cm²/g")
    print(f"    nabla_rad        = {nabla_rad_0:.4f}  (nabla_ad = 0.4)")
    print(f"    CONVECTIVE       = {nabla_rad_0 > 0.4}  (Schwarzschild criterion)")

    # 6) Try a short integration (just 10% of mass)
    print("[6] Test integration to 10% of M ...")
    if USE_ANALYTICAL_EPS:
        def ode_test(m, y):
            return stellar_structure_rhs_4eq(m, y, comp,
                                              use_convection=USE_CONV,
                                              use_analytical_eps=True)
    else:
        def ode_test(m, y):
            return stellar_structure_rhs_4eq(m, y, comp,
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
        print(f"    L(0.1M)          = {L_10:.6e} erg/s  ({L_10/L_SUN:.4e} L_sun)")
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
    r_final_diag = integrate_star_4eq(P_c_test, T_c_test, M, m_0, comp,
                                       use_convection=USE_CONV,
                                       return_full=False,
                                       use_analytical_eps=USE_ANALYTICAL_EPS)
    print(f"    R_final          = {r_final_diag:.6e} cm  ({r_final_diag/R_SUN:.4f} R_sun)")
    print("=" * 80 + "\n")

    print("=" * 80)
    print("SHOOTING METHOD: Optimizing Pc and Tc to match surface BCs (4eq system)")
    print(f"  use_convection = {USE_CONV},  analytical_eps = {USE_ANALYTICAL_EPS}")
    print(f"  T_surf_target = {T_SURF_SUN:.1f} K,  P_surf_target = {P_SURF_SUN:.3e} dyn/cm²")
    print("=" * 80)
    print(f"Total mass: {M/M_sun:.4f} M_sun")
    print("=" * 80 + "\n")
    # Optimize Pc and Tc to match surface T and P
    Pc_opt, Tc_opt, result = optimize_Pc_Tc_4eq(T_SURF_SUN, P_SURF_SUN, M, m_0, comp,
                                                 P_c_guess=Pc_poly, T_c_guess=Tc_poly,
                                                 use_convection=USE_CONV, verbose=True,
                                                 use_analytical_eps=USE_ANALYTICAL_EPS)
    print(f"\n>>> OPTIMAL: Pc = {Pc_opt:.4e} dyn/cm², Tc = {Tc_opt:.4e} K")
    # Integrate with optimal params
    r_final, sol = integrate_star_4eq(Pc_opt, Tc_opt, M, m_0, comp, 
                                       use_convection=USE_CONV,
                                       return_full=True,
                                       use_analytical_eps=USE_ANALYTICAL_EPS)
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

    # Compute nabla_rad, nabla_ad, and kappa at every evaluation point
    nabla_rad_sol = np.zeros(len(m_eval))
    nabla_ad_sol  = np.zeros(len(m_eval))
    kappa_sol     = np.zeros(len(m_eval))
    for _i in range(len(m_eval)):
        _rho = eos_rho(P_sol[_i], T_sol[_i], comp)
        _kap = kramer_opacity(_rho, T_sol[_i], X_SOLAR, Y_SOLAR, Z_SOLAR)
        kappa_sol[_i] = _kap
        _nad = ef.nabla_ad(5.0 / 3.0)   # ideal monatomic gas
        nabla_ad_sol[_i] = _nad
        if m_eval[_i] > 0:
            nabla_rad_sol[_i] = (3.0 * _kap * L_sol[_i] * P_sol[_i]) / \
                                 (16.0 * pi * a_rad * c_light * G * m_eval[_i] * T_sol[_i]**4)
        else:
            nabla_rad_sol[_i] = 0.0

    # Plot vs m/M
    q = m_eval / M  # fractional mass coordinate
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
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

    # --- Convection vs Radiation Plot ---
    grad_rad_pts = []
    grad_ad_pts = []
    for idx_p in range(len(m_eval)):
        rho_p = eos_rho(P_sol[idx_p], T_sol[idx_p], comp)
        kap_p = kramer_opacity(rho_p, T_sol[idx_p], X_SOLAR, Y_SOLAR, Z_SOLAR)
        n_ad_p = ef.nabla_ad(5.0 / 3.0)   # ideal monatomic gas
        if m_eval[idx_p] > 0:
            n_rad_p = (3.0 * kap_p * L_sol[idx_p] * P_sol[idx_p]) / (16.0 * pi * a_rad * c_light * G * m_eval[idx_p] * T_sol[idx_p]**4)
        else:
            n_rad_p = 0.0
        grad_rad_pts.append(n_rad_p)
        grad_ad_pts.append(n_ad_p)
    
    grad_rad_pts = np.array(grad_rad_pts)
    grad_ad_pts = np.array(grad_ad_pts)
    
    axes[2, 0].plot(q, grad_rad_pts, 'r-', label="nabla_rad")
    axes[2, 0].plot(q, grad_ad_pts, 'b--', label="nabla_ad")
    axes[2, 0].set_yscale('log')
    axes[2, 0].set_xlabel('m / M')
    axes[2, 0].set_ylabel('Gradient Value')
    axes[2, 0].set_title('Stability Analysis')
    axes[2, 0].legend()
    
    is_conv_zone = (grad_rad_pts > grad_ad_pts).astype(float)
    axes[2, 1].fill_between(q, 0, is_conv_zone, color='orange', alpha=0.3, label='Convective Zone')
    axes[2, 1].set_xlabel('m / M')
    axes[2, 1].set_yticks([0, 1])
    axes[2, 1].set_yticklabels(['Radiative', 'Convective'])
    axes[2, 1].set_title('Transport Mode')
    axes[2, 1].legend()
    
    plt.tight_layout()
    plotpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shooting_method_solution_4eq.png')
    plt.savefig(plotpath, dpi=150)
    print(f"\n>>> Plot saved to {plotpath}")
    plt.show()

    # ================================================================
    # WRITE ALL KEY OUTPUT TO .txt FILE (immune to burn() thread spam)
    # ================================================================
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_conv_182_output.txt')
    with open(outpath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POLYTROPE n=3 INITIAL GUESS\n")
        f.write("=" * 80 + "\n")
        f.write(f"  M = {M/M_sun:.4f} M_sun,  R_target = {R_TARGET/R_SUN:.4f} R_sun\n")
        f.write(f"  P_c  = {Pc_poly:.4e} dyn/cm²\n")
        f.write(f"  T_c  = {Tc_poly:.4e} K\n")
        f.write(f"  rho_c= {rhoc_poly:.4e} g/cm³\n")
        f.write("=" * 80 + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("SOLUTION SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Central pressure: {Pc_opt:.4e} dyn/cm²\n")
        f.write(f"Central temperature: {Tc_opt:.4e} K\n")
        f.write(f"Central density: {eos_rho(Pc_opt, Tc_opt, comp):.4e} g/cm³\n")
        f.write(f"Central radius: {r_sol[0]:.4e} cm = {r_sol[0]/R_SUN:.6f} R_sun\n")
        f.write(f"Surface radius: {r_sol[-1]:.4e} cm = {r_sol[-1]/R_SUN:.4f} R_sun\n")
        f.write(f"Surface pressure: {P_sol[-1]:.4e} dyn/cm²\n")
        f.write(f"Surface temperature: {T_sol[-1]:.4e} K  (target: {T_SURF_SUN:.1f} K, err: {(T_sol[-1]-T_SURF_SUN)/T_SURF_SUN*100:.2f}%)\n")
        f.write(f"Surface luminosity: {L_sol[-1]/L_SUN:.4f} L_sun\n")
        f.write(f"Target T_surf: {T_SURF_SUN:.1f} K,  Target P_surf: {P_SURF_SUN:.3e} dyn/cm²\n")
        f.write(f"P_surf error: {(P_sol[-1]-P_SURF_SUN)/P_SURF_SUN*100:.2f}%\n")
        f.write(f"T_surf error: {(T_sol[-1]-T_SURF_SUN)/T_SURF_SUN*100:.2f}%\n")
        f.write("=" * 80 + "\n\n")

        hdr = (f"{'Shell':>6} {'m/M_sun':>12} {'r/R_sun':>12} {'P [dyn/cm²]':>14} "
               f"{'T [K]':>12} {'rho [g/cm³]':>14} {'L [erg/s]':>14} "
               f"{'κ [cm²/g]':>12} {'∇_rad':>10} {'∇_ad':>8} {'mode':>8}\n")
        sep = "=" * 155 + "\n"
        dash = "-" * 155 + "\n"

        f.write(sep)
        f.write("FIRST 10 SHELLS (near center)\n")
        f.write(sep)
        f.write(hdr)
        f.write(dash)
        for i in range(min(10, len(m_eval))):
            mode = "CONV" if nabla_rad_sol[i] > nabla_ad_sol[i] else "rad"
            f.write(f"{i:>6} {m_eval[i]/M_sun:>12.6e} {r_sol[i]/R_SUN:>12.6e} {P_sol[i]:>14.4e} {T_sol[i]:>12.4e} "
                    f"{rho_sol[i]:>14.4e} {L_sol[i]:>14.4e} {kappa_sol[i]:>12.4e} {nabla_rad_sol[i]:>10.4f} {nabla_ad_sol[i]:>8.4f} {mode:>8}\n")
        f.write(sep + "\n")

        f.write(sep)
        f.write("LAST 10 SHELLS (near surface)\n")
        f.write(sep)
        f.write(hdr)
        f.write(dash)
        for i in range(max(0, len(m_eval)-10), len(m_eval)):
            mode = "CONV" if nabla_rad_sol[i] > nabla_ad_sol[i] else "rad"
            f.write(f"{i:>6} {m_eval[i]/M_sun:>12.6e} {r_sol[i]/R_SUN:>12.6e} {P_sol[i]:>14.4e} {T_sol[i]:>12.4e} "
                    f"{rho_sol[i]:>14.4e} {L_sol[i]:>14.4e} {kappa_sol[i]:>12.4e} {nabla_rad_sol[i]:>10.4f} {nabla_ad_sol[i]:>8.4f} {mode:>8}\n")
        f.write(sep + "\n")

    print(f"\n>>> Output saved to {outpath}")