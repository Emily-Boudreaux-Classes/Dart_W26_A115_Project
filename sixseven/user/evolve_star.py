#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-evolve a 1 M_sun stellar model.

Each successful timestep does:
1. burn the inner burn shells with the previous-step composition,
2. mix those shells with the transport module,
3. re-integrate the stellar structure using the updated composition,
4. append the timestep to the text log and save a restart checkpoint.

Run from the project root:
    python -m sixseven.user.evolve_star
"""

import gc
import os
import sys

import numpy as np

from sixseven.eos import eos_functions as ef
from sixseven.nuclear.nuc_burn import burn, init_composition
from sixseven.radiation.radiate import kramer_opacity
from sixseven.structure.test_conv_182 import (
    G,
    L_SUN,
    M_sun,
    N_BURN,
    P_SURF_SUN,
    R_SUN,
    X_SOLAR,
    Y_SOLAR,
    Z_SOLAR,
    a_rad,
    c_light,
    eos_rho,
    integrate_star_4eq,
    optimize_Pc_Tc_4eq,
    k_B,
    m_H,
    mu,
    pi,
    sigma_SB,
)
from sixseven.timestep.timestep import dyn_timestep
from sixseven.transport.transport_simple import apply_diffusion, compute_diffusion_coefficients

SPECIES = ["H-1", "He-4", "C-12"]

SEC_PER_YEAR = 3.156e7
SEC_PER_MYR = 3.156e13
SEC_PER_GYR = 3.156e16

TARGET_AGE_GYR = 1.0
TARGET_TIME = TARGET_AGE_GYR * SEC_PER_GYR
T_EFF_INIT_TARGET = 5772.0
DT_INIT = 1e14
DT_MIN = 1e12
DT_MAX = 2e14
HFACTOR = 1e15
N_EVAL = 400
N_EVOLVE_BURN = N_BURN
MAX_RETRIES = 4
MAX_STEPS = 5000
OPT_MAXITER = 50
OPT_MAX_ERR = 5e-3
MAX_STEP_PC_FRAC = 0.20
MAX_STEP_TC_FRAC = 0.20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "evolve_star_output.txt")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "evolve_star_checkpoint.npz")


def suppress_output():
    out_fd = sys.stdout.fileno()
    err_fd = sys.stderr.fileno()
    saved_out = os.dup(out_fd)
    saved_err = os.dup(err_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, out_fd)
    os.dup2(devnull, err_fd)
    os.close(devnull)
    return saved_out, saved_err


def restore_output(saved_out, saved_err):
    os.dup2(saved_out, sys.stdout.fileno())
    os.dup2(saved_err, sys.stderr.fileno())
    os.close(saved_out)
    os.close(saved_err)


def build_structure_dict(m_arr, r, P, T, rho, L, kappa, nabla_rad, nabla_ad):
    g_local = G * m_arr / np.maximum(r, 1.0) ** 2
    Hp = P / (rho * np.maximum(g_local, 1e-30))
    is_conv = nabla_rad > nabla_ad
    flux = L / (4.0 * pi * np.maximum(r, 1.0) ** 2)
    v_mlt = np.where(is_conv, np.cbrt(flux / np.maximum(rho, 1e-99)), 0.0)
    grad_mu = np.full_like(m_arr, 1e-6)
    K_cond = 16.0 * sigma_SB * T ** 3 / (
        3.0 * np.maximum(kappa, 0.01) * np.maximum(rho, 1e-99)
    )
    Cp = (5.0 / 2.0) * k_B / (mu * m_H) * np.ones_like(m_arr)
    return {
        "m": m_arr,
        "Hp": Hp,
        "v_mlt": v_mlt,
        "is_convective": is_conv,
        "grad_rad": nabla_rad,
        "grad_ad": nabla_ad,
        "grad_mu": grad_mu,
        "K": K_cond,
        "Cp": Cp,
        "rho": rho,
        "T": T,
    }


def evaluate_model(sol, M, m_0):
    m_eval = np.logspace(np.log10(m_0), np.log10(M), N_EVAL)
    m_eval[0] = m_0
    m_eval[-1] = M
    y = sol.sol(m_eval)
    r = np.exp(y[0])
    P = np.exp(y[1])
    L = y[2]
    T = np.exp(y[3])
    rho = np.array([eos_rho(Pi, Ti) for Pi, Ti in zip(P, T)])
    kappa = np.array([
        kramer_opacity(ri, Ti, X_SOLAR, Y_SOLAR, Z_SOLAR)
        for ri, Ti in zip(rho, T)
    ])
    nad = ef.nabla_ad(5.0 / 3.0)
    nabla_ad = np.full(N_EVAL, nad)
    nabla_rad = np.zeros(N_EVAL)
    for i in range(N_EVAL):
        if m_eval[i] > 0:
            nabla_rad[i] = (3.0 * kappa[i] * L[i] * P[i]) / (
                16.0 * pi * a_rad * c_light * G * m_eval[i] * T[i] ** 4
            )
    return m_eval, r, P, T, rho, L, kappa, nabla_rad, nabla_ad


def eddington_bc(r_surf, L_surf, M):
    T_eff = (L_surf / (4.0 * pi * r_surf ** 2 * sigma_SB)) ** 0.25
    g_surf = G * M / r_surf ** 2
    kap = kramer_opacity(eos_rho(P_SURF_SUN, T_eff), T_eff, X_SOLAR, Y_SOLAR, Z_SOLAR)
    for _ in range(3):
        P_phot = (2.0 / 3.0) * g_surf / kap
        kap = kramer_opacity(eos_rho(P_phot, T_eff), T_eff, X_SOLAR, Y_SOLAR, Z_SOLAR)
    P_phot = (2.0 / 3.0) * g_surf / kap
    return T_eff, P_phot


def print_summary(step_num, dt, Pc, Tc, R, L, T_eff, P_phot):
    print(
        f"  step {step_num:>4d}:  dt={dt:.2e} s  "
        f"Pc={Pc:.3e}  Tc={Tc:.3e}  "
        f"R={R / R_SUN:.4f} R_sun  L={L / L_SUN:.4f} L_sun  "
        f"Teff={T_eff:.1f} K  Pphot={P_phot:.2e}"
    )


def build_composition_array(n_shells):
    comp0 = init_composition()
    X_comp = np.zeros((n_shells, len(SPECIES)))
    for j, sym in enumerate(SPECIES):
        X_comp[:, j] = comp0.getMolarAbundance(sym)
    mu_center = comp0.getMeanParticleMass()
    return X_comp, mu_center


def build_composition_list(X_comp):
    comps = []
    for i in range(len(X_comp)):
        comp = init_composition()
        for j, sym in enumerate(SPECIES):
            comp.setMolarAbundance(sym, X_comp[i, j])
        comps.append(comp)
    return comps


def write_output_header(mode):
    with open(OUTPUT_PATH, mode) as f:
        if mode == "w":
            f.write("=" * 120 + "\n")
            f.write("STELLAR EVOLUTION TIMESTEPS\n")
            f.write("=" * 120 + "\n")
            f.write(
                f"M = 1.0 M_sun, target age = {TARGET_AGE_GYR:.3f} Gyr, "
                f"N_EVOLVE_BURN = {N_EVOLVE_BURN}\n"
            )
            f.write("Each successful timestep is appended immediately.\n")
            f.write("=" * 120 + "\n\n")
            hdr = (
                f"{'step':>4s}  {'dt [s]':>12s}  {'t_sim [s]':>12s}  {'t [Myr]':>10s}  "
                f"{'Pc [dyn/cm2]':>12s}  {'Tc [K]':>12s}  {'R/R_sun':>10s}  "
                f"{'L/L_sun':>10s}  {'T_eff [K]':>10s}  {'P_phot':>12s}  "
                f"{'X_c(H1)':>10s}  {'Y_c(He4)':>10s}\n"
            )
            f.write(hdr)
            f.write("-" * 120 + "\n")


def append_output_line(step, dt, t_sim, Pc, Tc, R, L, T_eff, P_phot, X_c, Y_c):
    line = (
        f"{step:>4d}  {dt:>12.4e}  {t_sim:>12.4e}  {t_sim / SEC_PER_MYR:>10.4f}  "
        f"{Pc:>12.4e}  {Tc:>12.4e}  {R / R_SUN:>10.4f}  "
        f"{L / L_SUN:>10.4f}  {T_eff:>10.1f}  {P_phot:>12.4e}  "
        f"{X_c:>10.6f}  {Y_c:>10.6f}"
    )
    with open(OUTPUT_PATH, "a") as f:
        f.write(line + "\n")
        f.flush()
    return line


def save_checkpoint(
    step,
    t_sim,
    dt,
    Pc_opt,
    Tc_opt,
    mu_center,
    m_eval,
    r,
    P,
    T,
    rho,
    L,
    kappa,
    nabla_rad,
    nabla_ad,
    X_comp,
    u_prev,
    T_eff,
    P_phot,
):
    np.savez_compressed(
        CHECKPOINT_PATH,
        step=step,
        t_sim=t_sim,
        dt=dt,
        Pc_opt=Pc_opt,
        Tc_opt=Tc_opt,
        mu_center=mu_center,
        m_eval=m_eval,
        r=r,
        P=P,
        T=T,
        rho=rho,
        L=L,
        kappa=kappa,
        nabla_rad=nabla_rad,
        nabla_ad=nabla_ad,
        X_comp=X_comp,
        u_prev=u_prev,
        T_eff=T_eff,
        P_phot=P_phot,
    )


def load_checkpoint():
    data = np.load(CHECKPOINT_PATH)
    return {
        "step": int(data["step"]),
        "t_sim": float(data["t_sim"]),
        "dt": float(data["dt"]),
        "Pc_opt": float(data["Pc_opt"]),
        "Tc_opt": float(data["Tc_opt"]),
        "mu_center": float(data["mu_center"]),
        "m_eval": data["m_eval"],
        "r": data["r"],
        "P": data["P"],
        "T": data["T"],
        "rho": data["rho"],
        "L": data["L"],
        "kappa": data["kappa"],
        "nabla_rad": data["nabla_rad"],
        "nabla_ad": data["nabla_ad"],
        "X_comp": data["X_comp"],
        "u_prev": data["u_prev"],
        "T_eff": float(data["T_eff"]),
        "P_phot": float(data["P_phot"]),
    }


def burn_and_mix_step(T, rho, m_eval, r, P, L, kappa, nabla_rad, nabla_ad, X_comp, dt, mu_center):
    burn_comps = build_composition_list(X_comp)
    saved = suppress_output()
    try:
        burn_results = burn(
            temps=T[:N_EVOLVE_BURN],
            rhos=rho[:N_EVOLVE_BURN],
            time=dt,
            comps=burn_comps,
        )
    finally:
        restore_output(*saved)

    X_next = X_comp.copy()
    mu_burn = np.full(N_EVOLVE_BURN, mu_center)
    for i, res in enumerate(burn_results):
        try:
            mu_burn[i] = res.composition.getMeanParticleMass()
            for j, sym in enumerate(SPECIES):
                X_next[i, j] = res.composition.getMolarAbundance(sym)
        except RuntimeError:
            pass

    del burn_results, burn_comps
    gc.collect()

    burn_structure = build_structure_dict(
        m_eval[:N_EVOLVE_BURN],
        r[:N_EVOLVE_BURN],
        P[:N_EVOLVE_BURN],
        T[:N_EVOLVE_BURN],
        rho[:N_EVOLVE_BURN],
        L[:N_EVOLVE_BURN],
        kappa[:N_EVOLVE_BURN],
        nabla_rad[:N_EVOLVE_BURN],
        nabla_ad[:N_EVOLVE_BURN],
    )
    D = compute_diffusion_coefficients(burn_structure)
    X_next = apply_diffusion(X_next, D, m_eval[:N_EVOLVE_BURN], dt)
    return X_next, mu_burn[0]


def solve_structure(Pc_guess, Tc_guess, T_eff_target, P_phot_target, M, m_0, X_comp):
    burn_comps = build_composition_list(X_comp)
    central_comp = burn_comps[0]
    saved = suppress_output()
    try:
        Pc_opt, Tc_opt, opt_result = optimize_Pc_Tc_4eq(
            T_eff_target,
            P_phot_target,
            M,
            m_0,
            central_comp,
            P_c_guess=Pc_guess,
            T_c_guess=Tc_guess,
            use_convection=True,
            verbose=False,
            burn_comps=burn_comps,
            maxiter=OPT_MAXITER,
        )

        # Guardrail: reject optimizer outputs that clearly miss the target.
        # objective = (Δlog10 Teff)^2 + (Δlog10 Pphot)^2.
        # Also reject if optimizer hit maxiter and remains far from target.
        if (not np.isfinite(opt_result.fun)) or (opt_result.fun > OPT_MAX_ERR):
            return None

        # Guardrail: avoid jumping to a different solution branch in one step.
        # If optimizer moves central values too far from the continuation guess,
        # force outer loop retry with smaller dt.
        if abs(Pc_opt - Pc_guess) / max(Pc_guess, 1e-99) > MAX_STEP_PC_FRAC:
            return None
        if abs(Tc_opt - Tc_guess) / max(Tc_guess, 1e-99) > MAX_STEP_TC_FRAC:
            return None

        _, sol = integrate_star_4eq(
            Pc_opt,
            Tc_opt,
            M,
            m_0,
            central_comp,
            use_convection=True,
            return_full=True,
            burn_comps=burn_comps,
        )
    finally:
        restore_output(*saved)

    if sol is None:
        return None

    state = evaluate_model(sol, M, m_0)
    del sol, burn_comps
    gc.collect()
    return Pc_opt, Tc_opt, state


def main():
    M = 1.0 * M_sun
    m_0 = 1e-8 * M

    if os.path.exists(CHECKPOINT_PATH):
        state = load_checkpoint()
        step = state["step"]
        t_sim = state["t_sim"]
        dt = state["dt"]
        Pc_opt = state["Pc_opt"]
        Tc_opt = state["Tc_opt"]
        mu_center = state["mu_center"]
        m_eval = state["m_eval"]
        r = state["r"]
        P = state["P"]
        T = state["T"]
        rho = state["rho"]
        L = state["L"]
        kappa = state["kappa"]
        nabla_rad = state["nabla_rad"]
        nabla_ad = state["nabla_ad"]
        X_comp = state["X_comp"]
        u_prev = state["u_prev"]
        T_eff = state["T_eff"]
        P_phot = state["P_phot"]
        print("=" * 80)
        print("RESUMING FROM CHECKPOINT")
        print(f"  step = {step},  t = {t_sim / SEC_PER_MYR:.3f} Myr")
        print(f"  Pc = {Pc_opt:.4e}, Tc = {Tc_opt:.4e}, Teff = {T_eff:.1f} K")
        print("=" * 80)
    else:
        step = 0
        t_sim = 0.0
        dt = DT_INIT
        Pc_opt = 1.9505e17
        Tc_opt = 1.4341e07
        print("=" * 80)
        print("STEP 0: Loading initial structure")
        print(f"  Pc = {Pc_opt:.4e}, Tc = {Tc_opt:.4e}  (from test_conv_182)")
        print("=" * 80)
        X_comp, mu_center = build_composition_array(N_EVOLVE_BURN)
        solved = solve_structure(Pc_opt, Tc_opt, T_EFF_INIT_TARGET, P_SURF_SUN, M, m_0, X_comp)
        if solved is None:
            raise RuntimeError("Initial structure solve failed.")
        Pc_opt, Tc_opt, structure = solved
        m_eval, r, P, T, rho, L, kappa, nabla_rad, nabla_ad = structure
        T_eff, P_phot = eddington_bc(r[-1], L[-1], M)
        u_prev = np.array([T, rho])
        write_output_header("w")
        append_output_line(
            0,
            0.0,
            0.0,
            Pc_opt,
            Tc_opt,
            r[-1],
            L[-1],
            T_eff,
            P_phot,
            X_comp[0, 0],
            X_comp[0, 1],
        )
        save_checkpoint(
            step,
            t_sim,
            dt,
            Pc_opt,
            Tc_opt,
            mu_center,
            m_eval,
            r,
            P,
            T,
            rho,
            L,
            kappa,
            nabla_rad,
            nabla_ad,
            X_comp,
            u_prev,
            T_eff,
            P_phot,
        )
        print_summary(0, 0.0, Pc_opt, Tc_opt, r[-1], L[-1], T_eff, P_phot)

    print("\n" + "=" * 80)
    print(f"TIME EVOLUTION TO {TARGET_AGE_GYR:.3f} Gyr")
    print("=" * 80)

    if not os.path.exists(OUTPUT_PATH):
        write_output_header("w")

    while t_sim < TARGET_TIME and step < MAX_STEPS:
        dt_trial = min(dt, DT_MAX, TARGET_TIME - t_sim)
        state_before = (
            Pc_opt,
            Tc_opt,
            mu_center,
            m_eval.copy(),
            r.copy(),
            P.copy(),
            T.copy(),
            rho.copy(),
            L.copy(),
            kappa.copy(),
            nabla_rad.copy(),
            nabla_ad.copy(),
            X_comp.copy(),
            u_prev.copy(),
        )

        success = False
        for _ in range(MAX_RETRIES):
            try:
                X_next, mu_new = burn_and_mix_step(
                    T,
                    rho,
                    m_eval,
                    r,
                    P,
                    L,
                    kappa,
                    nabla_rad,
                    nabla_ad,
                    X_comp,
                    dt_trial,
                    mu_center,
                )
                mu_ratio = mu_new / mu_center if (mu_new > 0 and mu_center > 0) else 1.0
                Pc_guess = Pc_opt * mu_ratio ** 2
                Tc_guess = Tc_opt * mu_ratio
                solved = solve_structure(
                    Pc_guess,
                    Tc_guess,
                    T_eff,
                    P_phot,
                    M,
                    m_0,
                    X_next,
                )
                if solved is None:
                    raise RuntimeError("structure solve failed")
                Pc_new, Tc_new, structure = solved
                m_eval, r, P, T, rho, L, kappa, nabla_rad, nabla_ad = structure
                T_eff, P_phot = eddington_bc(r[-1], L[-1], M)
                if not np.isfinite(T_eff) or not np.isfinite(P_phot):
                    raise RuntimeError("non-finite surface state")
                Pc_opt = Pc_new
                Tc_opt = Tc_new
                mu_center = mu_new
                X_comp = X_next
                success = True
                break
            except Exception:
                dt_trial *= 0.5
                if dt_trial < DT_MIN:
                    break
                (
                    Pc_opt,
                    Tc_opt,
                    mu_center,
                    m_eval,
                    r,
                    P,
                    T,
                    rho,
                    L,
                    kappa,
                    nabla_rad,
                    nabla_ad,
                    X_comp,
                    u_prev,
                ) = state_before

        if not success:
            print(f"  step {step + 1}: failed after retries; last dt={dt_trial:.2e} s")
            break

        t_sim += dt_trial
        step += 1
        X_c = X_comp[0, 0]
        Y_c = X_comp[0, 1]
        print_summary(step, dt_trial, Pc_opt, Tc_opt, r[-1], L[-1], T_eff, P_phot)
        print(
            f"           t_sim = {t_sim:.3e} s = {t_sim / SEC_PER_MYR:.3f} Myr"
            f"   X_c(H1)={X_c:.6f}  Y_c(He4)={Y_c:.6f}"
        )
        append_output_line(
            step,
            dt_trial,
            t_sim,
            Pc_opt,
            Tc_opt,
            r[-1],
            L[-1],
            T_eff,
            P_phot,
            X_c,
            Y_c,
        )
        u_new = np.array([T, rho])
        du = u_new - u_prev
        dt, _, _ = dyn_timestep(u_new, du, dt_trial, hfactor=HFACTOR, min_step=DT_MIN)
        dt = min(max(dt, DT_MIN), DT_MAX)
        u_prev = u_new
        gc.collect()
        save_checkpoint(
            step,
            t_sim,
            dt,
            Pc_opt,
            Tc_opt,
            mu_center,
            m_eval,
            r,
            P,
            T,
            rho,
            L,
            kappa,
            nabla_rad,
            nabla_ad,
            X_comp,
            u_prev,
            T_eff,
            P_phot,
        )

    print(f"\n>>> Output saved to {OUTPUT_PATH}")
    print(f">>> Checkpoint saved to {CHECKPOINT_PATH}")
    print("\n" + "=" * 80)
    print("EVOLUTION STATUS")
    print("=" * 80)
    print(f"  Evolved time: {t_sim:.3e} s = {t_sim / SEC_PER_MYR:.3f} Myr")
    print(f"  Target time : {TARGET_TIME:.3e} s = {TARGET_AGE_GYR:.3f} Gyr")
    print(f"  Final Pc    = {Pc_opt:.4e} dyn/cm²")
    print(f"  Final Tc    = {Tc_opt:.4e} K")
    print(f"  Final R     = {r[-1] / R_SUN:.4f} R_sun")
    print(f"  Final L     = {L[-1] / L_SUN:.4f} L_sun")
    print(f"  Final Teff  = {T_eff:.1f} K")
    print("=" * 80)


if __name__ == "__main__":
    main()
