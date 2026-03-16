#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the stellar-evolution output track in HR space.

Reads sixseven/user/evolve_star_output.txt and writes:
    sixseven/user/evolve_star_hr.png

Run from the project root:
    python -m sixseven.user.plot_hr_track
"""

import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "evolve_star_output.txt")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "evolve_star_hr.png")

T_EFF_SUN = 5772.0
L_SUN_UNITS = 1.0


def load_track(path):
    rows = []
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("=") or line.startswith("-"):
                continue
            if line.startswith("STELLAR") or line.startswith("M =") or line.startswith("Each successful"):
                continue
            if line.startswith("step"):
                continue
            parts = line.split()
            if len(parts) != 12:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    if not rows:
        raise RuntimeError(f"No timestep rows found in {path}")

    data = np.array(rows)
    return {
        "step": data[:, 0],
        "dt": data[:, 1],
        "t_sim": data[:, 2],
        "t_myr": data[:, 3],
        "Pc": data[:, 4],
        "Tc": data[:, 5],
        "R": data[:, 6],
        "L": data[:, 7],
        "Teff": data[:, 8],
        "Pphot": data[:, 9],
        "Xc": data[:, 10],
        "Yc": data[:, 11],
    }


def main():
    track = load_track(INPUT_PATH)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    scatter = ax.scatter(
        track["Teff"],
        track["L"],
        c=track["t_myr"],
        cmap="viridis",
        s=36,
        edgecolors="none",
        zorder=3,
    )
    ax.plot(track["Teff"], track["L"], color="0.35", linewidth=1.2, zorder=2)

    ax.scatter(
        [T_EFF_SUN],
        [L_SUN_UNITS],
        color="goldenrod",
        edgecolors="black",
        linewidths=0.8,
        s=90,
        marker="*",
        zorder=5,
        label="Sun",
    )

    ax.annotate(
        "Sun",
        xy=(T_EFF_SUN, L_SUN_UNITS),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_xlabel("Effective Temperature [K]")
    ax.set_ylabel("Luminosity [L_sun]")
    ax.set_title("Evolution Track in HR Space")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Age [Myr]")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    print(f"Saved HR diagram to {OUTPUT_PATH}")

    # ---- Time-series panel plot ----
    panels_path = os.path.join(SCRIPT_DIR, "evolve_star_panels.png")
    fig2, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=180)

    # Core temperature vs time
    ax1 = axes[0, 0]
    ax1.plot(track["t_myr"], track["Tc"], "o-", markersize=3, color="firebrick")
    ax1.set_xlabel("Age [Myr]")
    ax1.set_ylabel("Core Temperature [K]")
    ax1.set_title("Core Temperature vs Time")
    ax1.grid(True, alpha=0.25)

    # Core hydrogen fraction vs time
    ax2 = axes[0, 1]
    ax2.plot(track["t_myr"], track["Xc"], "o-", markersize=3, color="steelblue")
    ax2.set_xlabel("Age [Myr]")
    ax2.set_ylabel("X$_c$(H-1)")
    ax2.set_title("Core Hydrogen Fraction vs Time")
    ax2.grid(True, alpha=0.25)

    # Radius vs time
    ax3 = axes[1, 0]
    ax3.plot(track["t_myr"], track["R"], "o-", markersize=3, color="darkorange")
    ax3.set_xlabel("Age [Myr]")
    ax3.set_ylabel("Radius [R$_\\odot$]")
    ax3.set_title("Radius vs Time")
    ax3.grid(True, alpha=0.25)

    # Surface luminosity vs time
    ax4 = axes[1, 1]
    ax4.plot(track["t_myr"], track["L"], "o-", markersize=3, color="darkviolet")
    ax4.set_xlabel("Age [Myr]")
    ax4.set_ylabel("Luminosity [L$_\\odot$]")
    ax4.set_title("Surface Luminosity vs Time")
    ax4.grid(True, alpha=0.25)

    fig2.tight_layout()
    fig2.savefig(panels_path)
    print(f"Saved panel plots to {panels_path}")


if __name__ == "__main__":
    main()
