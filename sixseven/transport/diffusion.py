from fipy import Grid1D, CellVariable
import numpy as np

from modules.transport.transport import transport_step
from modules.nuclear import nuc_burn
from modules.eos import eos_functions

# -----------------------------
# Grid setup
# -----------------------------

nx = 100
mesh = Grid1D(nx=nx, dx=1.0 / nx)

species = ["H1", "He4"]

# FiPy variables for each species
Y = {s: CellVariable(mesh=mesh, value=0.1) for s in species}

# -----------------------------
# Stellar structure
# -----------------------------

m = mesh.cellCenters[0].value  # "mass" coordinate
structure = {
    "m": m,
    "rho": np.ones(nx),
    "Hp": np.linspace(5e9, 5e10, nx),
    "K": np.ones(nx),
    "Cp": np.ones(nx),
    "T": 1.5e7 * np.ones(nx),  # typical MS star core temp
    "grad_rad": 0.4 * np.ones(nx),
    "grad_ad": 0.2 * np.ones(nx),
    "grad_mu": 0.01 * np.ones(nx),
    "is_convective": np.zeros(nx, dtype=bool),
}

#convective region test
structure["is_convective"][30:50] = True

# Mixing length
alpha_MLT = 1.8
structure["l_mlt"] = alpha_MLT * structure["Hp"]

# Convective velocities
v = np.zeros(nx)
v[structure["is_convective"]] = 1e2  # reduce for stability
structure["v_mlt"] = v

mu = 0.61  # mean molecular weight
P = 1e17 * np.ones(nx)  # arbitrary pressure
structure["rho"] = eos_functions.simple_eos(P, mu, structure["T"])

# -----------------------------
# Convert FiPy → ndarray
# -----------------------------

X = np.vstack([Y[s].value for s in species]).T  # shape (nx, n_species)

# -----------------------------
# Time step
# -----------------------------
# small dt to satisfy CFL for diffusion
dt = 1e-12

# Initial Sasha composition object
comp_objects = None

# -----------------------------
# Transport + Nuclear burn
# -----------------------------

X_new = transport_step(
    X,
    structure,
    dt,
    comp_objects=comp_objects,
    f_ov=0.01,
    alpha_sc=0.01,
    alpha_th=1.0,
    instantaneous_convection=True,
)

# -----------------------------
# Putting back into FiPy
# -----------------------------

for i, s in enumerate(species):
    Y[s].setValue(X_new[:, i])

# -----------------------------
# Print diagnostics
# -----------------------------
print("X_new shape:", X_new.shape)
print("X_new sample values:", X_new[30:50])  # convective region
