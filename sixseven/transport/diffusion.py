from fipy import Grid1D, CellVariable, FaceVariable, TransientTerm, DiffusionTerm
import numpy as np

from test_transport import get_diffusion_profile

# This is a stand in for your more robust diffusion function
#def Dumnmy_get_diffusion(nx):
   # D = np.ones(nx) *1e-2
    #D[int(0.3*nx):int(0.5*nx)] += 1e-2 # Very approximate effect of a convective zone
    #return D

# Setting up grid / discritization. Most of this will be informed from inputs you recive from other modules
m_min, m_max = 0, 1.0  
nx = 100

# If you choose to use FiPy then you will need to work in its framework, this means you need to take the grid of 
# masses george has generated and turn them into a Grid1D

# There are alternate Grid1D constructors which let you take in a preexisting grid, try to look these up in the
# FiPy documentation if you decide to go this route
mesh = Grid1D(nx=nx, dx=(m_max - m_min) / nx)

# These are the initial conditions, I have done some very approximate stuff here to make an interesting
# demo case. Generally you will recive these from either george or sasha
species = ['H1', 'He4']
Y = {s: CellVariable(mesh=mesh, value=0.1, hasOld=True) for s in species}

initial_H1 = np.where(mesh.cellCenters[0] < 0.5, 0.7, 0.2)
Y['H1'].setValue(initial_H1)

initial_He4 = np.where(mesh.cellCenters[0] < 0.5, 0.3, 0.8)
Y['He4'].setValue(initial_He4)

# I am using this python module FiPy, its a Finite Volume PDE solver
# Finite volumes are good for this class of problem as they deal with conservation laws inherently
# and tend to be more numerically stable than finite difference. 

# These are variables which are different for each cell. Note how I am just using numpy arrays as the
# values. You already calculate D so you might use that as the value
r_cell = CellVariable(mesh=mesh, value=np.linspace(0.1, 1.0, nx))
rho_cell = CellVariable(mesh=mesh, value=np.ones(nx))
# ----------------------------------
# Construct structure dict for transport
# ----------------------------------

m = mesh.cellCenters[0].value
r = r_cell.value

Hp = np.linspace(5e9, 5e10, nx)

grad_rad = 0.4 * np.ones(nx)
grad_ad  = 0.2 * np.ones(nx)
grad_mu  = 0.01 * np.ones(nx)

is_convective = np.zeros(nx, dtype=bool)
is_convective[int(0.3*nx):int(0.5*nx)] = True

alpha_MLT = 1.8
l_mlt = alpha_MLT * Hp

v_mlt = np.zeros(nx)
superadiabatic = grad_rad - grad_ad
v_mlt[is_convective] = 3e5 * np.sqrt(
    np.maximum(superadiabatic[is_convective], 1e-4)
)

structure = {
    "m": m,
    "r": r,
    "rho": rho_cell.value,
    "Hp": Hp,
    "K": np.ones(nx),
    "Cp": np.ones(nx),
    "grad_rad": grad_rad,
    "grad_ad": grad_ad,
    "grad_mu": grad_mu,
    "is_convective": is_convective,
    "l_mlt": l_mlt,
    "v_mlt": v_mlt,
}

# ----------------------------------
# Call YOUR diffusion module
# ----------------------------------

D_profile = get_diffusion_profile(structure)

D_cell = CellVariable(mesh=mesh, value=D_profile)

A_face = (4 * np.pi * r_cell.faceValue**2 * rho_cell.faceValue)**2 * D_cell.faceValue

eqs = {s: TransientTerm() == DiffusionTerm(coeff=A_face) for s in species}



# This is just some plotting code to make visualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0, m_max)
ax.set_ylim(0, 1.1)
ax.set_xlabel('Mass Coordinate (m)')
ax.set_ylabel('Molar Abundance (Yi)')
ax.set_title('Species Abundance Evolution')

lines = {}
for name in species:
    line, = ax.plot(mesh.cellCenters[0], Y[name].value, label=name, lw=2)
    lines[name] = line

ax.legend()

# Diffusion timescales are very different than nuclear timescales, therefore it is often
# useful to solve these systems seperately (operator-splitting) using a much finer timestep
dt = 1e-1

# In this case I am letting matplotlib handle the timestepping loop, this is useful for animation
# however, not great for actual code. You could replace this by just the inner for loop
def update(frame):
    for s in species: # Solve indeendencly for each species. This removes cross-diffusion
        # Here you might do something like 
        # new_abundances = nuclear(...)
        # Y[s].setValue(new_abundances)

        # Some complexity to think about here, though using
        # a coroutine may make this a bit simpler to implement

        Y[s].updateOld()
        eqs[s].solve(var=Y[s], dt=dt)
        
        lines[s].set_ydata(Y[s].value)
    
    return list(lines.values())

animation = FuncAnimation(fig, update, frames=2000, interval=50, blit=True)

plt.show()

# Some things to try
#   1. Play around with getting your Diffision Function in there
#   2. See how changing the timestep size affects results
#   3. Try to get sashs outputs into here (they will come in as R_nuc) and then to return new abundances to george
#   4. Check mass conservation, solvers have a tendency to let mass "leak" out, when this happens re-normalization stages are often required
Collapse














