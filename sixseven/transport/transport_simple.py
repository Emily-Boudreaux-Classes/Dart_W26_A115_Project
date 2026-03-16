import numpy as np
from fipy import Grid1D, CellVariable, FaceVariable, TransientTerm, DiffusionTerm
from sixseven.transport.diffusion import A_face

print("Hello! This is the transport module. It handles compositional transport via diffusion.")
# DIFFUSION COEFFICIENTS
def compute_diffusion_coefficients(structure,
                                   alpha_sc=0.01,
                                   alpha_th=1.0):
    """
    Computes the total diffusion coefficient profile.

    This includes contributions from:
        1. Convective mixing (MLT approximation)
        2. Semiconvection
        3. Thermohaline mixing

    The structure dictionary must contain all required
    thermodynamic and gradient quantities.
    """

    m = structure["m"]                    # mass coordinate
    Hp = structure["Hp"]                  # pressure scale height
    v_mlt = structure["v_mlt"]            # convective velocity
    is_conv = structure["is_convective"]  # boolean mask for convective zones

    grad_rad = structure["grad_rad"]      # radiative temperature gradient
    grad_ad = structure["grad_ad"]        # adiabatic gradient
    grad_mu = structure["grad_mu"]        # molecular weight gradient

    K = structure["K"]                    # thermal conductivity
    Cp = structure["Cp"]                  # specific heat at constant pressure
    rho = structure["rho"]                # density
    T = structure["T"]                    # temperature

    nx = len(m)

    # Initialize diffusion arrays
    D_conv = np.zeros(nx) 
    D_sc   = np.zeros(nx) 
    D_th   = np.zeros(nx)

    # 1. Convective diffusion
    # Standard MLT approximation:
    #   D ~ (1/3) * v_mlt * mixing_length
    # Here we use Hp as the mixing length scale.
    D_conv[is_conv] = (1.0 / 3.0) * v_mlt[is_conv] * Hp[is_conv]

    # 2. Semiconvection
    # Condition:
    #   grad_ad < grad_rad < grad_ad + grad_mu

    # Physically: region unstable to composition but stable
    # against full convection.
    # We use a simple diffusion approximation scaled by alpha_sc.
    semi_mask = (
        (grad_rad > grad_ad) &
        (grad_rad < grad_ad + grad_mu)
    )

    D_sc[semi_mask] = (
        alpha_sc *
        K[semi_mask] /
        (rho[semi_mask] * Cp[semi_mask]) *
        (grad_rad[semi_mask] - grad_ad[semi_mask]) /
        grad_mu[semi_mask]
    )

    # 3. Thermohaline mixing
    # Condition:
    #   grad_mu < 0

    # Physically: heavy material sitting on top of lighter material
    # (inverse µ-gradient instability).

    # Modeled as diffusive mixing scaled by alpha_th.
    th_mask = grad_mu < 0

    D_th[th_mask] = (
        alpha_th *
        K[th_mask] /
        (rho[th_mask] * Cp[th_mask]) *
        (-grad_mu[th_mask])
    )

    # Total diffusion coefficient
    D_total = D_conv + D_sc + D_th

    return D_total

# EXPLICIT DIFFUSION STEP

def apply_diffusion(X, D, m, dt):
    """
    Performs one diffusion timestep in our mass coordinate.

    Parameters
    X : ndarray (n_shells, n_species)
        Current mass fractions.
    D : ndarray (n_shells,)
        Total diffusion coefficient profile.
    m : ndarray (n_shells,)
        Mass coordinate.
    dt : float
        Timestep (seconds).

    Returns
    -------
    X_new : ndarray
        Updated mass fractions.
    """

    nx, ns = X.shape
    X_new = X.copy()

    # Loop over each nuclear species independently
    for s in range(ns):

        # Spatial gradient of composition
        # ∂X/∂m
        gradX = np.gradient(X[:, s], m)

        # Diffusive flux:
        # F = -D ∂X/∂m
        flux = -D * gradX

        # Divergence of flux:
        # ∂F/∂m
        div_flux = np.gradient(flux, m)

        # Explicit time update:
        # X_new = X_old + dt * (∂F/∂m)
        X_new[:, s] += dt * div_flux

    # Enforce physical bounds
    # Mass fractions must stay between 0 and 1.
    # This will prevent numerical overshoot.
    X_new[X_new < 0] = 0.0
    X_new[X_new > 1] = 1.0

    return X_new


# What to call!

def transport_step(comps, structure, dt,
                   alpha_sc=0.01,
                   alpha_th=1.0):
    """
    Runs ONE timestep of compositional transport.

    This function:
        1. Computes diffusion coefficients from structure
        2. Applies diffusion update to composition (AKA gives out new composition)

    It does NOT (because Ben handles this in timestep):
        - call nuclear burning
        - compute temperature
        - compute density
        - update internal energy

    It ONLY modifies composition based on the current structure
    and the specified diffusion physics given to me.

    Parameters in here:

    X : ndarray (n_shells, n_species)
        Current mass fractions.

    structure : dict
        Must contain:
            m, Hp, v_mlt, is_convective,
            grad_rad, grad_ad, grad_mu,
            K, Cp, rho, T, and r (from George)

    dt : float
        Timestep (seconds).

    Gives:

    X_new : ndarray
        Updated composition after one timestep.
    """
    mesh = Grid1D(structure["dM"])
    symbols = ["H-1", "He-4", "C-12"]
    Y = {}
    X = np.zeros(shape=(len(comps), len(symbols)))
    for i, comp in enumerate(comps):          
        for j, sym in enumerate(symbols):    
            X[i, j] = comp.composition.getMolarAbundance(sym)
            Y[sym] = CellVariable(
                mesh=mesh,
                value=X[:, j],
            )          
    # Compute total diffusion coefficient profile
    D = compute_diffusion_coefficients(
        structure,
        alpha_sc=alpha_sc,
        alpha_th=alpha_th
    )
    D_cell = CellVariable(mesh=mesh, value=D)
    rho_cell = CellVariable(mesh=mesh, value=structure["rho"])
    r_cell= CellVariable(mesh=mesh, value=structure["r"])

    A_face = (4 * np.pi * r_cell.faceValue**2 * rho_cell.faceValue)**2 * D_cell.faceValue

    eqs = {symbol: TransientTerm() == DiffusionTerm(coeff=A_face) for symbol in symbols}

    for sym in symbols: 

        Y[sym].updateOld()
        eqs[sym].solve(var=Y[sym], dt=dt)

    
    new_comps = comps
    for shellID, comp in enumerate(comps):
        for j, sym in enumerate(symbols):
            new_comps[shellID].composition.setMolarAbundance(sym, Y[sym].value[shellID])

    return new_comps