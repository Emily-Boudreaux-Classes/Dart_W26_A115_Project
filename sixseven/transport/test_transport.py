from turtle import lt
import numpy as np
print("FILE LOADED")


class TransportModule:
    def __init__(
        self,
        f_ov=0.01,
        alpha_sc=0.01,
        alpha_th=1.0,
        instantaneous_convection=True,
    ):
        self.f_ov = f_ov
        self.alpha_sc = alpha_sc
        self.alpha_th = alpha_th
        self.instantaneous_convection = instantaneous_convection

    # Diffusion coefficients

    def D_conv(self, v_mlt, l_mlt):
        return (1.0 / 3.0) * v_mlt * l_mlt

    def D_overshoot(self, D0, z, Hp):
        return D0 * np.exp(-2.0 * z / (self.f_ov * Hp))

    def D_semiconv(self, K, rho, Cp, grad_rad, grad_ad, grad_mu):
        D = np.zeros_like(K)
        mask = grad_mu > 0
        D[mask] = (
            self.alpha_sc
            * K[mask]
            / (Cp[mask] * rho[mask])
            * (grad_rad[mask] - grad_ad[mask])
            / grad_mu[mask]
        )
        return np.maximum(D, 0.0)

    def D_thermohaline(self, K, rho, Cp, grad_rad, grad_ad, grad_mu, phi_over_delta=1.0):
        D = np.zeros_like(K)
        unstable = (
            (phi_over_delta * grad_mu <= grad_rad - grad_ad)
            & (grad_rad - grad_ad <= 0)
        )
        D[unstable] = (
            -self.alpha_th
            * (3 * K[unstable])
            / (2 * rho[unstable] * Cp[unstable])
            * (phi_over_delta * grad_mu[unstable])
            / (grad_ad[unstable] - grad_rad[unstable])
        )
        return np.maximum(D, 0.0)

    # Main diffusion builder

    def compute_diffusion_coefficients(self, structure):
        n = len(structure["m"])
        D = np.zeros(n)

        # Convective mixing
        D_conv = self.D_conv(structure["v_mlt"], structure["l_mlt"])
        D += D_conv * structure["is_convective"]

        # Semiconvection
        D += self.D_semiconv(
            structure["K"],
            structure["rho"],
            structure["Cp"],
            structure["grad_rad"],
            structure["grad_ad"],
            structure["grad_mu"],
        )

        # Thermohaline
        D += self.D_thermohaline(
            structure["K"],
            structure["rho"],
            structure["Cp"],
            structure["grad_rad"],
            structure["grad_ad"],
            structure["grad_mu"],
        )

        return D

    # Updating composition

    def step(self, X, structure, dt):
        """
        Advancing composition by one timestep.
        X: (n_shells, n_species)
        """

        if self.instantaneous_convection:
            X = self._instantaneous_mix(X, structure["is_convective"])

        D = self.compute_diffusion_coefficients(structure)

        # diffusion placeholder
        X_new = X.copy()
        m = structure["m"]

        for i in range(1, len(m) - 1):
            dm = m[i + 1] - m[i]
            X_new[i] += (
                dt
                * D[i]
                * (X[i + 1] - 2 * X[i] + X[i - 1])
                / dm**2
            )

        return X_new

    def _instantaneous_mix(self, X, conv_mask):
        X_new = X.copy()
        i = 0
        while i < len(conv_mask):
            if conv_mask[i]:
                j = i
                while j < len(conv_mask) and conv_mask[j]:
                    j += 1
                X_new[i:j] = np.mean(X[i:j], axis=0)
                i = j
            else:
                i += 1
        return X_new
    
def get_diffusion_profile(structure,
                          f_ov=0.01,
                          alpha_sc=0.01,
                          alpha_th=1.0,
                          instantaneous_convection=True):
    """
    Public interface for other modules.

    Returns:
        D(m) array
    """

    transport = TransportModule(
        f_ov=f_ov,
        alpha_sc=alpha_sc,
        alpha_th=alpha_th,
        instantaneous_convection=instantaneous_convection,
    )

    return transport.compute_diffusion_coefficients(structure)

if __name__ == "__main__":
    print("MAIN BLOCK ENTERED")
    n_shells = 100
    n_species = 3

    structure = {
    "m": np.linspace(0, 1, n_shells),
    "r": np.linspace(0.1, 1.0, n_shells),
    "rho": np.ones(n_shells),
    "T": np.ones(n_shells),

    # make Hp more realistic than unity
    "Hp": np.linspace(5e9, 5e10, n_shells),  # cm

    "K": np.ones(n_shells),
    "Cp": np.ones(n_shells),
    "grad_rad": 0.4 * np.ones(n_shells),
    "grad_ad": 0.2 * np.ones(n_shells),
    "grad_mu": 0.01 * np.ones(n_shells),
    "is_convective": np.zeros(n_shells, dtype=bool),
}

    # define convective region
    structure["is_convective"][20:40] = True


# --------------------------
# Mixing length + velocity
# --------------------------

    alpha_MLT = 1.8

# mixing length ℓ = α Hp
    l_mlt = alpha_MLT * structure["Hp"]

# convective velocity estimate (cm/s)
    v_mlt = np.zeros(n_shells)

    superadiabatic = structure["grad_rad"] - structure["grad_ad"]
    conv = structure["is_convective"]

    v_mlt[conv] = 3e5 * np.sqrt(np.maximum(superadiabatic[conv], 1e-4))

# putting into structure
    structure["l_mlt"] = l_mlt
    structure["v_mlt"] = v_mlt

    X = np.zeros((n_shells, n_species))
    X[:, 0] = 0.7
    X[:, 1] = 0.28
    X[:, 2] = 0.02

    transport = TransportModule()

    X_new = transport.step(X, structure, dt=1e-3)

    print("Transport step completed.")


    import matplotlib.pyplot as plt


    D = transport.compute_diffusion_coefficients(structure)

    m = structure["m"]


    plt.figure()
    plt.plot(m, D)
    plt.xlabel("Mass coordinate")
    plt.ylabel("Diffusion coefficient D")
    plt.title("Total Mixing Diffusion Coefficient")

    species = 0  

    plt.figure()
    plt.plot(m, X[:, species], label="Before")
    plt.plot(m, X_new[:, species], label="After")
    plt.xlabel("Mass coordinate")
    plt.ylabel("Mass fraction")
    plt.title(f"Species {species} profile")
    plt.legend()


    plt.figure()
    plt.plot(m, structure["is_convective"].astype(int))
    plt.xlabel("Mass coordinate")
    plt.ylabel("Convective = 1")
    plt.title("Convective Zones")
# =====================================
# Sensitivity to mixing length alpha_MLT
# =====================================

    alpha_grid = np.linspace(0.8, 3.0, 10)

    max_changes = []

    species = 0

    for alpha in alpha_grid:

        l_mlt = alpha * structure["Hp"]

        v_mlt = np.zeros(n_shells)
        superadiabatic = structure["grad_rad"] - structure["grad_ad"]
        conv = structure["is_convective"]

        v_mlt[conv] = 3e5 * np.sqrt(np.maximum(superadiabatic[conv], 1e-4))

        structure["l_mlt"] = l_mlt
        structure["v_mlt"] = v_mlt

        X_new = transport.step(X, structure, dt=1e-3)

        # measure maximum absolute change
        dX_max = np.max(np.abs(X_new[:, species] - X[:, species]))

        max_changes.append(dX_max)

    plt.figure()
    plt.plot(alpha_grid, max_changes)
    plt.xlabel(r"Mixing length parameter $\alpha_{\rm MLT}$")
    plt.ylabel(r"max |ΔX| (species 0)")
    plt.title("Sensitivity of Mixing Strength to α_MLT")
    plt.show()