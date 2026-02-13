import numpy as np

# Cassie EOS
from modules.eos import eos_functions

# Sasha nuclear
from modules.nuclear import nuc_burn

print("FILE LOADED")


class TransportModule:
    """
    Transport module.

    Inputs:
        structure (dict) from structure+EOS+radiation
        X (mass fractions per shell)
        dt from Ben

    Outputs:
        X_new
        D_profile
        constraints dict for timestepper function
    """

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

    # --------------------------------------------------
    # Diffusion coefficients
    # --------------------------------------------------

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

    def D_thermohaline(self, K, rho, Cp, grad_rad, grad_ad, grad_mu,
                       phi_over_delta=1.0):
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


    def compute_diffusion_coefficients(self, structure):

        n = len(structure["m"])
        D = np.zeros(n)

        D_conv = self.D_conv(
            structure["v_mlt"],
            structure["l_mlt"]
        )

        D += D_conv * structure["is_convective"]

        D += self.D_semiconv(
            structure["K"],
            structure["rho"],
            structure["Cp"],
            structure["grad_rad"],
            structure["grad_ad"],
            structure["grad_mu"],
        )

        D += self.D_thermohaline(
            structure["K"],
            structure["rho"],
            structure["Cp"],
            structure["grad_rad"],
            structure["grad_ad"],
            structure["grad_mu"],
        )

        return D

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, X, structure, dt, comp_objects=None):
        """
        X : (n_shells, n_species) mass fractions
        comp_objects : Sasha Composition per shell (optional)
        """

        # ---- Instantaneous convection
        if self.instantaneous_convection:
            X = self._instantaneous_mix(
                X, structure["is_convective"]
            )

        # ---- Diffusion
        D = self.compute_diffusion_coefficients(structure)

        X_new = X.copy()
        m = structure["m"]

        for i in range(1, len(m) - 1):
            dm = m[i + 1] - m[i]
            X_new[i] += (
                dt * D[i]
                * (X[i + 1] - 2 * X[i] + X[i - 1])
                / dm**2
            )

        # --------------------------------------------------
        # Nuclear burning from Sasha
        # --------------------------------------------------

        eps_nuc, mu, new_comps = nuc_burn.burn(
            structure["T"],
            structure["rho"],
            dt,
            comp_objects,
        )

        constraints = {
            "max_dX": np.max(np.abs(X_new - X)),
            "eps_nuc": eps_nuc,
        }

        return X_new, D, new_comps, constraints


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



def get_diffusion_profile(structure, **kwargs):

    transport = TransportModule(**kwargs)
    return transport.compute_diffusion_coefficients(structure)

def transport_step(X, structure, dt, comp_objects=None, **kwargs):
    """
    """

    transport = TransportModule(**kwargs)

    X_new, _, _, _ = transport.step(
        X,
        structure,
        dt,
        comp_objects=comp_objects,
    )

    return X_new
