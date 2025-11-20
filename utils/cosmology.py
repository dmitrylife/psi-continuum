# utils/cosmology.py

import numpy as np

# Fixed radiation density parameter (matches values used in literature)
Or_fix_default = 9.2e-5


# --------------------------------------------------
#  Normalized Hubble functions: H(z) = H0 * E(z)
# --------------------------------------------------

def E_LCDM(z, H0, Om, Ok, Or=Or_fix_default):
    """
    Normalized ΛCDM Hubble function E(z).

    E_LCDM^2(z) = Ω_m (1+z)^3 + Ω_r (1+z)^4 + Ω_k (1+z)^2 + Ω_Λ,
    where Ω_Λ = 1 − Ω_m − Ω_r − Ω_k.

    H0 is included only for interface compatibility.
    """
    z = np.asarray(z, dtype=float)

    # Compute Λ component
    Ol = 1.0 - Om - Ok - Or

    # Total density-like term
    rho = (
        Om * (1 + z)**3 +
        Or * (1 + z)**4 +
        Ok * (1 + z)**2 +
        Ol
    )

    # Numerical safety: avoid tiny negative values
    rho = np.maximum(rho, 0.0)
    return np.sqrt(rho)


def E_PSI(z, H0, Om, Ok, eps0, n, Or=Or_fix_default, Opsi0=0.0):
    """
    Normalized ΨCDM Hubble function.

      E^2(z) = Ω_m(1+z)^3 + Ω_r(1+z)^4 + Ω_k(1+z)^2
               + Ω_Ψ,0 + ε0(1+z)^n

    In current tests Ω_Ψ,0 = 0.0 by default.
    """
    z = np.asarray(z, dtype=float)

    psi_term = Opsi0 + eps0 * (1 + z)**n

    rho = (
        Om * (1 + z)**3 +
        Or * (1 + z)**4 +
        Ok * (1 + z)**2 +
        psi_term
    )

    rho = np.maximum(rho, 0.0)
    return np.sqrt(rho)


# --------------------------------------------------
#  χ² over H(z) dataset
# --------------------------------------------------

def chi2_H(z_arr, H_obs, sigma_H, H0, Efun):
    """
    Compute χ² for H(z) measurements:

      H_theory(z) = H0 * Efun(z)
      χ² = Σ_i [ (H_obs_i − H_theory_i) / σ_i ]²

    Inputs must be 1D arrays of equal length.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    H_obs = np.asarray(H_obs, dtype=float)
    sigma_H = np.asarray(sigma_H, dtype=float)

    # Vectorized computation for speed and clarity
    Ez = Efun(z_arr)
    H_theory = H0 * Ez

    r = (H_obs - H_theory) / sigma_H
    return float(np.sum(r * r))
