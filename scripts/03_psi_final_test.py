# scripts/03_psi_final_test.py

# ===============================================================
# 03_psi_final_test.py
#
#  Ψ-Continuum Final Comparison:
#  - Compare ΛCDM vs ΨCDM on:
#        1) H(z) dataset
#        2) Pantheon+ SN dataset (diagonal-only version)
#
#  Produces:
#     - χ² contributions for both datasets
#     - Total χ² for each model
#     - Residual plots
#     - Ratio HΨ / HΛ
#     - ΩΨ(z)
#     - meta.txt summary
#
#  Fully cleaned and GitHub-ready version.
# ===============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Make repo root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from utils.cosmology import E_LCDM, E_PSI, chi2_H
from utils.io_loaders import load_H_csv


# --------------------------------------------------------------
# SETTINGS
# --------------------------------------------------------------

HZ_CSV_PATH = os.path.join(ROOT, "data", "hz", "H(z).csv")
PANTHEON_PATH = os.path.join(ROOT, "data", "Pantheon+", "Pantheon+SH0ES.dat")
SAVE_DIR = os.path.join(ROOT, "results", "psi_final_outputs")

# Cosmology parameters
Om = 0.30
Ok = 0.0
Or = 9.2e-5

# ΨCDM parameters
eps0 = 0.10
n_psi = 0.00
Opsi0 = 0.00

# H0 values (SN-only allows freedom via M)
H0_LCDM = 68.0
H0_PSI  = 72.0

c_kms = 299792.458


# --------------------------------------------------------------
# Luminosity distance (generic)
# --------------------------------------------------------------

def lum_distance(z, H0, Efun, Ok):
    """Luminosity distance D_L(z) for curved or flat geometry."""
    def invE(zz):
        return 1.0 / (Efun(zz) + 1e-12)

    Dc, _ = quad(invE, 0, z, limit=300, epsabs=1e-8, epsrel=1e-8)
    Dc = (c_kms / H0) * Dc

    if abs(Ok) < 1e-12:
        Dt = Dc
    else:
        sq = np.sqrt(abs(Ok))
        x = sq * H0 * Dc / c_kms
        Dt = (c_kms / H0) / sq * (np.sinh(x) if Ok > 0 else np.sin(x))

    return (1 + z) * Dt


# --------------------------------------------------------------
# Simplified Pantheon+ loader (diagonal-only)
# --------------------------------------------------------------

def load_pantheon_diagonal(path):
    """
    Load Pantheon+ SH0ES catalog:
    - zCMB -> z
    - m_b_corr -> mu
    - m_b_corr_err_DIAG -> sigma

    This is the diagonal-only version (for final comparison plots).
    """
    if not os.path.exists(path):
        print("Pantheon+ file not found — skipping SN part.")
        return None

    df = pd.read_csv(path, sep=r"\s+", comment="#")

    if not {"zCMB", "m_b_corr", "m_b_corr_err_DIAG"} <= set(df.columns):
        print("Pantheon+ file missing required columns.")
        return None

    df = df.rename(columns={
        "zCMB": "z",
        "m_b_corr": "mu",
        "m_b_corr_err_DIAG": "sigma_mu"
    })

    return df[["z", "mu", "sigma_mu"]]


# --------------------------------------------------------------
# MAIN PROCEDURE
# --------------------------------------------------------------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------------------- Load H(z) ----------------------
    dfH = load_H_csv(HZ_CSV_PATH)
    zH, H_obs, sigma_H = dfH["z"].values, dfH["H"].values, dfH["sigma_H"].values
    print(f"Loaded H(z): {len(dfH)} points")

    # E(z) wrappers
    E_L = lambda z: E_LCDM(z, H0_LCDM, Om, Ok)
    E_P = lambda z: E_PSI(z, H0_PSI, Om, Ok, eps0, n_psi, Opsi0=Opsi0)

    # χ²_H for both models
    chiL_H = chi2_H(zH, H_obs, sigma_H, H0_LCDM, E_L)
    chiP_H = chi2_H(zH, H_obs, sigma_H, H0_PSI,  E_P)


    # ---------------------- Load Pantheon+ ----------------------
    dfS = load_pantheon_diagonal(PANTHEON_PATH)

    if dfS is not None:
        zS   = dfS["z"].values
        mu_o = dfS["mu"].values
        sig  = dfS["sigma_mu"].values
        w    = 1.0 / sig**2

        # ΛCDM distances
        DL_L = np.array([lum_distance(z, H0_LCDM, E_L, Ok) for z in zS])
        mu_L0 = 5 * np.log10(DL_L) + 25
        M_L = np.sum(w * (mu_o - mu_L0)) / np.sum(w)
        chiL_SN = np.sum(((mu_o - (mu_L0 + M_L)) / sig)**2)

        # ΨCDM distances
        DL_P = np.array([lum_distance(z, H0_PSI, E_P, Ok) for z in zS])
        mu_P0 = 5 * np.log10(DL_P) + 25
        M_P = np.sum(w * (mu_o - mu_P0)) / np.sum(w)
        chiP_SN = np.sum(((mu_o - (mu_P0 + M_P)) / sig)**2)

        print(f"Loaded Pantheon+: {len(dfS)} SNe")
    else:
        chiL_SN = chiP_SN = 0


    # ---------------------- Summary ----------------------
    chiL_tot = chiL_H + chiL_SN
    chiP_tot = chiP_H + chiP_SN

    print("\n" + "═"*60)
    print("               Ψ-CONTINUUM FINAL TEST SUMMARY")
    print("═"*60)
    print(f"H(z):       Λ = {chiL_H:.2f}     Ψ = {chiP_H:.2f}")
    print(f"Pantheon+:  Λ = {chiL_SN:.2f}     Ψ = {chiP_SN:.2f}")
    print("\nTOTAL χ²:")
    print(f"ΛCDM = {chiL_tot:.2f}")
    print(f"ΨCDM = {chiP_tot:.2f}")
    print(f"Δχ²(Ψ − Λ) = {chiP_tot - chiL_tot:+.2f}")
    print("═"*60)


    # ------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------

    # 1. H(z) curves
    z_plot = np.linspace(0, 2.5, 400)
    H_L = H0_LCDM * np.array([E_L(z) for z in z_plot])
    H_P = H0_PSI  * np.array([E_P(z) for z in z_plot])

    plt.figure(figsize=(8,5))
    plt.errorbar(zH, H_obs, sigma_H, fmt="o", ms=4, label="H(z) data")
    plt.plot(z_plot, H_L, "-",  label=f"ΛCDM (H0={H0_LCDM:.1f})")
    plt.plot(z_plot, H_P, "--", label=f"ΨCDM (eps0={eps0}, n={n_psi})")
    plt.xlabel("z"); plt.ylabel("H(z) [km/s/Mpc]")
    plt.title("Hubble Expansion History: ΛCDM vs ΨCDM")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/hz_curves.png", dpi=200)

    # 2. H(z) residuals
    H_Lobs = H0_LCDM * np.array([E_L(z) for z in zH])
    H_Pobs = H0_PSI  * np.array([E_P(z) for z in zH])

    plt.figure(figsize=(8,4))
    plt.axhline(0, color="k", lw=1)
    plt.errorbar(zH, H_obs - H_Lobs, sigma_H, fmt="o", label="ΛCDM residuals")
    plt.errorbar(zH, H_obs - H_Pobs, sigma_H, fmt="s", label="ΨCDM residuals")
    plt.xlabel("z"); plt.ylabel("ΔH")
    plt.title("H(z) Residuals")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/hz_residuals.png", dpi=200)

    # 3. Pantheon residuals
    if dfS is not None:
        dL = mu_o - (mu_L0 + M_L)
        dP = mu_o - (mu_P0 + M_P)

        plt.figure(figsize=(9,5))
        plt.axhline(0, color="k")
        plt.scatter(zS, dL, s=12, label="ΛCDM")
        plt.scatter(zS, dP, s=12, label="ΨCDM")
        plt.xlabel("z"); plt.ylabel("Δμ")
        plt.title("Pantheon+ Residuals: ΛCDM vs ΨCDM")
        plt.suptitle(
            f"(ΛCDM: H₀={H0_LCDM},  ΨCDM: H₀={H0_PSI},  ε₀={eps0},  n={n_psi})",
            y=0.98, fontsize=9
        )
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/sn_residuals.png", dpi=200)

        # Difference
        plt.figure(figsize=(9,5))
        plt.axhline(0, color="k")
        plt.scatter(zS, dP - dL, s=12)
        plt.xlabel("z"); plt.ylabel("Ψ − Λ")
        plt.title("Pantheon+ Residual Difference (ΨCDM − ΛCDM)")
        plt.suptitle(f"(ε₀={eps0}, n={n_psi})", y=0.98, fontsize=9)
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/sn_residual_diff.png", dpi=200)

    # 4. Ratio HΨ / HΛ
    R = H_P / H_L
    plt.figure(figsize=(8,4))
    plt.plot(z_plot, R)
    plt.axhline(1, color="k", ls=":")
    plt.xlabel("z"); plt.ylabel("HΨ / HΛ (dimensionless)")
    plt.title("Ratio of Expansion Rates: HΨ/HΛ")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/ratio_Rz.png", dpi=200)

    # 5. ΩΨ(z)
    OmegaPsi = Opsi0 + eps0 * (1 + z_plot)**n_psi
    plt.figure(figsize=(8,4))
    plt.plot(z_plot, OmegaPsi)
    plt.xlabel("z"); plt.ylabel("ΩΨ(z)")
    plt.title(f"Ψ-Component Energy Density Evolution  (ε₀={eps0}, n={n_psi})")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/omega_psi.png", dpi=200)


    # ------------------------------------------------------
    # META SUMMARY (improved formatted version)
    # ------------------------------------------------------

    meta = os.path.join(SAVE_DIR, "meta.txt")
    with open(meta, "w") as f:
        f.write("=============================================\n")
        f.write("         Ψ-Continuum Final Test Summary\n")
        f.write("=============================================\n\n")

        f.write("Data sets used:\n")
        f.write(f"  • H(z): {len(dfH)} measurements\n")
        if dfS is not None:
            f.write(f"  • Pantheon+: {len(dfS)} supernovae (diagonal-only version)\n")
        f.write("\n")

        f.write("Model Parameters:\n")
        f.write(f"  ΛCDM:\n")
        f.write(f"     H₀ = {H0_LCDM:.2f} km/s/Mpc\n")
        f.write(f"     Ωₘ = {Om:.3f}\n\n")

        f.write(f"  ΨCDM:\n")
        f.write(f"     H₀     = {H0_PSI:.2f} km/s/Mpc\n")
        f.write(f"     ε₀     = {eps0:.3f}\n")
        f.write(f"     n      = {n_psi:.3f}\n")
        f.write(f"     ΩΨ₀    = {Opsi0:.3f}\n\n")

        f.write("χ² contributions:\n")
        f.write(f"  H(z):      Λ = {chiL_H:.2f}     Ψ = {chiP_H:.2f}\n")
        f.write(f"  Pantheon+: Λ = {chiL_SN:.2f}     Ψ = {chiP_SN:.2f}\n\n")

        f.write("Total χ²:\n")
        f.write(f"  ΛCDM total = {chiL_tot:.2f}\n")
        f.write(f"  ΨCDM total = {chiP_tot:.2f}\n")
        f.write(f"  Δχ² (Ψ − Λ) = {chiP_tot - chiL_tot:+.3f}\n\n")

        # Automatically interpret Δχ²
        dchi = chiP_tot - chiL_tot
        f.write("Interpretation:\n")
        if dchi > 0:
            f.write(f"  ΛCDM provides a better fit by Δχ² = {dchi:.2f}.\n")
        else:
            f.write(f"  ΨCDM provides a better fit by Δχ² = {-dchi:.2f}.\n")

        f.write("=============================================\n")


# --------------------------------------------------------------

if __name__ == "__main__":
    main()
