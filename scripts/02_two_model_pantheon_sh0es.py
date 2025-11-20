# scripts/02_two_model_pantheon_sh0es.py

# ===============================================================
#  Pantheon+SH0ES: Comparison of two fixed cosmological models
#
#  - SN-only analysis, no fitting (parameters hard-coded)
#  - Uses full Pantheon+ covariance with analytic marginalization over M
#  - Compares:
#        (1) Flat ΛCDM
#        (2) ΨCDM (toy model from psi-continuum project)
#  - Plots residuals and residual difference
#
#  This version is cleaned, structured and GitHub-ready.
# ===============================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --------------------------------------------------------------
# 0) Make psi-continuum root importable
# --------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print("ROOT =", ROOT)
print("DIR CONTENTS =", os.listdir(ROOT))

sys.path.append(ROOT)
DATA_DIR = os.path.join(ROOT, "data", "Pantheon+")


OUTDIR = os.path.join(ROOT, "results", "pantheon_compare")
os.makedirs(OUTDIR, exist_ok=True)

from utils.cosmology import E_LCDM, E_PSI

c_kms = 299792.458  # km/s


# --------------------------------------------------------------
# 1) User model parameters (EDIT THESE)
# --------------------------------------------------------------

# --- ΛCDM parameters ---
LCDM_Om = 0.30

# --- ΨCDM toy parameters ---
PSI_Om    = 0.30
PSI_Opsi0 = 0.00
PSI_eps0  = 0.10
PSI_n     = 0.00

# --- Hubble constants for clean residual comparison ---
# Pantheon+ best-fit ΛCDM H0 ≈ 68.0 km/s/Mpc
H0_LCDM = 68.0

# ΨCDM effective H0 from your grid-scan heatmap (eps0=0.1, n=0)
# For eps0=0.1, n=0 → H0 ≈ 70.5 km/s/Mpc (from your Figure H0_heatmap)
H0_PSI = 70.5

CATALOG = os.path.join(DATA_DIR, "Pantheon+SH0ES.dat")
COVFILE = os.path.join(DATA_DIR, "Pantheon+SH0ES_STAT+SYS.cov")


# --------------------------------------------------------------
# 2) Load Pantheon+ catalog and select Hubble-flow sample
# --------------------------------------------------------------

df = pd.read_csv(CATALOG, sep=r"\s+")

required = ["zCMB", "m_b_corr", "IS_CALIBRATOR", "USED_IN_SH0ES_HF"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns in catalog: {missing}")

mask_hf = (df["USED_IN_SH0ES_HF"] == 1) & (df["IS_CALIBRATOR"] == 0)
idx_hf = np.where(mask_hf)[0]

if idx_hf.size == 0:
    raise RuntimeError("No Hubble-flow SNe found.")

z = df.loc[mask_hf, "zCMB"].to_numpy(float)
mu_obs = df.loc[mask_hf, "m_b_corr"].to_numpy(float)
N = len(z)

print(f"✔ Hubble-flow SNe loaded: {N}")


# --------------------------------------------------------------
# 3) Load covariance and slice to HF subset
# --------------------------------------------------------------

def load_cov(filename, Ncat):
    """
    Loads Pantheon+ covariance matrix.
    Supports both full NxN format and lower-triangle packed format.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename, "r") as f:
        first = f.readline().strip()

        try:
            # Attempt: first line is N
            Nfile = int(first)
            data = np.fromfile(f, sep=" ")

            if data.size == Nfile * Nfile:
                C = data.reshape((Nfile, Nfile))

            elif data.size == Nfile * (Nfile + 1) // 2:
                # Lower triangle packed
                C = np.zeros((Nfile, Nfile))
                k = 0
                for i in range(Nfile):
                    for j in range(i + 1):
                        C[i, j] = data[k]
                        C[j, i] = data[k]
                        k += 1

            else:
                raise ValueError("Unexpected covariance size.")

        except ValueError:
            # Not an integer → try square
            nums = [float(first)] + [float(x) for x in f.read().split()]
            arr = np.array(nums, float)
            root = int(np.sqrt(arr.size))
            if root * root != arr.size:
                raise ValueError("Covariance file is not square.")
            C = arr.reshape((root, root))

    if C.shape[0] != Ncat:
        raise ValueError(
            f"Covariance size mismatch: expected {Ncat}, got {C.shape[0]}."
        )

    return 0.5 * (C + C.T)  # symmetrize small asymmetries


C_full = load_cov(COVFILE, df.shape[0])
C = C_full[np.ix_(idx_hf, idx_hf)]

invC = np.linalg.inv(C)
ones = np.ones(N)


# --------------------------------------------------------------
# 4) Cosmology helpers (distances & μ)
# --------------------------------------------------------------

def Dc_from_E(z, Efunc):
    """
    Comoving distance Dc(z) for a given expansion law E(z).
    H0 is already embedded inside the wrapped E(z) function.
    """
    val, _ = quad(lambda qq: 1.0 / Efunc(qq),
                  0.0, z, limit=300, epsabs=1e-8, epsrel=1e-8)
    return c_kms * val


def DL_from_E(z, Efunc):
    """
    Luminosity distance in flat geometry:
        D_L = (1+z) * D_C
    """
    return (1.0 + z) * Dc_from_E(z, Efunc)


def mu_theory(z_arr, Efunc):
    """
    Distance modulus:
        μ = 5 log10(D_L / Mpc) + 25
    """
    DL = np.array([DL_from_E(zi, Efunc) for zi in z_arr])
    return 5.0 * np.log10(DL) + 25.0


# ---- Wrapped normalized expansion laws with model-specific H0 ----

def E_LCDM_wrap(z):
    return E_LCDM(z, H0_LCDM, LCDM_Om, 0.0)

def E_PSI_wrap(z):
    return E_PSI(z, H0_PSI, PSI_Om, 0.0, PSI_eps0, PSI_n, Opsi0=PSI_Opsi0)


# --------------------------------------------------------------
# 5) χ² with analytic marginalization over M
# --------------------------------------------------------------

def chi2_marg(mu_obs, mu_th, invC, ones):
    """
    Analytic marginalization over absolute magnitude M:

        χ² = Δ^T C⁻¹ Δ  -  (1^T C⁻¹ Δ)² / (1^T C⁻¹ 1)
    """
    d = mu_obs - mu_th
    Cd = invC @ d
    A = ones @ invC @ ones
    B = ones @ Cd
    chi2 = float(d @ Cd - (B * B) / A)
    Mbest = float(B / A)
    return chi2, Mbest


# --------------------------------------------------------------
# 6) Evaluate two fixed models
# --------------------------------------------------------------

# ΛCDM
mu_LCDM = mu_theory(z, E_LCDM_wrap)
chi2_LCDM, M_LCDM = chi2_marg(mu_obs, mu_LCDM, invC, ones)

# ΨCDM
mu_PSI = mu_theory(z, E_PSI_wrap)
chi2_PSI, M_PSI = chi2_marg(mu_obs, mu_PSI, invC, ones)

dof = N - 1  # M marginalized, no fitted parameters

print("\n=== Two fixed models (no fitting) ===")
print(f"[ΛCDM]  Ωm={LCDM_Om:.3f} → χ²_marg={chi2_LCDM:.2f}  (χ²/dof={chi2_LCDM/dof:.2f})")
print(f"[ΨCDM]  Ωm={PSI_Om:.3f}, Opsi0={PSI_Opsi0:.3f}, eps0={PSI_eps0:.3f}, n={PSI_n:.2f} "
      f"→ χ²_marg={chi2_PSI:.2f}  (χ²/dof={chi2_PSI/dof:.2f})")


# --------------------------------------------------------------
# 7) Plot residuals
# --------------------------------------------------------------

res_LCDM = mu_obs - (mu_LCDM + M_LCDM)
res_PSI   = mu_obs - (mu_PSI + M_PSI)

plt.figure(figsize=(9, 5))
plt.scatter(
    z, res_LCDM, s=12, alpha=0.8,
    label=f"ΛCDM residuals (Ωm={LCDM_Om:.2f}, H₀={H0_LCDM:.1f})"
)
plt.scatter(
    z, res_PSI, s=12, alpha=0.8,
    label=f"ΨCDM residuals (ε₀={PSI_eps0:.2f}, n={PSI_n:.2f}, H₀={H0_PSI:.1f})"
)
plt.axhline(0, ls="--", color="gray")
plt.xlabel("z (CMB)")
plt.ylabel(r"$\Delta \mu$")
plt.title("Pantheon+SH0ES residuals: ΛCDM vs ΨCDM (fixed models)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "residuals_compare.png"), dpi=200)
plt.show()


# Difference curve
plt.figure(figsize=(9, 4))
plt.scatter(z, res_PSI - res_LCDM, s=12)
plt.axhline(0, ls="--", color="gray")
plt.xlabel("z (CMB)")
plt.ylabel(r"$(\Delta\mu)_{\Psi} - (\Delta\mu)_{\Lambda}$")
plt.title("Residual difference (ΨCDM − ΛCDM)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "residuals_diff.png"), dpi=200)
plt.show()

# --------------------------------------------------------------
# 8) META SUMMARY (saved to meta.txt)
# --------------------------------------------------------------

meta_path = os.path.join(OUTDIR, "meta.txt")

with open(meta_path, "w") as f:
    f.write("===============================================\n")
    f.write("     Pantheon+SH0ES — Two Fixed Models Test\n")
    f.write("===============================================\n\n")

    f.write("Models compared:\n")
    f.write("  ΛCDM:\n")
    f.write(f"     Ωₘ = {LCDM_Om:.3f}\n")
    f.write(f"     H₀ = {H0_LCDM:.2f} km/s/Mpc\n")
    f.write(f"     Best-fit M = {M_LCDM:.5f}\n\n")

    f.write("  ΨCDM:\n")
    f.write(f"     Ωₘ   = {PSI_Om:.3f}\n")
    f.write(f"     ε₀   = {PSI_eps0:.3f}\n")
    f.write(f"     n    = {PSI_n:.3f}\n")
    f.write(f"     H₀   = {H0_PSI:.2f} km/s/Mpc\n")
    f.write(f"     Best-fit M = {M_PSI:.5f}\n\n")

    f.write(f"Hubble-flow SNe used: {N} objects\n")
    f.write(f"Degrees of freedom (analytic M): dof = {dof}\n\n")

    f.write("χ² values:\n")
    f.write(f"  ΛCDM: χ² = {chi2_LCDM:.3f}   (χ²/dof = {chi2_LCDM/dof:.3f})\n")
    f.write(f"  ΨCDM: χ² = {chi2_PSI:.3f}   (χ²/dof = {chi2_PSI/dof:.3f})\n")
    f.write(f"  Δχ² (Ψ − Λ) = {chi2_PSI - chi2_LCDM:+.3f}\n\n")

    # Interpretation
    dchi = chi2_PSI - chi2_LCDM
    f.write("Interpretation:\n")
    if abs(dchi) < 1:
        f.write("  The two models are statistically indistinguishable in this SN-only test.\n")
    if dchi > 0:
        f.write(f"  ΛCDM provides a slightly better fit by Δχ² = {dchi:.2f}.\n")
    else:
        f.write(f"  ΨCDM provides a slightly better fit by Δχ² = {-dchi:.2f}.\n")

    f.write("\nNotes:\n")
    f.write("  • ΨCDM parameters (ε₀, n) were fixed and NOT fitted to the SN data.\n")
    f.write("  • H₀ values are model-specific but fully absorbed by analytic marginalization over M.\n")
    f.write("  • This test isolates shape differences in μ(z), independent of absolute calibration.\n")
    f.write("===============================================\n")

print(f"Saved meta summary to {meta_path}")

print("✔ Done.")
