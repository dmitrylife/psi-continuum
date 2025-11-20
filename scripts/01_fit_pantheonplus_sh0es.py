# scripts/01_fit_pantheonplus_sh0es.py

# ===============================================================
#  Pantheon+SH0ES: SN-only fit of Ωm in flat ΛCDM using full COV
#
#  - Loads Pantheon+SH0ES.dat
#  - Loads Pantheon+SH0ES_STAT+SYS.cov
#  - Selects Hubble-flow sample: USED_IN_SH0ES_HF == 1 and not calibrators
#  - Computes χ² with analytic marginalization over M (absolute magnitude)
#  - Minimizes χ² over Ωm (H0 cancels out for SN-only fits)
#  - Plots residuals with best-fit Ωm and M
#
#  This version is fully cleaned, structured and safe for GitHub.
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "Pantheon+")

# Determine project root and output directory
OUTDIR = os.path.join(ROOT, "results", "pantheon_fit")
os.makedirs(OUTDIR, exist_ok=True)


# --------------------------------------------------------------
# 0) Constants
# --------------------------------------------------------------
c_kms = 299792.458  # speed of light in km/s


# --------------------------------------------------------------
# 1) Flat ΛCDM expansion function and distances
# --------------------------------------------------------------

def E_flat_LCDM(z, Om):
    """
    Normalized expansion rate E(z) in flat ΛCDM:
        E^2 = Ωm(1+z)^3 + ΩΛ
    Radiation and curvature are negligible for SN redshift range.
    """
    Ol = 1.0 - Om
    E2 = Om * (1 + z)**3 + Ol
    return np.sqrt(E2)


def comoving_distance(z, Om, H0=70.0):
    """
    Comoving distance in Mpc.
    H0 cancels out in SN-only fits (absorbed into M), so H0 is arbitrary.
    """
    def integrand(zz):
        return 1.0 / E_flat_LCDM(zz, Om)

    # SN range is small → integration is easy. Add safe tolerances.
    val, _ = quad(integrand, 0.0, z, limit=200, epsabs=1e-8, epsrel=1e-8)
    return (c_kms / H0) * val


def luminosity_distance(z, Om, H0=70.0):
    """
    Luminosity distance in flat space: D_L = (1+z) * D_C.
    """
    return (1.0 + z) * comoving_distance(z, Om, H0)


def mu_theory(z_arr, Om, H0=70.0):
    """
    Theoretical distance modulus μ = 5 log10(D_L / Mpc) + 25.
    """
    DL = np.array([luminosity_distance(zi, Om, H0) for zi in z_arr])
    return 5.0 * np.log10(DL) + 25.0


# --------------------------------------------------------------
# 2) Covariance loader (handles both full matrix and packed form)
# --------------------------------------------------------------

def load_cov(filename, N_expected):
    """
    Load Pantheon+ covariance matrix.

    Handles:
      1) N on first line, followed by either:
         - N*N elements   (full matrix)
         - N(N+1)/2       (lower triangle packed)
      2) Otherwise: auto-detect square matrix from flat list.

    Returns symmetric (N x N) array.
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
            # First line is not integer → parse all floats
            nums = [float(first)] + [float(x) for x in f.read().split()]
            arr = np.array(nums, dtype=float)

            root = int(np.sqrt(arr.size))
            if root * root != arr.size:
                raise ValueError("Covariance file is not square.")
            C = arr.reshape((root, root))

    if C.shape[0] != N_expected:
        raise RuntimeError(
            f"Covariance size mismatch: expected {N_expected}, got {C.shape[0]}.\n"
            "You must slice the covariance to the SN subset."
        )

    # Symmetrize tiny numerical asymmetries
    return 0.5 * (C + C.T)


# --------------------------------------------------------------
# 3) Load Pantheon+ data and extract Hubble-flow sample
# --------------------------------------------------------------

df = pd.read_csv(os.path.join(DATA_DIR, "Pantheon+SH0ES.dat"), sep=r"\s+")

required = ["zCMB", "m_b_corr", "IS_CALIBRATOR", "USED_IN_SH0ES_HF"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# Select Hubble-flow SNe with SH0ES flags
mask_hf = (df["USED_IN_SH0ES_HF"] == 1) & (df["IS_CALIBRATOR"] == 0)
idx_hf = np.where(mask_hf)[0]

if len(idx_hf) == 0:
    raise RuntimeError("No Hubble-flow SN found.")

z = df.loc[mask_hf, "zCMB"].to_numpy(float)
mu_obs = df.loc[mask_hf, "m_b_corr"].to_numpy(float)
N = len(z)
print(f"✔ Hubble-flow SNe loaded: {N}")


# Load full covariance and slice it
cov_path = os.path.join(DATA_DIR, "Pantheon+SH0ES_STAT+SYS.cov")
C_full = load_cov(cov_path, df.shape[0])
C = C_full[np.ix_(idx_hf, idx_hf)]
invC = np.linalg.inv(C)
ones = np.ones(N)


# --------------------------------------------------------------
# 4) χ² with analytic marginalization over M
# --------------------------------------------------------------

def chi2_marg(Om):
    """
    χ² marginalized over the absolute magnitude M:
      χ² = Δ^T C⁻¹ Δ - (1^T C⁻¹ Δ)² / (1^T C⁻¹ 1)
    """
    mu_th = mu_theory(z, Om)
    d = mu_obs - mu_th

    Cd = invC @ d
    A = ones @ invC @ ones
    B = ones @ Cd

    return float(d @ Cd - (B * B) / A)


# --------------------------------------------------------------
# 5) Fit Ωm and compute residuals
# --------------------------------------------------------------

res = minimize(lambda p: chi2_marg(p[0]), x0=[0.3],
               bounds=[(0.05, 0.6)], method="L-BFGS-B")

Om_best = float(res.x[0])
chi2_best = float(res.fun)
dof = N - 1   # Only Ωm is a fitted parameter; M is analytically marginalized

print("\n=== Pantheon+SH0ES SN-only fit (flat ΛCDM) ===")
print(f"Best-fit Ωm = {Om_best:.4f}")
print(f"χ²_marg = {chi2_best:.2f}   (N={N}, dof={dof})")

# Compute best-fit M for residuals
mu_th_best = mu_theory(z, Om_best)
M_best = float((ones @ invC @ (mu_obs - mu_th_best)) / (ones @ invC @ ones))
residuals = mu_obs - (mu_th_best + M_best)


# --------------------------------------------------------------
# 6) Plot residuals
# --------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(z, residuals, s=12)
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("z (CMB)")
plt.ylabel(r"$\Delta\mu = \mu_{\rm obs} - [\mu_{\rm th} + M_{\rm best}]$")
plt.title(f"Pantheon+SH0ES residuals, Ωm = {Om_best:.3f}")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save figure to project structure
outfile = os.path.join(OUTDIR, f"pantheon_residuals_Om_{Om_best:.3f}.png")
plt.savefig(outfile, dpi=200)
print(f"\n📁 Residual plot saved to: {outfile}")
plt.show()

# --------------------------------------------------------------
# META SUMMARY (saved to meta.txt)
# --------------------------------------------------------------

meta_path = os.path.join(OUTDIR, "meta.txt")
with open(meta_path, "w") as f:
    f.write("===============================================\n")
    f.write("      Pantheon+SH0ES — SN-only Ωₘ Fit\n")
    f.write("===============================================\n\n")

    f.write("Fitted model:\n")
    f.write("  Flat ΛCDM (radiation & curvature neglected for SN redshift range)\n\n")

    f.write("Results:\n")
    f.write(f"  Best-fit Ωₘ        = {Om_best:.4f}\n")
    f.write(f"  Best-fit M         = {M_best:.5f}\n")
    f.write(f"  χ²_marg            = {chi2_best:.2f}\n")
    f.write(f"  Number of SNe      = {N}\n")
    f.write(f"  Degrees of freedom = {dof}\n")
    f.write(f"  χ²/dof             = {chi2_best/dof:.3f}\n\n")

    f.write("Interpretation:\n")
    if chi2_best/dof < 1.2:
        f.write("  • The fit quality is good; ΛCDM describes the Hubble-flow sample well.\n")
    else:
        f.write("  • Fit quality is marginal; model may be under tension with SN data.\n")

    f.write("  • Analytic marginalization over M removes dependence on H₀,\n")
    f.write("    making this a pure test of the *shape* of μ(z).\n\n")

    f.write("Notes:\n")
    f.write("  • Only Hubble-flow SNe (USED_IN_SH0ES_HF = 1, non-calibrators)\n")
    f.write("    were included, following the Pantheon+ SH0ES methodology.\n")
    f.write("  • Full Pantheon+ covariance matrix was used.\n")
    f.write("  • No cosmological parameter besides Ωₘ was fitted.\n")
    f.write("===============================================\n")

print(f"Saved meta summary to {meta_path}")

print("\n✔ Done.")
