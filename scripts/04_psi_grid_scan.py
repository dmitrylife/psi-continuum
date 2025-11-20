# scripts/04_psi_grid_scan.py

# ===============================================================
# 04_psi_grid_scan.py
# Ψ-continuum: 2D scan over (eps0, n) with H0 minimization
# Dmitry Klimov & GPT, 2025
# Cleaned, optimized, and repository-consistent version
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os
import sys

# ---------------------------------------------------------------
# Import utils/ correctly
# ---------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from utils.cosmology import E_LCDM, E_PSI, chi2_H
from utils.io_loaders import load_H_csv


# ===============================================================
# Grid scan: eps0 × n
# ===============================================================

def scan_grid(dfH,
              Om=0.3,
              Ok=0.0,
              Or=9.2e-5,
              eps_min=0.0,
              eps_max=0.3,
              Neps=31,
              n_min=-2.0,
              n_max=+2.0,
              Nn=81):

    """Scan eps0 × n and minimize H0 at each point."""

    eps_grid = np.linspace(eps_min, eps_max, Neps)
    n_grid   = np.linspace(n_min,  n_max,  Nn)

    chi2_map = np.zeros((Nn, Neps))
    H0_map   = np.zeros((Nn, Neps))

    zH        = dfH["z"].values
    H_obs     = dfH["H"].values
    sigma_H   = dfH["sigma_H"].values

    # -----------------------------------------------------------
    # ΛCDM baseline χ²
    # -----------------------------------------------------------
    def chi2_LCDM_H0(H0):
        Efun = lambda z: E_LCDM(z, H0, Om, Ok, Or)
        return chi2_H(zH, H_obs, sigma_H, H0, Efun)

    res_LCDM = minimize_scalar(chi2_LCDM_H0, bounds=(60, 80), method="bounded")
    H0_LCDM  = res_LCDM.x
    chi2_LCDM = res_LCDM.fun

    print(f"ΛCDM baseline: H0*={H0_LCDM:.3f}, χ²={chi2_LCDM:.2f}")

    # -----------------------------------------------------------
    # 2D grid scan
    # -----------------------------------------------------------
    for i, n in enumerate(n_grid):
        # print(f"row {i+1}/{Nn}: n={n:+.3f}...")
        print(f"[{i+1:03d}/{Nn}] n = {n:+.3f}")

        for j, eps0 in enumerate(eps_grid):

            def chi2_PSI_H0(H0):
                Efun = lambda z: E_PSI(z, H0, Om, Ok, eps0, n, Or=Or, Opsi0=0.0)
                return chi2_H(zH, H_obs, sigma_H, H0, Efun)

            res = minimize_scalar(chi2_PSI_H0, bounds=(60, 80), method="bounded")

            H0_map[i, j]   = res.x
            chi2_map[i, j] = res.fun

    return eps_grid, n_grid, chi2_map, H0_map, chi2_LCDM, H0_LCDM


# ===============================================================
# Plotting
# ===============================================================

def plot_results(eps_grid, n_grid, chi2_map, H0_map, chi2_LCDM, H0_LCDM, outdir):

    os.makedirs(outdir, exist_ok=True)

    EPS, N = np.meshgrid(eps_grid, n_grid)

    # Ψ model minimum
    chi2_min = np.min(chi2_map)
    imin, jmin = np.unravel_index(np.argmin(chi2_map), chi2_map.shape)

    eps_best = eps_grid[jmin]
    n_best   = n_grid[imin]
    H0_best  = H0_map[imin, jmin]

    print(f"Ψ best: eps0={eps_best:.4f}, n={n_best:.3f}")
    print(f"H0*={H0_best:.2f}, χ²={chi2_min:.2f}")
    print(f"Δχ² = {chi2_min - chi2_LCDM:+.2f}")

    # -----------------------------------------------------------
    # Δχ² heatmap
    # -----------------------------------------------------------
    plt.figure(figsize=(7,5))
    im = plt.contourf(EPS, N, chi2_map - chi2_LCDM, levels=20, cmap="inferno")
    plt.colorbar(im, label="Δχ²")
    plt.contour(EPS, N, chi2_map - chi2_LCDM, levels=[1,4,9], colors="white")
    plt.scatter(eps_best, n_best, s=120, marker="*", color="white")
    plt.xlabel("ε₀"); plt.ylabel("n")
    plt.title("Ψ-continuum: Δχ²(eps0, n)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "chi2_heatmap.png"), dpi=200)

    # -----------------------------------------------------------
    # H0 heatmap
    # -----------------------------------------------------------
    plt.figure(figsize=(7,5))
    im2 = plt.contourf(EPS, N, H0_map, levels=20, cmap="viridis")
    plt.colorbar(im2, label="H0 [km/s/Mpc]")
    plt.contour(EPS, N, H0_map, levels=[H0_LCDM], colors="white", linestyles="--")
    plt.scatter(eps_best, n_best, s=120, marker="*", color="white")
    plt.xlabel("ε₀"); plt.ylabel("n")
    plt.title("H0(eps0, n) — ΛCDM shown dashed")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "H0_heatmap.png"), dpi=200)

    print(f"Saved plots to {outdir}")

    # -----------------------------------------------------------
    # META SUMMARY (formatted)
    # -----------------------------------------------------------
    meta_path = os.path.join(outdir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write("===============================================\n")
        f.write("         Ψ-Continuum Grid Scan Summary\n")
        f.write("===============================================\n\n")

        f.write("Parameter grid:\n")
        f.write(f"  ε₀ range : [{eps_grid[0]:.3f}, {eps_grid[-1]:.3f}]  "
                f"({len(eps_grid)} steps)\n")
        f.write(f"  n   range : [{n_grid[0]:.3f}, {n_grid[-1]:.3f}]  "
                f"({len(n_grid)} steps)\n")
        f.write("  Minimization: H₀ optimized at each grid point\n\n")

        f.write("Best-fit ΨCDM parameters (H(z) only):\n")
        f.write(f"  ε₀*  = {eps_best:.4f}\n")
        f.write(f"  n*   = {n_best:.4f}\n")
        f.write(f"  H₀*  = {H0_best:.4f}  km/s/Mpc\n")
        f.write(f"  χ²*  = {chi2_min:.3f}\n\n")

        f.write("ΛCDM baseline:\n")
        f.write(f"  H₀*  = {H0_LCDM:.4f}  km/s/Mpc\n")
        f.write(f"  χ²   = {chi2_LCDM:.3f}\n\n")

        f.write("Comparison:\n")
        dchi = chi2_min - chi2_LCDM
        f.write(f"  Δχ²(Ψ − Λ) = {dchi:+.3f}\n")

        if dchi > 0:
            f.write("  → ΛCDM provides a better fit to H(z).\n")
        else:
            f.write("  → ΨCDM provides a better fit to H(z).\n")
        f.write("\n")

        f.write("Physical interpretation:\n")
        if n_best > 1:
            f.write("  • Best-fit n > 1 implies a rapidly increasing Ψ-density\n")
            f.write("    at higher redshift (non-standard behavior).\n")
        elif n_best > 0:
            f.write("  • Ψ-density grows with redshift, mimicking early dark energy.\n")
        elif n_best == 0:
            f.write("  • Constant Ψ-density (Λ-like behaviour).\n")
        else:
            f.write("  • Ψ-density decreases with redshift (ultralight-like behavior).\n")
        f.write("\n")

        f.write("Notes:\n")
        f.write("  • Only H(z) dataset was used.\n")
        f.write("  • No SN or BAO constraints included in this grid.\n")
        f.write("  • This scan tests background expansion only.\n")
        f.write("  • Perturbations and structure formation are not included.\n")
        f.write("===============================================\n")

    print(f"Saved meta summary to {meta_path}")


# ===============================================================
# MAIN
# ===============================================================

def main():

    hz_path = os.path.join(ROOT, "data", "hz", "H(z).csv")
    dfH = load_H_csv(hz_path)
    print(f"Loaded H(z): {len(dfH)} points")

    eps_grid, n_grid, chi2_map, H0_map, chi2_LCDM, H0_LCDM = scan_grid(dfH)

    outdir = os.path.join(ROOT, "results", "grid_scan")
    plot_results(eps_grid, n_grid, chi2_map, H0_map, chi2_LCDM, H0_LCDM, outdir)


if __name__ == "__main__":
    main()
