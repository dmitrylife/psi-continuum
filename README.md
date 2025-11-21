# Psi-Continuum Cosmology  
### A Phenomenological Extension of ΛCDM Based on Non-Equilibrium Response  
**Author:** Dmitry Vasilevich Klimov
**Project website:** https://psi-continuum.org
**Year:** 2025

## 📘 Overview

**Psi-Continuum Cosmology (ΨCDM)** is a phenomenological extension of ΛCDM that introduces a **non-equilibrium response component** into the background expansion history.

The goal is **not** to replace ΛCDM. Instead, ΨCDM quantifies how much flexibility exists in late-time cosmology while remaining consistent with:

- Pantheon+SH0ES Supernovae
- Cosmic Chronometers H(z)

This repository contains:

- fully reproducible Python analysis scripts
- comparisons of ΛCDM and ΨCDM
- a grid scan of (varepsilon_0, n)
- automatically generated figures for the article
- clean utility modules for cosmology, data loading, and χ² computation

## 📂 Repository Structure

```
psi-continuum/
├── data/
│   ├── Pantheon+/
│   │   ├── Pantheon+SH0ES.dat
│   │   └── Pantheon+SH0ES_STAT+SYS.cov
│   │
│   └── hz/
│       └── H(z).csv
│
├── scripts/
│   ├── 01_fit_pantheonplus_sh0es.py
│   ├── 02_two_model_pantheon_sh0es.py
│   ├── 03_psi_final_test.py
│   └── 04_psi_grid_scan.py
│
├── utils/
│   ├── cosmology.py
│   ├── io_loaders.py
│   └── psi_equations.py
│
├── results/
│   ├── pantheon_fit/
│   ├── pantheon_compare/
│   ├── psi_final_outputs/
│   └── grid_scan/
│
├── theory/
│
├── README.md
└── requirements.txt
```

## 🧪 Scientific Scripts

### **1. 01_fit_pantheonplus_sh0es.py**  
Fits (Omega_m) using Pantheon+SH0ES (Hubble-flow SN subset) with **full covariance** and analytic marginalization over (M).

### **2. 02_two_model_pantheon_sh0es.py**  
Direct comparison of fixed ΛCDM vs ΨCDM models using the same SN dataset.

### **3. 03_psi_final_test.py**  
Combined comparison using:
- H(z) chronometers
- Pantheon+ diagonal-only SN data
Outputs all figures used in the article.

### **4. 04_psi_grid_scan.py**  
Performs a 2D scan over (varepsilon_0, n) while minimizing (H_0) at each grid point.

### **Optional (not included in publication):**  
The BAO module is not used due to incomplete data from the author. Its release is planned for a future issue.

## 📊 Summary of Numerical Results

### **Pantheon+SH0ES SN-only (full covariance)**  
- Best-fit matter density:
  \(\Omega_m \approx 0.497\)
- \(\chi^2 = 240.81\)
- dof ≈ 276

### **Two-model comparison (fixed models)**  
\[
\Delta\chi^2 \approx +0.64
\]  
ΛCDM slightly preferred (as expected).

### **Grid scan (H(z)-only)**  
Best parameters:
- \( \varepsilon_0 = 0.3 \)
- \( n = 1.05 \)
- \( H_0 \approx 72 \)
- Δχ² ≈ +38.75 relative to ΛCDM

## ▶️ Installation & Usage

```bash
python3 -m venv sci_venv
source sci_venv/bin/activate
pip install -r requirements.txt

# Example full comparison run
python scripts/03_psi_final_test.py
```

Python ≥ 3.10 recommended.

## 📝 Limitations

- Only background expansion is considered
- **Perturbation theory is not implemented yet**
- BAO dataset incomplete (future update)
- ΨCDM is phenomenological, not a field theory

## 📚 Citation

If you use this repository or figures in your research:

**Dmitry Vasilevich Klimov (2025).
*Psi-Continuum Cosmology: A Phenomenological Extension of ΛCDM Based on Non-Equilibrium Response and a Unified State Field.***

Zenodo. https://doi.org/10.5281/zenodo.17666099

## 📮 Contact

📧 Email: **d.klimov.psi@gmail.com**
🌐 Website: **https://psi-continuum.org**
