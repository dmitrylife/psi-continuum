# utils/io_loaders.py

import numpy as np
import pandas as pd
import os


def parse_H_value(text):
    """
    Parse strings like '83 ± 8' or '83+/-8' into (H, sigma).

    If the format cannot be parsed safely, return (nan, nan).
    """
    if text is None:
        return np.nan, np.nan

    s = str(text).strip().replace("−", "-")  # normalize minus sign

    # Detect error delimiter
    if "±" in s:
        parts = s.split("±")
    elif "+/-" in s:
        parts = s.split("+/-")
    else:
        return np.nan, np.nan

    if len(parts) != 2:
        return np.nan, np.nan

    try:
        h = float(parts[0].strip())
        sig = float(parts[1].strip())
        return h, sig
    except ValueError:
        return np.nan, np.nan


def load_H_csv(path):
    """
    Load H(z) dataset from a CSV file with columns:
      - 'z'
      - 'H(z) [km/s/Mpc]'  (string with 'value ± sigma')

    Returns a DataFrame with:
      z, H, sigma_H

    Rows with invalid H±σ values are skipped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df_raw = pd.read_csv(path)

    if "z" not in df_raw.columns:
        raise RuntimeError("Missing column 'z' in H(z) CSV.")
    if "H(z) [km/s/Mpc]" not in df_raw.columns:
        raise RuntimeError("Missing column 'H(z) [km/s/Mpc]' in H(z) CSV.")

    Z = df_raw["z"].astype(float).values
    H_list = []
    S_list = []

    for val in df_raw["H(z) [km/s/Mpc]"]:
        h, s = parse_H_value(val)
        if np.isfinite(h) and np.isfinite(s):
            H_list.append(h)
            S_list.append(s)

    if len(H_list) == 0:
        raise RuntimeError("No valid 'H ± σ' rows found.")

    # Trim Z if some rows were invalid
    Z = Z[:len(H_list)]

    return pd.DataFrame(
        {"z": Z, "H": H_list, "sigma_H": S_list}
    ).sort_values("z").reset_index(drop=True)
