import pandas as pd
import numpy as np

# ----------------------------
# Load your real reservoir data
# ----------------------------
df = pd.read_csv("reservoir/data/synthetic_oil_data.csv")  
# Must contain at least: porosity OR oil_saturation OR reservoir_flag

np.random.seed(42)

# ----------------------------
# If porosity exists (best case)
# ----------------------------
if "porosity" in df.columns:
    poro = df["porosity"].values

else:
    # If only reservoir flag exists
    poro = np.where(
        df["reservoir"] == 1,
        np.random.uniform(0.18, 0.30, len(df)),
        np.random.uniform(0.05, 0.12, len(df))
    )
    df["porosity"] = poro

# ----------------------------
# Generate seismic attributes
# ----------------------------

# Amplitude ↑ with porosity
df["amplitude"] = (
    0.8 * poro +
    np.random.normal(0, 0.03, len(df))
)

# RMS amplitude (always positive)
df["rms"] = np.abs(
    1.2 * poro +
    np.random.normal(0, 0.05, len(df))
)

# Frequency ↓ in hydrocarbon zones
df["frequency"] = (
    45 - 60 * poro +
    np.random.normal(0, 3, len(df))
)

# Acoustic impedance ↓ with porosity
df["impedance"] = (
    9000 - 20000 * poro +
    np.random.normal(0, 300, len(df))
)

# Coherence ↑ in clean reservoirs
df["coherence"] = np.clip(
    0.6 + 0.8 * poro + np.random.normal(0, 0.05, len(df)),
    0, 1
)

# ----------------------------
# Save final ML-ready dataset
# ----------------------------
df.to_csv("reservoir/data/synthetic_oil_data.csv", index=False)

print("✅ Synthetic seismic attributes generated")
print(df.head())
