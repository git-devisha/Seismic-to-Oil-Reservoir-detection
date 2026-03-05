import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from preprocess import load_and_merge, preprocess
from model import build_model

# Paths
SEISMIC_PATH = "reservoir/data/synthetic_oil_data.csv"
WELL_PATH = "data/well_logs.csv"

# Load + preprocess
data = pd.read_csv("reservoir/data/synthetic_oil_data.csv")
X = data[
    ["amplitude", "rms", "frequency", "impedance", "coherence"]
]

y = data["porosity"]   # or oil_saturation / reservoir flag

X, y, scaler = preprocess(data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = build_model()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Save
joblib.dump(model, "reservoir_model.pkl")
joblib.dump(scaler, "scaler.pkl")

import matplotlib.pyplot as plt

plt.scatter(data["porosity"], data["impedance"])
plt.xlabel("Porosity")
plt.ylabel("Impedance")
plt.title("Physical Consistency Check")
plt.show()
