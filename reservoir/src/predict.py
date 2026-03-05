import pandas as pd
import joblib

model = joblib.load("reservoir_model.pkl")
scaler = joblib.load("scaler.pkl")

new_seismic = pd.read_csv("reservoir/data/synthetic_oil_data.csv")
X_new = scaler.transform(new_seismic)

porosity_pred = model.predict(X_new)
new_seismic["predicted_porosity"] = porosity_pred

new_seismic.to_csv("predicted_reservoir.csv", index=False)
