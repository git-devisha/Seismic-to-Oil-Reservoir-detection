import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_merge(seismic_path, well_path):
    seismic = pd.read_csv(seismic_path)
    wells = pd.read_csv(well_path)

    data = seismic.merge(wells, on=["x", "y"])
    return data

def preprocess(data):
    X = data.drop(columns=["porosity", "x", "y"])
    y = data["porosity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
