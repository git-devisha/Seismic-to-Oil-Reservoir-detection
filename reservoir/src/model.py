from sklearn.ensemble import RandomForestRegressor

def build_model():
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    return model
