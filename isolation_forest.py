from sklearn.ensemble import IsolationForest

def train_isolation_forest(X):
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(X)
    return (preds == -1).astype(int)
