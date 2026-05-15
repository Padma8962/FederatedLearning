import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import SMOTE


def load_data(path):
    data = pd.read_csv(path)

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")
    sm = SMOTE()
    X, y = sm.fit_resample(X, y)

    return X.astype("float32"), y.astype("float32")