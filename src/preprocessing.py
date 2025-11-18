from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess(df: pd.DataFrame, features: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df_scaled = df.fit_transform(df[features])
    return pd.DataFrame(df_scaled, columns=features)

