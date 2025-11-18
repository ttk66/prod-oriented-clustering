from sklearn.cluster import KMeans
import pandas as pd

def cluster_data(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(df)
    return df, kmeans