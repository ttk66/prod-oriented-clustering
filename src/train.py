import mlflow
import mlflow.sklearn
from src.config import VERSION, MLFLOW_TRACKING_URI, RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.clustering import cluster_data

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris_clustering")

    df = load_data()
    df_scaled = preprocess(df, FEATURES)

    with mlflow.start_run(run_name=f"clustering_v{VERSION}"):
        clustered_df, model = cluster_data(df_scaled, n_clusters=3, random_state=RANDOM_STATE)

        mlflow.sklearn.log_model(model, artifact_path="kmeans_model")
        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("features", FEATURES)
        mlflow.log_param("version", VERSION)

        # Логируем первые несколько строк для проверки
        mlflow.log_text(clustered_df.head().to_csv(index=False), "clustered_head.csv")

        print("Кластеризация завершена. MLflow логирование прошло успешно.")

if __name__ == "__main__":
    main()