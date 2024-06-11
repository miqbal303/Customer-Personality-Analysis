import os
import pandas as pd
from sklearn.cluster import KMeans
import mlflow
import matplotlib.pyplot as plt
from zenml.steps import step
from project.logger import logging

@step
def clustering(transformed_data_path: str) -> str:
    data = pd.read_csv(transformed_data_path)
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(data)

    output_path = os.path.join("artifacts", "clusters.csv")
    pd.DataFrame(clusters, columns=["Cluster"]).to_csv(output_path, index=False)
    
    # Log artifacts and visualizations to MLflow
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=clusters, cmap='viridis')
    plt.title('Customer Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt_path = os.path.join("artifacts", "cluster_visualization.png")
    plt.savefig(plt_path)
    mlflow.log_artifact(plt_path)
    plt.close()

    mlflow.log_artifact(output_path)

    return output_path
