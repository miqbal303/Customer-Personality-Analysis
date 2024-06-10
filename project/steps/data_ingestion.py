import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from zenml.steps import step, BaseParameters
from logger import logging
import mlflow
from exception import CustomException

class DataIngestionParams(BaseParameters):
    data_path: str

@step
def data_ingestion(params: DataIngestionParams) -> str:
    try:
        logging.info("Starting data ingestion process.")
        df = pd.read_csv(params.data_path, sep=";")
        raw_data_path = os.path.join("artifacts", "raw.csv")
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        df.to_csv(raw_data_path, index=False)
        
        # Correlation matrix plot
        numeric_df = df.select_dtypes(include=[float, int])
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(25, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        corr_matrix_path = os.path.join("artifacts", "corr_matrix.png")
        plt.savefig(corr_matrix_path)
        plt.close()
        mlflow.log_artifact(corr_matrix_path)

        # Age distribution plot
        sns.histplot(df['Year_Birth'], bins=30, kde=True)
        plt.title('Age Distribution')
        age_dist_path = os.path.join("artifacts", "age_distribution.png")
        plt.savefig(age_dist_path)
        plt.close()
        mlflow.log_artifact(age_dist_path)

        # Total spending distribution plot
        df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
        sns.histplot(df['Total_Spending'], bins=30, kde=True)
        plt.title('Total Spending Distribution')
        total_spending_dist_path = os.path.join("artifacts", "total_spending_distribution.png")
        plt.savefig(total_spending_dist_path)
        plt.close()
        mlflow.log_artifact(total_spending_dist_path)

        logging.info("Data ingestion process completed successfully.")
        return raw_data_path
    except Exception as e:
        raise CustomException(e, sys)
