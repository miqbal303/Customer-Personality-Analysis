import os
import sys
from project.steps.data_ingestion import data_ingestion, DataIngestionParams
from project.steps.data_transformation import data_transformation
from project.steps.cluster_formation import clustering
from project.steps.model_training import model_training
import mlflow
from mlflow.exceptions import MlflowException
from project.pipelines.training_pipeline import training_pipeline
from project.logger import logging
from project.exception import CustomException



if __name__ == "__main__":
    try:
        # Create instances of the steps with parameters
        data_ingestion_step = data_ingestion(params=DataIngestionParams(data_path='project/data/marketing_campaign.csv'))
        data_transformation_step = data_transformation()
        clustering_step = clustering()
        model_training_step = model_training()

        # Run the pipeline with MLflow tracking
        with mlflow.start_run():
            training_pipeline(
                data_ingestion=data_ingestion_step,
                data_transformation=data_transformation_step,
                clustering=clustering_step,
                model_training=model_training_step
            ).run()
    except Exception as e:
        logging.error(f"Pipeline failed due to: {str(e)}")
        raise CustomException(e, sys)