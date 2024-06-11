import sys
from zenml.pipelines import pipeline
from project.steps.data_ingestion import data_ingestion, DataIngestionParams
from project.steps.data_transformation import data_transformation
from project.steps.cluster_formation import clustering
from project.steps.model_training import model_training
import mlflow
from mlflow.exceptions import MlflowException
from logger import logging
from exception import CustomException

# Set the MLflow tracking URI to point to the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Function to create experiment if it doesn't exist
def get_or_create_experiment(experiment_name):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logging.info(f"Experiment '{experiment_name}' not found. Creating new experiment.")
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        logging.info(f"Using experiment '{experiment_name}' with ID {experiment_id}")
        return experiment_id
    except MlflowException as e:
        logging.error(f"An error occurred: {e}")
        raise CustomException(e, sys)

# Set the MLflow experiment name
experiment_name = "customer_analysis_pipeline"
experiment_id = get_or_create_experiment(experiment_name)

# Set the experiment ID to be used in the pipeline
mlflow.set_experiment(experiment_id=experiment_id)

@pipeline
def training_pipeline(data_ingestion, data_transformation, clustering, model_training):
    raw_data_path = data_ingestion()
    transformed_data_path, preprocessor_path = data_transformation(raw_data_path=raw_data_path)
    cluster_file_path = clustering(transformed_data_path=transformed_data_path)
    model_training(transformed_data_path=transformed_data_path, cluster_file_path=cluster_file_path)


