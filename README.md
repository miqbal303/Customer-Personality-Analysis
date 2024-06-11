# Customer-Personality-Analysis
ZenML MLOps framework

## Work In Progress

conda create -p venv python=3.9 -y && conda activate venv\

conda activate venv\

zenml init


Run MLFlow Server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

