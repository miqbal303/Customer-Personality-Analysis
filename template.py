import os

# Define the directory structure
dirs = [
    "project/steps",
    "project/pipelines",
    "project/deployment",
    "project/data",
    "project/artifacts"
]

# Define the files to create
files = [
    "project/steps/__init__.py",
    "project/steps/data_ingestion.py",
    "project/steps/data_transformation.py",
    "project/steps/clustering.py",
    "project/steps/model_training.py",
    "project/pipelines/__init__.py",
    "project/pipelines/training_pipeline.py",
    "project/deployment/__init__.py",
    "project/deployment/deploy_to_sagemaker.py",
    "project/deployment/inference.py",
    "project/data/marketing_campaign.csv",
    "project/run_pipeline.py",
    "project/logger.py"
]

# Create directories
for dir in dirs:
    os.makedirs(dir, exist_ok=True)
    print(f"Created directory: {dir}")

# Create files
for file in files:
    with open(file, 'w') as f:
        f.write("")  # Create an empty file
    print(f"Created file: {file}")
