from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Create an instance of the pipeline
    pipeline_instance = training_pipeline()

    # Run the pipeline
    pipeline_instance.run()
