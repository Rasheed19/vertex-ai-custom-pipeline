from kfp.dsl import pipeline

@pipeline(name="custom-model-bq-batch-prediction-pipeline")
def custom_model_bq_batch_prediction_pipeline(
    project: str,
    location: str,
    dataset_display_name: str,
    dataset_bq_source: str,
    training_job_dispaly_name: str,
    gcs_staging_directory: str,
    python_package_gcs_uri: str,
    python_package_module_name: str,
    training_split: float,
    test_split: float,
    training_container_uri: str,
    serving_container_uri: str,
    training_bigquery_destination: str,
    model_display_name: str,
    batch_prediction_display_name: str,
    batch_prediction_instances_format: str,
    batch_prediction_predictions_format: str,
    batch_prediction_source_uri: str,
    batch_prediction_destination_uri: str,
    batch_prediction_machine_type: str = "n1-standard-4",
    batch_prediction_batch_size: int = 1000,
):
    from google_cloud_pipeline_components.aiplatform import (
        CustomPythonPackageTrainingJobRunOp, ModelBatchPredictOp,
        TabularDatasetCreateOp)

    # Create the dataset
    dataset_create_op = TabularDatasetCreateOp(
        project=project,
        location=location,
        display_name=dataset_display_name,
        bq_source=dataset_bq_source,
    )

    # Run the training task
    train_op = CustomPythonPackageTrainingJobRunOp(
        display_name=training_job_dispaly_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=python_package_module_name,
        container_uri=training_container_uri,
        model_display_name=model_display_name,
        model_serving_container_image_uri=serving_container_uri,
        dataset=dataset_create_op.outputs["dataset"],
        base_output_dir=gcs_staging_directory,
        bigquery_destination=training_bigquery_destination,
        training_fraction_split=training_split,
        test_fraction_split=test_split,
        staging_bucket=gcs_staging_directory,
    )

    # Run the batch prediction task
    _ = ModelBatchPredictOp(
        project=project,
        location=location,
        model=train_op.outputs["model"],
        instances_format=batch_prediction_instances_format,
        bigquery_source_input_uri=batch_prediction_source_uri,
        predictions_format=batch_prediction_predictions_format,
        bigquery_destination_output_uri=batch_prediction_destination_uri,
        job_display_name=batch_prediction_display_name,
        machine_type=batch_prediction_machine_type,
        manual_batch_tuning_parameters_batch_size=batch_prediction_batch_size,
    )