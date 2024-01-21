from google.cloud import aiplatform, bigquery
from dotenv import dotenv_values
import click
from utils import get_query_create_test_set, load_yaml_file
from pipelines import custom_model_bq_batch_prediction_pipeline, pipeline_compiler


@click.command(
    help="""
    Vertex AI custom pipeline with classification problem case.

    This project is based on the Google's demo which can be 
    found in https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/custom_tabular_train_batch_pred_bq_pipeline.ipynb


    Run the model training pipeline with various
    options.

    Examples:

    \b
    # Run the pipeline with default options
    python run.py
                
    \b
    # Enable caching; default is False
    python run.py --no-cache

    """
)
@click.option(
    "--cache/--no-cache",
    default=False,
    help="Enable caching or not",
)
def main(cache: bool = False):
    config = dotenv_values(".env")
    train_config = load_yaml_file("./config/training_config.yaml")

    # Initialize Vertex AI SDK
    aiplatform.init(
        project=config["PROJECT_ID"],
        location=config["REGION"],
        staging_bucket=config["BUCKET_URI"],
    )

    # Initialize BigQuery client
    bq_client = bigquery.Client(
        project=config["PROJECT_ID"],
        credentials=aiplatform.initializer.global_config.credentials,
    )

    # Source of the dataset
    DATA_SOURCE = "bq://bigquery-public-data.ml_datasets.census_adult_income"
    # Set name for the managed Vertex AI dataset
    DATASET_DISPLAY_NAME = "adult_census_dataset"
    # BigQuery Dataset name
    BQ_DATASET_ID = "income_prediction"
    # Set name for the BigQuery source table for batch prediction
    BQ_INPUT_TABLE = "income_test_data"
    # Set the size(%) of the train set
    TRAIN_SPLIT = 0.9  # we want 10% of the data for batch prediction; the rest will also be spliited latter
    # Provide the container for training the model
    TRAINING_CONTAINER = (
        "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest"
    )
    # Provide the container for serving the model
    SERVING_CONTAINER = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-23:latest"
    # Set the display name for training job
    TRAINING_JOB_DISPLAY_NAME = "income_classify_train_job"
    # Model display name for Vertex AI Model Registry
    MODEL_DISPLAY_NAME = "income_classify_model"
    # Set the name for batch prediction job
    BATCH_PREDICTION_JOB_NAME = "income_classify_batch_pred"
    # Dispaly name for the Vertex AI Pipeline
    PIPELINE_DISPLAY_NAME = "income_classfiy_batch_pred_pipeline"
    # Filename to compile the pipeline to
    PIPELINE_FILE_NAME = f"{PIPELINE_DISPLAY_NAME}.json"

    # Create a BQ dataset
    bq_dataset = bigquery.Dataset(f"{config['PROJECT_ID']}.{BQ_DATASET_ID}")
    bq_dataset = bq_client.create_dataset(bq_dataset)
    print(f"Created dataset {bq_client.project}.{bq_dataset.dataset_id}")

    # Query to create a test set from the source table
    query = get_query_create_test_set(
        project_id=config["PROJECT_ID"],
        bq_dataset_id=BQ_DATASET_ID,
        bq_input_table=BQ_INPUT_TABLE,
        train_split=TRAIN_SPLIT,
    )
    # Run the query
    _ = bq_client.query(query)

    # Call and compile pipeline
    pipeline_compiler(
        pipeline=custom_model_bq_batch_prediction_pipeline,
        config_path_name=f"./config/{PIPELINE_FILE_NAME}",
    )

    # Define the parameters for running the pipeline
    parameters = {
        "project": config["PROJECT_ID"],
        "location": config["REGION"],
        "dataset_display_name": DATASET_DISPLAY_NAME,
        "dataset_bq_source": DATA_SOURCE,
        "training_job_dispaly_name": TRAINING_JOB_DISPLAY_NAME,
        "gcs_staging_directory": config["BUCKET_URI"],
        "python_package_gcs_uri": f"{config['BUCKET_URI']}/training_package/trainer-0.1.tar.gz",
        "python_package_module_name": "trainer.task",
        "training_split": train_config["train_split"],
        "test_split": 1 - train_config["train_split"],
        "training_container_uri": TRAINING_CONTAINER,
        "serving_container_uri": SERVING_CONTAINER,
        "training_bigquery_destination": f"bq://{config['PROJECT_ID']}",
        "model_display_name": MODEL_DISPLAY_NAME,
        "batch_prediction_display_name": BATCH_PREDICTION_JOB_NAME,
        "batch_prediction_instances_format": "bigquery",
        "batch_prediction_predictions_format": "bigquery",
        "batch_prediction_source_uri": f"bq://{config['PROJECT_ID']}.{BQ_DATASET_ID}.{BQ_INPUT_TABLE}",
        "batch_prediction_destination_uri": f"bq://{config['PROJECT_ID']}.{BQ_DATASET_ID}",
    }

    # Create a Vertex AI Pipeline job
    job = aiplatform.PipelineJob(
        display_name=PIPELINE_DISPLAY_NAME,
        template_path=f"./config/{PIPELINE_FILE_NAME}",
        parameter_values=parameters,
        enable_caching=cache,
    )
    # Run the pipeline job
    job.submit()


if __name__ == "__main__":
    main()
