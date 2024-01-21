import random
import string
import yaml
import pandas as pd
from datetime import datetime
from google.cloud import bigquery, aiplatform
from typing_extensions import Any

# Generate a uuid of a specifed length(default=8)
def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def get_query_create_test_set(
    project_id: str, bq_dataset_id: str, bq_input_table: str, train_split: float
) -> str:
    return f"""
    CREATE OR REPLACE TABLE
    `{project_id}.{bq_dataset_id}.{bq_input_table}` AS

    SELECT
    * EXCEPT (pseudo_random, income_bracket)
    FROM (
    SELECT
        *,
        RAND() AS pseudo_random 
    FROM
        `bigquery-public-data.ml_datasets.census_adult_income` )
    WHERE pseudo_random > {train_split}
    """

def load_yaml_file(path: str) -> Any:
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data


def fetch_batch_prediction_result(
        project_id: str,
        bq_dataset_id: str,
        batch_prediction_job_name: str,
) -> pd.DataFrame:
    
    # Load the batch prediction job details using the display name
    [batch_prediction_job] = aiplatform.BatchPredictionJob.list(
        filter=f'display_name="{batch_prediction_job_name}"'
    )
    # Fetch the name of the output table
    output_table = batch_prediction_job.output_info.bigquery_output_table
    print("Predictions table ID:", output_table)

    # Define the query
    query = f"""
        SELECT prediction FROM `{project_id}.{bq_dataset_id}.{output_table}`
    """
    # Initialize BigQuery client
    bq_client = bigquery.Client(
        project=project_id,
        credentials=aiplatform.initializer.global_config.credentials,
    )
   
    # Fetch the data into a dataframe
    return bq_client.query(query).to_dataframe()



def get_current_time():
    now = datetime.now()

    return f"{now}"
