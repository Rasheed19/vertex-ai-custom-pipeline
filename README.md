# vertex-ai-custom-pipeline
Vertex custom training with Vertex AI using a prebuilt Docker container and BigQuery source data.  

This project is based on the Google's demo which can be 
found in https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/pipelines/custom_tabular_train_batch_pred_bq_pipeline.ipynb

## Folder analysis
1. `config` contains the component and pipeline configuration files
1. `components` contains veterx component python files
1. `pipelines` contains veterx pipeline python files
1. `utils` contains helper functions 

## Set up for running locally
1. Clone the repository by running
    ```
    git clone https://github.com/Rasheed19/vertex-ai-custom-pipeline
    ```
1. To agree with the custom job pipeline, rename the root folder to `python_package`.
1. Navigate to the root folder, i.e., `python_package` and create a python virtual environment by running
    ```
    python3.10 -m venv .venv
    ``` 
1. Activate the virtual environment by running
    ```
    source .venv/bin/activate
    ```
1. Upgrade `pip` by running 
   ```
   pip install --upgrade pip
   ``` 
1. Install all the required Python libraries by running 
    ```
    pip install -r requirements.txt
    ```
1. Create a file named `.env` in the root folder and store the following variables related to your GCP:
    ```
    PROJECT_ID=your-project-id
    REGION=your-project-region
    BUCKET_URI=gs://your-project-name
    SERVICE_ACCOUNT=your-service-account
   ```
1. In the root directory, create a source distribution for the training by running:
   ```
   python3 setup.py sdist --formats=gztar
   ```
   This should create `dist/trainer-0.1.tar.gz.`

1. Copy the created source distribution to your cloud storage bucket by running:
   ```
   gsutil cp -r python_package/dist/* BUCKET_URI/training_package/
   ```
   where the `BUCKET_URI` is your bucket URI

1. Run the following commands in your terminal to configure the pipeline run on the Vertex AI (make sure 
   `gcloud CLI` is installed on your computer):
   1. Login:
       ```
       gcloud auth login
       ```
   1.  Configure the login to use your prefered project:
        ```
        gcloud config set project your-prpject-id
        ```
    1. Get and save your user account credentials:
          ```
          gcloud auth application-default login
          ```
    1. Grant access to the pipeline to use your storage bucket
        ```
        gsutil iam ch serviceAccount:your-service-account:roles/storage.objectCreator gs://your-project-name
        ```

        ```
        gsutil iam ch user:your-gmail-address:objectCreator gs://your-project-name
        ```

1. Then run the pipeline that trains, registers, and performs batch prediction on the Vertex AI by running one of the following customised commands in your terminal:
    1. Run the pipeline with default options
        ```
        python run.py
        ```
            
    1. Run the pipeline without caching:
       ```
       python run.py --no-cache
       ````
1. To fetch the batch prediction data, run the following command:
   ```
    python run_fetch_batch_prediction.py
   ```
