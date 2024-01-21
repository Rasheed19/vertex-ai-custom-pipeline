import os
import joblib
import argparse
from google.cloud import storage
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier

"""
Note that this prject uses a prebuilt Docker container and thus
must follow all the rules of Vertex AI. All the preprocessing and 
training codes must be included in this python file.
"""

# Read environmental variables
PROJECT = os.getenv("CLOUD_ML_PROJECT_ID")
TRAINING_DATA_URI = os.getenv("AIP_TRAINING_DATA_URI")

# Set Bigquery Client
bq_client = bigquery.Client(project=PROJECT)
storage_client = storage.Client(project=PROJECT)

# Define the constants
TARGET = 'income_bracket'
ARTIFACTS_PATH = os.getenv("AIP_MODEL_DIR")
# Get the bucket name from the model dir
BUCKET_NAME = ARTIFACTS_PATH.replace("gs://","").split("/")[0]

MODEL_FILENAME = 'model.joblib' 
# Define the format of your input data, excluding the target column.
# These are the columns from the census data files.
COLUMNS = [
    'age',
    'workclass',
    'functional_weight',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country'
]
# Categorical columns are columns that need to be turned into a numerical value to be used by scikit-learn
CATEGORICAL_COLUMNS = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native_country'
]

# Function to fetch the data from BigQuery
def download_table(bq_table_uri: str):
    prefix = "bq://"
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix):]

    table = bigquery.TableReference.from_string(bq_table_uri)
    rows = bq_client.list_rows(
        table,
    )
    return rows.to_dataframe(create_bqstorage_client=False)

# Function to upload local files to GCS
def upload_model(bucket_name: str,
                filename: str):
     # Upload the saved model file to GCS
    bucket = storage_client.get_bucket(bucket_name)
    storage_path = os.path.join(ARTIFACTS_PATH, filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
    blob.upload_from_filename(filename)
    

if __name__ == '__main__':
    # Load the training data
    X_train = download_table(TRAINING_DATA_URI)

    # Remove the column we are trying to predict ('income-level') from our features list
    # Convert the Dataframe to a lists of lists
    train_features = X_train.drop(TARGET, axis=1).to_numpy().tolist()
    # Create our training labels list, convert the Dataframe to a lists of lists
    train_labels = X_train[TARGET].to_numpy().tolist()

    # Since the census data set has categorical features, we need to convert
    # them to numerical values. We use a list of pipelines to convert each
    # categorical column and then use FeatureUnion to combine them before calling
    # the RandomForestClassifier.
    categorical_pipelines = []

    # Each categorical column needs to be extracted individually and converted to a numerical value.
    # To do this, each categorical column use a pipeline that extracts one feature column via
    # SelectKBest(k=1) and a LabelBinarizer() to convert the categorical value to a numerical one.
    # A scores array (created below) selects and extracts the feature column. The scores array is
    # created by iterating over the COLUMNS and checking if it is a CATEGORICAL_COLUMN.
    for i, col in enumerate(COLUMNS):
        if col in CATEGORICAL_COLUMNS:
            # Create a scores array to get the individual categorical column.
            # Example:
            #  data = [39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married', 'Adm-clerical',
            #         'Not-in-family', 'White', 'Male', 2174, 0, 40, 'United-States']
            #  scores = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #
            # Returns: [['Sate-gov']]
            scores = []
            # Build the scores array
            for j in range(len(COLUMNS)):
                if i == j: # This column is the categorical column we want to extract.
                    scores.append(1) # Set to 1 to select this column
                else: # Every other column should be ignored.
                    scores.append(0)
            skb = SelectKBest(k=1)
            skb.scores_ = scores
            # Convert the categorical column to a numerical value
            lbn = LabelBinarizer()
            r = skb.transform(train_features)
            lbn.fit(r)
            # Create the pipeline to extract the categorical feature
            categorical_pipelines.append(
                (
                    f'categorical-{i}',
                    Pipeline([(f'SKB-{i}', skb), (f'LBN-{i}', lbn)]),
                )
            )

    # Create pipeline to extract the numerical features
    skb = SelectKBest(k=6)
    # From COLUMNS use the features that are numerical
    skb.scores_ = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    categorical_pipelines.append(('numerical', skb))

    # Combine all the features using FeatureUnion
    preprocess = FeatureUnion(categorical_pipelines)

    # Create the classifier
    classifier = RandomForestClassifier()

    # Transform the features and fit them to the classifier
    classifier.fit(preprocess.transform(train_features), train_labels)

    # Create the overall model as a single pipeline
    pipeline = Pipeline([
        ('union', preprocess),
        ('classifier', classifier)
    ])

    # Save the pipeline locally
    joblib.dump(pipeline, MODEL_FILENAME)  # will I need to create a folder to save the model locally

    # Upload the locally saved model to GCS
    upload_model(bucket_name = BUCKET_NAME, 
                 filename=MODEL_FILENAME
                )