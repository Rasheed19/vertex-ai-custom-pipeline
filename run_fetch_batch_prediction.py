from utils import fetch_batch_prediction_result, get_current_time
from dotenv import dotenv_values


def main():
    config = dotenv_values(".env")
    batch_result = fetch_batch_prediction_result(
        project_id=config["PROJECT_ID"],
        bq_dataset_id="income_prediction",
        batch_prediction_job_name="income_classify_batch_pred",
    )

    batch_result.to_csv(f"./data/batch_prediction_{get_current_time()}.csv")

    print(batch_result)


if __name__ == "__main__":
    main()
