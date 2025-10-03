import mlflow
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

client = mlflow.MlflowClient()

exp = client.get_experiment_by_name("E2E_DVC")
exp_id = exp.experiment_id
artifact_loc = exp.artifact_location

best_run = client.search_runs(
    experiment_ids = [exp_id],
    max_results = 1,
    order_by = ["metrics.accuracy DESC"]
)[0]

# print(best_run)
run_id = best_run.info.run_id
run_name = best_run.info.run_name
run_uri = f"runs:/{run_id}/{run_name}"

model = mlflow.pyfunc.load_model(run_uri)

X_test = pd.read_csv("data/split/X_test.csv")
y_test = pd.read_csv("data/split/y_test.csv")

pred_vals = model.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, pred_vals),
    "Precision": precision_score(y_test, pred_vals, average="macro"),
    "Recall": precision_score(y_test, pred_vals, average="macro")
}

os.makedirs("data/metrics", exist_ok=True)

with open("data/metrics/metrics.json", "w") as file:
    json.dump(metrics, file)