import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load features and target
X = pd.read_csv("data/raw/X.csv")
y = pd.read_csv("data/raw/y.csv")

# Extract correct target column
if "insurance_premium_category" in y.columns:
    y = y["insurance_premium_category"]
else:
    raise ValueError("Expected column 'insurance_premium_category' not found in target file.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Column groups
numerical = ["income_lpa", "bmi"]
nominal = ["occupation"]
ordinal = ["lifestyle_risk", "age_group", "city_tier"]

ordinal_categories = [
    ["low", "medium", "high"],                      # lifestyle_risk
    ["young", "adult", "middle_aged", "senior"],   # age_group
    ["1", "2", "3"]                                 # city_tier
]

# Preprocessing for LR and KNC
preprocessing = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
    ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal)
])

# Preprocessing for RFC (passthrough numerical)
preprocessing_rfc = ColumnTransformer([
    ("num", "passthrough", numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
    ("ord", OrdinalEncoder(categories=ordinal_categories), ordinal)
])

# Model pipelines
Models = {}

Models["LR"] = Pipeline([
    ("preprocessing", preprocessing),
    ("model", LogisticRegression())
])
Models["LR"].fit(X_train, y_train)

Models["KNC"] = Pipeline([
    ("preprocessing", preprocessing),
    ("model", KNeighborsClassifier())
])
Models["KNC"].fit(X_train, y_train)

Models["RFC"] = Pipeline([
    ("preprocessing", preprocessing_rfc),
    ("model", RandomForestClassifier())
])
Models["RFC"].fit(X_train, y_train)

# MLflow setup
mlflow.set_experiment("E2E_DVC")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

for model_name, model in Models.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        y_pred = model.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro")
        })
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        mlflow.log_artifacts("insurance.csv")

# Save splits
os.makedirs("data/split", exist_ok=True)
X_train.to_csv("data/split/X_train.csv", index=False)
X_test.to_csv("data/split/X_test.csv", index=False)
y_train.to_csv("data/split/y_train.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)

print("Models trained successfully and splits saved!")
