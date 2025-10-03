import pandas as pd
import mlflow
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

X = pd.read_csv("data/raw/X.csv")
y = pd.read_csv("data/raw/y.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

preprocessing = ColumnTransformer([
    ("comments", TfidfVectorizer(), "clean_comment")  # no need for list here
])

Models = {}

# KNC Model

KNC = Pipeline([
    ("preprocessing", preprocessing),
    ("model", KNeighborsClassifier())
])
KNC.fit(X_train, y_train)
Models["KNC"] = KNC

# NBC Model

NBC = Pipeline([
    ("preprocessing", preprocessing),
    ("model", MultinomialNB())
])
NBC.fit(X_train, y_train)
Models["NBC"] = NBC

# RFC Model

RFC = Pipeline([
    ("preprocessing", preprocessing),
    ("model", RandomForestClassifier())
])
RFC.fit(X_train, y_train)
Models["RFC"] = RFC

df = pd.concat([X, y], axis = 1)
df.to_csv("reddit.csv")

mlflow.set_experiment("E2E_DVC")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

for model_name, model in Models.items():
    with mlflow.start_run(run_name = model_name):
        # log params
        model_params = model.get_params()
        mlflow.log_params(model_params)

        # log metrics
        pred_vals = model.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, pred_vals),
            "precision": precision_score(y_test, pred_vals, average = "macro"),
            "recall": recall_score(y_test, pred_vals, average = "macro")
        })

        # log model
        mlflow.sklearn.log_model(model, artifact_path = model_name)

        # log artifacts
        mlflow.log_artifacts("reddit.csv")

os.makedirs("data/split", exist_ok=True)

X_train.to_csv("data/split/X_train.csv")
X_test.to_csv("data/split/X_test.csv")
y_train.to_csv("data/split/y_train.csv")
y_test.to_csv("data/split/y_test.csv")