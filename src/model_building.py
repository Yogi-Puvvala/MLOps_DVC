import pandas as pd
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------------
# Load Data
# -------------------------------
X = pd.read_csv("data/raw/X.csv")
y = pd.read_csv("data/raw/y.csv").iloc[:, 0]  # ensure y is 1D

text_col = "clean_comment"
X[text_col] = X[text_col].fillna("")

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
# Preprocessing
# -------------------------------
# HashingVectorizer is memory-friendly for large datasets
preprocessing_hash = ColumnTransformer([
    ("text", HashingVectorizer(n_features=5000, alternate_sign=False), text_col)
])

# TF-IDF for MultinomialNB with limited max_features
preprocessing_tfidf = ColumnTransformer([
    ("text", TfidfVectorizer(max_features=5000), text_col)
])

# -------------------------------
# Define Models
# -------------------------------
Models = {}

# 1️⃣ SGDClassifier
sgd = Pipeline([
    ("preprocessing", preprocessing_hash),
    ("model", SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=42))
])
sgd.fit(X_train, y_train)
Models["SGD"] = sgd

# 2️⃣ Passive Aggressive Classifier
pac = Pipeline([
    ("preprocessing", preprocessing_hash),
    ("model", PassiveAggressiveClassifier(max_iter=500, random_state=42))
])
pac.fit(X_train, y_train)
Models["PAC"] = pac

# 3️⃣ MultinomialNB
mnb = Pipeline([
    ("preprocessing", preprocessing_tfidf),
    ("model", MultinomialNB())
])
mnb.fit(X_train, y_train)
Models["MNB"] = mnb

# -------------------------------
# Save combined CSV for MLflow
# -------------------------------
df = pd.concat([X, y.rename("target")], axis=1)
df.to_csv("reddit.csv", index=False)

# -------------------------------
# MLflow Logging
# -------------------------------
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
        mlflow.log_artifacts("reddit.csv")

# -------------------------------
# Save train/test splits
# -------------------------------
os.makedirs("data/split", exist_ok=True)
X_train.to_csv("data/split/X_train.csv", index=False)
X_test.to_csv("data/split/X_test.csv", index=False)
y_train.to_csv("data/split/y_train.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)

print("Models trained successfully and splits saved!")
