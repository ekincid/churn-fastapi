import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# 🔧 MLflow tracking URI: localhost:5000'de çalışan sqlite bazlı MLflow UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco-Churn")

# Veri yükle
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Temizlik
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(float)
df["tenure"] = df["tenure"].astype(float)
df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Giriş ve hedef ayrımı
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# Sayısal ve kategorik sütunlar
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = list(set(X.columns) - set(num_cols))

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Tam pipeline: preprocessing + model
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Veri bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow deney başlat
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    acc = float(np.round(accuracy_score(y_test, y_pred), 4))
    loss = float(np.round(log_loss(y_test, y_prob), 4))

    mlflow.log_param("model_type", "LogisticRegression + pipeline")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("log_loss", loss)

    input_example = X_train.iloc[[0]].copy()
    signature = infer_signature(X_train, pipeline.predict_proba(X_train))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        registered_model_name="telco_churn_model",
        input_example=input_example,
        signature=signature
    )
