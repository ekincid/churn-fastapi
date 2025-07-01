import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = "models:/telco_churn_model/Production"
output_dir = "exported_model"

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=output_dir)

print(f"Model '{model_uri}' klasöre indirildi: {output_dir}")
