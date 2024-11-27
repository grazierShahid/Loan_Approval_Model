import mlflow
import os
from mlflow.tracking import MlflowClient
import joblib
import pandas as pd
import warnings
import numpy as np

# Ignore warnings
warnings.filterwarnings("ignore")

def create_input_example():
    return pd.DataFrame({
        'income': [50000],
        'age': [35],
        'loan_amount': [200000],
        'credit_score': [700],
        'loan_term': [360],
        'employment_status': ['Employed'],
        'loan_purpose': ['Home'],
        'debt_to_income': [0.3]
    })

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set experiment name
experiment_name = "loan_approval_prediction"
mlflow.set_experiment(experiment_name)

# Create MLflow client
client = MlflowClient()

try:
    # Start MLflow run
    with mlflow.start_run(run_name="model_comparison") as run:
        input_example = create_input_example()
        
        # Load your models and metrics
        models_path = "./artifacts/models/"
        metrics_path = "./artifacts/models/model_metrics.csv"
        
        # Load metrics if available
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            print("Loaded metrics:", metrics_df.head())  # Debug print
            
            # Log metrics for each model
            for column in metrics_df.columns:
                try:
                    metrics_dict = metrics_df[column].to_dict()
                    mlflow.log_metrics({
                        f"{column}_metrics": metrics_dict[0] if len(metrics_dict) > 0 else 0
                    })
                except Exception as e:
                    print(f"Error logging metrics for {column}: {str(e)}")
        
        # Log models
        for model_file in os.listdir(models_path):
            if model_file.endswith('.joblib'):
                try:
                    model_name = model_file.replace('.joblib', '')
                    model_path = os.path.join(models_path, model_file)
                    
                    # Load model
                    model = joblib.load(model_path)
                    
                    # Log model with signature and input example
                    mlflow.sklearn.log_model(
                        model,
                        f"models/{model_name}",
                        input_example=input_example
                    )
                    print(f"Successfully logged model: {model_name}")
                    
                except Exception as e:
                    print(f"Error loading model {model_name}: {str(e)}")
                    continue

        # Register best model (assuming random_forest is best)
        model_name = "loan_approval_model"
        model_uri = f"runs:/{run.info.run_id}/models/random_forest"
        
        try:
            # Register the model
            registered_model = mlflow.register_model(model_uri, model_name)
            
            # Add model description and tags
            client.update_registered_model(
                name=model_name,
                description="Loan approval prediction model with preprocessing pipeline"
            )
            
            # Add tags
            client.set_registered_model_tag(
                name=model_name,
                key="type",
                value="classification"
            )
            
            # Transition to production
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production"
            )
            
            print(f"Successfully registered and tagged model: {model_name}")
            
        except Exception as e:
            print(f"Error registering model: {str(e)}")

        # Log dataset
        try:
            if os.path.exists("./dataset/loan_approval_dataset.csv"):
                mlflow.log_artifact("./dataset/loan_approval_dataset.csv", "data")
                print("Successfully logged dataset")
            else:
                print("Dataset file not found")
        except Exception as e:
            print(f"Error logging dataset: {str(e)}")

        # Log pipeline summaries
        pipeline_dir = "./artifacts/"
        for file in os.listdir(pipeline_dir):
            if file.startswith('pipeline_summary_'):
                mlflow.log_artifact(os.path.join(pipeline_dir, file), "pipeline_summaries")
                print(f"Successfully logged pipeline summary: {file}")

except Exception as e:
    print(f"Error in MLflow tracking setup: {str(e)}")

print("MLflow tracking setup completed!")