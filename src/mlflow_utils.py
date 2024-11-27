import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import json

class MLflowTracker:
    def __init__(self, experiment_name="loan_approval_experiment"):
        self.client = MlflowClient()
        self.experiment_name = experiment_name
        
        # Set up experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)

    def log_data_characteristics(self, df: pd.DataFrame, data_version: str):
        """Log data characteristics and versioning information"""
        with mlflow.start_run(nested=True) as run:
            # Log basic data stats
            mlflow.log_params({
                "data_version": data_version,
                "n_rows": len(df),
                "n_features": len(df.columns)
            })
            
            # Log feature distributions
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    mlflow.log_metrics({
                        f"{column}_mean": df[column].mean(),
                        f"{column}_std": df[column].std(),
                        f"{column}_missing": df[column].isnull().sum()
                    })
            
            # Log feature names and types
            feature_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
            mlflow.log_dict(feature_info, "feature_info.json")

    def log_model_training(self, model, model_params, metrics, preprocessor=None):
        """Log model training details and artifacts"""
        with mlflow.start_run(nested=True) as run:
            # Log parameters
            mlflow.log_params(model_params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model with preprocessor
            if preprocessor:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    preprocessor=preprocessor
                )
            else:
                mlflow.sklearn.log_model(model, "model")
            
            return run.info.run_id

    def register_best_model(self, run_id, model_name="loan_approval_model"):
        """Register the best model in MLflow Model Registry"""
        result = mlflow.register_model(
            f"runs:/{run_id}/model",
            model_name
        )
        
        # Add model description and tags
        self.client.update_registered_model(
            name=model_name,
            description="Loan approval prediction model with preprocessing pipeline"
        )
        
        # Transition to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        
        return result.version