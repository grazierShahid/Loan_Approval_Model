import pandas as pd
import os
from typing import Tuple
import joblib

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.ingested_data_dir = "artifacts/ingested_data"
        
    def create_directories(self):
        """Create necessary directories for storing ingested data"""
        os.makedirs(self.ingested_data_dir, exist_ok=True)
    
    def save_ingested_data(self, X: pd.DataFrame, y: pd.Series):
        """Save the ingested data to artifacts directory"""
        try:
            # Save features and target separately
            X.to_csv(os.path.join(self.ingested_data_dir, "X.csv"), index=False)
            y.to_csv(os.path.join(self.ingested_data_dir, "y.csv"), index=False)
            
            # Save column information
            feature_info = {
                'numerical_features': X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_features': X.select_dtypes(include=['object']).columns.tolist()
            }
            joblib.dump(feature_info, os.path.join(self.ingested_data_dir, "feature_info.joblib"))
            
            print(f"Ingested data saved to {self.ingested_data_dir}")
            
        except Exception as e:
            print(f"Error saving ingested data: {str(e)}")
            raise e
    
    def initiate_data_ingestion(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and perform initial cleaning of the dataset
        
        Returns:
            Tuple containing features (X) and target variable (y)
        """
        try:
            # Create directories
            self.create_directories()
            
            # Load the dataset
            df = pd.read_csv(self.data_path)
            
            # Initial cleaning
            df.drop(columns=['loan_id'], inplace=True)
            df.rename(columns=lambda x: x.strip(), inplace=True)
            
            # Clean categorical columns
            categorical_features = ['education', 'self_employed']
            for col in categorical_features:
                if col in df.columns:
                    df[col] = df[col].str.strip()
            
            # Handle missing values in categorical features
            df[categorical_features].fillna('Unknown', inplace=True)
            
            # Separate features and target
            X = df.drop(columns=['loan_status'])
            y = df['loan_status'].apply(lambda x: 1 if x.strip() == 'Approved' else 0)
            
            # Save the ingested data
            self.save_ingested_data(X, y)
            
            print("Data ingestion completed successfully")
            return X, y
            
        except Exception as e:
            print(f"Error in data ingestion: {str(e)}")
            raise e

if __name__ == "__main__":
    data_ingestion = DataIngestion('dataset/loan_approval_dataset.csv')
    X, y = data_ingestion.initiate_data_ingestion()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")