import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path: str, test_data_path: str, output_dir: str):
        """Initialize the ModelTester with model and data paths.

        Args:
            model_path (str): Path to the trained model file
            test_data_path (str): Path to the test data CSV file
            output_dir (str): Directory to save test results
        """
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.model = None
        self.X_test = None
        self.y_test = None

    def load_model(self):
        """Load the trained model."""
        try:
            self.model = load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_test_data(self):
        """Load and prepare test data."""
        try:
            # Load test data
            test_df = pd.read_csv(self.test_data_path)
            
            # Separate features and target
            self.y_test = test_df['classification']
            self.X_test = test_df.drop('classification', axis=1)
            
            logger.info(f"Test data loaded with shape: {self.X_test.shape}")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise

    def evaluate_model(self):
        """Evaluate model performance using various metrics."""
        try:
            # Make predictions
            y_pred_proba = self.model.predict(self.X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted')
            }

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            logger.info("Model Performance Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")

            return metrics, conf_matrix

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save_results(self, metrics: dict, conf_matrix: np.ndarray):
        """Save evaluation results to files."""
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics to CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_path = self.output_dir / 'test_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)

            # Save confusion matrix to CSV
            conf_matrix_df = pd.DataFrame(conf_matrix)
            conf_matrix_path = self.output_dir / 'confusion_matrix.csv'
            conf_matrix_df.to_csv(conf_matrix_path, index=False)

            logger.info(f"Test metrics saved to: {metrics_path}")
            logger.info(f"Confusion matrix saved to: {conf_matrix_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def test(self):
        """Execute the complete model testing pipeline."""
        logger.info("Starting model testing process...")
        
        self.load_model()
        self.load_test_data()
        metrics, conf_matrix = self.evaluate_model()
        self.save_results(metrics, conf_matrix)
        
        logger.info("Model testing completed successfully")
        return metrics

def main():
    # Define paths
    model_path = "src/models/web_attack_model.h5"  # Updated to match the saved model name in train_model.py
    test_data_path = "data/split/test_data.csv"
    output_dir = "results"
    
    # Initialize and run the tester
    tester = ModelTester(model_path, test_data_path, output_dir)
    try:
        metrics = tester.test()
    except Exception as e:
        logger.error(f"Error during model testing: {e}")

if __name__ == "__main__":
    main()