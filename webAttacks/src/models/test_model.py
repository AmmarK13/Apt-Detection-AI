import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

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
        """Evaluate model performance using various metrics and attack pattern analysis."""
        try:
            # Make predictions
            y_pred_proba = self.model.predict(self.X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate overall metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted')
            }

            # Analyze performance for each attack pattern using the correct feature names
            attack_patterns = {
                'sql_injection': 'sql_indicator',
                'xss': 'xss_indicator',
                'path_traversal': 'path_traversal',
                'command_injection': 'cmd_injection'
            }
            for attack_type, feature_name in attack_patterns.items():
                if feature_name in self.X_test.columns:
                    pattern_mask = self.X_test[feature_name] == 1
                    if pattern_mask.any():
                        pattern_metrics = {
                            f'{attack_type}_precision': precision_score(self.y_test[pattern_mask], y_pred[pattern_mask], average='binary'),
                            f'{attack_type}_recall': recall_score(self.y_test[pattern_mask], y_pred[pattern_mask], average='binary'),
                            f'{attack_type}_f1': f1_score(self.y_test[pattern_mask], y_pred[pattern_mask], average='binary')
                        }
                        metrics.update(pattern_metrics)
                    logger.info(f"Performance for {attack_type}: {pattern_metrics}")


            # Calculate confusion matrix
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            logger.info("Model Performance Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.4f}")

            return metrics, conf_matrix, y_pred

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save_results(self, metrics: dict, conf_matrix: np.ndarray, y_pred):
        """Save evaluation results to files and generate visualizations."""
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

            # Generate and save confusion matrix visualization
            self.plot_confusion_matrix(conf_matrix)
            
            # Generate and save metrics visualization
            self.plot_metrics(metrics)

            logger.info(f"Test metrics saved to: {metrics_path}")
            logger.info(f"Confusion matrix saved to: {conf_matrix_path}")
            logger.info(f"Visualizations saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
            
    def plot_confusion_matrix(self, conf_matrix):
        """Plot and save confusion matrix visualization."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix_viz.png')
        plt.close()
        
    def plot_metrics(self, metrics):
        """Plot and save metrics visualization."""
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            metrics.keys(),
            metrics.values(),
            color=['#2C7BB6', '#D7191C', '#FDAE61', '#ABD9E9']
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
            
        plt.ylim(0, 1.1)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_viz.png')
        plt.close()

    def test(self):
        """Execute the complete model testing pipeline."""
        logger.info("Starting model testing process...")
        
        self.load_model()
        self.load_test_data()
        metrics, conf_matrix, y_pred = self.evaluate_model()
        self.save_results(metrics, conf_matrix, y_pred)
        
        logger.info("Model testing completed successfully")
        return metrics

def main():
    # Define paths
    model_path = "src/models/web_attack_model.h5"  # Updated path to match where train_model.py saves the model
    test_data_path = "data/features/engineered_features.csv"  # Using the engineered features
    output_dir = "results"
    
    # Initialize and run the tester
    tester = ModelTester(model_path, test_data_path, output_dir)
    try:
        metrics = tester.test()
    except Exception as e:
        logger.error(f"Error during model testing: {e}")

if __name__ == "__main__":
    main()