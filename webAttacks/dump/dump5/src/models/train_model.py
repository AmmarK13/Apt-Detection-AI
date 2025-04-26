import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebAttackModel:
    def __init__(self, input_file: str, output_dir: str):
        """Initialize the WebAttackModel with input and output paths.

        Args:
            input_file (str): Path to the feature engineered CSV file
            output_dir (str): Directory to save model artifacts
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load the feature engineered dataset."""
        try:
            self.df = pd.read_csv(self.input_file)
            logger.info(f"Loaded feature engineered dataset with shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def prepare_data(self):
        """Prepare data for model training."""
        try:
            # Separate features and target
            X = self.df.drop('classification', axis=1)
            y = self.df['classification']

            # Split data into train and test sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Training set shape: {self.X_train.shape}")
            logger.info(f"Test set shape: {self.X_test.shape}")
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise

    def build_model(self):
        """Build an enhanced neural network model optimized for attack detection."""
        try:
            self.model = Sequential([
                # Input layer with increased capacity for new features
                Dense(512, activation='relu', input_shape=(self.X_train.shape[1],),
                      kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.4),
                
                # First hidden layer for pattern recognition
                Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Second hidden layer for attack pattern analysis
                Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Third layer for high-level feature abstraction
                Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.2),
                
                # Output layer
                Dense(1, activation='sigmoid')
            ])

            # Use Adam optimizer with custom learning rate
            optimizer = Adam(learning_rate=0.001)
            
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            logger.info("Model built successfully")
        except Exception as e:
            logger.error(f"Error in model building: {e}")
            raise

    def train_model(self, epochs=100, batch_size=64):
        """Train the model with advanced callbacks and monitoring."""
        try:
            # Early stopping with increased patience
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Learning rate reduction on plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001
            )

            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )

            # Save training history plot
            self.plot_training_history(history)
            logger.info("Model training completed")
            return history
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def evaluate_model(self):
        """Evaluate the model and generate performance metrics."""
        try:
            # Generate predictions
            y_pred = (self.model.predict(self.X_test) > 0.5).astype(int)

            # Calculate metrics
            report = classification_report(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            # Plot confusion matrix
            self.plot_confusion_matrix(conf_matrix)

            logger.info("\nClassification Report:")
            logger.info(f"\n{report}")

            # Save model
            model_path = self.output_dir / 'web_attack_model.h5'
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise

    def plot_training_history(self, history):
        """Plot training history."""
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()

    def plot_confusion_matrix(self, conf_matrix):
        """Plot confusion matrix."""
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
        plt.savefig(self.output_dir / 'confusion_matrix.png')
        plt.close()

def main():
    # Define input and output paths
    input_file = "data/features/engineered_features.csv"
    output_dir = "src/models"

    # Initialize and run the model
    attack_model = WebAttackModel(input_file, output_dir)
    try:
        attack_model.load_data()
        attack_model.prepare_data()
        attack_model.build_model()
        attack_model.train_model()
        attack_model.evaluate_model()
        logger.info("Model training and evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error during model training: {e}")

if __name__ == "__main__":
    main()