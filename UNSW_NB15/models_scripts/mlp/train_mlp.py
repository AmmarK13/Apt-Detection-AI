import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load training and validation data
X_train = pd.read_csv('UNSW_NB15/prepared_data/X_train.csv')
y_train = pd.read_csv('UNSW_NB15/prepared_data/y_train.csv').values.ravel()
X_val = pd.read_csv('UNSW_NB15/prepared_data/X_val.csv')
y_val = pd.read_csv('UNSW_NB15/prepared_data/y_val.csv').values.ravel()

# ✅ Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)  # Use the same scaler as training!

# Define Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # Input Layer
    keras.layers.Dense(32, activation="relu"),  # Hidden Layer
    keras.layers.Dense(1, activation="sigmoid")  # Output Layer (binary classification)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model and the scaler
model.save("UNSW_NB15/models/mlp_model.h5")

joblib.dump(scaler, "UNSW_NB15/models/scaler.pkl")  # Save the scaler for later use

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")
