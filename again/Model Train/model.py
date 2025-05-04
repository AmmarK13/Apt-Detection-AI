import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import joblib  # For saving the model

# === File Paths for Balanced Datasets ===
file_paths = [
    r"D:\4th semester\SE\project\Dataset\df_equal_FTP_BruteForce.csv",
    r"D:\4th semester\SE\project\Dataset\df_equal_SSH_BruteForce.csv"
]

# === Output Directory to Save Models ===
model_output_dir = r"D:\4th semester\SE\project\Models_heavy"
os.makedirs(model_output_dir, exist_ok=True)

# === Training Loop ===
for file_path in file_paths:
    df = pd.read_csv(file_path)

    # === Split features and target ===
    X = df.drop(columns=["Label"])
    y = df["Label"]

    # === Encode Labels: 0 = Benign, 1 = Attack ===
    y_encoded = LabelEncoder().fit_transform(y)

    # === Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # === Train model ===
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # === Evaluate on Training Data ===
    y_train_pred = model.predict(X_train)
    print(f"\n==== Training on dataset: {os.path.basename(file_path)} ====\n")
    print("Training Accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Training Precision:", precision_score(y_train, y_train_pred))
    print("Training Recall:   ", recall_score(y_train, y_train_pred))
    print("Training F1 Score: ", f1_score(y_train, y_train_pred))

    # === Evaluate on Test Data ===
    y_pred = model.predict(X_test)
    print("\n==== Evaluating model trained on: {} ====\n".format(os.path.basename(file_path)))
    print("Test Accuracy:  ", accuracy_score(y_test, y_pred))
    print("Test Precision: ", precision_score(y_test, y_pred))
    print("Test Recall:    ", recall_score(y_test, y_pred))
    print("Test F1 Score:  ", f1_score(y_test, y_pred))

    # === Save the model ===
    model_name = os.path.basename(file_path).replace(".csv", "_model.pkl")
    model_path = os.path.join(model_output_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")
