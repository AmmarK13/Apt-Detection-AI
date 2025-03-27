### **🔥 All Your Datasets Are Labeled! ✅**

Since all datasets contain a **"Label"** column (or similar like **"classification"**), we can now proceed with **Supervised Learning** to train a model for **Threat Detection**.

---

## **📌 Next Steps: Training a Machine Learning Model**

### **1️⃣ Preprocessing the Data**

Before training, we need to:  
✅ Drop unnecessary columns (`Timestamp`, `Source IP`, `Destination IP`, etc.)  
✅ Convert categorical labels (`BENIGN`, `DDoS`, etc.) into numbers  
✅ Normalize numerical features

Here’s the **optimized preprocessing script**:

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set cleaned data directory
cleaned_dir = "data/cleaned/"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith(".csv")]

# Initialize an empty list to store dataframes
dfs = []

# Load and preprocess each dataset
for file in csv_files:
    file_path = os.path.join(cleaned_dir, file)
    
    try:
        # Load dataset
        df = pd.read_csv(file_path, low_memory=False)
        print(f"\n📂 Processing {file} - {df.shape[0]} rows, {df.shape[1]} columns")

        # Drop unnecessary columns
        columns_to_drop = ["Timestamp", "Source IP", "Destination IP", "Flow ID"]  # Drop non-numeric columns if they exist
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors="ignore")

        # Check if there's a label column
        possible_labels = ["attack", "attack_type", "label", "category", "class"]
        label_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in possible_labels):
                label_col = col
                break
        
        if label_col:
            # Encode categorical labels into numbers
            df[label_col] = LabelEncoder().fit_transform(df[label_col])
            df.rename(columns={label_col: "Label"}, inplace=True)
            dfs.append(df)
            print(f"✅ Processed {file} with label column: {label_col}")
        else:
            print(f"❌ No label column found in {file}, skipping...")
    
    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")

# Merge all processed datasets
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"\n🚀 Final Merged Dataset: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    
    # Separate features and target
    X = final_df.drop(columns=["Label"])  # Features
    y = final_df["Label"]  # Target (attack type)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("✅ All data preprocessed and ready for model training!")
else:
    print("❌ No valid datasets found for training.")


```

---

### **2️⃣ Train a Random Forest Classifier**

We start with **Random Forest**, a powerful supervised learning model.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"🔥 Model Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
```

---

### **3️⃣ What’s Next?**

✅ **Check accuracy** (above **85% is good**)  
✅ **Try more models** → **XGBoost, SVM, Neural Networks**  
✅ **Deploy the model** for real-time **Intrusion Detection**

---

