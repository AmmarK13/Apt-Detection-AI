### **Project Title**: APT DETECTION AI 
### Attacks Handled: DDOS Attacks 

### Dataset: Network Intrusion dataset(CIC-IDS- 2017)
### **Prepared By**: [Khadijah Farooqi]
### **Date**: [2nd May 2025]

---

### **1. Objectives**

Ensure that the machine learning pipeline:

- Delivers high detection accuracy.
- Handles noisy or incomplete data gracefully.
- Is robust, maintainable, and scalable.

---

### **2. Standards to Follow**

- **PEP8** for Python code formatting.    
- **CRISP-DM** methodology for ML lifecycle.
- **IEEE 829** for test documentation (adapted).
- **Data privacy** ensured by anonymizing input samples.

---

### **3. SQA Activities**

| Activity                                  | Description                                                                         | Tools                           |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------- |
| **Data Quality Checks**                   | Handle missing values, drop low variance features, scale and balance data as needed | pandas, numpy                   |
| **Step-by-step Preprocessing Validation** | Validate each preprocessing step from missing value handling to label check         | Custom Python functions         |
| **Model Training and Testing**            | Ensure model is trained and evaluated on properly split data                        | scikit-learn                    |
| **Transformation for New Data**           | Apply the same transformation to test data as training                              | pandas, saved training metadata |
| **Unit Testing**                          | Test each transformation and model method                                           | pytest                          |
| **Functional Testing**                    | End-to-end pipeline test with known datasets                                        | Python                          |
| **Performance Testing**                   | Check for model speed and efficiency                                                | time, memory-profiler           |
| **Logging and Error Handling**            | Log issues at each preprocessing and prediction step                                | logging module                  |
### **4. Detailed ML Pipeline Steps (QA Checklist)**

|Step|QA Criteria|Status|
|---|---|---|
|1. Missing Value Handling|Are NaNs replaced using strategy (e.g., mean or sampled)?|✅|
|2. Low Variance Check|Are low variance features identified?|✅|
|3. Feature Drop|Are irrelevant/constant features removed?|✅|
|4. Scaling Need Check|Is data distribution checked before scaling?|✅|
|5. Feature Scaling|Are features scaled appropriately (e.g., MinMaxScaler)?|✅|
|6. Balancing Data|Is class distribution checked before balancing?|✅|
|7. Label Balancing|Is label column rebalanced using under/over sampling?|✅|
|8. Label Check|Are labels binary or multiclass and correct?|✅|
|9. Splitting Data|Is data split into train/test with shuffling?|✅|
|10. Training Model|Is the model trained using cleaned and processed data?|✅|
|11. Testing Model|Is accuracy and performance evaluated using test data?|✅|
|12. Data Transformation|Is transformation applied consistently for new/test data?|✅|

---

### **5. Test Plan**

|Test Case ID|Test Description|Expected Result|
|---|---|---|
|TC01|Load dataset and check for missing values|Missing values handled|
|TC02|Identify and drop low variance features|Low info features removed|
|TC03|Determine if scaling is needed|Scaling decision made|
|TC04|Apply scaling if required|Scaled values between defined range|
|TC05|Check class imbalance|Class distribution printed|
|TC06|Apply balancing techniques|Balanced dataset created|
|TC07|Check labels after balancing|Labels verified|
|TC08|Split into train and test|Train/Test split with consistent shape|
|TC09|Train model on training data|Model trained with no errors|
|TC10|Evaluate model on test data|Accuracy/precision/recall printed|
|TC11|Run data_transform on raw input|Transformed input returned|
|TC12|Predict on transformed test data|Predictions generated without error|

---

### **6. Tools & Environment**

- **Programming Language**: Python 3.11
- **Libraries**: scikit-learn, pandas, numpy, joblib
- **Environment**: Parrot OS / Linux
- **IDE**: Visual Studio Code
- **Testing**: pytest
- **Version Control**: Git 

---

### **7. Roles and Responsibilities**

| Role          | Name       | Responsibility                           |
| ------------- | ---------- | ---------------------------------------- |
| Data Engineer | [Khadijah] | Preprocessing, Transformation, Balancing |
| ML Engineer   | [Khadijah] | Model Training and Evaluation            |
| QA Engineer   | [Khadijah] | SQA Activities, Testing, Validation      |

---