# Web Attack Detection AI Model

This project implements an AI model for detecting web attacks using the CSIC database. The project follows a structured approach with separate components for data cleaning, feature engineering, and model development using a Random Forest classifier.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned and processed data
│   ├── features/         # Feature engineered datasets
│   └── split/            # Train/test split datasets
├── code/                 # Source code
│   ├── analysis/         # Data analysis scripts
│   ├── cleaning/         # Data cleaning scripts
│   ├── encode/           # Encoding modules
│   │   ├── categorical/  # Categorical encoding methods
│   │   └── numerical/    # Numerical encoding methods
│   ├── model/            # Model training and evaluation
│   └── pip/              # Pipeline components
├── src/                  # Legacy source code
├── notebooks/           # Jupyter notebooks for EDA
├── docs/                # Documentation files
└── requirements.txt     # Project dependencies
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Components

### 1. Data Cleaning
Handles data preprocessing, including:
- Removing duplicates
- Handling missing values
- Data type validation
- Anomaly detection

### 2. Feature Engineering
- Hybrid encoding (combination of one-hot and label encoding based on cardinality)
- Feature scaling using MinMax scaler
- Feature selection and transformation

### 3. Model Development
- Random Forest classifier implementation
- Model training and evaluation
- Performance metrics (accuracy, precision, recall, F1 score)
- Confusion matrix visualization

## Pipeline Architecture

The project implements a modular pipeline architecture with the following components:

1. **Pipeline Manager**: Orchestrates the execution of all pipeline steps
2. **Data Cleaning Step**: Preprocesses the raw CSIC dataset
3. **Hybrid Encoding Step**: Applies appropriate encoding to categorical features
4. **MinMax Scaler Step**: Normalizes numerical features
5. **Model Step**: Trains and evaluates the Random Forest model

## Usage

### Running the Complete Pipeline

To execute the entire data processing and model training pipeline:

```bash
python newMethodMain.py
```

This script will:
1. Clean the raw CSIC dataset
2. Apply hybrid encoding to categorical features
3. Scale numerical features using MinMax scaling
4. Train and evaluate the Random Forest model

### Individual Component Execution

1. Data Cleaning:
```bash
python src/cleaning/clean_data.py
```

2. Feature Engineering:
```bash
python src/features/build_features.py
```

3. Model Training:
```bash
python src/models/train_model.py
```

## Model Performance

The Random Forest model achieves high performance in detecting web attacks with metrics including:
- Accuracy
- Precision
- Recall
- F1 Score

Detailed performance metrics are generated during model evaluation.

## Contributing

Please follow the project structure and coding conventions when contributing to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.