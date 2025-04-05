# Web Attack Detection AI Model

This project implements an AI model for detecting web attacks using the CSIC database. The project follows a structured approach with separate components for data cleaning, feature engineering, and model development.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned and processed data
│   └── features/         # Feature engineered datasets
├── src/                  # Source code
│   ├── cleaning/         # Data cleaning scripts
│   ├── features/         # Feature engineering code
│   ├── models/           # Model training and evaluation
│   └── utils/           # Utility functions and helpers
├── notebooks/           # Jupyter notebooks for EDA
├── tests/               # Unit tests
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
- Text feature extraction
- Pattern recognition
- Feature scaling and normalization
- Feature selection

### 3. Model Development
- Model training
- Hyperparameter tuning
- Model evaluation
- Performance metrics

## Usage

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

## Contributing

Please follow the project structure and coding conventions when contributing to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.