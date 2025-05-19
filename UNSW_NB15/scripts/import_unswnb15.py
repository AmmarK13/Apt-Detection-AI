import kaggle

# Download UNSW-NB15 dataset from Kaggle
kaggle.api.dataset_download_files('dhoogla/unswnb15', path='data/', unzip=True)
print("Dataset downloaded to 'data/' successfully")
