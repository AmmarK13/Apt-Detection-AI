import kaggle

# Download CIC-IDS-2017 dataset
kaggle.api.dataset_download_files('chethuhn/network-intrusion-dataset', path='data/', unzip=True)

# Download UNSW-NB15 dataset
kaggle.api.dataset_download_files('dhoogla/unswnb15', path='data/', unzip=True)

# Download CSIC-2010 Web Application Attacks dataset
kaggle.api.dataset_download_files('ispangler/csic-2010-web-application-attacks', path='data/', unzip=True)

# Download CSE-CIC-IDS2018 dataset
kaggle.api.dataset_download_files('solarmainframe/ids-intrusion-csv', path='data/', unzip=True)

print("Datasets downloaded successfully!")
