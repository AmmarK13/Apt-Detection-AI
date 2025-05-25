import pandas as pd

# === File Path ===
file_path =r"D:\4th semester\SE\project\Dataset\afewstepsleft.csv"   # Replace with your dataset path

# === Load the dataset ===
df = pd.read_csv(file_path)

# === Randomize the rows ===
df = df.sample(frac=1)  # Randomize rows' sequence

# === Specify attack types to balance against "Benign" ===
attack_labels = ["FTP-BruteForce", "SSH-Bruteforce"]  # Add more attack types if needed

# Initialize an empty dictionary to store balanced data
balanced_data = {}

# === Loop through each attack type and balance the dataset ===
for attack in attack_labels:
    df_attack = df[df["Label"] == attack]
    df_benign = df[df["Label"] == "Benign"][:df_attack.shape[0]]  # Balance "Benign" samples with attack samples
    
    # Concatenate the "Benign" data with the attack data
    balanced_df = pd.concat([df_benign, df_attack], axis=0)
    
    # Add the balanced dataframe to the dictionary
    balanced_data[attack] = balanced_df

    # Output the balanced dataset
    output_path = f"D:\\4th semester\\SE\\project\\Dataset\\balanced_{attack}.csv"
    balanced_df.to_csv(output_path, index=False)
    print(f"Balanced dataset for {attack} saved at {output_path}")

