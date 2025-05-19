import pandas as pd
import os


def drop_1_uniquevalue_cols(input_path,output_path):
    df= pd.read_csv(input_path)

    cols_with_one_val = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=cols_with_one_val, inplace=True)
    
    print(f"Dropped {len(cols_with_one_val)} columns with only one unique value:")
    for col in cols_with_one_val:
        print(f" - {col}")
    
    df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")



input_path= r"D:\4th semester\SE\project\Dataset\Processed_Timestamp_Encoded.csv"
output_path= r"D:\4th semester\SE\project\Dataset\Dropped_1_unique.csv"


drop_1_uniquevalue_cols(input_path,output_path)