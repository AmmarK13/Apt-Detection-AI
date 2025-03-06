import polars as pl

# Convert Training Set
df_train = pl.read_parquet("data/UNSW_NB15_training-set.parquet")
df_train.write_csv("data/UNSW_NB15_training-set.csv")

# Convert Testing Set
df_test = pl.read_parquet("data/UNSW_NB15_testing-set.parquet")
df_test.write_csv("data/UNSW_NB15_testing-set.csv")

