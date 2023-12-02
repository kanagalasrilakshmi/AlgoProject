import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define the number of rows and columns in the dataset
num_rows = 1000000
num_columns = 100

# Create a random dataset with duplicates
data = np.random.randint(1, 10, size=(num_rows, num_columns))
df = pd.DataFrame(data, columns=[f'Column_{i}' for i in range(1, num_columns + 1)])

# Introduce duplicates in the dataset
duplicate_indices = np.random.choice(df.index, size=int(0.1 * num_rows), replace=False)
df.loc[duplicate_indices] = df.iloc[np.random.choice(df.index, size=len(duplicate_indices))].values

# Reset index
df.reset_index(drop=True, inplace=True)

df.groupby(df.columns.tolist(),as_index=False).size()

df.to_csv('../data/data.csv', index=False)
