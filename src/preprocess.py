import pandas as pd

# Load dataset
data = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded Successfully!")

# Check missing values
print(data.isnull().sum())

# Remove duplicates
data = data.drop_duplicates()

print("Duplicates Removed!")

# Save cleaned data
data.to_csv("data/processed_data.csv", index=False)

print("Processed Data Saved Successfully!")