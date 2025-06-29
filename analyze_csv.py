import pandas as pd

# Load the CSV file
df = pd.read_csv("gpt-3.5-cleaned.csv")

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))
print("Non-NaN annotated_texts:", df['annotated_texts'].notna().sum())
print("\nFirst 10 rows:")
print(df.head(10))
print("\nSample non-NaN annotated_texts:")
non_nan_df = df[df['annotated_texts'].notna()]
print(non_nan_df.head(10)) 