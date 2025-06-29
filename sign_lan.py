import pandas as pd
import json

# Load the dataset
df = pd.read_csv("gpt-3.5-cleaned.csv")
# Use annotated_texts as the sign value, filter out NaN
sign_dict = {row['annotated_texts'].lower(): row['annotated_texts'].lower() 
             for index, row in df.iterrows() 
             if pd.notna(row['annotated_texts'])}

# Remove duplicates and save as JSON
sign_dict = {k: v for k, v in sign_dict.items() if k}  # Filter out empty or None
with open("sign_language_dict.json", "w") as f:
    json.dump(sign_dict, f, indent=4)

print("Dictionary created with", len(sign_dict), "entries.")