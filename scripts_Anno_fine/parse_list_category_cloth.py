import os
import pandas as pd

input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "list_category_cloth.txt")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)
output_csv_path = os.path.join(output_folder, "categories_clothes_fine.csv")

def parse_list_category_cloth(file_path):
    with open(file_path, 'r') as file:
        num_categories = int(file.readline().strip())
        headers = file.readline().strip().split()
        data = []
        for line in file:
            parts = line.strip().split()
            category_name = " ".join(parts[:-1])
            category_type = int(parts[-1])
            data.append([category_name, category_type])
    df = pd.DataFrame(data, columns=headers)
    return df

df = parse_list_category_cloth(input_file_path)
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())
