import os
import pandas as pd

input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "test_attr.txt")
cloth_attr_csv_path = os.path.join("dataset", "organized_data_1", "attributes_clothes_fine.csv")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)
output_csv_path = os.path.join(output_folder, "test_attributes.csv")

def parse_test_attr(file_path, cloth_attribute_names):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            attribute_labels = list(map(int, line.strip().split()))
            data.append(attribute_labels)
    df = pd.DataFrame(data, columns=cloth_attribute_names)
    return df

cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

df = parse_test_attr(input_file_path, cloth_attribute_names)

df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())
