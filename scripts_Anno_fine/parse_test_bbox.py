import os
import pandas as pd

input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "test_bbox.txt")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)
output_csv_path = os.path.join(output_folder, "test_bbox.csv")

def parse_test_bbox(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            parts = line.strip().split()
            x1, y1, x2, y2 = map(int, parts)
            data.append([x1, y1, x2, y2])
    df = pd.DataFrame(data, columns=["x_1", "y_1", "x_2", "y_2"])
    return df

df = parse_test_bbox(input_file_path)
df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())
