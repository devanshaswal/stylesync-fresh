import os
import pandas as pd

input_file_path = os.path.join("dataset", "Annotation", "Anno_fine", "list_attr_img.txt")
cloth_attr_csv_path = os.path.join("dataset", "organized_data_1", "attributes_clothes_fine.csv")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)
output_csv_path = os.path.join(output_folder, "attributes_images_fine.csv")

def parse_list_attr_img(file_path, cloth_attribute_names):
    with open(file_path, 'r') as file:
        num_images = int(file.readline().strip())
        headers = file.readline().strip().split()
        columns = ["image_name"] + cloth_attribute_names
        data = []
        for line in file:
            parts = line.strip().split()
            image_name = parts[0]
            attribute_labels = list(map(int, parts[1:]))
            data.append([image_name] + attribute_labels)

    df = pd.DataFrame(data, columns=columns)
    return df

cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

df = parse_list_attr_img(input_file_path, cloth_attribute_names)

df.to_csv(output_csv_path, index=False)
print(f"Parsed data saved to {output_csv_path}")
print(df.head())
