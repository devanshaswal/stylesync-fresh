import os

import pandas as pd

base_path = os.path.join("dataset", "Annotation", "Anno_fine")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)

cloth_attr_csv_path = os.path.join(output_folder, "attributes_clothes_fine.csv")
category_csv_path = os.path.join(output_folder, "categories_clothes_fine.csv")

cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

category_df = pd.read_csv(category_csv_path)
category_names = category_df["category_name"].tolist()

def parse_val_attr(file_path, attribute_names):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    return pd.DataFrame(data, columns=attribute_names)

def parse_val_bbox(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    return pd.DataFrame(data, columns=["x_1", "y_1", "x_2", "y_2"])

def parse_val_cate(file_path):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    return pd.DataFrame(data, columns=["category_label"])

def parse_val_landmarks(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    headers = [f"landmark_x_{i}" for i in range(1, 9)] + [f"landmark_y_{i}" for i in range(1, 9)]
    return pd.DataFrame(data, columns=headers)

def parse_val_images(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file]
    return pd.DataFrame(data, columns=["image_name"])

val_attr_df = parse_val_attr(os.path.join(base_path, "val_attr.txt"), cloth_attribute_names)
val_bbox_df = parse_val_bbox(os.path.join(base_path, "val_bbox.txt"))
val_cate_df = parse_val_cate(os.path.join(base_path, "val_cate.txt"))
val_landmarks_df = parse_val_landmarks(os.path.join(base_path, "val_landmarks.txt"))
val_images_df = parse_val_images(os.path.join(base_path, "val.txt"))

val_attr_df.to_csv(os.path.join(output_folder, "val_attributes.csv"), index=False)
val_bbox_df.to_csv(os.path.join(output_folder, "val_bbox.csv"), index=False)
val_cate_df.to_csv(os.path.join(output_folder, "val_categories.csv"), index=False)
val_landmarks_df.to_csv(os.path.join(output_folder, "val_landmarks.csv"), index=False)
val_images_df.to_csv(os.path.join(output_folder, "val_images.csv"), index=False)

print("All validation files parsed and saved successfully!")
