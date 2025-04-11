import os
import pandas as pd

base_path = os.path.join("dataset", "Annotation", "Anno_fine")
output_folder = os.path.join("dataset", "organized_data_1")
os.makedirs(output_folder, exist_ok=True)

cloth_attr_csv_path = os.path.join("dataset", "organized_data_1", "attributes_clothes_fine.csv")
category_csv_path = os.path.join("dataset", "organized_data_1", "categories_clothes_fine.csv")

cloth_df = pd.read_csv(cloth_attr_csv_path)
cloth_attribute_names = cloth_df["attribute_name"].tolist()

category_df = pd.read_csv(category_csv_path)
category_names = category_df["category_name"].tolist()

def parse_train_attr(file_path, attribute_names):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    return pd.DataFrame(data, columns=attribute_names)

def parse_train_bbox(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    return pd.DataFrame(data, columns=["x_1", "y_1", "x_2", "y_2"])

def parse_train_cate(file_path):
    with open(file_path, 'r') as file:
        data = [int(line.strip()) for line in file]
    return pd.DataFrame(data, columns=["category_label"])

def parse_train_landmarks(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(int, line.strip().split())) for line in file]
    headers = []
    for i in range(1, 9):
        headers.append(f"landmark_location_x_{i}")
        headers.append(f"landmark_location_y_{i}")
    return pd.DataFrame(data, columns=headers)

def parse_train_images(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file]
    return pd.DataFrame(data, columns=["image_name"])

train_attr_df = parse_train_attr(os.path.join(base_path, "train_attr.txt"), cloth_attribute_names)
train_bbox_df = parse_train_bbox(os.path.join(base_path, "train_bbox.txt"))
train_cate_df = parse_train_cate(os.path.join(base_path, "train_cate.txt"))
train_landmarks_df = parse_train_landmarks(os.path.join(base_path, "train_landmarks.txt"))
train_images_df = parse_train_images(os.path.join(base_path, "train.txt"))

train_attr_df.to_csv(os.path.join(output_folder, "train_attributes.csv"), index=False)
train_bbox_df.to_csv(os.path.join(output_folder, "train_bbox.csv"), index=False)
train_cate_df.to_csv(os.path.join(output_folder, "train_categories.csv"), index=False)
train_landmarks_df.to_csv(os.path.join(output_folder, "train_landmarks.csv"), index=False)
train_images_df.to_csv(os.path.join(output_folder, "train_images.csv"), index=False)

print("All training set files parsed and saved successfully!")
