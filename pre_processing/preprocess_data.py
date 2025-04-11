import hashlib
import logging
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.paths import FINAL_MERGED_CSV, PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def normalize_path(path, base_dir=None):
   
    if path.startswith("img/"):
        path = path[len("img/"):]
  
    path = os.path.normpath(path)
 
    return os.path.join(base_dir, path) if base_dir else path

def get_image_hash(image_path):
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def crop_and_save_image(row, image_dir, output_dir, target_size=(224, 224)):
    img_relative_path = normalize_path(row["image_name"])
    img_path = normalize_path(img_relative_path, base_dir=image_dir)
    
    if not os.path.exists(img_path):
        logger.warning(f"Image {img_path} not found. Skipping.")
        return None

    if row["x_1"] >= row["x_2"] or row["y_1"] >= row["y_2"]:
        logger.warning(f"Invalid bounding box for {row['image_name']}. Skipping.")
        return None
    
    try:
        image_hash = get_image_hash(img_path)
        save_folder = os.path.join(output_dir, os.path.dirname(img_relative_path))
        os.makedirs(save_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_relative_path))[0]  
        save_path = os.path.join(save_folder, f"{base_name}.jpg")

        
        if os.path.exists(save_path) and get_image_hash(save_path) == image_hash:
            return row["image_name"], target_size
        
        image = Image.open(img_path).convert("RGB")
        cropped = image.crop((row["x_1"], row["y_1"], row["x_2"], row["y_2"]))
        resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
        resized.save(save_path, format="JPEG", quality=100, subsampling=0)
        
        return row["image_name"], target_size
    except Exception as e:
        logger.error(f"Error processing {row['image_name']}: {e}")
        return None

def crop_and_save_images(df, image_dir, output_dir, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    image_sizes = {}

    logger.info(f"Processing images from {image_dir}...")

    args = [(row._asdict(), image_dir, output_dir, target_size) for row in df.itertuples(index=False)]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.starmap(crop_and_save_image, args), total=len(df), desc="Cropping and resizing images"))

    metadata = []
    for result in results:
        if result is not None:
            image_name, img_size = result
            image_sizes[image_name] = img_size
            
           
            original_image_name = image_name 
            save_path = os.path.join(output_dir, normalize_path(original_image_name))
            
            metadata.append([original_image_name, img_size, save_path])

  
    metadata_df = pd.DataFrame(metadata, columns=["image_name", "image_size", "cropped_image_path"])
    metadata_csv_path = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    logger.info(f"Metadata file saved at {metadata_csv_path} with {len(metadata_df)} entries.")

    return image_sizes


def create_landmark_heatmap(image_size, landmarks, sigma=5):
   
    height, width = image_size
    xx, yy = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")

    if len(landmarks) == 0:
        return np.zeros((height, width), dtype=np.float32)

  
    landmarks = np.array(landmarks)

    
    squared_distances = (xx[None, :, :] - landmarks[:, 0, None, None]) ** 2 + \
                        (yy[None, :, :] - landmarks[:, 1, None, None]) ** 2

    heatmap = np.exp(-squared_distances / (2 * sigma ** 2)).sum(axis=0)
    heatmap /= heatmap.max() if heatmap.max() > 0 else 1 

    return heatmap.astype(np.float32)

def process_heatmap_row(row, image_sizes, output_dir):
    """Process a single row for heatmap generation."""
    img_relative_path = normalize_path(row["image_name"])
    if row["image_name"] not in image_sizes:
        return None  

    img_size = image_sizes[row["image_name"]]
    subfolder = os.path.dirname(img_relative_path)


    landmarks = [(row[f"landmark_location_x_{i}"] * img_size[0], 
                  row[f"landmark_location_y_{i}"] * img_size[1])
                 for i in range(1, 9) 
                 if not pd.isna(row[f"landmark_location_x_{i}"]) and not pd.isna(row[f"landmark_location_y_{i}"])]

    if not landmarks:
        return None

    heatmap = create_landmark_heatmap(img_size, landmarks)

    base_name = os.path.splitext(os.path.basename(img_relative_path))[0] 
    save_folder = os.path.join(output_dir, subfolder)
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, f"{base_name}.npy"), heatmap)

    return row["image_name"]

def generate_and_save_heatmaps(df, image_sizes, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    args = [(dict(row._asdict()), image_sizes, output_dir) for row in df.itertuples(index=False)]


    logger.info("Generating heatmaps using multiprocessing...")
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.starmap(process_heatmap_row, args, chunksize=100), 
                  total=len(df), desc="Generating heatmaps"))

    logger.info("Heatmap generation complete!")


if __name__ == "__main__":
    logger.info(f"Using dataset: {FINAL_MERGED_CSV}")
    if os.path.exists(FINAL_MERGED_CSV):
        merged_df = pd.read_csv(FINAL_MERGED_CSV)
        
      
        logger.info("Generating metadata...")
        cropped_images_dir = os.path.join(PROCESSED_DATA_DIR, "cropped_images")
        metadata_csv_path = os.path.join(PROCESSED_DATA_DIR, "metadata.csv")

    
        metadata = []
        for root, _, files in os.walk(cropped_images_dir):
            for file in files:
                if file.endswith(".jpg"):
                   
                    img_relative_path = os.path.relpath(os.path.join(root, file), PROCESSED_DATA_DIR)
                    
                  
                    base_name = os.path.splitext(file)[0]
                 
                    metadata.append([img_relative_path, (224, 224), os.path.join(cropped_images_dir, img_relative_path)])

        metadata_df = pd.DataFrame(metadata, columns=["image_name", "image_size", "cropped_image_path"])
        metadata_df.to_csv(metadata_csv_path, index=False)
        logger.info(f"New metadata file saved at {metadata_csv_path} with {len(metadata_df)} entries.")

   
        logger.info("Processing complete!")
    else:
        logger.error(f"Dataset not found: {FINAL_MERGED_CSV}")


















