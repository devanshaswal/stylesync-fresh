import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import logging
import os
import numpy as np
from tqdm import tqdm
import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.model_resnet import FashionHybridModel, compute_f1_multilabel, compute_f1_multiclass
from utils.dataset_resnet import FashionDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resnet_eval_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ResNetEvaluator")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model_path = "best_model/best_model.pth"
image_dir = "data/processed/cropped_images"
heatmap_dir = "data/processed/heatmaps"
train_meta = "zsehran/metadata_updated_train.csv"
val_meta = "zsehran/metadata_updated_val.csv"
test_meta = "zsehran/metadata_updated_test.csv"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_dataset = FashionDataset(train_meta, image_dir, heatmap_dir, transform=transform, use_cache=False)
val_dataset = FashionDataset(val_meta, image_dir, heatmap_dir, transform=transform, use_cache=False)
test_dataset = FashionDataset(test_meta, image_dir, heatmap_dir, transform=transform, use_cache=False)


val_subset = Subset(val_dataset, list(range(0, int(len(val_dataset) * 0.15))))
test_subset = Subset(test_dataset, list(range(0, int(len(test_dataset) * 0.10))))


batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


model = FashionHybridModel()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


def evaluate(loader, name):
    logger.info(f"Evaluating on {name} set...")
    total_cat_correct = 0
    total_type_correct = 0
    total_attr_correct = 0
    total_samples = 0
    total_f1_attr = 0
    total_f1_cat = 0
    total_f1_type = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch['image'].to(device)
            heatmap = batch['heatmap'].to(device)
            attr = batch['attributes'].to(device)
            cat = batch['category_label'].to(device)
            ctype = batch['category_type'].to(device)

            outputs = model(image, heatmap)

            pred_cat = outputs['category_logits'].argmax(dim=1)
            pred_type = outputs['category_type_logits'].argmax(dim=1)
            pred_attr = (outputs['attribute_probs'] > 0.5).int()
            correct_attr = ((pred_attr == attr).all(dim=1)).sum().item()
            total_attr_correct += correct_attr


            total_cat_correct += (pred_cat == cat).sum().item()
            total_type_correct += (pred_type == ctype).sum().item()

            total_f1_attr += compute_f1_multilabel(pred_attr, attr, average='micro')
            total_f1_cat += compute_f1_multiclass(outputs['category_probs'], cat)
            total_f1_type += compute_f1_multiclass(outputs['category_type_probs'], ctype)
            total_samples += 1

    acc_cat = total_cat_correct / len(loader.dataset)
    acc_type = total_type_correct / len(loader.dataset)
    acc_attr = total_attr_correct / len(loader.dataset)
    avg_f1_cat = total_f1_cat / total_samples
    avg_f1_type = total_f1_type / total_samples 
    avg_f1_attr = total_f1_attr / total_samples

    logger.info(f"Results on {name} Set:")
    logger.info(f"  Category Accuracy   : {acc_cat:.4f}")
    logger.info(f"  Category Type Accuracy: {acc_type:.4f}")
    logger.info(f"  Attribute Accuracy     : {acc_attr:.4f}")
    logger.info(f"  F1 - Category        : {avg_f1_cat:.4f}")
    logger.info(f"  F1 - Category Type   : {avg_f1_type:.4f}")
    logger.info(f"  F1 - Attributes      : {avg_f1_attr:.4f}")



evaluate(test_loader, "Test")
