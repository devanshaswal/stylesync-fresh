import os
import sys
import torch
import logging
import random
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Set up project path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.vit_model import FashionViTModel
from utils.dataset_vit import FashionDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ViT-Test-Eval")

# Constants
TEST_METADATA = "zsehran/metadata_updated_test.csv"
IMAGES_DIR = "data/processed/cropped_images"
HEATMAPS_DIR = "data/processed/test_heatmaps"
CHECKPOINT_PATH = "best_model/vit_base_patch16_224_best.pth"

ATTRIBUTE_GROUPS = {
    'color_print': ['red', 'pink', 'floral', 'striped', 'stripe', 'print', 'printed', 'graphic', 'love', 'summer'],
    'neckline': ['v-neck', 'hooded', 'collar', 'sleeveless', 'strapless', 'racerback', 'muscle'],
    'silhouette_fit': ['slim', 'boxy', 'fit', 'skinny', 'shift', 'bodycon', 'maxi', 'mini', 'midi', 'a-line'],
    'style_construction': ['crochet', 'knit', 'woven', 'lace', 'denim', 'cotton', 'chiffon', 'mesh'],
    'details': ['pleated', 'pocket', 'button', 'drawstring', 'trim', 'hem', 'capri', 'sleeve', 'flare', 'skater',
                'sheath', 'shirt', 'pencil', 'classic', 'crop']
}
group_sizes = {k: len(v) for k, v in ATTRIBUTE_GROUPS.items()}


def main():
    logger.info("Starting ViT evaluation on 10% test subset")

    # Transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset
    full_test_dataset = FashionDataset(
        metadata_path=TEST_METADATA,
        cropped_images_dir=IMAGES_DIR,
        heatmaps_dir=HEATMAPS_DIR,
        transform=test_transform,
        attribute_groups=ATTRIBUTE_GROUPS,
        use_cache=True,
        cache_size=100,
        validate_files=False,
    )

    subset_size = int(0.10 * len(full_test_dataset))
    subset_indices = random.sample(range(len(full_test_dataset)), subset_size)
    test_dataset = Subset(full_test_dataset, subset_indices)
    logger.info(f"Using {len(test_dataset)} samples for testing (10% subset)")

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionViTModel(
        num_categories=46,
        num_category_types=3,
        group_sizes=group_sizes,
        vit_name='vit_base_patch16_224',
        use_pretrained=False,
        enable_deep_supervision=False
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    logger.info("Model loaded and ready for evaluation")

    # Evaluation containers
    all_preds_cat, all_targets_cat = [], []
    all_preds_type, all_targets_type = [], []
    group_preds, group_targets = {g: [] for g in group_sizes}, {g: [] for g in group_sizes}

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            try:
                logger.info(f"Evaluating batch {step + 1}/{len(test_loader)}")

                image = batch['image'].to(device)
                heatmap = batch['heatmap'].to(device)

                outputs = model(image, heatmap)

                # Category
                cat_pred = torch.argmax(outputs['category_logits'], dim=1)
                all_preds_cat.extend(cat_pred.cpu().numpy())
                all_targets_cat.extend(batch['category_label'].cpu().numpy())

                # Category Type
                type_pred = torch.argmax(outputs['category_type_logits'], dim=1)
                all_preds_type.extend(type_pred.cpu().numpy())
                all_targets_type.extend(batch['category_type'].cpu().numpy())

                # Attributes per group
                for group in group_sizes:
                    logits = outputs['attribute_preds'][group]
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    group_preds[group].append(pred.cpu())
                    group_targets[group].append(batch['attribute_targets'][group].cpu())

            except Exception as e:
                logger.warning(f"Skipped batch {step + 1} due to error: {str(e)}")

    # Compute classification scores
    cat_acc = accuracy_score(all_targets_cat, all_preds_cat)
    cat_f1 = f1_score(all_targets_cat, all_preds_cat, average='macro')
    type_acc = accuracy_score(all_targets_type, all_preds_type)
    type_f1 = f1_score(all_targets_type, all_preds_type, average='macro')

    print("\n--- ViT Test Evaluation Results (10% subset) ---")
    print(f"Category Accuracy: {cat_acc:.4f} | F1 (Macro): {cat_f1:.4f}")
    print(f"Type Accuracy: {type_acc:.4f} | F1 (Macro): {type_f1:.4f}")

    #  attribute F1 and accuracy per group
    group_f1_scores = {}
    group_accuracies = {}
    for group in group_sizes:
        pred = torch.cat(group_preds[group]).numpy()
        tgt = torch.cat(group_targets[group]).numpy()

        f1 = f1_score(tgt, pred, average='micro', zero_division=0)
        group_f1_scores[group] = f1

        exact_matches = np.all(pred == tgt, axis=1)
        acc = exact_matches.sum() / len(tgt)
        group_accuracies[group] = acc

        print(f"Attribute Group '{group}' - F1 (Micro): {f1:.4f} | Accuracy: {acc:.4f}")

    # Average attribute accuracy across all groups
    avg_attr_accuracy = sum(group_accuracies.values()) / len(group_accuracies)
    print(f"\nAverage Attribute Accuracy (across all groups): {avg_attr_accuracy:.4f}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
