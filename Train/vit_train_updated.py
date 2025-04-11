import os
import time
import torch
import logging
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import f1_score
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*use_reentrant.*")




from utils.dataset_vit import FashionDataset
from models.vit_model import FashionViTModel, FashionMultiTaskLoss


logging.basicConfig(
    level=logging.DEBUG,  
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FashionViTTraining")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs, grad_accum_steps):
    model.train()
    num_attr_groups = len(model.attribute_predictor.sub_heads)
    total_loss = 0.0
    total_samples = 0

    
    category_correct = 0
    type_correct = 0
    attribute_correct = 0
    f1_category = 0.0
    f1_type = 0.0
    f1_attributes = 0.0

    optimizer.zero_grad()

   
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} - Training", leave=False, mininterval=10)


    for step, batch in enumerate(progress_bar):
        try:
            images = batch['image']
            heatmaps = batch['heatmap']
            logger.debug(f"Batch {step} - images type: {type(images)}, shape: {getattr(images, 'shape', None)}")
            logger.debug(f"Batch {step} - heatmaps type: {type(heatmaps)}, shape: {getattr(heatmaps, 'shape', None)}")

            
            images = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)

            # logger.info(f"Batch {step} - Image Shape after GPU transfer: {images.shape}")
            # logger.info(f"Batch {step} - Heatmap Shape after GPU transfer: {heatmaps.shape}")

            category_labels = batch['category_label'].to(device, non_blocking=True)
            category_type_labels = batch['category_type'].to(device, non_blocking=True)
            attribute_targets = {k: v.to(device) for k, v in batch['attribute_targets'].items()}

            targets = {
                'category_label': category_labels,
                'category_type': category_type_labels,
                'attribute_targets': attribute_targets
            }
            if step % 20 == 0:
                num_all_zero = sum((v.sum(dim=1) == 0).sum().item() for v in attribute_targets.values())
                total = next(iter(attribute_targets.values())).shape[0]
                logger.info(f"[Batch {step}] All-zero attribute samples: {num_all_zero}/{total}")


            
            with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
                outputs = model(images, heatmaps)
                loss, loss_dict = criterion(outputs, targets, epoch)

            # Scale loss and backpropagate
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                loss.div(grad_accum_steps).backward()

           
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.debug(f"{name}: grad={param.grad.norm().item()}")

            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            
            _, pred_cat = torch.max(outputs['category_logits'], 1)
            _, pred_type = torch.max(outputs['category_type_logits'], 1)

            category_correct += (pred_cat == category_labels).sum().item()
            type_correct += (pred_type == category_type_labels).sum().item()

         
            f1_category += f1_score(category_labels.cpu().numpy(), pred_cat.cpu().numpy(), average='macro')
            f1_type += f1_score(category_type_labels.cpu().numpy(), pred_type.cpu().numpy(), average='macro')

            
            attr_preds = {k: (torch.sigmoid(v) > 0.5).float() for k, v in outputs['attribute_preds'].items()}
            attr_accs = []
            attr_f1s = []
            for group in attribute_targets:
                correct = (attr_preds[group] == attribute_targets[group]).all(dim=1).sum().item()
                acc = correct / max(1, len(attribute_targets[group]))  
                attr_accs.append(acc)
                # f1 = f1_score(attribute_targets[group].cpu().numpy(), attr_preds[group].cpu().numpy(), average='micro')
                f1 = f1_score(attribute_targets[group].cpu().numpy(), attr_preds[group].cpu().numpy(), average='micro', zero_division=0)

                attr_f1s.append(f1)

            attribute_correct += sum(attr_accs)
            f1_attributes += sum(attr_f1s)

            total_loss += loss.item()
            total_samples += images.size(0)

           
            if step % 50 == 0:
                try:
                    progress_bar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Cat Acc': f"{category_correct / max(1, total_samples):.2%}",
                        'Type Acc': f"{type_correct / max(1, total_samples):.2%}",
                        'Attr Acc': f"{attribute_correct / max(1, total_samples * len(attribute_targets)):.2%}"
                    })
                except Exception as e:
                    logger.warning(f"Could not update progress bar: {str(e)}")

           
            del images, heatmaps, category_labels, category_type_labels, attribute_targets, outputs, loss

        except Exception as e:
            logger.error(f"Error in batch {step}: {str(e)}", exc_info=True)
            continue  

    epoch_loss = total_loss / max(1, len(dataloader))  # Prevent division by zero
    cat_acc = category_correct / max(1, total_samples)
    type_acc = type_correct / max(1, total_samples)
    attr_acc = attribute_correct / max(1, total_samples * num_attr_groups)
    f1_cat_avg = f1_category / max(1, len(dataloader))
    f1_type_avg = f1_type / max(1, len(dataloader))
    f1_attr_avg = f1_attributes / max(1, len(dataloader))

   
    logger.info(
        f"Epoch {epoch}/{total_epochs} - "
        f"Loss: {epoch_loss:.4f}, "
        f"Cat Acc: {cat_acc:.2%}, "
        f"Type Acc: {type_acc:.2%}, "
        f"Attr Acc: {attr_acc:.2%}, "
        f"F1 Cat: {f1_cat_avg:.4f}, "
        f"F1 Type: {f1_type_avg:.4f}, "
        f"F1 Attr: {f1_attr_avg:.4f}"
    )

    return epoch_loss, cat_acc, type_acc, attr_acc, f1_cat_avg, f1_type_avg, f1_attr_avg, loss_dict


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    category_correct = 0
    type_correct = 0
    attribute_correct = 0
    f1_category = 0.0
    f1_type = 0.0
    f1_attributes = 0.0

    all_attr_preds = []
    all_attr_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            heatmaps = batch['heatmap'].to(device)
            category_labels = batch['category_label'].to(device)
            category_type_labels = batch['category_type'].to(device)
            attribute_targets = {k: v.to(device) for k, v in batch['attribute_targets'].items()}

            if len(attribute_targets) > 0:
                num_all_zero = sum((v.sum(dim=1) == 0).sum().item() for v in attribute_targets.values())
                total = next(iter(attribute_targets.values())).shape[0]
                logger.info(f"[Validation Batch] All-zero attribute samples: {num_all_zero}/{total}")

            targets = {
                'category_label': category_labels,
                'category_type': category_type_labels,
                'attribute_targets': attribute_targets
            }

            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images, heatmaps)
                loss, loss_dict = criterion(outputs, targets, epoch)

            # Compute
            _, pred_cat = torch.max(outputs['category_logits'], 1)
            _, pred_type = torch.max(outputs['category_type_logits'], 1)

            category_correct += (pred_cat == category_labels).sum().item()
            type_correct += (pred_type == category_type_labels).sum().item()

            # C F1 Scores
            f1_category += f1_score(category_labels.cpu().numpy(), pred_cat.cpu().numpy(), average='macro')
            f1_type += f1_score(category_type_labels.cpu().numpy(), pred_type.cpu().numpy(), average='macro')

            # AMetrics
            attr_preds = {k: (torch.sigmoid(v) > 0.5).float() for k, v in outputs['attribute_preds'].items()}
            all_attr_preds.append(attr_preds)
            all_attr_targets.append(attribute_targets)

            total_loss += loss.item()
            total_samples += images.size(0)

    
    all_attr_preds = {k: torch.cat([p[k] for p in all_attr_preds]) for k in all_attr_preds[0]}
    all_attr_targets = {k: torch.cat([t[k] for t in all_attr_targets]) for k in all_attr_targets[0]}

    f1_attr_avg = {
        group: f1_score(
            all_attr_targets[group].cpu().numpy(),
            all_attr_preds[group].cpu().numpy(),
            average='micro'
        )
        for group in all_attr_targets
    }
    f1_attributes = sum(f1_attr_avg.values()) / len(f1_attr_avg)


    val_loss = total_loss / len(dataloader)
    cat_acc = category_correct / total_samples
    type_acc = type_correct / total_samples
    attr_acc = attribute_correct / max(1, total_samples * len(attribute_targets))
    f1_cat_avg = f1_category / len(dataloader)
    f1_type_avg = f1_type / len(dataloader)
    f1_attr_avg = sum(f1_attr_avg.values()) / len(f1_attr_avg)

    return val_loss, cat_acc, type_acc, attr_acc, f1_cat_avg, f1_type_avg, f1_attributes, loss_dict


def main(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

  
    torch.backends.cudnn.benchmark = True

    scaler = torch.cuda.amp.GradScaler()



    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    epoch_dir = os.path.join(output_dir, 'Epochs')
    os.makedirs(epoch_dir, exist_ok=True)

    writer = SummaryWriter(output_dir)

    # Define attribute groups
    ATTRIBUTE_GROUPS = {
        'color_print': ['red', 'pink', 'floral', 'striped', 'stripe', 'print', 'printed', 'graphic', 'love', 'summer'],
        'neckline': ['v-neck', 'hooded', 'collar', 'sleeveless', 'strapless', 'racerback', 'muscle'],
        'silhouette_fit': ['slim', 'boxy', 'fit', 'skinny', 'shift', 'bodycon', 'maxi', 'mini', 'midi', 'a-line'],
        'style_construction': ['crochet', 'knit', 'woven', 'lace', 'denim', 'cotton', 'chiffon', 'mesh'],
        'details': ['pleated', 'pocket', 'button', 'drawstring', 'trim', 'hem', 'capri', 'sleeve', 'flare', 'skater', 'sheath', 'shirt', 'pencil', 'classic', 'crop']
    }
    group_sizes = {group: len(attributes) for group, attributes in ATTRIBUTE_GROUPS.items()}

    # Define data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        # transforms.GaussianBlur(kernel_size=3),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = FashionDataset(
        metadata_path=args.train_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        attribute_groups=ATTRIBUTE_GROUPS,
        use_cache=False,
        cache_size=100,
        validate_files=True,
    )

    val_dataset = FashionDataset(
        metadata_path=args.val_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=val_transform,
        attribute_groups=ATTRIBUTE_GROUPS,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    val_subset_size = int(0.3 * len(val_dataset))  # 30% of the validation dataset
    subset_indices = random.sample(range(len(val_dataset)), val_subset_size)
    val_dataset = Subset(val_dataset, subset_indices)
    logger.info(f"Validation subset size: {len(val_dataset)}")

    # Weighted Random Sampler
    category_counts = train_dataset.metadata['category_label'].value_counts().to_dict()
    max_count = max(category_counts.values())
    class_weights = {cat: max_count / count for cat, count in category_counts.items()}
    labels = train_dataset.metadata['category_label'].tolist()
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4
    )
   
    sample_batch = next(iter(train_loader))
    logger.info(f"Sample batch keys: {sample_batch.keys()}")
    logger.info(f"Image tensor shape: {sample_batch['image'].shape}")
    logger.info(f"Category labels: {sample_batch['category_label'][:5]}")
    logger.info(f"Category type labels: {sample_batch['category_type'][:5]}")
    logger.info(f"Heatmap shape: {sample_batch['heatmap'].shape}")
    for group, tensor in sample_batch['attribute_targets'].items():
        logger.info(f"Attributes [{group}] shape: {tensor.shape}")
        break  


    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

   
    model = FashionViTModel(
        num_categories=args.num_categories,
        num_category_types=args.num_category_types,
        group_sizes=group_sizes,
        vit_name=args.backbone,
        use_pretrained=True,
        enable_deep_supervision=True
    ).to(device)
    model.log_requires_grad()  

    model.freeze_backbone(freeze=True)
    logger.info("Frozen ViT backbone for first 5 epochs.")

   
    criterion = FashionMultiTaskLoss(
        category_weight=1.0,
        category_type_weight=0.5,
        attribute_weight=0.3,
        aux_weight=0.2,
        total_epochs=args.epochs,
        attr_groups=group_sizes
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.epochs > 5:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=5),
                CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
            ],
            milestones=[5]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


    best_val_loss = float('inf')
    patience = 5
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == 6:
            model.freeze_backbone(freeze=False)
            logger.info("Unfrozen ViT backbone for fine-tuning.")

        try:
     
            train_loss, *_ = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs, grad_accum_steps=4)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

         
            val_loss, *_ = validate(model, val_loader, criterion, device, epoch)
            logger.info(f"Epoch {epoch}: Validation Loss = {val_loss:.4f}")

         
            scheduler.step()

        
            epoch_model_path = os.path.join(epoch_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            logger.info(f"Saved epoch {epoch} model at {epoch_model_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, f"{args.backbone}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved at epoch {epoch} with val loss: {val_loss:.4f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        except Exception as e:
            logger.error(f"Unexpected error during training at epoch {epoch}: {str(e)}", exc_info=True)
            break  # Stop training if an error occurs

    writer.close()
    logger.info("Training completed!")

class Args:
    def __init__(self):
        self.train_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_train.csv'
        self.val_metadata = '/content/drive/MyDrive/stylesync/zsehran/metadata_updated_val.csv'
        self.images_dir = '/content/cropped_images1'
        self.heatmaps_dir = '/content/heatmaps1'
        self.backbone = 'vit_base_patch16_224'
        self.num_categories = 46
        self.num_category_types = 3
        self.num_attributes = 50
        self.batch_size = 512
        self.epochs = 20
        self.learning_rate = 3e-4
        self.num_workers = 2
        self.output_dir = '/content/drive/MyDrive/Runs/VIT'

if __name__ == "__main__":
    args = Args()
    main(args)