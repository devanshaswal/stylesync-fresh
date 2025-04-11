import argparse
import logging
import math
import os
import random
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, 
                             f1_score)
from torch.cuda.amp import GradScaler, autocast

# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_resnet import (FashionHybridModel, FashionMultiTaskLoss,
                                compute_f1_multiclass, compute_f1_multilabel)
from utils.dataset_resnet import FashionDataset

warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger('TrainingScript')

def print_cuda_info():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


def mixup_data(x, y, alpha=0.05):
    # Transfer to CPU
    x_cpu = x.cpu()
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x_cpu.size(0)
    index = torch.randperm(batch_size)
    mixed_x_cpu = lam * x_cpu + (1 - lam) * x_cpu[index, :]
    # Transfer back to original device
    mixed_x = mixed_x_cpu.to(x.device)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    indices = torch.randperm(x.size(0)).to(x.device)
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[indices]
    return x, y_a, y_b, lam



def unfreeze_backbone_layers(model, num_layers_to_unfreeze):
    module = model.module if hasattr(model, 'module') else model
    conv_layers = [m for m in module.image_encoder.modules() if isinstance(m, nn.Conv2d)]
    for layer in conv_layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    print(f"Unfroze last {num_layers_to_unfreeze} conv layers of the backbone.")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, total_epochs, scheduler):
    model.train()
    total_loss = 0
    loss_components = {
        'category': 0, 'category_type': 0, 'attribute': 0,
        'image_aux': 0, 'heatmap_aux': 0, 'total_aux': 0
    }
    total_f1_category = 0
    total_f1_category_type = 0
    total_f1_attributes = 0
    total_accuracy_category = 0
    total_accuracy_category_type = 0
    total_accuracy_attributes = 0
    total_samples = 0
    num_batches = len(train_loader)

    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch}")

    for batch_idx, batch in progress_bar:
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmap'].to(device, non_blocking=True)
        attributes = batch['attributes'].to(device, non_blocking=True)
        category_labels = batch['category_label'].to(device, non_blocking=True)

        if images.shape[0] == 0 or heatmaps.shape[0] == 0:
            raise RuntimeError(f"Empty batch at index {batch_idx}")

        targets = {
            'category_type_labels': batch['category_type'].to(device, non_blocking=True),
            'attribute_targets': attributes
        }

    
        rand_val = random.random()
        mixup_prob = 0.1
        cutmix_prob = 0.1
        if rand_val < mixup_prob:
            mixed_images, target_a, target_b, lam = mixup_data(images, category_labels, alpha=0.05)
        elif rand_val < mixup_prob + cutmix_prob:
            mixed_images, target_a, target_b, lam = cutmix_data(images, category_labels, alpha=1.0)
        else:
            mixed_images = images
            target_a = category_labels
            target_b = category_labels
            lam = 1.0

        try:
            with torch.amp.autocast('cuda'):
                outputs = model(mixed_images, heatmaps)
                loss_cat_a = criterion.category_loss_fn(outputs['category_logits'], target_a)
                loss_cat_b = criterion.category_loss_fn(outputs['category_logits'], target_b)
                mixup_loss = lam * loss_cat_a + (1 - lam) * loss_cat_b

                category_type_loss = criterion.category_type_loss_fn(
                    outputs['category_type_logits'], targets['category_type_labels'])
                attribute_loss = criterion.attribute_loss_fn(
                    outputs['attribute_preds'], targets['attribute_targets'])

                image_aux_loss = criterion.category_loss_fn(outputs['image_aux_logits'], target_a)
                heatmap_aux_loss = criterion.category_type_loss_fn(
                    outputs['heatmap_aux_logits'], targets['category_type_labels'])
                total_aux_loss = image_aux_loss + heatmap_aux_loss

                loss = (criterion.category_weight * mixup_loss +
                        criterion.category_type_weight * category_type_loss +
                        criterion.attribute_weight * attribute_loss +
                        criterion.aux_weight * total_aux_loss)

                loss_dict = {
                    'category_loss': mixup_loss.item(),
                    'category_type_loss': category_type_loss.item(),
                    'attribute_loss': attribute_loss.item(),
                    'image_aux_loss': image_aux_loss.item(),
                    'heatmap_aux_loss': heatmap_aux_loss.item(),
                    'total_aux_loss': total_aux_loss.item()
                }
        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            raise

   
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())

        avg_target = ((target_a.float() + target_b.float()) / 2).round().long()
        f1_category = compute_f1_multiclass(outputs['category_probs'], avg_target)
        accuracy_category = (outputs['category_logits'].argmax(dim=1) == avg_target).float().mean()

        f1_category_type = compute_f1_multiclass(outputs['category_type_probs'], targets['category_type_labels'])
        accuracy_category_type = (outputs['category_type_logits'].argmax(dim=1) == targets['category_type_labels']).float().mean()

        f1_attributes = compute_f1_multilabel(outputs['attribute_probs'], targets['attribute_targets'], average='micro')
        accuracy_attributes = (outputs['attribute_probs'] > 0.5).float().mean()

        total_loss += loss.item()
        total_f1_category += f1_category
        total_f1_category_type += f1_category_type
        total_f1_attributes += f1_attributes
        total_accuracy_category += accuracy_category
        total_accuracy_category_type += accuracy_category_type
        total_accuracy_attributes += accuracy_attributes
        total_samples += 1

        for k, v in loss_dict.items():
            key = k.replace('_loss', '')
            if key in loss_components:
                loss_components[key] += v
            else:
                logger.warning(f"Warning: Key {key} not found in loss_components.")

    avg_loss = total_loss / num_batches
    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples

    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    logger.info(f"Epoch {epoch} Completed | LR: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"Train F1 - Category: {avg_f1_category:.4f}, Type: {avg_f1_category_type:.4f}, Attr: {avg_f1_attributes:.4f}")
    print(f"Train Acc - Category: {avg_accuracy_category:.4f}, Type: {avg_accuracy_category_type:.4f}, Attr: {avg_accuracy_attributes:.4f}")
    logger.info(f"Train Loss: {avg_loss:.4f}")
    logger.info(f"Train F1 Category: {avg_f1_category:.4f}")
    logger.info(f"Train F1 Category Type: {avg_f1_category_type:.4f}")
    logger.info(f"Train F1 Attributes: {avg_f1_attributes:.4f}")
    logger.info(f"Train Accuracy Category: {avg_accuracy_category:.4f}")
    logger.info(f"Train Accuracy Type: {avg_accuracy_category_type:.4f}")
    logger.info(f"Train Accuracy Attr: {avg_accuracy_attributes:.4f}")
    logger.info(f"Loss Components: {avg_components}")

    return (avg_loss, avg_f1_category, avg_f1_category_type, avg_f1_attributes,
            avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components)


def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct_category = 0
    correct_type = 0
    correct_attributes = 0
    total = 0
    total_f1_category = 0
    total_f1_category_type = 0
    total_f1_attributes = 0
    total_accuracy_category = 0
    total_accuracy_category_type = 0
    total_accuracy_attributes = 0
    total_samples = 0

    loss_components = {'category': 0, 'category_type': 0, 'attribute': 0, 'compatibility': 0}
    all_attr_targets = []
    all_attr_preds = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            heatmaps = batch['heatmap'].to(device, non_blocking=True)
            attributes = batch['attributes'].to(device, non_blocking=True)

            targets = {
                'category_labels': batch['category_label'].to(device, non_blocking=True),
                'category_type_labels': batch['category_type'].to(device, non_blocking=True),
                'attribute_targets': attributes
            }
            if 'compatibility_targets' in batch:
                targets['compatibility_targets'] = batch['compatibility_targets'].to(device, non_blocking=True)
            else:
                targets['compatibility_targets'] = torch.zeros(images.size(0), 1).to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images, heatmaps)
                loss, loss_dict = criterion(outputs, targets, epoch)

            _, predicted_category = outputs['category_logits'].max(1)
            _, predicted_type = outputs['category_type_logits'].max(1)
            predicted_attributes = (outputs['attribute_probs'] > 0.3).float()

            total += targets['category_labels'].size(0)
            correct_category += predicted_category.eq(targets['category_labels']).sum().item()
            correct_type += predicted_type.eq(targets['category_type_labels']).sum().item()
            correct_attributes += ((predicted_attributes == targets['attribute_targets']).all(dim=1).sum().item())

            total_loss += loss.item()

            f1_category = compute_f1_multiclass(outputs['category_probs'], targets['category_labels'])
            accuracy_category = (predicted_category == targets['category_labels']).float().mean()

            f1_category_type = compute_f1_multiclass(outputs['category_type_probs'], targets['category_type_labels'])
            accuracy_category_type = (predicted_type == targets['category_type_labels']).float().mean()

            f1_attributes = compute_f1_multilabel(predicted_attributes, targets['attribute_targets'], average='micro')
            accuracy_attributes = (predicted_attributes == targets['attribute_targets']).float().mean()

            total_f1_category += f1_category
            total_f1_category_type += f1_category_type
            total_f1_attributes += f1_attributes
            total_accuracy_category += accuracy_category
            total_accuracy_category_type += accuracy_category_type
            total_accuracy_attributes += accuracy_attributes
            total_samples += 1

            all_attr_targets.append(targets['attribute_targets'].cpu().numpy())
            all_attr_preds.append((predicted_attributes).cpu().numpy())

            for k in loss_components.keys():
                loss_components[k] += loss_dict.get(k, 0)

    all_attr_targets = np.concatenate(all_attr_targets, axis=0)
    all_attr_preds = np.concatenate(all_attr_preds, axis=0)
    per_attribute_f1 = f1_score(all_attr_targets, all_attr_preds, average=None)
    logger.info(f"Per-Attribute F1 Scores: {per_attribute_f1}")

  
    worst_5_indices = np.argsort(per_attribute_f1)[:5]
 
    attribute_names = val_loader.dataset.dataset.attribute_columns if hasattr(val_loader.dataset, 'dataset') else train_dataset.attribute_columns
    logger.info("Worst performing attributes:")
    for idx in worst_5_indices:
        logger.info(f"- {attribute_names[idx]}: F1 = {per_attribute_f1[idx]:.4f}")

    avg_loss = total_loss / len(val_loader)
    category_acc = 100.0 * correct_category / total
    type_acc = 100.0 * correct_type / total
    attributes_acc = 100.0 * correct_attributes / total

    avg_f1_category = total_f1_category / total_samples
    avg_f1_category_type = total_f1_category_type / total_samples
    avg_f1_attributes = total_f1_attributes / total_samples
    avg_accuracy_category = total_accuracy_category / total_samples
    avg_accuracy_category_type = total_accuracy_category_type / total_samples
    avg_accuracy_attributes = total_accuracy_attributes / total_samples

    avg_components = {k: v / total_samples for k, v in loss_components.items()}

    print(f" Validation - Loss: {avg_loss:.4f}")
    print(f"  Validation F1 Scores - Category: {avg_f1_category:.4f}, "
          f"Type: {avg_f1_category_type:.4f}, Attributes: {avg_f1_attributes:.4f}")
    print(f" Validation Accuracy - Category: {avg_accuracy_category:.4f}, "
          f"Type: {avg_accuracy_category_type:.4f}, Attributes: {avg_accuracy_attributes:.4f}")

    logger.info(f"Validation Loss: {avg_loss:.4f}")
    logger.info(f"Validation F1 Scores: Category={avg_f1_category:.4f}, "
                f"Type={avg_f1_category_type:.4f}, Attr={avg_f1_attributes:.4f}")
    logger.info(f"Validation Accuracy: Category={avg_accuracy_category:.4f}, "
                f"Type={avg_accuracy_category_type:.4f}, Attr={avg_accuracy_attributes:.4f}")

    return (avg_loss, category_acc, type_acc, attributes_acc,
            avg_f1_category, avg_f1_category_type, avg_f1_attributes,
            avg_accuracy_category, avg_accuracy_category_type, avg_accuracy_attributes, avg_components)

def main(args):
    print_cuda_info()
    logger.info("Starting training script...")
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA Device: {torch.cuda.get_device_name(0)}")
        scaler = GradScaler()
    else:
        device = torch.device("cpu")
        print("Warning: No CUDA, using CPU!")
        scaler = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('/content/drive/MyDrive/Runs', f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    epoch_dir = os.path.join(output_dir, 'Epochs')
    model_dir = os.path.join(epoch_dir, 'Runs1')
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(output_dir)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = FashionDataset(
        metadata_path=args.train_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=train_transform,
        use_cache=False,
        cache_size=100,
        validate_files=True,
    )

    val_dataset = FashionDataset(
        metadata_path=args.val_metadata,
        cropped_images_dir=args.images_dir,
        heatmaps_dir=args.heatmaps_dir,
        transform=val_transform,
        use_cache=True,
        cache_size=100,
        validate_files=True,
    )

    val_subset_fraction = 0.2
    val_subset_size = int(val_subset_fraction * len(val_dataset))
    indices = list(range(len(val_dataset)))
    random.shuffle(indices)
    val_subset_indices = indices[:val_subset_size]
    val_dataset = Subset(val_dataset, val_subset_indices)
    print(f"Validation subset size: {len(val_dataset)}")

    assert train_dataset.metadata['category_label'].min() == 0, "Category labels must start from 0"
    assert train_dataset.metadata['category_label'].max() == 45, "Label range should be 0..45"
    logger.info("Category labels: OK (0..45)")


    attribute_counts = train_dataset.metadata[train_dataset.attribute_columns].sum(axis=0)
    max_count = attribute_counts.max()
    alpha_attributes = (max_count / (attribute_counts + 1e-6)).clip(upper=10.0)  # Cap at 10x
    alpha_attributes = torch.tensor(alpha_attributes.tolist(), dtype=torch.float32).to(device)

   
    category_counts = train_dataset.metadata['category_label'].value_counts().to_dict()
    for i in range(46):
        if i not in category_counts:
            category_counts[i] = 1
    max_count = max(category_counts.values())
    cap_value = 10.0
    class_weights = {cat: min(max_count / count, cap_value) for cat, count in category_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.metadata['category_label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False
    )

    num_categories = len(train_dataset.metadata['category_label'].unique())
    num_category_types = len(train_dataset.metadata['category_type'].unique())
    print(f"Unique categories: {num_categories}, unique category types: {num_category_types}")

    model = FashionHybridModel(
        num_categories=num_categories,
        num_category_types=num_category_types,
        num_attributes=args.num_attributes,
        backbone=args.backbone
    )

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    if hasattr(model, 'module'):
        model.module.freeze_backbone(freeze=False)
    else:
        model.freeze_backbone(freeze=False)

  
    criterion = FashionMultiTaskLoss(
        category_weight=1.0,
        attribute_weight=1.5,
        aux_weight=0.05,
        compatibility_weight=0.0,
        alpha_attributes=alpha_attributes 
    ).to(device)
    criterion.category_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)


    if hasattr(model, 'module'):
        backbone_params = list(model.module.image_encoder.parameters())
    else:
        backbone_params = list(model.image_encoder.parameters())
    other_params = [p for name, p in model.named_parameters() if 'image_encoder' not in name]

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': args.learning_rate * 0.25, 'weight_decay': 1e-4},
        {'params': other_params, 'lr': args.learning_rate, 'weight_decay': 1e-4}
    ], weight_decay=1e-4)

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group.setdefault('start_lr', group['lr'])
        group.setdefault('max_lr', group['lr'])
        group.setdefault('end_lr', group['max_lr'])

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.learning_rate * 0.15, args.learning_rate],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        cycle_momentum=False,
        pct_start=0.3
    )
    scaler = GradScaler()

    checkpoint_path = '/content/drive/MyDrive/Runs/run12/Epochs/Runs1/epoch_3.pth'
    start_epoch = 0
    best_val_loss = float('inf')

    best_attr_f1 = 0.0
    patience = 7  
    epochs_without_improvement = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer.param_groups[0]['max_lr'] = args.learning_rate * 0.1
        optimizer.param_groups[1]['max_lr'] = args.learning_rate
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, args.epochs):
  
        if epoch == 2:
            unfreeze_backbone_layers(model, num_layers_to_unfreeze=2)
        if epoch == 5:
            unfreeze_backbone_layers(model, num_layers_to_unfreeze=4)
        if epoch == 5:
            if hasattr(model, 'module'):
                model.module.freeze_backbone(False)
            else:
                model.freeze_backbone(False)

        res_train = train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, args.epochs, scheduler)
        (train_loss, train_f1_cat, train_f1_type, train_f1_attr,
         train_acc_cat, train_acc_type, train_acc_attr, _) = res_train

        res_val = validate(model, val_loader, criterion, device, epoch)
        (val_loss, val_cat_acc, val_type_acc, val_attr_acc,
         val_f1_cat, val_f1_type, val_f1_attr,
         val_acc_cat, val_acc_type, val_acc_attr, _) = res_val

        print(f"Validation Epoch {epoch} | Loss: {val_loss:.4f}")
        print(f"  Validation F1 - Cat: {val_f1_cat:.4f}, Type: {val_f1_type:.4f}, Attr: {val_f1_attr:.4f}")
        print(f"  Validation Acc - Cat: {val_acc_cat:.4f}, Type: {val_acc_type:.4f}, Attr: {val_acc_attr:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/train_category', train_f1_cat, epoch)
        writer.add_scalar('F1/val_category', val_f1_cat, epoch)
        writer.add_scalar('F1/train_category_type', train_f1_type, epoch)
        writer.add_scalar('F1/val_category_type', val_f1_type, epoch)
        writer.add_scalar('F1/val_attributes', val_f1_attr, epoch)  

     
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(model_dir, f'epoch_{epoch}.pth'))

        if val_f1_attr > best_attr_f1:
            best_attr_f1 = val_f1_attr
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(model_dir, 'best_model.pth'))
            logger.info(f"Epoch {epoch}: Improved attribute F1 to {val_f1_attr:.4f}. Model saved as best model.")
        else:
            epochs_without_improvement += 1
            logger.info(f"Epoch {epoch}: No improvement in attribute F1. Patience count: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in attribute F1.")
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        torch.cuda.empty_cache()
    writer.close()


class Args:
    def __init__(self):
        self.train_metadata = 'zsehran/metadata_updated_train.csv'
        self.val_metadata = 'zsehran/metadata_updated_val.csv'
        self.images_dir = 'data/processed/cropped_images'
        self.heatmaps_dir = 'data/processed/heatmaps'
        self.backbone = 'resnet50'
        self.num_categories = 46
        self.num_category_types = 3
        self.num_attributes = 50
        self.batch_size = 256
        self.epochs = 20
        self.learning_rate = 3e-4
        self.num_workers = 4

if __name__ == '__main__':
    args = Args()
    main(args)

