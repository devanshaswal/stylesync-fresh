import logging
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FashionHybridModel')


def compute_f1_multiclass(preds, targets, average='macro'):
    preds = preds.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average=average, zero_division=1)


def compute_f1_multilabel(preds, targets, average='micro'):
    preds = (preds > 0.5).int().cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average=average, zero_division=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()



class FocalLossMultiLabel(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLossMultiLabel, self).__init__()
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.to(logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        alpha = self.alpha.clone().to(logits.device).type_as(bce_loss)
        if alpha.ndim == 1 and bce_loss.ndim == 2:
            alpha = alpha.unsqueeze(0)
        one = torch.tensor(1.0, device=bce_loss.device, dtype=bce_loss.dtype)
        focal_loss = alpha * ((one - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()



class FashionHybridModel(nn.Module):
    def __init__(self, num_categories=46, num_category_types=3, num_attributes=50,
                 embed_dim=512, use_pretrained=True, dropout_rate=0.5, backbone='resnet50'):
        super(FashionHybridModel, self).__init__()
        logger.info(f"Initializing FashionHybridModel with backbone {backbone}")

        # Get intermediate features from ResNet's layer3
        resnet = models.resnet50(weights="IMAGENET1K_V1" if use_pretrained else None)
        # Extract features up to layer3 (children indices 0 to 6)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:7])
        self.image_feature_dim = 1024  # layer3 outputs 1024 channels

        # new adapter to reduce channel depth
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),  # Reduce from 1024 to 512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

       
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # newly added layer
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.heatmap_feature_dim = 128  

      
        total_feature_dim = 512 + 128  # 640
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

        # Classification heads
        self.category_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.BatchNorm1d(embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim//2, num_categories)
        )

        self.category_type_classifier = nn.Linear(embed_dim, num_category_types)
        self.attribute_predictor = nn.Sequential(
            nn.Linear(embed_dim, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attributes)
        )

        # Auxiliary classifiers for deep supervision
        self.image_aux_classifier = nn.Linear(512, num_categories)
        self.heatmap_aux_classifier = nn.Linear(128, num_category_types)

        logger.info("Model initialization complete")

    def forward(self, image, heatmap):
        # Extractiing image features from ResNet's layer3 
        img_feat = self.image_encoder(image)              # Shape: [B, 1024, H, W]
        img_feat = self.feature_adapter(img_feat)           # Shape: [B, 512, 1, 1]
        img_feat = torch.flatten(img_feat, 1)               # Shape: [B, 512]


        heatmap_feat = self.heatmap_encoder(heatmap)        # Shape: [B, 128, 1, 1]
        heatmap_feat = torch.flatten(heatmap_feat, 1)       # Shape: [B, 128]

      
        fused_feat = self.fusion(torch.cat([img_feat, heatmap_feat], dim=1))  # Shape: [B, embed_dim]

    
        attn_scores = torch.matmul(fused_feat.unsqueeze(1), fused_feat.unsqueeze(2))
        attn_scores = attn_scores / math.sqrt(fused_feat.size(1))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        fused_feat = torch.matmul(attn_weights.squeeze(1), fused_feat)

       
        image_aux_logits = self.image_aux_classifier(img_feat)
        heatmap_aux_logits = self.heatmap_aux_classifier(heatmap_feat)

        # Main classification outputs
        category_logits = self.category_classifier(fused_feat)
        category_type_logits = self.category_type_classifier(fused_feat)
        attribute_preds = self.attribute_predictor(fused_feat)

        outputs = {
            'category_logits': category_logits,
            'category_probs': F.softmax(category_logits, dim=1),
            'category_type_logits': category_type_logits,
            'category_type_probs': F.softmax(category_type_logits, dim=1),
            'attribute_preds': attribute_preds,
            'attribute_probs': torch.sigmoid(attribute_preds),
            'image_aux_logits': image_aux_logits,
            'heatmap_aux_logits': heatmap_aux_logits
        }
        return outputs

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze
        trainable_params = sum(p.requires_grad for p in self.image_encoder.parameters())
        logger.info(f"Image encoder backbone has been {'frozen' if freeze else 'unfrozen'}. Trainable params: {trainable_params}")


class FashionMultiTaskLoss(nn.Module):
    def __init__(self, category_weight=1.0, category_type_weight=0.5,
                 attribute_weight=0.3, compatibility_weight=0.0,
                 aux_weight=0.3, total_epochs=30, alpha_attributes=1.0):
        super(FashionMultiTaskLoss, self).__init__()
        print("Initialized FashionMultiTaskLoss with alpha_attributes =", alpha_attributes)

        self.category_weight = category_weight
        self.category_type_weight = category_type_weight
        self.attribute_weight = attribute_weight
        self.aux_weight = aux_weight
        self.total_epochs = total_epochs  # store total_epochs

        self.category_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.category_type_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        # Replace BCEWithLogitsLoss with FocalLossMultiLabel for attribute loss
        self.attribute_loss_fn = FocalLossMultiLabel(alpha=alpha_attributes, gamma=2.0)

    def forward(self, outputs, targets, epoch):
        category_loss = self.category_loss_fn(outputs['category_logits'], targets['category_labels'])
        category_type_loss = self.category_type_loss_fn(outputs['category_type_logits'], targets['category_type_labels'])
        attribute_loss = self.attribute_loss_fn(outputs['attribute_preds'], targets['attribute_targets'])

        image_aux_loss = self.category_loss_fn(outputs['image_aux_logits'], targets['category_labels'])
        heatmap_aux_loss = self.category_type_loss_fn(outputs['heatmap_aux_logits'], targets['category_type_labels'])
        total_aux_loss = image_aux_loss + heatmap_aux_loss

        scaling_factor = max(0.1, 1.0 - (epoch / self.total_epochs))
        aux_weight = self.aux_weight * scaling_factor  # Dynamic aux loss weight

        total_loss = (
            self.category_weight * category_loss +
            self.category_type_weight * category_type_loss +
            self.attribute_weight * attribute_loss +
            aux_weight * total_aux_loss
        )

        loss_dict = {
            'category_loss': category_loss.item(),
            'category_type_loss': category_type_loss.item(),
            'attribute_loss': attribute_loss.item(),
            'image_aux_loss': image_aux_loss.item(),
            'heatmap_aux_loss': heatmap_aux_loss.item(),
            'total_aux_loss': total_aux_loss.item()
        }

        return total_loss, loss_dict


def log_validation_f1(avg_f1_category, avg_f1_category_type, avg_f1_attributes):
    logger.info(f"Validation F1 Scores: Category={avg_f1_category:.4f}, "
                f"Type={avg_f1_category_type:.4f}, Attr={avg_f1_attributes:.4f}")
