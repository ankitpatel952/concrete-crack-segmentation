# loss functions for crack segmentation
# cracks are like 2-5% of the image at most, so plain BCE just learns to predict "no crack"
# focal loss punishes the easy background pixels less, dice cares about overlap directly
# combining both worked better than either alone in my experiments

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    # smooth=1 just keeps it from blowing up on empty masks
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice_score


class BinaryFocalLoss(nn.Module):
    # focal loss from the RetinaNet paper — gamma=2, alpha=0.25 is what they recommend
    # basically adds (1-p)^gamma weight so the network stops coasting on easy negatives
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_factor * torch.pow((1.0 - p_t), self.gamma)

        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HybridFocalDiceLoss(nn.Module):
    # 50/50 blend — tried 0.7/0.3 favoring dice but equal weighting was more stable
    def __init__(self, lambda_focal: float = 0.5, lambda_dice: float = 0.5,
                 alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.dice = SoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_focal = self.focal(logits, targets)
        l_dice = self.dice(logits, targets)
        return self.lambda_focal * l_focal + self.lambda_dice * l_dice
