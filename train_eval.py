# training and validation loop
# uses cosine annealing since the loss landscape for segmentation is pretty bumpy near the end
# AMP is enabled on CUDA — on CPU/MPS it just runs normally

import os
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ResNet34CrackUNet
from loss import HybridFocalDiceLoss
from dataset import ConcreteCrackDataset
from metrics import compute_batch_metrics


def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    total_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            b_metrics = compute_batch_metrics(logits, masks)
            for k in total_metrics:
                total_metrics[k] += b_metrics[k] * images.size(0)

    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_metrics = {k: v / dataset_size for k, v in total_metrics.items()}
    return epoch_loss, epoch_metrics


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        b_metrics = compute_batch_metrics(logits, masks)
        for k in total_metrics:
            total_metrics[k] += b_metrics[k] * images.size(0)

    dataset_size = len(loader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_metrics = {k: v / dataset_size for k, v in total_metrics.items()}
    return epoch_loss, epoch_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ResNet34-UNet for Crack Segmentation")
    parser.add_argument("--train_img", type=str, default="data/train/images")
    parser.add_argument("--train_mask", type=str, default="data/train/masks")
    parser.add_argument("--val_img", type=str, default="data/val/images")
    parser.add_argument("--val_mask", type=str, default="data/val/masks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ConcreteCrackDataset(args.train_img, args.train_mask, patch_size=args.patch_size, is_train=True)
    val_dataset = ConcreteCrackDataset(args.val_img, args.val_mask, patch_size=args.patch_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ResNet34CrackUNet(pretrained=True).to(device)
    criterion = HybridFocalDiceLoss(lambda_focal=0.5, lambda_dice=0.5, alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_iou = 0.0

    print("--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} IoU: {train_metrics['iou']:.4f} F1: {train_metrics['f1']:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_metrics['iou']:.4f} F1: {val_metrics['f1']:.4f}")

        # save whenever validation IoU improves
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_path = os.path.join(args.output_dir, "best_crack_unet_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": best_val_iou,
                "val_f1": val_metrics["f1"]
            }, save_path)
            print(f"  --> Saved new best checkpoint with Val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
