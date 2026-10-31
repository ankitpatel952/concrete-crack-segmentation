# evaluation metrics for binary segmentation
# standard stuff: precision, recall, F1 (same as dice), IoU (jaccard)
# also ODS and OIS which are used in edge/crack detection benchmarks

import numpy as np
import torch


def compute_batch_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    tn = ((1.0 - preds_flat) * (1.0 - targets_flat)).sum().item()

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def compute_ods_ois(predictions_list, ground_truths_list, thresholds=np.linspace(0.1, 0.9, 17)):
    # ODS = one threshold that works best across the whole dataset
    # OIS = best threshold picked per image individually (upper bound basically)
    # OIS will always be >= ODS, gap tells you how much per-image tuning would help

    best_ods_f1 = 0.0
    best_ods_thresh = 0.5

    for thresh in thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0
        for prob, gt in zip(predictions_list, ground_truths_list):
            pred = (prob > thresh).astype(np.float32)
            gt_f = gt.astype(np.float32)
            tp = np.sum(pred * gt_f)
            fp = np.sum(pred * (1.0 - gt_f))
            fn = np.sum((1.0 - pred) * gt_f)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        p = total_tp / (total_tp + total_fp + 1e-7)
        r = total_tp / (total_tp + total_fn + 1e-7)
        f1 = (2.0 * p * r) / (p + r + 1e-7)
        if f1 > best_ods_f1:
            best_ods_f1 = f1
            best_ods_thresh = thresh

    # per-image best
    ois_f1_sum = 0.0
    for prob, gt in zip(predictions_list, ground_truths_list):
        best_img_f1 = 0.0
        gt_f = gt.astype(np.float32)
        for thresh in thresholds:
            pred = (prob > thresh).astype(np.float32)
            tp = np.sum(pred * gt_f)
            fp = np.sum(pred * (1.0 - gt_f))
            fn = np.sum((1.0 - pred) * gt_f)
            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            f1 = (2.0 * p * r) / (p + r + 1e-7)
            if f1 > best_img_f1:
                best_img_f1 = f1
        ois_f1_sum += best_img_f1

    ois_f1 = ois_f1_sum / max(len(predictions_list), 1)

    return {
        "ODS_F1": best_ods_f1,
        "ODS_threshold": best_ods_thresh,
        "OIS_F1": ois_f1
    }
