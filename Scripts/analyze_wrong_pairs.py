import os
import argparse
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import model
from Loaders.LFW_loader import LFW
from config import LFW_ALIGNED_DATA_DIR
from eval import parse_lfw_pairs, getThreshold

def find_best_checkpoint(log_file='train.log', ckpt_dir='your_save_dir_here'):
    best_acc = 0.0
    best_epoch = None
    current_epoch = None
    # Patterns
    epoch_pattern = re.compile(r'Epoch\s+(\d+)/')           # Matches "Epoch 1/XX"
    test_epoch_pattern = re.compile(r'Test Epoch:\s+(\d+)') # Matches "Test Epoch: 1"
    acc_pattern = re.compile(r'LFW Accuracy:\s+([\d.]+)%')
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Update current epoch when we see it
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue
            test_match = test_epoch_pattern.search(line)
            if test_match:
                current_epoch = int(test_match.group(1))
                continue
            # When we see accuracy, use the last known current_epoch
            acc_match = acc_pattern.search(line)
            if acc_match and current_epoch is not None:
                acc = float(acc_match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = current_epoch

    if best_epoch is None:
        raise ValueError("No LFW accuracy found in the log file.")
    ckpt_path = os.path.join(ckpt_dir, f'{best_epoch:03d}.ckpt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Best LFW Accuracy: {best_acc:.3f}% at epoch {best_epoch}")
    print(f"Loading checkpoint: {ckpt_path}")
    return ckpt_path

def extract_features(model_path, lfw_dir, batch_size=128):
    net = model.MobileFacenet().cuda()
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['net_state_dict'])
    net.eval()

    nl, nr, folds, flags = parse_lfw_pairs(lfw_dir)
    dataset = LFW(nl, nr)
    loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
        shuffle=False, num_workers=8, drop_last=False)

    all_left = []
    all_right = []
    with torch.no_grad():
        for batch in loader:
            B = batch.size(0)                     # number of pairs
            batch = batch.cuda(non_blocking=True)
            batch = batch.view(B * 2, 3, 112, 112)
            embeddings = net(batch)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.view(B, 2, -1)

            left = embeddings[:, 0, :].cpu().numpy()
            right = embeddings[:, 1, :].cpu().numpy()
            all_left.append(left)
            all_right.append(right)

    fl = np.concatenate(all_left, axis=0)
    fr = np.concatenate(all_right, axis=0)

    # Safety checks
    assert fl.shape == fr.shape, f"Shape mismatch: fl={fl.shape}, fr={fr.shape}"
    assert fl.shape[0] == len(flags), "Feature count != number of LFW pairs"
    return fl, fr, nl, nr, np.array(folds), np.array(flags)


def find_wrong_pairs_10fold(fl, fr, folds, flags):
    scores = np.sum(fl * fr, axis=1)
    scores = np.clip(scores, -1.0, 1.0)
    false_positives = []
    false_negatives = []

    for i in range(10):
        val_mask = folds != i
        test_mask = folds == i

        threshold = getThreshold(scores[val_mask], flags[val_mask], 10000)
        test_indices = np.where(test_mask)[0]
        test_scores = scores[test_mask]
        test_flags = flags[test_mask]
        preds_same = test_scores > threshold
        gt_same = test_flags > 0

        # FN: should be SAME, predicted DIFFERENT
        fn_local = np.where((gt_same == 1) & (preds_same == 0))[0]

        # FP: should be DIFFERENT, predicted SAME
        fp_local = np.where((gt_same == 0) & (preds_same == 1))[0]

        false_negatives.extend(test_indices[fn_local].tolist())
        false_positives.extend(test_indices[fp_local].tolist())
    
    # To see the worst mistakes first
    false_negatives = sorted(false_negatives, key=lambda i: scores[i])
    false_positives = sorted(false_positives, key=lambda i: -scores[i])
    
    total_wrong = len(false_positives) + len(false_negatives)
    print(f"Total wrong pairs: {total_wrong} / 6000 ({total_wrong / 60:.2f}%)")
    print(f"False Negatives (SAME → DIFF): {len(false_negatives)}")
    print(f"False Positives (DIFF → SAME): {len(false_positives)}")

    return false_negatives, false_positives, scores

def display_wrong_pairs(indices, nl, nr, scores, kind="FN", max_show=16):
    assert kind in ("FN", "FP")
    if kind == "FN":
        status = "FN (SAME→DIFF)"
        color = "red"
    else:
        status = "FP (DIFF→SAME)"
        color = "orange"
    cols = 4
    n_show = min(len(indices), max_show)
    rows = int(np.ceil(n_show / cols))
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i in range(n_show):
        idx = indices[i]
        left_path = nl[idx]
        right_path = nr[idx]
        score = scores[idx]
        img_l = Image.open(left_path).resize((112, 112))
        img_r = Image.open(right_path).resize((112, 112))
        combined = np.hstack((np.array(img_l), np.array(img_r)))
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(combined)
        ax.axis("off")
        ax.set_title(f"#{idx}  s={score:.3f}",fontsize=10,color=color,pad=3)

    plt.suptitle(f"{status} — {n_show} examples",fontsize=14,y=0.98)
    plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.05,hspace=0.25,wspace=0.05)
    plt.show()

def plot_confusion_matrix_seaborn(cm):
    labels = ["SAME", "DIFF"]
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,
        yticklabels=labels,cbar=False,annot_kws={"fontsize": 12})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("LFW Confusion Matrix (10-fold)", fontsize=14)
    plt.tight_layout()
    plt.show()

def compute_confusion_matrix_10fold(fl, fr, folds, flags, plot=True):
    scores = np.sum(fl * fr, axis=1)
    scores = np.clip(scores, -1.0, 1.0)
    TP = TN = FP = FN = 0
    for i in range(10):
        val_mask = folds != i
        test_mask = folds == i
        threshold = getThreshold(scores[val_mask], flags[val_mask], 10000)
        test_scores = scores[test_mask]
        test_flags = flags[test_mask] > 0   # ground truth SAME / DIFF
        preds_same = test_scores > threshold
        TP += np.sum((test_flags == 1) & (preds_same == 1))
        FN += np.sum((test_flags == 1) & (preds_same == 0))
        FP += np.sum((test_flags == 0) & (preds_same == 1))
        TN += np.sum((test_flags == 0) & (preds_same == 0))
    cm = np.array([[TP, FN],[FP, TN]])
    print("\nConfusion Matrix (LFW, 10-fold aggregated):")
    print("             Pred SAME   Pred DIFF")
    print(f"Actual SAME     {TP:5d}       {FN:5d}")
    print(f"Actual DIFF     {FP:5d}       {TN:5d}")
    acc = (TP + TN) / np.sum(cm) * 100
    print(f"Verification Accuracy from CM: {acc:.3f}%")
    if plot:
        plot_confusion_matrix_seaborn(cm)
    return cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze wrong LFW pairs using best model")
    parser.add_argument('--log_file', type=str, default='./results/CASIA_20251225_122726/train.log', help='Path to training log')
    parser.add_argument('--ckpt_dir', type=str, default='./results/CASIA_20251225_122726', help='Directory containing .ckpt files')
    parser.add_argument('--lfw_dir', type=str, default=LFW_ALIGNED_DATA_DIR)
    parser.add_argument('--max_show', type=int, default=20, help='Max wrong pairs to display')
    args = parser.parse_args()
    # Step 1: Find best checkpoint
    best_ckpt = find_best_checkpoint(args.log_file, args.ckpt_dir)
    # Step 2: Extract features
    fl, fr, nl, nr, folds, flags = extract_features(best_ckpt, args.lfw_dir)
    # Step 3: Find wrong pairs
    false_negatives, false_positives, scores = find_wrong_pairs_10fold(fl, fr, folds, flags)
    # Step 4: Display them
    if len(false_negatives) > 0: display_wrong_pairs(false_negatives, nl, nr, scores, kind="FN")
    else: print("Perfect! No false negatives found.")
    if len(false_positives) > 0: display_wrong_pairs(false_positives, nl, nr, scores, kind="FP")
    else: print("Perfect! No false negatives found.")
    # Step 5: Confusion Matrix
    cm = compute_confusion_matrix_10fold(fl, fr, folds, flags, plot=True)