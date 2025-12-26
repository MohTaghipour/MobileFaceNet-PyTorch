import os
import numpy as np
import csv
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import model
from Loaders.LFW_loader import LFW
from config import LFW_ALIGNED_DATA_DIR
import argparse

def parse_lfw_pairs(root, csv_file='pairs.csv', folder_name='lfw-deepfunneled/lfw-deepfunneled'):

    csv_path = os.path.join(root, csv_file)  
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")

    nameLs = []
    nameRs = []
    folds = []
    flags = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # ← Skip The header line
        for i, row in enumerate(reader):
            row = [x.strip() for x in row if x.strip()]  # Clean empty cells
            if len(row) == 3:       # Same person: name, img1, img2
                name, idx1, idx2 = row
                p1 = f"{name}/{name}_{int(idx1):04d}.jpg"
                p2 = f"{name}/{name}_{int(idx2):04d}.jpg"
                flag = 1
            elif len(row) == 4:     # Different people: name1, idx1, name2, idx2
                name1, idx1, name2, idx2 = row
                p1 = f"{name1}/{name1}_{int(idx1):04d}.jpg"
                p2 = f"{name2}/{name2}_{int(idx2):04d}.jpg"
                flag = -1
            else:
                print(f"Skipping invalid row {i+1}: {row}")
                continue

            # Full paths
            nameL = os.path.join(root, folder_name, p1)
            nameR = os.path.join(root, folder_name, p2)
            nameLs.append(nameL)
            nameRs.append(nameR)
            folds.append(i // 600)   # 10-fold: 600 pairs per fold
            flags.append(flag)

    print(f"Loaded {len(nameLs)} LFW pairs (10-fold) from {csv_file}")
    return [nameLs, nameRs, folds, flags]

def getAccuracy(scores, flags, threshold):
    if len(scores) == 0: return 0.0
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    thresholds = np.linspace(-1, 1, 2*thrNum+1)
    accuracies = [getAccuracy(scores, flags, t) for t in thresholds]
    best_idx = np.argmax(accuracies)
    return thresholds[best_idx]

def evaluation_10_fold(featureLs, featureRs, fold, flags):
    """
    10-fold face verification evaluation.
    Args:
        featureLs: numpy array of left embeddings, shape (N, D)
        featureRs: numpy array of right embeddings, shape (N, D)
        fold: numpy array of fold indices, shape (N,)
        flags: numpy array of ground-truth labels, shape (N,)
    Returns:
        ACCs: numpy array of length 10 with fold accuracies
    """
    ACCs = np.zeros(10)
    featureLs = np.asarray(featureLs)
    featureRs = np.asarray(featureRs)
    fold = np.asarray(fold).astype(int)
    flags = np.asarray(flags).astype(int)
    scores = np.sum(featureLs * featureRs, axis=1)
    scores = np.clip(scores, -1.0, 1.0)
    for i in range(10):
        valFold = fold != i
        testFold = fold == i
        threshold = getThreshold(scores[valFold], flags[valFold], 10000)
        ACCs[i] = getAccuracy(scores[testFold], flags[testFold], threshold)
    return ACCs

def extract_features(dataset_pairs, batch_size=32, resume=None, gpu=True):
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    nl, nr, folds, flags = dataset_pairs
    dataset = LFW(nl, nr)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=8, drop_last=False)
    net = model.MobileFacenet().to(device)
    if resume:
        ckpt = torch.load(resume, map_location=device)
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    featureLs, featureRs = [], []
    with torch.no_grad():
        for left, right in loader:   # ← مهم
            left = left.to(device)
            right = right.to(device)

            embL = F.normalize(net(left), dim=1).cpu().numpy()
            embR = F.normalize(net(right), dim=1).cpu().numpy()

            featureLs.append(embL)
            featureRs.append(embR)

    featureLs = np.concatenate(featureLs, axis=0)
    featureRs = np.concatenate(featureRs, axis=0)
    folds = np.asarray(folds)
    flags = np.asarray(flags)
    return featureLs, featureRs, folds, flags

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Face verification evaluation")
    parser.add_argument('--lfw_dir', type=str, default=LFW_ALIGNED_DATA_DIR)
    parser.add_argument('--resume', type=str, default='./results/CASIA_20251225_122726/024.ckpt', help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    pairs = parse_lfw_pairs(args.lfw_dir)
    featureLs, featureRs, folds, flags = extract_features(pairs, batch_size=args.batch_size, resume=args.resume)
    accs = evaluation_10_fold(featureLs, featureRs, folds, flags)
    for i, a in enumerate(accs):
        print(f"Fold {i+1}: {a*100:.2f}%")
    print(f"Average accuracy: {np.mean(accs)*100:.2f}%")