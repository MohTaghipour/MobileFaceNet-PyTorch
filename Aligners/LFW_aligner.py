import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
import shutil

from config import LFW_DATA_DIR, LFW_ALIGNED_DATA_DIR

def align_face(img_np, landmarks, output_size=112):
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    dst[:, 0] += (output_size - 112) / 2
    dst[:, 1] += (output_size - 112) / 2
    src = landmarks.astype(np.float32)

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return cv2.resize(img_np, (output_size, output_size))
    return cv2.warpAffine(img_np, M, (output_size, output_size), borderValue=128)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(
        image_size=112,
        margin=0,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device
    )

    os.makedirs(LFW_ALIGNED_DATA_DIR, exist_ok=True)
    csv_path = os.path.join(LFW_DATA_DIR, 'pairs.csv')

    # Gather all image files using os.walk
    all_files = []
    for root, _, files in os.walk(LFW_DATA_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, LFW_DATA_DIR)
                dst_path = os.path.join(LFW_ALIGNED_DATA_DIR, rel_path)
                all_files.append((src_path, dst_path))
    print(f"Found {len(all_files)} images. Processing in batches...")

    batch_size = 128
    no_face_count = 0

    for i in tqdm(range(0, len(all_files), batch_size)):
        batch_files = all_files[i:i+batch_size]
        imgs = []
        dst_paths = []

        for src_path, dst_path in batch_files:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                img = Image.open(src_path).convert('RGB')
                imgs.append(img)
                dst_paths.append(dst_path)
            except Exception as e:
                print(f"Skipping {src_path}: {e}")

        if not imgs:
            continue

        # Detect faces in batch
        with torch.no_grad():
            boxes, _, landmarks_batch = mtcnn.detect(imgs, landmarks=True)

        for idx, img in enumerate(imgs):
            img_np = np.array(img)
            landmarks = landmarks_batch[idx] if landmarks_batch is not None else None

            if landmarks is not None:
                aligned = align_face(img_np, landmarks[0])
            else:
                no_face_count += 1
                aligned = cv2.resize(img_np, (112, 112))

            Image.fromarray(aligned.astype(np.uint8)).save(dst_paths[idx])

    shutil.copy(csv_path, LFW_ALIGNED_DATA_DIR)
    print(f"LFW alignment complete! {no_face_count} images had no detected faces.")