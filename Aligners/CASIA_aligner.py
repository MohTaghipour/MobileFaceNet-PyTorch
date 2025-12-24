import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from facenet_pytorch import MTCNN
import torch
from config import CASIA_DATA_DIR, CASIA_ALIGNED_DATA_DIR

def align_face(img_np, landmarks, output_size=112):
    """
    Align face using 5-point landmarks and affine transformation.
    """
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
    aligned = cv2.warpAffine(img_np, M, (output_size, output_size), borderValue=128)
    return aligned

if __name__ == '__main__':
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

    os.makedirs(CASIA_ALIGNED_DATA_DIR, exist_ok=True)
    txt_path = os.path.join(CASIA_DATA_DIR, 'casia-webface.txt')

    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if len(line.strip().split()) >= 2]

    print("Preprocessing CASIA-WebFace in batches...")
    batch_size = 128  # Adjust it based on GPU memory
    no_face_count = 0

    for i in tqdm(range(0, len(lines), batch_size)):
        batch_lines = lines[i:i+batch_size]
        imgs = []
        src_paths = []
        dst_paths = []

        # Load batch images
        for line in batch_lines:
            rel_path = line.split()[1]
            src_path = os.path.join(CASIA_DATA_DIR, rel_path)
            dst_path = os.path.join(CASIA_ALIGNED_DATA_DIR, rel_path)

            if not os.path.exists(src_path):
                continue

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img = Image.open(src_path).convert('RGB')
            imgs.append(img)
            src_paths.append(src_path)
            dst_paths.append(dst_path)

        # Batch face detection
        with torch.no_grad():
            boxes, _, landmarks_batch = mtcnn.detect(imgs, landmarks=True)

        # Process each image in batch
        for idx, img in enumerate(imgs):
            img_np = np.array(img)
            landmarks = landmarks_batch[idx] if landmarks_batch is not None else None

            if landmarks is not None:
                aligned = align_face(img_np, landmarks[0])
            else:
                no_face_count += 1
                aligned = cv2.resize(img_np, (112, 112))

            Image.fromarray(aligned.astype(np.uint8)).save(dst_paths[idx])

    shutil.copy(txt_path, CASIA_ALIGNED_DATA_DIR)
    print(f"CASIA-WebFace alignment complete! {no_face_count} images had no detected faces.")