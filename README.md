## MobileFaceNet with Modern Pytorch

This project is a heavily modified version of the original implementation by Xiaoccer:

https://github.com/Xiaoccer/MobileFaceNet_Pytorch

“This project is still under development, so please use it with caution.”

Currently it is based on:

- Python 3.12.12
- CUDA: 12.1
- PyTorch: 2.2.2
- torchvision: 0.17.1

This version includes:

- Full upgrade to PyTorch 2.x
- Modern project structure
- Many bug fixes
- New training utilities
- Preprocessing Codes

In order to train use this dataset:
https://www.kaggle.com/datasets/ntl0601/casia-webface

For test:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

1. Create conda environment using the provided yaml file

2. Update `config.py` according to your setup.

3. run LFW_aligner.py and CASIA_aligner.py

4. Run `train.py`. You should see output similar to the following (depending on your `batch_size` and number of GPUs):

_Loaded 460412 images from 10537 identities._

_Loaded 6000 LFW pairs (10-fold) from pairs.csv_

Starting epoch 1/30  (3596 batches total)
Epoch 1: 100%|████████████████████████████████████████████████████████████████████| 3596/3596 [13:29<00:00,  4.44it/s, loss=18.5955, avg_loss=22.7953]
Epoch 1/30 | Loss: 22.7953 | Time: 14.1m
Test Epoch: 1 ...
 LFW Accuracy: 88.400% ± 1.209%
Saved checkpoint: 001.ckpt

etc.

### The progress updates will continue for all batches in each epoch.
