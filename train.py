import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from config import BATCH_SIZE, LEARNING_RATE, SAVE_FREQ, RESUME, FINAL_EMBEDDING_SIZE, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, QUICK_TEST 
from config import CASIA_ALIGNED_DATA_DIR, LFW_ALIGNED_DATA_DIR, SAVE_DIR
import model
from utils import init_log
from Loaders.CASIA_loader import CASIA
from Loaders.LFW_loader import LFW
import torch.optim as optim
import time
from eval import parse_lfw_pairs, evaluation_10_fold
import numpy as np
from tqdm import tqdm

def main():

    # GPU Initialization
    cudnn.benchmark = True
    cudnn.enabled = True
    multi_gpus = torch.cuda.device_count() > 1
    print("Number of visible GPUs:", torch.cuda.device_count())

    # Other inits
    start_epoch = 1
    if RESUME:
        save_dir = os.path.dirname(RESUME)
    else:
        save_dir = os.path.join(SAVE_DIR, MODEL_PRE + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(save_dir, exist_ok=True)
    logging = init_log(save_dir)
    _print = logging.info

    # Datasets
    trainset = CASIA(root=CASIA_ALIGNED_DATA_DIR)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )
    nl, nr, folds, flags = parse_lfw_pairs(root=LFW_ALIGNED_DATA_DIR)
    testdataset = LFW(nl, nr)
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=6, pin_memory=True, persistent_workers=True, drop_last=False
    )

    # Model
    net = model.MobileFacenet().cuda()
    ArcMargin = model.ArcMarginProduct(FINAL_EMBEDDING_SIZE, trainset.class_nums).cuda()
    if RESUME:
        ckpt = torch.load(RESUME,weights_only=True)
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        _print(f"Resumed from epoch {start_epoch}")

    # Optimizer + Scheduler
    prelu_params = [p for m in net.modules() if isinstance(m, nn.PReLU) for p in m.parameters()]
    ignored_params = (list(map(id, net.linear1.parameters())) + list(map(id, ArcMargin.weight)) + [id(p) for p in prelu_params])
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': ArcMargin.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=LEARNING_RATE, momentum=0.9, nesterov=True)

    # Reduce LR based on epoch
    milestones = [int(TOTAL_EPOCH * 0.6), int(TOTAL_EPOCH * 0.867), int(TOTAL_EPOCH * 0.967)]  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=milestones, gamma=0.1, last_epoch=-1)
    if multi_gpus:
        net = DataParallel(net)
        ArcMargin = DataParallel(ArcMargin)

    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    best_epoch = 0
    num_batches = len(trainloader)

    for epoch in range(start_epoch, TOTAL_EPOCH + 1):
        net.train()
        train_loss = 0.0
        total = 0
        start_time = time.time()
        _print("="*60)
        _print(f"Starting epoch {epoch}/{TOTAL_EPOCH}  ({num_batches} batches total)")
        progress_bar = tqdm(enumerate(trainloader),total=num_batches,desc=f"Epoch {epoch}",
            leave=True,ncols=150,bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            miniters=1, mininterval=0.0, postfix={'loss': '-', 'avg_loss': '-'})
        
        for batch_idx, (img, label) in progress_bar:
            if QUICK_TEST and batch_idx > 100: break
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            optimizer_ft.zero_grad(set_to_none=True)
            feats = net(img)
            feats = F.normalize(feats, p=2, dim=1) 
            output = ArcMargin(feats, label)
            loss = criterion(output, label)
            loss.backward()
            optimizer_ft.step()
            batch_loss = loss.item()
            train_loss += batch_loss * img.size(0)
            total += img.size(0)
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{train_loss / total:.4f}'
            })
        train_loss /= total
        _print(f"Epoch {epoch}/{TOTAL_EPOCH} | Loss: {train_loss:.4f} | "
               f"Time: {(time.time()-start_time)/60:.1f}m")

        # LFW evaluation
        if epoch % TEST_FREQ == 0 or epoch == TOTAL_EPOCH:
            net.eval()
            _print(f'Test Epoch: {epoch} ...')
            all_left = []
            all_right = []
            with torch.no_grad():
                for batch in testloader:            # batch shape: (B, 2, 3, 112, 112)
                    B = batch.size(0)
                    batch = batch.view(B*2, 3, 112, 112).cuda(non_blocking=True)
                    embeddings = net(batch)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.view(B, 2, -1)  # reshape back to (B, 2, 128)
                    left = embeddings[:,0,:]
                    right = embeddings[:,1,:]
                    all_left.append(left.cpu())
                    all_right.append(right.cpu())
            # Concatenate all
            fl = torch.cat(all_left).numpy()        # (6000, 128)
            fr = torch.cat(all_right).numpy()       # (6000, 128)
            # Run 10-fold evaluation
            accs = evaluation_10_fold(fl, fr, folds, flags)
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            _print(f'LFW Accuracy: {mean_acc:.3f}% ± {std_acc:.3f}%')
            # Track best accuracy
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_epoch = epoch

        scheduler.step()

        # Save model
        if epoch % SAVE_FREQ == 0 or epoch == TOTAL_EPOCH:
            state_dict = net.module.state_dict() if multi_gpus else net.state_dict()
            torch.save({'epoch': epoch,'net_state_dict': state_dict}
                       ,os.path.join(save_dir, '%03d.ckpt' % epoch))
            _print(f"Saved checkpoint: {epoch:03d}.ckpt")

    _print('finishing training')
    _print(f"Best LFW Accuracy: {best_acc:.3f}% at epoch {best_epoch}")

if __name__ == '__main__':
    main()