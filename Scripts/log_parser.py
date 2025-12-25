import re
import matplotlib.pyplot as plt

def parse_log_file(log_path):
    epochs = []
    losses = []
    mean_accs = []
    std_accs = []
    loss_pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d.]+)')
    acc_pattern = re.compile(r'LFW Accuracy:\s+([\d.]+)%\s+±\s+([\d.]+)%')
    current_epoch = None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            loss_match = loss_pattern.search(line)
            if loss_match:
                epoch = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                current_epoch = epoch
                epochs.append(epoch)
                losses.append(loss)
                if len(mean_accs) < len(epochs):
                    mean_accs.append(None)
                    std_accs.append(None)

            acc_match = acc_pattern.search(line)
            if acc_match and current_epoch is not None:
                mean_acc = float(acc_match.group(1))
                std_acc = float(acc_match.group(2))
                if len(mean_accs) < len(epochs):
                    mean_accs.append(mean_acc)
                    std_accs.append(std_acc)
                else:
                    mean_accs[-1] = mean_acc
                    std_accs[-1] = std_acc

    return epochs, losses, mean_accs, std_accs

if __name__ == '__main__':
    
    log_file = './results/CASIA_20251225_122726/train.log'
    epochs, losses, mean_accs, std_accs = parse_log_file(log_file)

    # Filter evaluated epochs
    eval_epochs = [e for e, m in zip(epochs, mean_accs) if m is not None]
    eval_mean = [m for m in mean_accs if m is not None]
    eval_std = [s for s in std_accs if s is not None]

    # Find the epoch with highest accuracy
    if eval_mean:
        best_idx = eval_mean.index(max(eval_mean))
        best_epoch = eval_epochs[best_idx]
        best_acc = eval_mean[best_idx]
        best_std = eval_std[best_idx]
    else:
        best_epoch = None
        best_acc = None

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Training Loss
    ax1.plot(epochs, losses, marker='o', linestyle='-', color='tab:blue', markersize=4)
    ax1.set_title('Training Loss per Epoch', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    # Plot 2: LFW Accuracy with error band
    ax2.plot(eval_epochs, eval_mean, marker='o', linestyle='-', color='tab:green', label='Mean Accuracy', markersize=6)

    # Add shaded area for ± std
    ax2.fill_between(eval_epochs,
                    [m - s for m, s in zip(eval_mean, eval_std)],
                    [m + s for m, s in zip(eval_mean, eval_std)],
                    alpha=0.2, color='tab:green')

    # Highlight the BEST accuracy point
    if best_epoch is not None:
        ax2.plot(best_epoch, best_acc,
                marker='*', markersize=15, color='red', markeredgecolor='black', linewidth=1,
                label=f'Best: {best_acc:.3f}% (epoch {best_epoch})')
        # Optional: add text annotation
        ax2.annotate(f'Best\n{best_acc:.3f}% @ epoch {best_epoch}',
                    xy=(best_epoch, best_acc),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax2.set_title('LFW Accuracy per Evaluated Epoch', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Training Progress Summary', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()