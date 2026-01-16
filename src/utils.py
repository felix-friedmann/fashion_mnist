import matplotlib.pyplot as plt
from src.config import *

def plot_training_curves(train_losses, val_losses, val_accuracies, output_dir):
    """
    Plot training and validation curves.
    :param train_losses: Training losses.
    :param val_losses: Validation losses.
    :param val_accuracies: Validation accuracies.
    :param output_dir: Output directory.
    """

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # accuracy plot
    ax2.plot(epochs, val_accuracies, 'g-o', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves_{TIME}.png", dpi=150)
    plt.close()
