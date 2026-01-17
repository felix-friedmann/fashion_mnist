import logging
import matplotlib.pyplot as plt
import torch
from src.config import *

def setup_logging(level=logging.INFO):
    """
    Sets up the logging framework.
    :param level: Logging filter. Default: logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_device():
    """
    Returns the device to run the model on.
    :return: The gpu device or cpu if no usable gpu is available.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


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


def print_model_summary(model):
    """
    Logging model summary.
    :param model: The model.
    """

    logger = logging.getLogger()
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("Model Summary".center(50, '='))
    logger.info("Architecture:")
    logger.info("Conv: 1 -> 32 -> 64 -> 128 | Fc: 1152 -> 512 -> 10")
    logger.info("Dropout: Conv (0.1, 0.1, 0.15) | Fc (0.3)")
    logger.info("BatchNorm: True")
    logger.info(f"Total params: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB)")

    logger.info("Training Config:")
    logger.info(f"Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | Epochs: {NUM_EPOCHS}")
    logger.info("Optimizer: SGD (momentum=0.9, weight_decay=1e-4)")
    logger.info("Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)")

