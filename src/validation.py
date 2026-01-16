import logging
import torch
from src.config import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, data_loader, device, output_dir, epoch=None, criterion=None):
    """
    Evaluates a model on the given dataloader.
    :param model: The model to be evaluated.
    :param data_loader: The dataloader.
    :param device: The device to be used for evaluation.
    :param output_dir: The directory to print the confusion matrix to.
    :param epoch: The current epoch.
    :param criterion: The training criterion.
    """

    logger = logging.getLogger(__name__)

    num_correct = 0
    num_samples = 0
    num_class_correct = [0 for _ in range(NUM_CLASSES)]
    num_class_samples = [0 for _ in range(NUM_CLASSES)]

    all_preds = []
    all_targets = []

    running_val_loss = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            scores = model(data)
            _, predicted = torch.max(scores, 1)

            if criterion is not None:
                loss = criterion(scores, targets)
                running_val_loss += loss.item()
                num_batches += 1

            num_samples += targets.size(0)
            num_correct += torch.sum(predicted == targets)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # last batch could be smaller so len(targets)
            for i in range(len(targets)):
                target = targets[i]
                preds = predicted[i]
                if target == preds:
                    num_class_correct[target] += 1
                num_class_samples[target] += 1

        acc = (100.0 * num_correct / num_samples).item()
        logger.info(f"Accuracy of the network: {acc:.2f}%")

        for i in range(NUM_CLASSES):
            class_acc = 100.0 * num_class_correct[i] / num_class_samples[i]
            logger.info(f"Accuracy of class {CLASSES[i]}: {class_acc:.2f}%")

    avg_val_loss = running_val_loss / num_batches if num_batches > 0 else 0.0

    if output_dir is not None:
        logger.info("Plotting confusion matrix...")
        cm = confusion_matrix(all_targets, all_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        if epoch is not None:
            plt.title(f"Confusion Matrix Epoch {epoch+1}/{NUM_EPOCHS}")
            plt.savefig(f"{output_dir}/confusion_matrix_epoch{epoch+1}.png", dpi=150)
        else:
            plt.title(f"Confusion Matrix Final")
            plt.savefig(f"{output_dir}/confusion_matrix_final.png", dpi=150)

        plt.close()

    return avg_val_loss, acc
