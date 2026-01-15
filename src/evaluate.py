import logging
import torch
from src.config import *

def evaluate_model(model, data_loader, device):
    """
    Evaluates a model on the given dataloader.
    :param model: The model to be evaluated.
    :param data_loader: The dataloader.
    :param device: The device to be used for evaluation.
    """

    logger = logging.getLogger(__name__)

    num_correct = 0
    num_samples = 0
    num_class_correct = [0 for _ in range(NUM_CLASSES)]
    num_class_samples = [0 for _ in range(NUM_CLASSES)]

    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            scores = model(data)
            _, predicted = torch.max(scores, 1)
            num_samples += targets.size(0)
            num_correct += (predicted == targets).sum()

            # last batch could be smaller so len(targets)
            for i in range(len(targets)):
                target = targets[i]
                preds = predicted[i]
                if target == preds:
                    num_class_correct[target] += 1
                num_class_samples[target] += 1

        acc = 100.0 * num_correct / num_samples
        logger.info(f"Accuracy of the network: {acc:.2f}%")

        for i in range(NUM_CLASSES):
            acc = 100.0 * num_class_correct[i] / num_class_samples[i]
            logger.info(f"Accuracy of class {CLASSES[i]}: {acc:.2f}%")
