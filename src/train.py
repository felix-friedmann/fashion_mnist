import logging
import os
from torch import nn, optim
from src.config import *
from src.validation import evaluate_model
from src.utils import plot_training_curves
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(model, train_loader, test_loader, device, plot=False):
    """
    Training the model on the training set.
    :param model: The model to be trained.
    :param train_loader: The training data loader.
    :param test_loader: The testing data loader.
    :param device: The device to run the model on.
    :param plot: Whether to plot the training curves.
    """

    logger = logging.getLogger(__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        num_batches = 0

        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}...")

        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # fwd
            scores = model(data)
            loss = criterion(scores, target)

            # bwd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tracking
            running_loss += loss.item()
            num_batches += 1

        # average training loss
        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)

        # evaluating model
        logger.info(f"Validating model of epoch {epoch + 1}/{NUM_EPOCHS}...")
        val_loss, val_acc = evaluate_model(model, test_loader, device, False, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            logger.info(f"New learning rate: {old_lr} -> {new_lr}")

        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} Summary: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

    if plot:
        output = f"graphs"
        os.makedirs(output, exist_ok=True)
        plot_training_curves(train_losses, val_losses, val_accuracies, output)

    logger.info("Finished training.")
    return train_losses, val_losses, val_accuracies
