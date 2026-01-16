import logging
from torch import nn, optim
from src.config import *
from src.evaluate import evaluate_model

def train_model(model, train_loader, test_loader, device, output_dir):
    """
    Training the model on the training set.
    :param model: The model to be trained.
    :param train_loader: The training data loader.
    :param test_loader: The testing data loader.
    :param device: The device to run the model on.
    :param output_dir: The directory to print the confusion matrix to.
    """

    logger = logging.getLogger(__name__)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        model.train()
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

            if idx % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Batch {idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        logger.info(f"Evaluating model of epoch {epoch + 1}/{NUM_EPOCHS}...")
        evaluate_model(model, test_loader, device, output_dir, epoch)

    logger.info("Finished training.")


