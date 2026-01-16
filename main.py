import argparse
import torch
import logging
import torchvision.datasets as datasets
import src.cnn as cnn
from src.config import *
from torch.utils.data import DataLoader
from src.train import train_model
from src.evaluate import evaluate_model
from datetime import datetime
import os

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Choose the minimum logging level to be displayed.')
    parser.add_argument('--conf', action='store_true', help='Prints confusion matrix.')
    parser.add_argument('--train-acc', action='store_true', help='Prints training accuracy.')
    args = parser.parse_args()

    # logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # fetch data
    logger.info("Fetching and augmenting Fashion MNIST data from Pytorch datasets...")
    train_dataset, train_loader, test_dataset, test_loader = load_data()

    # get device and model
    device = get_device()
    logger.info(f"Fetching model to device: {device.type}")
    model = cnn.CNN().to(device)

    # create conf matrix directory
    output_dir = None
    if args.conf:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"conf_matrix/conf_matrix_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

    # train model
    logger.info("Training model...")
    train_model(model, train_loader, test_loader, device, output_dir)

    logger.info("Evaluating model...")

    if args.train_acc:
        logger.info("Final train accuracy:")
        evaluate_model(model, train_loader, device, output_dir)

    logger.info("Final test accuracy:")
    evaluate_model(model, test_loader, device, output_dir)


def setup_logging(level=logging.INFO):
    """
    Sets up the logging framework.
    :param level: Logging filter. Default: logging.INFO.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_data():
    """
    Loads the training and testing datasets from Pytorch datasets.
    :return: The training and testing dataset.
    """
    train_dataset = datasets.FashionMNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.FashionMNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


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


if __name__ == '__main__':
    main()