import argparse
import torch
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import src.cnn as cnn
from src.config import *
from torch.utils.data import DataLoader
from src.eda import run_eda
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--eda', action='store_true', help='Runs minimal exploratory data analysis.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Choose the minimum logging level to be displayed.')
    args = parser.parse_args()

    # logging
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # seeds
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # fetch data
    logger.info("Fetching Fashion MNIST data from Pytorch datasets...")
    train_dataset, train_loader, test_dataset, test_loader = load_data()

    # run minimal eda (needed?)
    if args.eda:
        logger.info("Running minimal exploratory data analysis...")
        run_eda(train_dataset, test_dataset)

    device = get_device()
    logger.info(f"Fetching model to device: {device.type}")
    model = cnn.CNN().to(device)

    logger.info("Training model...")
    train_model(model, train_loader, test_loader, device)

    logger.info("Evaluating model...")
    logger.info("Final train accuracy:")
    evaluate_model(model, train_loader, device)
    logger.info("Final test accuracy:")
    evaluate_model(model, test_loader, device)


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