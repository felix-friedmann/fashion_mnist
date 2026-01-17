import argparse
import torchvision.datasets as datasets
from torchvision.datasets import FashionMNIST
import src.cnn as cnn
from torch.utils.data import DataLoader
from src.train import train_model
from src.validation import evaluate_model
from src.utils import *
from src.config import *

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Choose the minimum logging level to be displayed.')
    parser.add_argument('--conf', action='store_true', help='Prints confusion matrix.')
    parser.add_argument('--train-acc', action='store_true', help='Prints training accuracy.')
    parser.add_argument('--plot-training', action='store_true', help='Plots training data.')
    parser.add_argument('--model-summary', action='store_true', help='Prints model summary.')
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

    # train model
    logger.info("Training model...")
    train_model(model, train_loader, test_loader, device, args.plot_training)

    logger.info("Evaluating model...")
    if args.train_acc:
        logger.info("Final train accuracy:")
        evaluate_model(model, train_loader, device)

    logger.info("Final test accuracy:")
    evaluate_model(model, test_loader, device, args.conf)

    if args.model_summary:
        print_model_summary(model)


def load_data() -> tuple[FashionMNIST, DataLoader, FashionMNIST, DataLoader]:
    """
    Loads the training and testing datasets from Pytorch datasets.
    :return: The training and testing dataset.
    """
    train_dataset = datasets.FashionMNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.FashionMNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


if __name__ == '__main__':
    main()