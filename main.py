import argparse
import os
import torchvision.datasets as datasets
from torchvision.datasets import FashionMNIST
import src.cnn as cnn
from torch.utils.data import DataLoader
from src.train import train_model
from src.validation import evaluate_model
from src.utils import *
from src.config import *
from src.gradcam import generate_gradcam

def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Choose the minimum logging level to be displayed.')
    parser.add_argument('--conf', action='store_true', help='Prints confusion matrix.')
    parser.add_argument('--acc', action='store_true', help='Prints model accuracy for train and test.')
    parser.add_argument('--plot-training', action='store_true', help='Plots training data.')
    parser.add_argument('--model-summary', action='store_true', help='Prints model summary.')
    parser.add_argument('--onnx', action='store_true', help='Exports onnx model.')
    parser.add_argument('--grad-cam', type=int, default=None, help='Generates Grad-CAM heatmap for given class.')
    parser.add_argument('--samples', type=int, default=10, help='Number of Grad-CAM samples to generate.')
    parser.add_argument('--aug-smooth', action='store_true', help='Only valid with --grad-cam.')
    parser.add_argument('--eigen-smooth', action='store_true', help='Only valid with --grad-cam.')
    parser.add_argument('--save-model', type=str, default=None, help='Saves trained model.')
    parser.add_argument('--load-model', type=str, default=None, help='Loads the given model.')
    args = parser.parse_args()

    if (args.samples != 10 or args.aug_smooth or args.eigen_smooth) and args.grad_cam is None:
        parser.error('--samples, --aug_smooth and --eigen_smooth require --grad_cam.')

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

    if args.load_model is not None:
        model = load_model(args.load_model)
    else:
        model = cnn.CNN().to(device)

        # train model
        logger.info("Training model...")
        train_model(model, train_loader, test_loader, device, args.plot_training)

        if args.save_model is not None:
            os.makedirs("models", exist_ok=True)
            filename = args.save_model if args.save_model.endswith(".pth") else args.save_model + ".pth"
            torch.save(model.state_dict(), f"models/{filename}")

    # Grad-CAM
    if args.grad_cam is not None:
        logger.info("Plotting Grad-CAM...")
        generate_gradcam(model, test_loader, target_class=args.grad_cam, num_samples=args.samples, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)

    # evaluate model
    if args.acc:
        logger.info("Evaluating model...")

        logger.info("Final train accuracy:")
        evaluate_model(model, train_loader, device)

        logger.info("Final test accuracy:")
        evaluate_model(model, test_loader, device, args.conf)

    # model summary
    if args.model_summary:
        print_model_summary(model)

    # ONNX export
    if args.onnx:
        logger.info("Exporting onnx model...")
        export_onnx(model)


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