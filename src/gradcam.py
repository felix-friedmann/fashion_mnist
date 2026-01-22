import logging
import os
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.utils import *
from src.config import *

def gradcam(model: CNN, input_tensor: Tensor, targets: list[ClassifierOutputTarget] = None, aug_smooth: bool = False, eigen_smooth: bool = False) -> np.ndarray:
    """
    Generate the Grad-CAM.
    :param model: CNN model.
    :param input_tensor: The input image.
    :param targets: List of targets.
    :param aug_smooth: Whether to apply aug smoothing.
    :param eigen_smooth: Whether to apply eigen smoothing.
    :return: Grad-CAM.
    """

    target_layer = [model.bn3]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]

        img = input_tensor.squeeze().cpu().numpy()
        img_normalized = (img - img.min()) / (img.max() - img.min())
        rgb_img = np.stack([img_normalized] * 3, axis=-1)

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return visualization


def save_gradcam_visualization(visualization, true_label: int, pred_label: int, output_dir: str, filename: str):
    """
    Save the Grad-CAM visualization.
    :param visualization: Grad-CAM visualization.
    :param true_label: True label.
    :param pred_label: Predicted label.
    :param output_dir: Output directory.
    :param filename: Output filename.
    """

    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.title(f'True: {CLASSES[true_label]}, Pred: {CLASSES[pred_label]}')
    plt.axis('off')

    plt.savefig(f"{output_dir}/{filename}", bbox_inches='tight', dpi=150)
    plt.close()


def generate_gradcam(model: CNN, data_loader: DataLoader, target_classes: list[int], num_samples: int, aug_smooth: bool, eigen_smooth: bool):
    """
    Generate Grad-CAM visualization.
    :param model: CNN model.
    :param data_loader: The image data.
    :param target_classes: Target classes.
    :param num_samples: Number of samples.
    :param aug_smooth: Whether to apply aug smoothing.
    :param eigen_smooth: Whether to apply eigen smoothing.
    """

    logger = logging.getLogger()
    for target in target_classes:
        if target < 0 or target >= len(CLASSES):
            logger.error(f"Target class {target} does not exist. Skipping non existing class {target}.")
            continue

    model.eval()
    device = get_device()

    samples_found = {cls: 0 for cls in target_classes}

    for data, target in data_loader:
        # break if all classes have num_samples examples
        if all(count >= num_samples for count in samples_found.values()):
            break

        for target_class in target_classes:
            # jump to next class
            if samples_found[target_class] >= num_samples:
                continue

            # mask is tensor of bool -> all True where target == target_class
            mask = target == target_class

            if not mask.any():
                continue

            # current batch
            class_images = data[mask].to(device)
            class_labels = target[mask]

            output = model(class_images)
            preds = output.argmax(dim=1)

            for idx in range(class_images.shape[0]):
                if samples_found[target_class] >= num_samples:
                    break

                # slice to keep batch dimension!?
                image = class_images[idx:idx+1]
                true_label = class_labels[idx].item()
                pred_label = preds[idx].item()

                targets = [ClassifierOutputTarget(target_class)]
                visualization = gradcam(model, image, targets, aug_smooth, eigen_smooth)

                output_dir_class = f"gradcam/{CLASSES[target_class]}"
                os.makedirs(output_dir_class, exist_ok=True)

                filename = f"gradcam_{CLASSES[target_class]}_{samples_found[target_class]}.png"

                save_gradcam_visualization(visualization, true_label, pred_label, output_dir_class, filename)

                samples_found[target_class] += 1
