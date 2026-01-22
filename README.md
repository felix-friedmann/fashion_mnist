![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![WIP](https://img.shields.io/badge/status-WIP-orange)

# Fashion MNIST Image Recognition

Deep learning image classifier for the Fashion-MNIST dataset. Achieved model improvement from 84.23% to 92.74% test accuracy through 
regularization techniques and architecture improvements.

## Data

Fashion-Mnist dataset, 60.000 training images, 10.000 test images with 10 classes. Each image is a 28x28 grayscale image, for more information see 
the [fashion-mnist repository](https://github.com/zalandoresearch/fashion-mnist) of Zalando research.

## Architecture

To achieve the best accuracy different model architectures have been tried, all of them are a 3-layer convolutional neural network
(for more details to the different models see this [file](architectures.md)). The model that achieved the highest accuracy has the following architecture:

![Model Architecture](docs/model.onnx.png)

It is a 3-layer CNN with 1 -> 32, 32 -> 64 and 64 -> 128 channels and each one with a 3x3 kernel. Each layer is batch normed with the corresponding
number of output channels, activated through Relu and pooled by 2x2 max pooling. After reshaping follow two fully connected layers with 1152 -> 512 -> 10.
The first to convolutional layers have a dropout rate of 0.1, the third of 0.15 and the first fully connected layer of 0.3. The model trains with a batch
size of 64, 30 epochs and a starter learning rate of 0.001 with a ReduceLROnPlateau scheduler. The criterion is a CrossEntropyLoss while the optimizer is 
stochastic gradient descent with weight decay.

The final model tends to overfit lightly in the last epochs though the difference in loss stays within 0.0695:

![Training Curves](docs/training_curves.png)

While the footwear is classified with high confidence, shirts overlap with classes that contain similar features:

![Confusion Matrix](docs/confusion_matrix.png)

| Class       | Test Acc | Train Acc |
|-------------|----------|-----------|
| T-shirt/Top | 84.90%   | 92.00%    |
| Trouser     | 98.90%   | 99.82%    |
| Pullover    | 87.70%   | 93.37%    |
| Dress       | 94.10%   | 97.87%    |
| Coat        | 90.60%   | 95.62%    |
| Sandal      | 98.20%   | 99.83%    |
| Shirt       | 79.70%   | 92.32%    |
| Sneaker     | 98.00%   | 98.80%    |
| Bag         | 98.30%   | 99.52%    |
| Ankle boot  | 97.00%   | 98.93%    |

## Grad-CAM Visualization

To visualize what the CNN has learned, a Grad-CAM is implemented using the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library
that follows [Selvaraju et al. (2019)](https://arxiv.org/pdf/1610.02391). As advised by most of the literature, the last convolutional layer is chosen as
the target layer or in this case the batch norm of conversion layer 3.

![T-shirt/Top Grad-CAM](docs/gradcam/tshirt_top_gradcam.png)

![Trouser Grad-CAM](docs/gradcam/trouser_gradcam.png)

![Pullover Grad-CAM](docs/gradcam/pullover_gradcam.png)

![Dress Grad-CAM](docs/gradcam/dress_gradcam.png)

![Coat Grad-CAM](docs/gradcam/coat_gradcam.png)

![Sandal Grad-CAM](docs/gradcam/sandal_gradcam.png)

![Shirt Grad-CAM](docs/gradcam/shirt_gradcam.png)

![Sneaker Grad-CAM](docs/gradcam/sneaker_gradcam.png)

![Bag Grad-CAM](docs/gradcam/bag_gradcam.png)

![Ankle boot Grad-CAM](docs/gradcam/ankleboot_gradcam.png)

## Experiments

| Nr | Model     | Augmentation                                                           | LR               | Epochs | Train Acc | Test Acc | Notes                                                             |
|----|-----------|------------------------------------------------------------------------|------------------|--------|-----------|----------|-------------------------------------------------------------------|
| 1  | CNN-small | None                                                                   | 0.001            | 10     | 85.37%    | 84.23%   | Baseline, worst: Shirt (51.30%)                                   |
| 2  | CNN-small | RandomAffine/Erasing on Classes `T-shirt`, `Pullover`, `Coat`, `Shirt` | 0.001            | 10     | 83.19%    | 82.40%   | Fashion MNIST dataset too clean, augmentation only hurts          |
| 3  | CNN-mid   | None                                                                   | 0.001            | 10     | 87.01%    | 86.07%   | Worst: Shirt (54.50%), no plateau in loss graph                   |
| 4  | CNN-mid   | None                                                                   | 0.001            | 20     | 90.14%    | 88.70%   | Loss plateaus at around 0.33-0.31                                 |
| 5  | CNN-big   | None                                                                   | 0.001            | 20     | 89.28%    | 87.84%   | Light overfitting                                                 |
| 6  | CNN-big   | None                                                                   | 0.001            | 20     | 87.60%    | 86.25%   | Added dropout (conv: 0.2, fc: 0.4), wrong dropout for conv no 2d  |
| 7  | CNN-big   | None                                                                   | 0.001            | 20     | 88.25%    | 87.08%   | Changed to right conv dropout, added weight decay on optimizer    |
| 8  | CNN-big   | None                                                                   | 0.001 + schedule | 25     | 85.50%    | 84.76%   | Raised epochs and added lr scheduler, batch to 128                |
| 9  | CNN-big   | None                                                                   | 0.001 + schedule | 25     | 89.04%    | 87.91%   | dropped batch back to 64, too much regularization?                |
| 10 | CNN-mid   | None                                                                   | 0.001 + schedule | 25     | 90.21%    | 88.46%   | no dropout                                                        |
| 11 | CNN-mid   | None                                                                   | 0.001 + schedule | 25     | 89.99%    | 88.38%   | fc dropout 0.2                                                    |
| 12 | CNN-mid   | None                                                                   | 0.001 + schedule | 25     | 99.76%    | 91.78%   | batch normalization, strong overfitting                           |
| 13 | CNN-mid   | None                                                                   | 0.001 + schedule | 20     | 95.86%    | 92.37%   | small conv (0.1) and fc (0.2) dropout                             |
| 14 | CNN-mid   | None                                                                   | 0.001 + schedule | 30     | 97.42%    | 92.85%   |                                                                   |
| 15 | CNN-mid   | None                                                                   | 0.001 + schedule | 25     | 98.01%    | 92.72%   | dropped batch size to 32, bigger overfit                          |
| 16 | CNN-mid   | None                                                                   | 0.001 + schedule | 30     | 96.81%    | 92.74%   | small dropout changes, batch back to 64, less overfit than Nr. 14 |

## Installation and Usage

This project requires **Python >= 3.9.6**. Clone the repository and install dependencies:

```bash
git clone https://github.com/felix-friedmann/fashion_mnist.git
cd fashion_mnist

# Optional: use a venv for isolation
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Usage

The main script allows you to train, evaluate and export the CNN model for Fashion-MNIST classification.

#### Available Flags

| Flag                   | Description                                                          |
|------------------------|----------------------------------------------------------------------|
| `--log-level <LEVEL>`  | Set logging verbosity (DEBUG, INFO, WARNING, ERROR). Default: `INFO` |
| `--acc`                | Test model accuracy on train and test set                            |
| `--conf`               | Generate confusion matrix (saved to `conf_matrix/`)                  |
| `--plot-training`      | Plot training curves (saved to `graphs/`)                            |
| `--model-summary`      | Print model architecture summary                                     |
| `--onnx`               | Export trained model to ONNX format                                  |
| `--grad-cam <classes>` | Generate Grad-CAM for the given classes (saved to `gradcam/<class>`) |
| `--samples`            | The number of Grad-CAM samples to generate. Default: 10              |
| `--aug-smooth`         | Reduces noise in the Grad-CAM through small changes of the images    |
| `--eigen-smooth`       | Reduces noise in the Grad-CAM through eigenvector projection         |
| `--save-model <name>`  | Saves the trained model to `models/<name>.pth`                       |
| `--load-model <name>`  | Loads model from `models/<name>.pth` instead of training a new one   |

#### Examples

```bash
# Train model and evaluate it on the test dataset
python main.py

# Run the main script only with warning logs or higher
python main.py --log-level WARNING

# Train the model, generate confusion matrix and training curves
# and export the model to ONNX.
python main.py --conf --plot-training --onnx
```

### Repository Overview

`main.py` - Main script for data loading, model training and evaluation  
`src/augment.py` - Provides the possibility of data augmentation  
`src/cnn.py` - CNN model architecture  
`src/config.py` - Configuration parameters  
`gradcam.py` - Grad-CAM implementation  
`src/train.py` - Model training and train validation  
`src/utils.py` - Helper functions  
`src/validation.py` - Model evaluation
