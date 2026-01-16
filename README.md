![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![WIP](https://img.shields.io/badge/status-WIP-orange)

# Fashion MNIST Image Recognition

## Experiments

| Model                             | Augmentation                                                           | LR    | Epochs | Train Acc | Test Acc | Notes                                                    |
|-----------------------------------|------------------------------------------------------------------------|-------|--------|-----------|----------|----------------------------------------------------------|
| CNN (3 conv, maxpool, 2 fc, relu) | None                                                                   | 0.001 | 10     | 85.37%    | 84.23%   | Baseline, worst: Shirt (51.30%)                          |
| CNN (3 conv, maxpool, 2 fc, relu) | RandomAffine/Erasing on Classes `T-shirt`, `Pullover`, `Coat`, `Shirt` | 0.001 | 10     | 83.19%    | 82.40%   | Fashion MNIST dataset too clean, augmentation only hurts |