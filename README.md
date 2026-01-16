![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![WIP](https://img.shields.io/badge/status-WIP-orange)

# Fashion MNIST Image Recognition

## Experiments

| Model     | Augmentation                                                           | LR               | Epochs | Train Acc | Test Acc | Notes                                                            |
|-----------|------------------------------------------------------------------------|------------------|--------|-----------|----------|------------------------------------------------------------------|
| CNN-small | None                                                                   | 0.001            | 10     | 85.37%    | 84.23%   | Baseline, worst: Shirt (51.30%)                                  |
| CNN-small | RandomAffine/Erasing on Classes `T-shirt`, `Pullover`, `Coat`, `Shirt` | 0.001            | 10     | 83.19%    | 82.40%   | Fashion MNIST dataset too clean, augmentation only hurts         |
| CNN-mid   | None                                                                   | 0.001            | 10     | 87.01%    | 86.07%   | Worst: Shirt (54.50%), no plateau in loss graph                  |
| CNN-mid   | None                                                                   | 0.001            | 20     | 90.14%    | 88.70%   | Loss plateaus at around 0.33-0.31                                |
| CNN-big   | None                                                                   | 0.001            | 20     | 89.28%    | 87.84%   | Light overfitting                                                |
| CNN-big   | None                                                                   | 0.001            | 20     | 87.60%    | 86.25%   | Added dropout (conv: 0.2, fc: 0.4), wrong dropout for conv no 2d |
| CNN-big   | None                                                                   | 0.001            | 20     | 88.25%    | 87.08%   | Changed to right conv dropout, added weight decay on optimizer   |
| CNN-big   | None                                                                   | 0.001 + schedule | 25     | 85.50%    | 84.76%   | Raised epochs and added lr scheduler, batch to 128               |
| CNN-big   | None                                                                   | 0.001 + schedule | 25     | 89.04%    | 87.91%   | dropped batch back to 64, too much regularization?               |