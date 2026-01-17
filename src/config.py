import torchvision.transforms as transforms
from datetime import datetime

TIME: str = datetime.now().strftime("%Y%m%d_%H%M%S")

BATCH_SIZE: int = 64

SEED: int = 42

NUM_EPOCHS: int = 30

LEARNING_RATE: float = 0.001

NUM_CLASSES: int = 10

IN_CHANNELS: int = 1

CLASSES: list[str] = ['T-shirt/top',
                     'Trouser',
                     'Pullover',
                     'Dress',
                     'Coat',
                     'Sandal',
                     'Shirt',
                     'Sneaker',
                     'Bag',
                     'Ankle boot']

CONFUSED_CLASSES: list[int] = [0, 2, 4, 6]

BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor()
])

STRONG_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
])