import torch
from torchvision import transforms
import Augmentor
import random


### MODEL ### 

MODEL = 'model1'
HIDDEN = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
N_HEADS = 4
LENGTH = 32

### TRAINING ###

LR = 1e-4
BATCH_SIZE = 32
DROPOUT = 0.1
N_EPOCHS = 128
CHECKPOINT_FREQ = 10 # save checkpoint every 10 epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

# IMAGE SIZE
WIDTH = 256
HEIGHT = 64
CHANNELS = 3


### AUGMENTATIONS ###

p = Augmentor.Pipeline()
p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)

TRANSFORMS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(CHANNELS),
            p.torch_transform(),  # random distortion and shear
            transforms.ColorJitter(contrast=(0.5,1),saturation=(0.5,1)),
            transforms.RandomRotation(degrees=(-9, 9), fill=255),
            transforms.RandomAffine(10, None, [0.6 ,1] ,3 ,fillcolor=255),
            transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.9)),
            transforms.ToTensor()
        ])
