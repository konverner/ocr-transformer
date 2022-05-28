import torch
import random

LR = 1e-4
BATCH_SIZE = 16
HIDDEN = 512
ENC_LAYERS = 2
DEC_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.1
LENGTH = 32

# IMAGE SIZE
WIDTH = 256
HEIGHT = 64
CHANNELS = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
