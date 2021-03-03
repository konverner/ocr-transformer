import numpy as np
import matplotlib.pyplot as plt
import cv2, os, argparse, time, random, math
from torchvision import transforms, models
import torch
from config import *

# Перевести текст в массив индексов
def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]

# Датасет загрузки изображений и тексты
class TextLoader(torch.utils.data.Dataset):
    def __init__(self ,name_image ,label, char2idx,idx2char,eval=False):
        print('text loader:', len(name_image), len(label))
        self.name_image = name_image
        self.label = label
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((int(hp.height *1.05), int(hp.width *1.05))),
            transforms.RandomCrop((hp.height, hp.width)),
            transforms.RandomRotation(degrees=(-2, 2)),
            transforms.RandomAffine(10 ,None ,[0.6 ,1] ,3 ,fillcolor=255),
            #transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = self.name_image[index]
        if not self.eval:
            img = self.transform(img)
            img = img / img.max()
            img = img**(random.random( ) *0.7 + 0.6)
        else:
            img = np.transpose(img ,(2 ,0 ,1))
            img = img / img.max()

        label = text_to_labels(self.label[index], self.char2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        return len(self.label)
