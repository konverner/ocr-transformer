import numpy as np
import matplotlib.pyplot as plt
import cv2, os, argparse, time, random, math
from torchvision import transforms, models
from collections import Counter
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
            transforms.ColorJitter(contrast=(0.5,1),saturation=(0.5,1)),
            transforms.RandomRotation(degrees=(-2, 2)),
            #transforms.RandomAffine(10 ,None ,[0.6 ,1] ,3 ,fillcolor=255),
            #transforms.transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
            transforms.ToTensor()
        ])
    
    def random_exp(self,n=1,train=True,show=False):
        examples = []
        for k in range(n):
          i = random.randint(0,len(self.name_image))
          img = self.transform(self.name_image[i])
          img = img/img.max()
          img = img**(random.random( ) *0.7 + 0.6)
          examples.append(img)
        if show == True:
          fig=plt.figure(figsize=(8, 8))
          rows = int(n/4) + 2
          columns = int(n/8) + 2
          for j,exp in enumerate(examples):
            fig.add_subplot(rows, columns, j+1)
            plt.imshow(exp.permute(1, 2, 0))
        return examples
    
    def get_info(self):
        N = len(self.label)
        max_len = -1
        for label in self.label:
          if len(label) > max_len:
            max_len = len(label)
        counter = Counter(''.join(self.label))
        counter = dict(sorted(counter.items(), key=lambda item: item[1]))
        print('Size of dataset: {}\nMax length of expression: {}\nThe most common char: {}\n The least common char: {}'.format(\
        N,max_len,list(counter.items())[-1],list(counter.items())[0]))

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
