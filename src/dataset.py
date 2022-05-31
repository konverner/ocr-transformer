import torch
from collections import Counter
import Augmentor
from torchvision import transforms
import augmentations
from utils import *
from config import LENGTH, CHANNELS

### AUGMENTATIONS ###
vignet = augmentations.Vignetting()
cutout = augmentations.Cutout(min_size_ratio=[1, 4], max_size_ratio=[2, 5])
un = augmentations.UniformNoise()
tt = ToTensor()
ld = augmentations.LensDistortion()
######################

# text to array of indicies
def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]

# store list of images' names (in directory) and does some operations with images
class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, transforms, char2idx, idx2char, eval=False):
        """
        params
        ---
        images_name : list
            list of names of images (paths to images)
        labels : list
            list of labels to correspondent images from images_name list
        char2idx : dict
        idx2char : dict
        """
        self.images_name = images_name
        self.labels = labels
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.eval = eval
        self.transform = transforms

    def _transform(self, X):
        j = np.random.randint(0, 3, 1)[0]
        if j == 0:
            return self.transform(X)
        if j == 1:
            return tt(ld(vignet(X)))
        if j == 2:
            return tt(ld(un(X)))

    # returns (and shows) random data examples
    def random_exp(self, n=1, show=False, fix=False):
        """
        params
        ---
        n : int
            number of examples to return (to show)
        show : bool
            show or not
        fix : bool
            if it is true, function will return the same images again and again but with different augmentations

        returns
        ---
        examples : list of PIL.Image objects
        """
        examples = []
        if fix == True:
            for i in range(n):
                img = self._transform(self.images_name[i])
                print(self.label[i])
                img = img / img.max()
                img = img ** (random.random() * 0.7 + 0.6)
                examples.append(img)
        else:
            for k in range(n):
                i = random.randint(0, len(self.images_name))
                img = self._transform(self.images_name[i])
                print(self.label[i])
                img = img / img.max()
                img = img ** (random.random() * 0.7 + 0.6)
                examples.append(img)
        if show == True:
            fig = plt.figure(figsize=(8, 8))
            rows = int(n / 4) + 2
            columns = int(n / 8) + 2
            for j, exp in enumerate(examples):
                fig.add_subplot(rows, columns, j + 1)
                plt.imshow(exp.permute(1, 2, 0))
        return examples

    # shows some stats about dataset
    def get_info(self):
        N = len(self.labels)
        max_len = -1
        for label in self.labels:
            if len(label) > max_len:
                max_len = len(label)
        counter = Counter(''.join(self.labels))
        counter = dict(sorted(counter.items(), key=lambda item: item[1]))
        print(
            'Size of dataset: {}\nMax length of expression: {}\nThe most common char: {}\n The least common char: {}'.format( \
                N, max_len, list(counter.items())[-1], list(counter.items())[0]))

    def __getitem__(self, index):
        img = self.images_name[index]
        if not self.eval:
            img = self.transform(img)
            img = img / img.max()
            img = img ** (random.random() * 0.7 + 0.6)
        else:
            img = np.transpose(img, (2, 0, 1))
            img = img / img.max()

        label = text_to_labels(self.labels[index], self.char2idx)
        return (torch.FloatTensor(img), torch.LongTensor(label))

    def __len__(self):
        return len(self.labels)


# MAKE TEXT TO BE THE SAME LENGTH
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        y_padded = torch.LongTensor(LENGTH, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded
