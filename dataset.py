from collections import Counter
import Augmentor
from torchvision import transforms
import augmentations
from utilities import *

### AUGMENTATIONS ###
vignet = augmentations.Vignetting()
cutout = augmentations.Cutout(min_size_ratio=[1, 4], max_size_ratio=[2, 5])
un = augmentations.UniformNoise()
tt = ToTensor()
p = Augmentor.Pipeline()
ld = augmentations.LensDistortion()
p.shear(max_shear_left=2, max_shear_right=2, probability=0.7)
p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=11)
######################

# text to array of indicies
def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]

# store list of images' names (in directory) and does some operations with images
class TextLoader(torch.utils.data.Dataset):
    def __init__(self, images_name, labels, char2idx, idx2char, eval=False):
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
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            p.torch_transform(),  # random distortion and shear
            # transforms.Resize((int(hp.height *1.05), int(hp.width *1.05))),
            # transforms.RandomCrop((hp.height, hp.width)),
            # transforms.ColorJitter(contrast=(0.5,1),saturation=(0.5,1)),
            transforms.RandomRotation(degrees=(-9, 9), fill=255),
            # transforms.RandomAffine(10 ,None ,[0.6 ,1] ,3 ,fillcolor=255),
            transforms.transforms.GaussianBlur(3, sigma=(0.1, 1.9)),
            transforms.ToTensor()
        ])

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
        N = len(self.label)
        max_len = -1
        for label in self.label:
            if len(label) > max_len:
                max_len = len(label)
        counter = Counter(''.join(self.label))
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
