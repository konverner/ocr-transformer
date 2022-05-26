import os
import random
import string
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import editdistance
from tqdm import tqdm
from config import WIDTH, HEIGHT


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x) 

    
# convert images and labels into defined data structures
def process_data(image_dir, labels_dir, ignore=[]):
    """
    params
    ---
    image_dir : str
      path to directory with images

    labels_dir : str
      path to tsv file with labels

    returns
    ---

    img2label : dict
      keys are names of images and values are correspondent labels

    chars : list
      all unique chars used in data

    all_labels : list
    """

    chars = []
    img2label = dict()

    raw = open(labels_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        try:
            x = t.split('\t')
            flag = False
            for item in ignore:
                if item in x[1]:
                    flag = True
            if flag == False:
                img2label[image_dir + x[0]] = x[1]
                for char in x[1]:
                    if char not in chars:
                        chars.append(char)
        except:
            print('ValueError:', x)
            pass

    all_labels = sorted(list(set(list(img2label.values()))))
    chars.sort()
    chars = ['PAD', 'SOS'] + chars + ['EOS']

    return img2label, chars, all_labels


# MAKE TEXT TO BE THE SAME LENGTH
class TextCollate():
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])
        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))
            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        return x_padded, y_padded


# TRANSLATE INDICIES TO TEXT
def labels_to_text(s, idx2char):
    """
    params
    ---
    idx2char : dict
        keys : int
            indicies of characters
        values : str
            characters

    returns
    ---
    S : str
    """
    S = "".join([idx2char[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]


# COMPUTE CHARACTER ERROR RATE
def char_error_rate(p_seq1, p_seq2):
    """
    params
    ---
    p_seq1 : str
    p_seq2 : str

    returns
    ---
    cer : float
    """
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                             ''.join(c_seq2)) / max(len(c_seq1), len(c_seq2))


# RESIZE AND NORMALIZE IMAGE
def process_image(img):
    """
    params:
    ---
    img : np.array

    returns
    ---
    img : np.array
    """
    w, h, _ = img.shape
    new_w = HEIGHT
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = WIDTH
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


# GENERATE IMAGES FROM FOLDER
def generate_data(img_paths):
    """
    params
    ---
    names : list of str
        paths to images

    returns
    ---
    data_images : list of np.array
        images in np.array format
    """
    data_images = []
    for path in tqdm(img_paths):
        img = cv2.imread(path)
        try:
            img = process_image(img)
            data_images.append(img.astype('uint8'))
        except:
            print(path)
            img = process_image(img)
    return data_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, criterion, loader, logging=True):
    """
    params
    ---
    model : nn.Module
    criterion : nn.Object
    loader : torch.utils.data.DataLoader

    returns
    ---
    epoch_loss / len(loader) : float
        overall loss
    """
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (src, trg) in tqdm(loader):
            src, trg = src.cuda(), trg.cuda()
            output = model(src, trg[:-1, :])
            loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test(model, image_dir, label_dir, char2idx, idx2char, case=True, punct=False):
    """
    params
    ---
    model : pytorch model
    image_dir : str
        path to the folder with images
    label_dir : str
        path to the tsv file with labels
    char2idx : dict
    idx2char : dict
    case : bool
        if case is False then case of letter is ignored while comparing true and predicted transcript
    punct : bool
        if punct is False then punctution marks are ignored while comparing true and predicted transcript

    returns
    ---
    character_accuracy : float
    string_accuracy : float
    """
    img2label = dict()
    raw = open(label_dir, 'r', encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
        x = t.split('\t')
        img2label[image_dir + x[0]] = x[1]
    preds = prediction(model, image_dir, char2idx, idx2char)
    N = len(preds)

    wer = 0
    cer = 0

    for item in preds.items():
        print(item)
        img_name = item[0]
        true_trans = img2label[image_dir + img_name]
        predicted_trans = item[1]

        if 'ё' in true_trans:
            true_trans = true_trans.replace('ё', 'е')
        if 'ё' in predicted_trans['predicted_label']:
            predicted_trans['predicted_label'] = predicted_trans['predicted_label'].replace('ё', 'е')

        if not case:
            true_trans = true_trans.lower()
            predicted_trans['predicted_label'] = predicted_trans['predicted_label'].lower()

        if not punct:
            true_trans = true_trans.translate(str.maketrans('', '', string.punctuation))
            predicted_trans['predicted_label'] = predicted_trans['predicted_label'].translate(str.maketrans('', '', string.punctuation))

        if true_trans != predicted_trans['predicted_label']:
            print('true:', true_trans)
            print('predicted:', predicted_trans)
            print('cer:', char_error_rate(predicted_trans['predicted_label'], true_trans))
            print('---')
            wer += 1
            cer += char_error_rate(predicted_trans['predicted_label'], true_trans)

    character_accuracy = 1 - cer / N
    string_accuracy = 1 - (wer / N)
    return character_accuracy, string_accuracy


# MAKE PREDICTION
def prediction(model, test_dir, char2idx, idx2char):
    """
    params
    ---
    model : nn.Module
    test_dir : str
        path to directory with images
    char2idx : dict
        map from chars to indicies
    id2char : dict
        map from indicies to chars

    returns
    ---
    preds : dict
        key : name of image in directory
        value : dict with keys ['p_value', 'predicted_label']
    """
    preds = {}
    os.makedirs('/output', exist_ok=True)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(test_dir):
            img = cv2.imread(test_dir + filename)
            img = process_image(img).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0)
            if torch.cuda.is_available():
                src = src.cuda()

            x = model.backbone.conv1(src)
            x = model.backbone.bn1(x)
            x = model.backbone.relu(x)
            x = model.backbone.maxpool(x)

            x = model.backbone.layer1(x)
            x = model.backbone.layer2(x)
            x = model.backbone.layer3(x)
            x = model.backbone.layer4(x)

            x = model.backbone.fc(x)
            x = x.permute(0, 3, 1, 2).flatten(2).permute(1, 0, 2)
            memory = model.transformer.encoder(model.pos_encoder(x))

            p_values = 1
            out_indexes = [char2idx['SOS'], ]
            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)
                output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))

                out_token = output.argmax(2)[-1].item()
                p_values = p_values * torch.sigmoid(output[-1, 0, out_token]).item()
                out_indexes.append(out_token)
                if out_token == char2idx['EOS']:
                    break

            pred = labels_to_text(out_indexes[1:], idx2char)
            preds[filename] = {'predicted_label': pred, 'p_values': p_values}

    return preds


class ToTensor(object):
    def __init__(self, X_type=None, Y_type=None):
        self.X_type = X_type

    def __call__(self, X):
        X = X.transpose((2, 0, 1))
        X = torch.from_numpy(X)
        if self.X_type is not None:
            X = X.type(self.X_type)
        return X


# MAKE CONFUSION MATRIX ON SYMBOLS
def confused_chars(string_true, string_predict, conf_dict):
    """
  params
  ---
  string_true : str
  string_predict : str
  conf_dict : dict
    keys : str
        symbol
    values : list of pairs (str,int)
        the 1st element is symbol
        the 2nd element is how many times it was mistaken with symbol that is correspondent key
    e.g. ['O':[['0',15],['o',10]] means that symbol 'O' was mistaken with '0' 15 times and with 'o' 10 times

  returns
  ---
  conf_dict : dict

  """
    for i in range(len(string_true)):
        if string_true[i] != string_predict[i]:
            if string_true[i] not in conf_dict.keys():
                conf_dict[string_true[i]] = [[string_predict[i], 1]]
            else:
                flag = False
                for j in range(len(conf_dict[string_true[i]])):
                    if conf_dict[string_true[i]][j][0] == string_predict[i]:
                        conf_dict[string_true[i]][j][1] += 1
                        flag = True
                        break
                if flag == False:
                    conf_dict[string_true[i]].append([string_predict[i], 1])

    return conf_dict


# MAKE VISUALIZATION OF CONFUSION MATRIX
def print_confuse_dict(PATH: str):
    """
    PATH : path to pickle file with confuse matrix
    """
    PATH = PATH
    d = pickle.load(open(PATH, 'rb'))
    for d_i in d.items():
        print(d_i[0])
        xs, ys = [*zip(*d_i[1])]
        plt.bar(xs, ys, align='center')
        plt.show()
