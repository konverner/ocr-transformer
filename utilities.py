import re, io, copy, shutil, cv2, os, editdistance
from os.path import join
import numpy as np
from collections import Counter
from tqdm import tqdm
from config import *
import torch
import matplotlib.pyplot as plt
import wandb

def process_data(image_dir, labels_dir,ignore=[]):
    '''
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
    '''

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


def train_valid_split(img2label, val_part=0.3):
    '''
    params
    ---
    img2label : dict
        keys are

    returns
    ---
    imgs_val
    labels_val

    imgs_train
    labels_train
    '''

    imgs_val, labels_val = [], []
    imgs_train, labels_train = [], []

    N = int(len(img2label)*val_part)
    count = 0
    for i, item in enumerate(img2label.items()):
        if i < N:
            imgs_val.append(item[0])
            labels_val.append(item[1])
        else:
            imgs_train.append(item[0])
            labels_train.append(item[1])
    print('valid part:{}'.format(len(imgs_val)))
    print('train part:{}'.format(len(imgs_train)))
    return imgs_val,labels_val,imgs_train,labels_train


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


# Перевести индексы в текст
def labels_to_text(s, idx2p):
    S = "".join([idx2p[i] for i in s])
    if S.find('EOS') == -1:
        return S
    else:
        return S[:S.find('EOS')]


# compute CER
def char_error_rate(p_seq1, p_seq2):
    p_vocab = set(p_seq1 + p_seq2)
    p2c = dict(zip(p_vocab, range(len(p_vocab))))
    c_seq1 = [chr(p2c[p]) for p in p_seq1]
    c_seq2 = [chr(p2c[p]) for p in p_seq2]
    return editdistance.eval(''.join(c_seq1),
                             ''.join(c_seq2)) / max(len(c_seq1),len(c_seq2))


# подгружает изображения, меняет их до необходимого размера и нормирует."""
def process_image(img):
    # img  = np.stack([img, img, img], axis=-1)
    w, h, _ = img.shape

    new_w = hp.height
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h, _ = img.shape

    img = img.astype('float32')

    new_h = hp.width
    if h < new_h:
        add_zeros = np.full((w, new_h - h, 3), 255)
        img = np.concatenate((img, add_zeros), axis=1)

    if h > new_h:
        img = cv2.resize(img, (new_h, new_w))

    return img


def generate_data(names, image_dir):
    data_images = []
    for name in tqdm(names):
        img = cv2.imread(name)
        if type(img) == type(None):
            print("ValueError:",name)
        else:
            img = process_image(img)            
            data_images.append(img.astype('uint8'))
    return data_images


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# LOAD A STATE OF MODEL FROM chk_path
# if chk_path is empty then it initilizes state to zero
def load_from_checkpoint(model,chk_path):
    valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all = [], [], [], []
    epochs = 0
    best_eval_loss_cer = float('-inf')
    if chk_path:
        ckpt = torch.load(chk_path)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        if 'epochs' in ckpt:
            epochs = int(ckpt['epoch'])
        if 'valid_loss_all' in ckpt:
            valid_loss_all = ckpt['valid_loss_all']
        if 'best_eval_loss_cer' in ckpt:
            best_eval_loss_cer = ckpt['best_eval_loss_cer']
        if 'train_loss_all' in ckpt:
            train_loss_all = ckpt['train_loss_all']
        if 'eval_accuracy_all' in ckpt:
            eval_accuracy_all = ckpt['eval_accuracy_all']
        if 'eval_loss_cer_all' in ckpt:
            eval_loss_cer_all = ckpt['eval_loss_cer_all']
        print('weights have been loaded')
    return model, epochs, best_eval_loss_cer, valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all

def evaluate(model, criterion, iterator,logging=True):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (src, trg) in tqdm(iterator):
            src, trg = src.cuda(), trg.cuda()
            output = model(src, trg[:-1, :])
            loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def test(model,image_dir,trans_dir,char2idx,idx2char,case=True):
    img2label = dict()
    raw = open(trans_dir,'r',encoding='utf-8').read()
    temp = raw.split('\n')
    for t in temp:
      x = t.split('\t')
      img2label[image_dir + x[0]] = x[1]
    preds = prediction(model,image_dir,char2idx,idx2char)
    N = len(preds)

    wer = 0
    cer = 0

    for item in preds.items():
      print(item)
      img_name = item[0]
      true_trans = img2label[image_dir+img_name]
      predicted_trans = item[1]

      if 'ё' in true_trans:
        true_trans = true_trans.replace('ё','е')
      if 'ё' in predicted_trans:
        predicted_trans = predicted_trans.replace('ё','е')

      if not case:
        true_trans=true_trans.lower()
        predicted_trans=predicted_trans.lower()

      if true_trans != predicted_trans:
        print('true:', true_trans)
        print('predicted:', predicted_trans)
        print('cer:', char_error_rate(predicted_trans,true_trans))
        print('---')
        wer += 1
        cer += char_error_rate(predicted_trans,true_trans)

    return 1 - (wer/N), 1 - cer/N


# Предсказания
def prediction(model, test_dir,char2idx,idx2char):
    preds = {}
    os.makedirs('/output', exist_ok=True)
    model.eval()

    with torch.no_grad():
        for filename in os.listdir(test_dir):
            img = cv2.imread(test_dir + filename)
            img = process_image(img).astype('uint8')
            img = img / img.max()
            img = np.transpose(img, (2, 0, 1))

            src = torch.FloatTensor(img).unsqueeze(0).cuda()

            x = model.backbone.conv1(src)
            x = model.backbone.bn1(x)
            x = model.backbone.relu(x)
            x = model.backbone.maxpool(x)

            x = model.backbone.layer1(x)
            x = model.backbone.layer2(x)
            x = model.backbone.layer3(x)
            x = model.backbone.layer4(x)
            # x = model.backbone.avgpool(x)

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
            # print('pred:',p_values,pred)
            preds[filename] = pred

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


def confused_chars(string_true,string_predict,conf_dict):
  for i in range(len(string_true)):
    if string_true[i] != string_predict[i]:
      if string_true[i] not in conf_dict.keys():
        conf_dict[string_true[i]] = [[string_predict[i],1]]
      else:
        flag = False
        for j in range(len(conf_dict[string_true[i]])):
          if conf_dict[string_true[i]][j][0] == string_predict[i]:
            conf_dict[string_true[i]][j][1] += 1
            flag=True
            break
        if flag == False:
            conf_dict[string_true[i]].append([string_predict[i],1])

  return conf_dict

def get_mixed_data(pretrain_image_dir,pretrain_labels_dir,train_image_dir,train_labels_dir,pretrain_part=0.3):
  img2label1, chars1, all_words1 = process_data(pretrain_image_dir,pretrain_labels_dir) # PRETRAIN PART
  img2label2, chars2, all_words2 = process_data(train_image_dir,train_labels_dir) # TRAIN PART
  N = int(len(img2label2)*pretrain_part)
  img2label1_list = list(img2label1.items())
  for i in range(N):
    j = np.random.randint(0,N)
    item = img2label1_list[j]
    img2label2[item[0]] = item[1]
  return img2label2
