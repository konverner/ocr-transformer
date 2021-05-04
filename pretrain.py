import os
import handwritting_generator
from torchvision import transforms
import time
from dataset import *
from utilities import *
from train import *
from config import *
from train import *
import wandb
import pickle

def prepair_validation(PATH_TO_SOURCE_VALID,PATH_TEMP_VALID):
  g = handwritting_generator.Generator()
  g.upload_source(PATH_TO_SOURCE_VALID)
  chunk = g.generate_batch(3500)
  
  labels_file = open(PATH_TEMP_VALID+'labels_valid.tsv','w',encoding='utf-8')

  i = 0
  for x,y in chunk:
    x.save(PATH_TEMP_VALID+'temp_valid'+str(i)+'.png')
    if i == 3499:
      _ = labels_file.write('temp_valid'+str(i)+'.png'+'\t'+y)
    else:   
      _ = labels_file.write('temp_valid'+str(i)+'.png'+'\t'+y+'\n')
    i += 1
  labels_file.close()
  img2label, _, all_words = process_data(PATH_TEMP_VALID, PATH_TEMP_VALID+"labels_valid.tsv",ignore=[])

  X_val, y_val, _, _ = train_valid_split(img2label,val_part=1.0)
  return X_val, y_val

def pretrain(model,chars,epochs_per_chunk,batch_size,chk=None):
  '''
  model : nn.Module
  chars : list 
    list of chars to learn
  epochs_per_chunk : int
    how many epochs will be allocated to one chunk (15k) of pretrain data
  batch_size : int
  chk : str
    path to .pt checkpoint
  '''
  # CHECK WHETHER OR NOT CHECKPOINT IS PROVIDED
  if chk != None:
    model ,epochs, best_eval_loss_cer, valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all = load_from_checkpoint(model,chk)

  # CREATE MAPS FROM CHARACTERS TO INDICIES AND VISA VERSA
  char2idx = {char: idx for idx, char in enumerate(chars)}
  idx2char = {idx: char for idx, char in enumerate(chars)}
  print('Characters:', len(chars), ':', ' '.join(chars))

  optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
  criterion = nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  # DEFINE GENERATOR AND UPLOAD TEXT SOURCE FOR IT
  g = handwritting_generator.Generator()
  g.upload_source(PATH_TO_SOURCE)
  N = int(n_epochs/5)

  X_val, y_val = prepair_validation(PATH_TO_SOURCE_VALID,PATH_TEMP_VALID)
  X_val = generate_data(X_val,PATH_TEMP_VALID)
  val_dataset = TextLoader(X_val, y_val, char2idx,idx2char, eval=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                          batch_size=1, pin_memory=False,
                                          drop_last=False, collate_fn=TextCollate())

  for j in range(3):
    # GENERATE BATCH OF SYNTHATIC DATA
    print('new data chunk have generated')
    chunk = g.generate_batch(15000)
    
    # SAVE IMAGES AND CORRISPONDENT LABELS INTO DIRECTORY 
    if os.path.isfile(PATH_TEMP+'labels.tsv'):
      os.remove(PATH_TEMP+'labels.tsv')
    labels_file = open(PATH_TEMP+'labels.tsv','w',encoding='utf-8')

    i = 0
    for x,y in chunk:
      x.save(PATH_TEMP+'temp'+str(i)+'.png')
      if i == 14999:
        _ = labels_file.write('temp'+str(i)+'.png'+'\t'+y)
      else:
        _ = labels_file.write('temp'+str(i)+'.png'+'\t'+y+'\n')
      i += 1
    labels_file.close()

    # CREATE DATA LOADER FOR CURRENT CHUNK OF DATA
    img2label, _, all_words = process_data(PATH_TEMP, PATH_TEMP+"labels.tsv",ignore=[])

    _, _, X_train, y_train = train_valid_split(img2label,val_part=0.0)

    X_train = generate_data(X_train, PATH_TEMP)
    train_dataset = TextLoader(X_train, y_train, char2idx ,idx2char, eval=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                              batch_size=hp.batch_size, pin_memory=True,
                                              drop_last=True, collate_fn=TextCollate())

    valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all = [], [], [], []
    epochs, best_eval_loss_cer = 0, float('inf')

    # CREATE A DIRECTORY WITH LOGS
    os.makedirs(path.log, exist_ok=True)

    train_all(model,optimizer,criterion,scheduler,0,\
          best_eval_loss_cer,train_loader,\
          val_loader,valid_loss_all,train_loss_all,eval_loss_cer_all,\
          eval_accuracy_all,logging=True,epoch_limit=epochs_per_chunk)
