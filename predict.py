import sys
import torch
import random
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve())+'/src')

from const import DIR, PATH_TEST_DIR, PATH_TEST_LABELS, WEIGHTS_PATH, PREDICT_PATH
from config import MODEL, ALPHABET, N_HEADS, ENC_LAYERS, DEC_LAYERS, DEVICE, HIDDEN

from utils import generate_data, process_data 
from dataset import TextCollate, TextLoader
from utils import prediction

char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}

if MODEL == 'model1':
  from models import model2
  model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                          nhead=N_HEADS, dropout=0.0).to(DEVICE)
if MODEL == 'model2':
  from models import model2
  model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                          nhead=N_HEADS, dropout=0.0).to(DEVICE)

if WEIGHTS_PATH != None:
  print(f'loading weights from {WEIGHTS_PATH}')
  model.load_state_dict(torch.load(WEIGHTS_PATH))

preds = prediction(model, PREDICT_PATH, char2idx, idx2char)

f = open(DIR+'/predictions.tsv', 'w')
f.write('filename\tprediction\n')
for item in preds.items():
  f.write(item[0]+'\t'+item[1]+'\n')
f.close()
print(f'predictions are saved in {DIR}predictions.tsv')
