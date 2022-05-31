import sys
import torch
import random
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve())+'/src')

from const import PATH_TEST_DIR, PATH_TEST_LABELS,\
                  PATH_TRAIN_DIR, PATH_TRAIN_LABELS, CHECKPOINT_PATH
from config import MODEL, BATCH_SIZE, N_HEADS, \
                    ENC_LAYERS, DEC_LAYERS, LR, \
                    DEVICE, RANDOM_SEED, HIDDEN, \
                    DROPOUT, CHECKPOINT_FREQ, N_EPOCHS, \
                    ALPHABET, TRAIN_TRANSFORMS, TEST_TRANSFORMS
from utils import generate_data, process_data 
from dataset import TextCollate, TextLoader
from fit import fit

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
idx2char = {idx: char for idx, char in enumerate(ALPHABET)}

print(f"loading dataset {PATH_TRAIN_DIR} ...")
img2label, _, all_words = process_data(PATH_TRAIN_DIR, PATH_TRAIN_LABELS) 
img_names, labels = list(img2label.keys()), list(img2label.values())
X_train = generate_data(img_names)
y_train = labels

train_dataset = TextLoader(X_train, y_train, TRAIN_TRANSFORMS, char2idx, idx2char)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                           batch_size=BATCH_SIZE, pin_memory=True,
                                           drop_last=True, collate_fn=TextCollate())

print(f"loading dataset {PATH_TEST_DIR} ...")
img2label, _, all_words = process_data(PATH_TEST_DIR, PATH_TEST_LABELS) 
img_names, labels = list(img2label.keys()), list(img2label.values())
X_test = generate_data(img_names)
y_test = labels

test_dataset = TextLoader(X_test, y_test, TEST_TRANSFORMS, char2idx ,idx2char)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                           batch_size=BATCH_SIZE, pin_memory=True,
                                           drop_last=True, collate_fn=TextCollate())

if MODEL == 'model1':
  from models import model1
  model = model1.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                          nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)
if MODEL == 'model2':
  from models import model2
  model = model2.TransformerModel(len(ALPHABET), hidden=HIDDEN, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,   
                          nhead=N_HEADS, dropout=DROPOUT).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
print(f'checkpoints are saved in {CHECKPOINT_PATH} every {CHECKPOINT_FREQ} epochs')
for epoch in range(1, N_EPOCHS, CHECKPOINT_FREQ):
  fit(model, optimizer, criterion, test_loader, test_loader, epoch, epoch+CHECKPOINT_FREQ)
  torch.save(model.state_dict(), CHECKPOINT_PATH+'checkpoint_{}.pt'.format(epoch+CHECKPOINT_FREQ))
