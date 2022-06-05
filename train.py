import sys
import torch
import random
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve())+'/src')

from const import PATH_TEST_DIR, PATH_TEST_LABELS, FROM_CHECKPOINT_PATH, \
                  PATH_TRAIN_DIR, PATH_TRAIN_LABELS, CHECKPOINTS_PATH
from config import MODEL, BATCH_SIZE, N_HEADS, \
                    ENC_LAYERS, DEC_LAYERS, LR, \
                    DEVICE, RANDOM_SEED, HIDDEN, \
                    DROPOUT, CHECKPOINT_FREQ, N_EPOCHS, \
                    ALPHABET, TRAIN_TRANSFORMS, TEST_TRANSFORMS, \
                    OPTIMIZER_NAME, SCHUDULER_ON, PATIENCE
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

if FROM_CHECKPOINT_PATH != None:
  model.load_state_dict(torch.load(FROM_CHECKPOINT_PATH))
  print(f'loading from checkpoint {FROM_CHECKPOINT_PATH}')

criterion = torch.nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
optimizer = torch.optim.__getattribute__(OPTIMIZER_NAME)(model.parameters(), lr=LR)

if SCHUDULER_ON:
  scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE)
else:
  scheduler = None

print(f'checkpoints are saved in {CHECKPOINTS_PATH} every {CHECKPOINT_FREQ} epochs')
for epoch in range(1, N_EPOCHS, CHECKPOINT_FREQ):
  fit(model, optimizer, scheduler, criterion, train_loader, test_loader, epoch, epoch+CHECKPOINT_FREQ)
  torch.save(model.state_dict(), CHECKPOINTS_PATH+'checkpoint_{}.pt'.format(epoch+CHECKPOINT_FREQ))
