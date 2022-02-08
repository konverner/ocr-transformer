from dataset import *
from utilities import *
from train import *
from config import *
from model import *
from torch import optim


# CREATE A DIRECTORY WITH LOGS
os.makedirs(path.log, exist_ok=True)

# CREATE A DATASET
img2label, chars, all_words = process_data(path.train_image_dir, path.train_labels_dir)

# If chars are already given
if hp.chars:
  chars = hp.chars
print('Characters:', len(chars), ':', ' '.join(chars))

char2idx = {char: idx for idx, char in enumerate(chars)}
idx2char = {idx: char for idx, char in enumerate(chars)}

X_val, y_val, X_train, y_train = train_valid_split(img2label)

X_train = generate_data(X_train, path.train_image_dir)
X_val = generate_data(X_val, path.train_image_dir)

train_dataset = TextLoader(X_train, y_train, char2idx ,idx2char, eval=False)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                           batch_size=hp.batch_size, pin_memory=True,
                                           drop_last=True, collate_fn=TextCollate())

val_dataset = TextLoader(X_val, y_val, char2idx,idx2char, eval=True)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                         batch_size=1, pin_memory=False,
                                         drop_last=False, collate_fn=TextCollate())

valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all = [], [], [], []
epochs, best_eval_loss_cer = 0, float('inf')


# CREATE A MODEL
model = TransformerModel('resnet50', len(chars), hidden=hp.hidden, enc_layers=hp.enc_layers, dec_layers=hp.dec_layers,   
                         nhead=hp.nhead, dropout=hp.dropout, pretrained=True).to(device)


# INITILIZE A STATE
model ,epochs, best_eval_loss_cer, valid_loss_all, train_loss_all, eval_accuracy_all, eval_loss_cer_all = load_from_checkpoint(model,path.chk)


# TRAIN
optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

print(f'The model has {count_parameters(model):,} trainable parameters')

train_all(model,optimizer,criterion,scheduler,epochs,best_eval_loss_cer,train_loader,val_loader,valid_loss_all,train_loss_all,eval_loss_cer_all,eval_accuracy_all)
