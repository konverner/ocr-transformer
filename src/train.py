import time
from tqdm import tqdm
import numpy as np
import torch
from config import DEVICE
from utils import labels_to_text, char_error_rate, evaluate

def train(model, optimizer, criterion, train_loader):
    """
    params
    ---
    model : nn.Module
    optimizer : nn.Object
    criterion : nn.Object
    train_loader : torch.utils.data.DataLoader

    returns
    ---
    epoch_loss / len(train_loader) : float
        overall loss
    """
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(train_loader):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


# GENERAL FUNCTION FROM TRAINING AND VALIDATION
def fit(model, optimizer, criterion, train_loader, val_loader, epoch_limit):
    loss = {'train': [], 'valid': []}
    for epoch in range(0, epoch_limit):
      print(f'Epoch: {epoch + 1:02}')
      print("-----------train------------")
      #train_loss = train(model, optimizer, criterion, train_loader)
      #print("train loss :",train_loss)
      print("\n-----------valid------------")
      valid_loss = evaluate(model, criterion, val_loader)
      print("validation loss :",valid_loss)

      #loss['train'].append(train_loss)


def validate(model, loader,confuse_dict):
    """
    params
    ---
    model : nn.Module
    loader :
    confuse_dict : dict
        to keep track of model's mistakes on symbols

    returns
    ---
    cer_overall / len(loader) * 100 : float
    wer_overall / len(loader) * 100 : float
    confuse_dict : dict
    """
    idx2char = loader.dataset.idx2char
    char2idx = loader.dataset.char2idx
    model.eval()
    show_count = 0
    wer_overall = 0
    cer_overall = 0
    with torch.no_grad():
        for (src, trg) in loader:
            img = np.moveaxis(src[0].numpy(), 0, 2)
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

            out_indexes = [char2idx['SOS'], ]

            for i in range(100):
                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(DEVICE)

                output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
                out_token = output.argmax(2)[-1].item()
                out_indexes.append(out_token)
                if out_token == char2idx['EOS']:
                    break

            out_char = labels_to_text(out_indexes[1:], idx2char)
            real_char = labels_to_text(trg[1:, 0].numpy(), idx2char)
            wer_overall += int(real_char != out_char)

            if out_char:
                cer = char_error_rate(real_char, out_char)
            else:
                cer = 1

            cer_overall += cer
    
    return cer_overall / len(loader) * 100, wer_overall / len(loader) * 100, confuse_dict
