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

def pretrain(model,chars,n_epochs,batch_size,PATH_TO_SOURCE,best_eval_loss_cer=0):
  char2idx = {char: idx for idx, char in enumerate(chars)}
  idx2char = {idx: char for idx, char in enumerate(chars)}

  g = handwritting_generator.Generator()
  g.upload_source(PATH_TO_SOURCE)
  trans = transforms.ToTensor()
  count_bad = 0

  valid_loss_all = []
  train_loss_all = []


  optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
  criterion = nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  for epoch in range(n_epochs):
    print('----EPOCH {}----'.format(epoch))
    start_time = time.time()
    epoch_loss = 0
    batch = g.generate_batch(batch_size=batch_size)
    for x,y in batch:
      y = [char2idx[i] for i in y]
      x = trans(x)
      print(y)
      y = torch.Tensor(y)
      x, y = x.cuda(), y.cuda()
      optimizer.zero_grad()
      output = model(x, y[:-1, :])

      loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(y[1:, :], (-1,)))
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()

      train_loss_all = epoch_loss / len(iterator)

    print("train loss: {}".format(train_loss_all))

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x,y in batch:
          x, y = trans(x), trans(y)
          x, y = x.cuda(), u.cuda()
          output = model(x, y[:-1, :])
          loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(y[1:, :], (-1,)))
          epoch_loss += loss.item()

    valid_loss = epoch_loss / len(iterator)
    print("----EPOCH {}---\nvalid loss: {}".format(valid_loss))
    
    eval_loss_cer_all = []
    eval_accuracy_all = []

    model.eval()
    show_count = 20
    error_w = 0
    error_p = 0
    with torch.no_grad():
      for (x, y) in batch:
          img = np.moveaxis(x[0].numpy(), 0, 2)
          x = x.cuda()
          x = model.backbone.conv1(x)
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
              trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)

              output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))
              out_token = output.argmax(2)[-1].item()
              out_indexes.append(out_token)
              if out_token == char2idx['EOS']:
                  break

          out_char = labels_to_text(out_indexes[1:], idx2char)
          real_char = labels_to_text(trg[1:, 0].numpy(), idx2char)
          error_w += int(real_char != out_char)
          if out_char:
              cer = char_error_rate(real_char, out_char)
          else:
              cer = 1

          error_p += cer
          if show > show_count:
              if logging:
                  if logging:
                    wandb.log({'Validation Character Accuracy': (1-cer)*100})
                    wandb.log({"Validation Examples": wandb.Image(img, caption="Pred: {} Truth: {}".format(out_char, real_char))})
              show_count += 1
              print('Real:', real_char)
              print('Pred:', out_char)
              print(cer)
  
      
    eval_loss_cer, eval_accuracy = error_p / len(dataloader) * 100, error_w / len(dataloader) * 100
    eval_loss_cer_all.append(eval_loss_cer)
    eval_accuracy_all.append(eval_accuracy)

    if eval_loss_cer < best_eval_loss_cer:
      count_bad = 0
      best_eval_loss_cer = eval_loss_cer
      torch.save({
          'model': model.state_dict(),
          'epoch': epoch,
          'best_eval_loss_cer': best_eval_loss_cer,
          'valid_loss_all': valid_loss_all,
          'train_loss_all': train_loss_all,
          'eval_loss_cer_all': eval_loss_cer_all,
          'eval_accuracy_all': eval_accuracy_all,
      }, path.log+'resnet50_trans_%.3f.pt' % (best_eval_loss_cer))
      print('Save best model')
    else:
      count_bad += 1
      torch.save({
          'model': model.state_dict(),
          'epoch': epoch,
          'best_eval_loss_cer': best_eval_loss_cer,
          'valid_loss_all': valid_loss_all,
          'train_loss_all': train_loss_all,
          'eval_loss_cer_all': eval_loss_cer_all,
          'eval_accuracy_all': eval_accuracy_all,
      }, path.log+'resnet50_trans_last.pt')
      print('Save model')

    if logging:
        wandb.log({'Train loss WER': train_loss, "Validation loss WER": valid_loss, 'Validation Word Accuracy': 100 - eval_accuracy,
                'Validation loss CER': eval_loss_cer})

    print(f'Time: {time.time() - start_time}s')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val   Loss: {valid_loss:.4f}')
    print(f'Eval loss CER: {eval_loss_cer:.4f}')
    print(f'Eval accuracy: {100 - eval_accuracy:.4f}')
    if count_bad > 19:
        break

