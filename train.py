import time
from dataset import *
from utilities import *
from train import *
from config import *
from model import *

def train(model, optimizer, criterion, iterator):
    model.train()
    epoch_loss = 0
    counter = 0
    for src, trg in iterator:
        counter += 1
        if counter % 500 == 0:
            print('[', counter, '/', len(iterator), ']')
        #src, trg = src.cuda(), trg.cuda()

        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Общая функция обучения и валидации
def train_all(model,optimizer,criterion,scheduler,epochs,best_eval_loss_cer, train_loader, val_loader,valid_loss_all,train_loss_all,eval_loss_cer_all,eval_accuracy_all,logging=True):
    train_loss = 0
    count_bad = 0
    for epoch in range(epochs, 1000):
        print(f'Epoch: {epoch + 1:02}')
        start_time = time.time()
        print("-----------train------------")
        train_loss = train(model, optimizer, criterion, train_loader,logging=logging)
        print("\n-----------valid------------")
        valid_loss = evaluate(model, criterion, val_loader,logging=logging)
        print("-----------eval------------")
        eval_loss_cer, eval_accuracy = validate(model, val_loader, show=20,logging=logging)
        scheduler.step(eval_loss_cer)
        valid_loss_all.append(valid_loss)
        train_loss_all.append(train_loss)
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
            }, '/content/gdrive/MyDrive/log/resnet50_trans_%.3f.pt' % (best_eval_loss_cer))
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
            }, '/content/gdrive/MyDrive/log/resnet50_trans_last.pt')
            print('Save model')

        if logging:
            wandb.log({'train_loss_wer': train_loss, "valid_loss_wer": valid_loss, 'eval_accuracy_wer': 100 - eval_accuracy,
                   'eval_loss_cer': eval_loss_cer})

        print(f'Time: {time.time() - start_time}s')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val   Loss: {valid_loss:.4f}')
        print(f'Eval  CER: {eval_loss_cer:.4f}')
        print(f'Eval accuracy: {eval_accuracy:.4f}')
        plt.clf()
        plt.plot(valid_loss_all[-20:])
        plt.plot(train_loss_all[-20:])
        plt.savefig('/content/gdrive/MyDrive/log/all_loss.png')
        plt.clf()
        plt.plot(eval_loss_cer_all[-20:])
        plt.savefig('/content/gdrive/MyDrive/log/loss_cer.png')
        plt.clf()
        plt.plot(eval_accuracy_all[-20:])
        plt.savefig('/content/gdrive/MyDrive/log/eval_accuracy.png')
        if count_bad > 19:
            break



def validate(model, dataloader,show=70,logging=True):
    idx2char = dataloader.dataset.idx2char
    char2idx = dataloader.dataset.char2idx
    model.eval()
    show_count = 0
    error_w = 0
    error_p = 0
    with torch.no_grad():
        for (src, trg) in dataloader:
            img = np.moveaxis(src[0].numpy(), 0, 2)
            #src = src.cuda()
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
                # plt.imshow(img)
                # plt.show()
                if logging:
                    wandb.log({"Validation Examples": wandb.Image(img, caption="Pred: {} Truth: {}".format(out_char, real_char))})
                show_count += 1
                print('Real:', real_char)
                print('Pred:', out_char)
                print(cer)

    return error_p / len(dataloader) * 100, error_w / len(dataloader) * 100
