import math
import torch
import torch.nn as nn
from torchvision import models
from utils import PositionalEncoding, count_parameters


class TransformerModel(nn.Module):
    def __init__(self, outtoken, hidden, enc_layers=1, dec_layers=1, nhead=1, dropout=0.1, pretrained=True):
        super(TransformerModel, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Conv2d(2048, int(hidden/2), 1)

        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.decoder = nn.Embedding(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers,
                                          num_decoder_layers=dec_layers, dim_feedforward=hidden * 4, dropout=dropout,
                                          activation='relu')

        self.fc_out = nn.Linear(hidden, outtoken)
        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None
        
        print('transformer layers: {}'.format(enc_layers))
        print('transformer heads: {}'.format(nhead))
        print('backbone: resnet50')
        print('dropout: {}'.format(dropout))
        print(f'{count_parameters(self):,} trainable parameters')

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        '''
        params
        ---
        src : Tensor [64, 3, 64, 256] : [B,C,H,W]
            B - batch, C - channel, H - height, W - width
        trg : Tensor [13, 64] : [L,B]
            L - max length of label
        '''
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device) 
        x = self.backbone.conv1(src)

        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x) # [64, 2048, 2, 8] : [B,C,H,W]
            
        x = self.backbone.fc(x) # [64, 256, 2, 8] : [B,C,H,W]
        x = x.permute(0, 3, 1, 2) # [64, 8, 256, 2] : [B,W,C,H]
        x = x.flatten(2) # [64, 8, 512] : [B,W,CH]
        x = x.permute(1, 0, 2) # [8, 64, 512] : [W,B,CH]
        
        src_pad_mask = self.make_len_mask(x[:, :, 0])
        src = self.pos_encoder(x) # [8, 64, 512]
        trg_pad_mask = self.make_len_mask(trg)
        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
                                  memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
                                  memory_key_padding_mask=src_pad_mask) # [13, 64, 512] : [L,B,CH]
        output = self.fc_out(output) # [13, 64, 92] : [L,B,H]

        return output
