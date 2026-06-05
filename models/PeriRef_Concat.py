import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import Embedding_Both, Embedding_Long, Embedding_Short
from layers.Transformer_EncDec import CrossEncoder, CrossEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.RevIN import InstanceNorm


class Cyclical(nn.Module):

    def __init__(self, configs):
        super(Cyclical, self).__init__()
        # ensure at least one of short_len or long_len is positive
        assert not (getattr(configs, 'short_len', 0) == 0 and getattr(configs, 'long_len', 0) == 0), \
            "short_len and long_len cannot both be 0"

        self.embedding = nn.Linear(configs.seq_len + configs.short_len * len(configs.short_periods) + \
                                   configs.long_len * len(configs.long_periods), configs.d_model)

        self.encoder_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(configs.d_model, configs.d_ff),
                    nn.ReLU(),
                    nn.Linear(configs.d_ff, configs.d_model)
                ) for l in range(configs.e_layers)
            ] 
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.norm_layer = nn.LayerNorm(configs.d_model) 

        self.x_projector = nn.Linear(configs.d_model, configs.seq_len)
        self.y_projector = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x, short_ref, long_ref, mask=None):
        B, C, L = x.shape
        short_ref = short_ref.reshape(B, -1).unsqueeze(1).repeat(1, C, 1)
        long_ref = long_ref.reshape(B, -1).unsqueeze(1).repeat(1, C, 1)
        x_enc = self.embedding(torch.concat([x, short_ref, long_ref], dim=-1))
        # x_enc, cross = self.embedding(x, short_ref, long_ref)
        # x_enc = x_enc + cross.sum(dim=1).unsqueeze(1)
        
        for encode_layer in self.encoder_layers:
            x_enc = encode_layer(x_enc)
            x_enc = x_enc + self.dropout(x_enc)
            x_enc = self.norm_layer(x_enc)

        enc_out = x_enc

        sx = self.x_projector(enc_out)
        sy = self.y_projector(enc_out)

        return sx, sy


class Residual(nn.Module):

    def __init__(self, seq_len, pred_len, d_model, N):
        super(Residual, self).__init__()
        self.seq_len = seq_len

        if N > 4:  
            self.model = nn.Sequential(
                nn.Linear(seq_len, d_model),
                nn.ReLU(),
                nn.Linear(d_model, pred_len)
            )
        else:
            self.model = nn.Linear(seq_len, pred_len)

    def forward(self, tx, mask):
        tx_fft = torch.fft.rfft(tx)
        tx_fft = tx_fft * mask
        tx = torch.fft.irfft(tx_fft, n=self.seq_len)

        ty = self.model(tx)

        return ty
        

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm

        self.norm = InstanceNorm(1, configs.use_norm)        
        self.cyclical = Cyclical(configs)
        self.residual = Residual(configs.seq_len, configs.pred_len, configs.d_model, configs.N)        

    def forward(self, x, short_ref, long_ref, mask):

        if self.use_norm:
            x = self.norm(x, 'norm') 

        x = x.permute(0, 2, 1)
        sx, sy = self.cyclical(x, short_ref, long_ref)
        ty = self.residual(x - sx, mask)
        dec_out = (sy + ty).permute(0, 2, 1)
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')
                              
        return dec_out, None
