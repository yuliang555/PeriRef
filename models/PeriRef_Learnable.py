import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import Embedding_Both, Embedding_Long, Embedding_Short
from layers.Transformer_EncDec import CrossEncoder, CrossEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.RevIN import InstanceNorm
import math
        

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.ref_len = configs.ref_len
        self.norm = InstanceNorm(1, configs.use_norm)

        self.N_short = len(configs.short_periods)
        self.N_long = len(configs.long_periods)        
        assert not (self.N_short == 0 and self.N_long == 0), "short_periods and long_periods cannot both be []"

        # self.offset = torch.arange(configs.ref_len, dtype=torch.float32).unsqueeze(0)
        short_periods = torch.log(0.3 * torch.tensor(configs.short_periods, dtype=torch.float32))
        long_periods = torch.log(0.3 * torch.tensor(configs.long_periods, dtype=torch.float32))
        self.short_periods = nn.Parameter(short_periods, requires_grad=True)
        self.long_periods = nn.Parameter(long_periods, requires_grad=True)
        
        if self.N_short > 0 and self.N_long > 0:
            Embedding = Embedding_Both
        elif self.N_short > 0:
            Embedding = Embedding_Short
        elif self.N_long > 0:
            Embedding = Embedding_Long

        self.embedding = Embedding(configs.use_pos, configs.enc_in, configs.seq_len, configs.ref_len, configs.d_model)
        # self.embedding = Embedding(configs.use_pos, configs.enc_in, configs.seq_len, 2, configs.d_model)

        self.encoder = CrossEncoder(
            [
                CrossEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        ) 

        self.projector = nn.Linear(configs.d_model, configs.pred_len)        

    def forward(self, x, time_idx, mask=None):

        if self.use_norm:
            x = self.norm(x, 'norm') 

        x = x.permute(0, 2, 1)

        offset = torch.arange(self.ref_len, dtype=torch.float32).unsqueeze(0).to(x.device)
        t = (time_idx.unsqueeze(-1) + offset).unsqueeze(1)                      # B, 1, L

        short_periods = torch.exp(self.short_periods).view(1, self.N_short, 1)
        long_periods = torch.exp(self.long_periods).view(1, self.N_long, 1)
        short_phase = 2 * math.pi * t / short_periods                           # B, N, L
        long_phase = 2 * math.pi * t / long_periods

        short_ref = torch.cat([torch.sin(short_phase), torch.cos(short_phase)], dim=1)
        long_ref = torch.cat([torch.sin(long_phase), torch.cos(long_phase)], dim=1)

        x_enc, cross = self.embedding(x, short_ref, long_ref)
        enc_out, attns = self.encoder(x_enc, cross, attn_mask=mask)
        dec_out = self.projector(enc_out).permute(0, 2, 1)
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')
                              
        return dec_out, attns[-1]
