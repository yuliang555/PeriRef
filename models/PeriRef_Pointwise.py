import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import Embedding_Fourier
from layers.Transformer_EncDec import CrossEncoder, CrossEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.RevIN import InstanceNorm


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.norm = InstanceNorm(1, configs.use_norm)

        N_short = len(configs.short_periods)
        N_long = len(configs.long_periods)        
        assert not (N_short == 0 and N_long == 0), "short_periods and long_periods cannot both be []"

        self.embedding = Embedding_Fourier(configs.use_pos, configs.enc_in, configs.seq_len, 2*(N_short+N_long), configs.d_model)

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

    def forward(self, x, short_ref, long_ref, mask=None):

        if self.use_norm:
            x = self.norm(x, 'norm') 

        x = x.permute(0, 2, 1)
        x_enc, cross = self.embedding(x, short_ref, long_ref)
        enc_out, attns = self.encoder(x_enc, cross, attn_mask=mask)
        dec_out = self.projector(enc_out).permute(0, 2, 1)
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')
                              
        return dec_out, attns[-1]