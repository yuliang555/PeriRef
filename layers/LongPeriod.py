import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import CrossEncoder, CrossEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class LongPeriod(nn.Module):

    def __init__(self, configs):
        super(LongPeriod, self).__init__()
        self.use_pos = configs.use_pos
        self.long_len = configs.long_len

        if configs.use_pos:
            self.pos = nn.Parameter(torch.randn(configs.seq_len, configs.enc_in))

        self.x_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.long_embedding = nn.Linear(configs.long_len, configs.d_model)

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

        self.x_projector = nn.Linear(configs.d_model, configs.seq_len)
        self.y_projector = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x, long_ref, mask=None):
        # x: B, L, C;  long_ref: B, N, L
        # sx: B, L, C;  sy: B, T, C

        if self.use_pos:
            x = x + self.pos

        x_enc = self.x_embedding(x.permute(0, 2, 1))
        cross = self.long_embedding(long_ref)

        enc_out, attns = self.encoder(x_enc, cross, attn_mask=mask)

        sx = self.x_projector(enc_out).permute(0, 2, 1)
        sy = self.y_projector(enc_out).permute(0, 2, 1)

        return sx, sy

