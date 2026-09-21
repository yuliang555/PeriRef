import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import InstanceNorm
        

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.use_norm = configs.use_norm
        self.norm = InstanceNorm(1, configs.use_norm)

        self.N_short = len(configs.short_periods)
        self.N_long = len(configs.long_periods)        
        assert not (self.N_short == 0 and self.N_long == 0), "short_periods and long_periods cannot both be []"

        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        self.x_embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.short_embedding = nn.Linear(configs.ref_len, configs.d_model)
        self.long_embedding = nn.Linear(configs.ref_len, configs.d_model)

        self.value = nn.Linear(configs.d_model, configs.d_model) 

        self.weight = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, 2 * (self.N_short + self.N_long))
        )  

        self.ffn = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model)
        )
        
        self.projector = nn.Linear(configs.d_model, configs.pred_len)        

    def forward(self, x, short_ref, long_ref, mask=None):

        if self.use_norm:
            x = self.norm(x, 'norm') 

        x_enc = self.x_embedding(x.permute(0, 2, 1))
        short_enc = self.short_embedding(short_ref)
        long_enc = self.long_embedding(long_ref)
        basis = self.value(torch.cat([short_enc, long_enc], dim=1))

        weight = torch.softmax(self.weight(x_enc), dim=-1)        
        new_x = torch.einsum('bcn,bnd->bcd', weight, basis)

        x_enc = x_enc + self.dropout(new_x)
        x_enc = self.norm1(x_enc)
        enc_out = self.ffn(x_enc)
        enc_out = x_enc + self.dropout(enc_out)
        enc_out = self.norm2(enc_out)

        dec_out = self.projector(enc_out).permute(0, 2, 1)
            
        if self.use_norm:
            dec_out = self.norm(dec_out, 'denorm')
                              
        return dec_out, None
