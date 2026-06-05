import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='Informer',
                    help='model name, options: [Autoformer, Informer, Transformer]')
parser.add_argument('--use_long', type=int, default=0, help='status')

# data loader
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='/home/yl/datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='/home/yl/checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')

# MFRS
parser.add_argument('--short_periods', nargs="*", type=float, default=[], help='reference series length')
parser.add_argument('--long_periods', nargs="*", type=float, default=[], help='reference series length')
parser.add_argument('--short_type', type=str, default="sin", help='reference series type, options:[sin, swatooth, reactangle, pulse]')
parser.add_argument('--long_type', type=str, default="pulse", help='reference series type, options:[sin, swatooth, reactangle, pulse]')
parser.add_argument('--short_len', type=int, default=96, help="harmonic base pattern number")
parser.add_argument('--long_len', type=int, default=256, help="long-term pattern number")
parser.add_argument('--alpha', type=float, default=0.1, help='Alignment parameter')
parser.add_argument('--use_norm', type=int, default=1, help='1: use revin or 0: no revin')
parser.add_argument('--use_pos', type=int, default=0, help='Whether to use the alignment module, 0: False, 1: True')
parser.add_argument('--N', type=int, default=8, help='Alignment parameter')
# parser.add_argument('--lamda', type=float, default=1., help='Alignment parameter')

# CycleNet.
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')
parser.add_argument('--seg_len', type=int, default=48, help='segment length')
parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period_len')

# FilterNet
parser.add_argument('--embed_size', default=128, type=int)
parser.add_argument('--hidden_size', default=256, type=int)

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=True)
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
    
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

args.short_len = int(max(args.short_periods) / 2) if len(args.short_periods) > 0 else 0
args.long_len = args.long_len if len(args.long_periods) > 0 else 0

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):

        if args.model == 'CycleNet':
            setting = '{}_{}_{}_lr={}_d={}_sl={}_pl={}'.format(            
                args.model,
                args.model_id,
                args.loss,
                args.learning_rate,
                args.d_model,
                args.seq_len,                      
                args.pred_len,      
            )
        elif args.model == 'iTransformer':
            setting = '{}_{}_{}_pos={}_lr={}_d={}_f={}_e={}_sl={}_pl={}_periods={}_long={}'.format(            
                args.model,
                args.model_id,
                args.loss,
                args.use_pos,
                args.learning_rate,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.seq_len,                       
                args.pred_len,
                args.long_periods,
                args.use_long        
            )
        elif args.model == 'FilterNet':
            setting = '{}_{}_{}_pos={}_lr={}_embed={}_hidden={}_sl={}_pl={}_periods={}_long={}'.format(            
                args.model,
                args.model_id,
                args.loss,
                args.use_pos,
                args.learning_rate,
                args.embed_size,
                args.hidden_size,
                args.seq_len,                       
                args.pred_len,
                args.long_periods,
                args.use_long        
            )                                     
        elif any(substr in args.model for substr in {'PeriRef'}):
            setting = '{}_{}_{}_pos={}_norm={}_lr={}_lradj={}_d={}_f={}_e={}_sl={}_pl={}_N={}_sp={}_lp={}_short={}_long={}_type={}_seed={}'.format(            
                args.model,
                args.model_id,
                args.loss,
                args.use_pos,
                args.use_norm,
                args.learning_rate,
                args.lradj,
                args.d_model,
                args.d_ff,
                args.e_layers,
                args.seq_len,                      
                args.pred_len,
                args.N,
                args.short_periods,
                args.long_periods,
                args.short_len,
                args.long_len,
                args.long_type,
                args.random_seed
            )


        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_pos={}_norm={}_lr={}_lradj={}_d={}_f={}_e={}_sl={}_pl={}_N={}_sp={}_lp={}_short={}_long={}_type={}_seed={}'.format(            
        args.model,
        args.model_id,
        args.loss,
        args.use_pos,
        args.use_norm,
        args.learning_rate,
        args.lradj,
        args.d_model,
        args.d_ff,
        args.e_layers,
        args.seq_len,                      
        args.pred_len,
        args.N,
        args.short_periods,
        args.long_periods,
        args.short_len,
        args.long_len,
        args.long_type,
        args.random_seed
    )

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
