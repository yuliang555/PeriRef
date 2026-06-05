from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, iTransformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, \
    LDLinear, SparseTSF, RLinear, RMLP, CycleiTransformer, FilterNet, PeriRef, PeriRef_Concat
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.draw import Attention

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class FALoss(nn.Module):
    def __init__(self, alpha):
        super(FALoss, self).__init__()
        self.alpha = alpha
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()    

    def forward(self, pred, true, pred_1, true_1):
        loss_value = self.mae(pred, true)

        pred_fft = torch.fft.rfft(pred_1, dim=1)[:, :48, :]
        true_fft = torch.fft.rfft(true_1, dim=1)[:, :48, :]
        loss_freq = self.mae(torch.abs(pred_fft - true_fft), torch.zeros_like(pred_fft))            
        
        loss = loss_value + loss_freq * self.alpha 
        
        return loss
    

class NewLoss(nn.Module):
    def __init__(self, alpha):
        super(NewLoss, self).__init__()
        self.alpha = alpha
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()    

    def forward(self, pred, true):
        loss_mse = self.mse(pred, true)
        loss_mae = self.mae(pred, true)            
        
        loss = loss_mse * self.alpha + loss_mae * (1 - self.alpha)
        
        return loss


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'iTransformer': iTransformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SegRNN': SegRNN,
            'CycleNet': CycleNet,
            'LDLinear': LDLinear,
            'SparseTSF': SparseTSF,
            'RLinear': RLinear,
            'RMLP': RMLP,
            'CycleiTransformer': CycleiTransformer,
            'PeriRef': PeriRef,
            'PeriRef_Concat': PeriRef_Concat,
            'FilterNet': FilterNet
        }                
        model = model_dict[self.args.model].Model(self.args).float()
        self._genarete_mask()
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _genarete_mask(self):
        mask_one = np.ones(self.args.N)
        mask_zero = np.zeros(self.args.seq_len // 2 + 1 - self.args.N)
        mask = np.concatenate([mask_one, mask_zero])

        self.MASK = torch.tensor(mask).float().to(self.device)
      
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':            
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss == 'new':
            criterion = NewLoss(self.args.alpha)
        elif self.args.loss == 'fa':
            criterion = FALoss(self.args.alpha)

        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_short_ref, batch_long_ref) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_short_ref = batch_short_ref.float().to(self.device)
                batch_long_ref = batch_long_ref.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PeriRef'}):
                            outputs, attn = self.model(batch_x, batch_short_ref, batch_long_ref, self.MASK)
                        elif any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle, batch_long_ref)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                else:
                    if any(substr in self.args.model for substr in {'PeriRef'}):
                        outputs, attn = self.model(batch_x, batch_short_ref, batch_long_ref, self.MASK)                    
                    elif any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle, batch_long_ref)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                                
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        gpu_memory = 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_short_ref, batch_long_ref) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_short_ref = batch_short_ref.float().to(self.device)
                batch_long_ref = batch_long_ref.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PeriRef'}):
                            outputs, attn = self.model(batch_x, batch_short_ref, batch_long_ref, self.MASK)                        
                        elif any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle, batch_long_ref)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'PeriRef'}):
                        outputs, attn = self.model(batch_x, batch_short_ref, batch_long_ref, self.MASK)                    
                    elif any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle, batch_long_ref)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # pred_1 = torch.concat([sx, sy], dim=2)
                    # true_1 = torch.concat([batch_x, batch_y], dim=1).permute(0, 2, 1)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            current_gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            gpu_memory = max(gpu_memory, current_gpu_memory)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, nn.MSELoss())

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print(f"gpu: {gpu_memory}")
        
        # if any(substr in self.args.model for substr in {'PeriRef'}):
        #     draw = Attention('viridis')

        #     attn = torch.mean(attn, dim=1)
        #     attn = attn.detach().cpu().numpy()[0, :16, :]            
        #     draw.save_heatmap(f'{self.args.model_id}_train.pdf', attn)

        #     # attention_path = f'{self.args.model_id}_train.csv'
        #     # if not os.path.exists(attention_path):
        #     #     os.makedirs(attention_path) 

        #     df = pandas.DataFrame(attn)
        #     df.to_csv(f'{self.args.model_id}_train.csv') 
       
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # inputx = []
        result_path = f'./Figure/result/{self.args.model_id}/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.model.eval()
        test_time = time.time()
        cpu_memory = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_short_ref, batch_long_ref) in enumerate(test_loader):
                batch_y = batch_y.float().to(self.device)

                batch_short_ref = batch_short_ref.float().to(self.device)
                batch_long_ref = batch_long_ref.float().to(self.device)
                batch_x = batch_x.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'PeriRef'}):
                            outputs, attn = self.model(batch_x, batch_short_ref, batch_long_ref, self.MASK)                        
                        elif any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle, batch_long_ref)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                else:
                    if any(substr in self.args.model for substr in {'PeriRef'}):
                        outputs, attn = self.model(batch_x, batch_short_ref, batch_long_ref, self.MASK)                    
                    elif any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle, batch_long_ref)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_long_ref)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 40 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(result_path, f'{i}_{self.args.pred_len}_{self.args.model}.pdf'))
                    # np.savetxt(os.path.join(result_path, str(i) + '.txt'), pd)
                    # np.savetxt(os.path.join(result_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)        


        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        metric_path = f"results/{self.args.model_id}_{self.args.gpu}.csv"
        if not os.path.exists(metric_path):
            open(metric_path, "w").close()                      
        df = pandas.DataFrame({
            "settings": [f"{setting}"],
            "MSE": [mse],
            "MAE": [mae],
        })
        if os.path.getsize(metric_path) == 0:
            df.to_csv(metric_path, index=None)
        else:
            df_old = pandas.read_csv(metric_path)
            df_new = pandas.concat([df_old, df], ignore_index=True)
            df_new.to_csv(metric_path, index=None) 
                    
        print('mse:{}, mae:{}'.format(mse, mae))
        # f = open(f"result_{self.args.pred_len}.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Cycle'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Cycle'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
