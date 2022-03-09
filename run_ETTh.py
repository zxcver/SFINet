import argparse
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from experiments.exp_ETTh import Exp_ETTh

parser = argparse.ArgumentParser(description='SFINet on ETT dataset')


parser.add_argument('--model', type=str, required=False, default='SFINet', help='model of the experiment')
### -------  dataset settings --------------
parser.add_argument('--data', type=str, required=False, default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1','wph1','wph2','wpm1','wpm2'], help='name of dataset')
parser.add_argument('--root_path', type=str, default='datasets/ETT-data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M', 'MS'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints_root', type=str, default='checkpoints', help='location of model checkpoints')
parser.add_argument('--events_root',type=str, default='events', help='location of model checkpoints')
parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')


### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multile gpus')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length of SFINet encoder, look back window')
parser.add_argument('--label_len', type=int, default=24, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length, horizon')
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0)

### -------  training settings --------------  
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae',help='loss function')
parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default =False, help='save the output results')
parser.add_argument('--model_name', type=str, default='SFINet')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', action='store_true', default=False)

### -------  model settings --------------  
parser.add_argument('--hidden_size', default=1.0, type=float, help='hidden channel of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--window_size', default=12, type=int, help='input size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--positionalEcoding', type=bool, default=False)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=1)


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'wph1': {'data': 'wph1.csv', 'T': 'A1grGenPowerForProcess_1sec', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'wpm1': {'data': 'wpm1.csv', 'T': 'A1grGenPowerForProcess_1sec', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'wph2': {'data': 'wph1.csv', 'T': 'A1grGenPowerForProcess_1sec', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'wpm2': {'data': 'wpm1.csv', 'T': 'A1grGenPowerForProcess_1sec', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]
args.checkpoints = os.path.join(args.checkpoints_root,args.data)
args.events = os.path.join(args.events_root, args.data)
args.folder1 = '{}_{}_{}'.format(args.features, args.seq_len, args.pred_len)
args.folder2 = 'lr{}_bs{}_hid{}_s1_dp{}'.format(args.lr,args.batch_size,args.hidden_size,args.dropout)

print('Args in experiment:')
print(args)

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp = Exp_ETTh

mae_, mse_, rmse_, mape_, mspe_, smape_, wmape_ = [],[],[],[],[],[],[]

if args.evaluate:
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s1_l{}_dp{}_inv{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size, args.levels,args.dropout,args.inverse)
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mae, mse, rmse, mape, mspe, smape, wmape, _, _  = exp.test(setting, evaluate=True)
    lines = 'Final mean normed mse:{:.4f}, mae:{:.4f}, wmape:{:.4f}'.format(np.mean(mse_), np.mean(mae_), np.mean(wmape_))
    print(lines)
    with open(os.path.join(args.checkpoints,args.folder1,args.folder2,setting,'summary_eval.txt'),'w') as summary2:
        summary2.writelines(lines)
else:
    # experiments times
    if args.itr:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s1_l{}_dp{}_inv{}_itr{}'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size, args.levels,args.dropout,args.inverse,ii)
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse, rmse, mape, mspe, smape, wmape, _, _ = exp.test(setting)
            mae_.append(mae)
            mse_.append(mse)
            rmse_.append(rmse)
            mape_.append(mape)
            mspe_.append(mspe)
            smape_.append(smape)
            wmape_.append(wmape)
            torch.cuda.empty_cache()

        print('Final mean normed mse:{:.4f}, mae:{:.4f}, wmape:{:.4f}'.format(
                np.mean(mse_), np.mean(mae_), np.mean(wmape_)))
        print('Final min normed mse:{:.4f}, mae:{:.4f},wmape:{:.4f}'.format(
                min(mse_), min(mae_), min(wmape_)))

    else:
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_bs{}_hid{}_s1_l{}_dp{}_inv{}_itr0'.format(args.model,args.data, args.features, args.seq_len, args.label_len, args.pred_len,args.lr,args.batch_size,args.hidden_size, args.levels,args.dropout,args.inverse)
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mae, mse, rmse, mape, mspe, smape, wmape, _, _   = exp.test(setting)
        lines = 'Final mean normed mse:{:.4f},mae:{:.4f},mape:{:.4f},mspe:{:.4f},smape:{:.4f},wmape:{:.4f}'.format(
            mse, mae, mape, mspe, smape, wmape)   
        print(lines)
        with open(os.path.join(args.checkpoints,args.folder1,args.folder2,setting,'summary_all.txt'),'w') as summary:
            summary.writelines(lines)
        summary.close()