
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))

class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,kernel = 5, dropout=0.5, 
                    groups = 1, hidden_size = 1, INN = True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                        kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                        kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                        kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                        kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                        kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                        kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                        kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                        kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]

        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))
            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)
            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return (c, d)

class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size, INN):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True,
                    kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size, INN):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , hidden_size = hidden_size, INN = INN)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        #even: B, T, D odd: B, T, D
        # batch, T ,dims
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) 

class SFINet_Tree(nn.Module):
    def __init__(self, in_planes,input_len, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        # cur level  
        # pre level all block num
        # cur level block num 
        # cur level out num
        self.structs = []
        fcs = []
        self.num_blocks = 0
        for i in range(num_levels):
            self.structs.append([i+1,self.num_blocks,2**i,2**(i+1)])
            self.num_blocks+=2**i
            if i != 0:
                for j in range(2**i):
                    fcs.append(
                        nn.Sequential(
                            nn.Linear(int(input_len/(2**i)), int(input_len/(2**(i-1))), bias=False),
                            nn.ReLU(inplace=True),
                            nn.Linear(int(input_len/(2**(i-1))), int(input_len/(2**i)), bias=False),
                            nn.Sigmoid()
                            )
                    )

        workingblocks = []
        for i in range(self.num_blocks):
            workingblocks.append(
                LevelSCINet(
                    in_planes = in_planes,
                    kernel_size = kernel_size,
                    dropout = dropout,
                    groups= groups,
                    hidden_size = hidden_size,
                    INN = INN)
            )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.workingblocks = nn.ModuleList(workingblocks)
        self.fcs = nn.ModuleList(fcs)
        
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D

    def forward(self, x):
        inout = []
        inout.append([x])
        for index_i,struct in enumerate(self.structs):
            cur_level,pre_blocks_num,cur_blocks_num,cur_out_num = struct
            cur_inout = []

            # add shuffle in every level
            if index_i != 0:
                group = len(inout[index_i])
                batchsize, num_channels, dims = inout[index_i][0].data.size()
                level_input = torch.cat(inout[index_i],1)
                level_input = level_input.reshape(batchsize, num_channels, group, dims)
                level_input = level_input.permute(0, 2, 1, 3)
                level_input = level_input.reshape(batchsize, num_channels*group, dims)
                level_input = level_input.split(num_channels, 1)
            else:
                level_input = inout[index_i]

            for index_j in range(cur_blocks_num):
                workingblock = self.workingblocks[pre_blocks_num+index_j]
                if index_i != 0:
                    fc_id = pre_blocks_num-1+index_j
                    fcblock = self.fcs[fc_id]
                    x_shuffle = level_input[index_j]
                    s_src = inout[index_i][index_j]
                    x_shuffle_scale = self.avg_pool(x_shuffle).view(batchsize, num_channels)
                    x_shuffle_scale = fcblock(x_shuffle_scale).view(batchsize, num_channels, 1)
                    xa_shuffle = x_shuffle * x_shuffle_scale.expand_as(x_shuffle)
                    input = s_src + xa_shuffle
                else:
                    input = level_input[index_j]
                x_even_update, x_odd_update= workingblock(input)
                cur_inout.extend([x_even_update,x_odd_update])

            inout.append(cur_inout)
            assert cur_out_num == len(cur_inout)
        
        # final inout 
        pooling_out = []
        pooling_out.append(inout[-1])
        for index_i,struct in enumerate(self.structs):
            cur_level,prea_blocks_num,cur_blocks_num,cur_out_num = struct
            cur_pooling = []
            for index_j in range(0,len(pooling_out[index_i]),2):
                haha = self.zip_up_the_pants(pooling_out[index_i][index_j], pooling_out[index_i][index_j+1])
                cur_pooling.append(haha)
            pooling_out.append(cur_pooling)
        
        return pooling_out[-1][0]

class EncoderTree(nn.Module):
    def __init__(self, in_planes,input_len, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super().__init__()
        self.levels=num_levels
        self.SFINet_Tree = SFINet_Tree(
            in_planes = in_planes,
            input_len = input_len,
            num_levels = num_levels,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN)

    def forward(self, x):
        x= self.SFINet_Tree(x)
        return x

class SFINet(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1,num_levels = 3,
                    concat_len = 0, groups = 1, kernel = 5, dropout = 0.5, single_step_output_One = 0, 
                    input_len_seg = 0, positionalE = False, modified = True):
        super(SFINet, self).__init__()
        self.output_len = output_len
        self.input_len = input_len
        self.input_dim = input_dim

        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE

        self.blocks1 = EncoderTree(
                        in_planes=self.input_dim,
                        input_len=self.input_len,
                        num_levels = self.num_levels,
                        kernel_size = self.kernel_size,
                        dropout = self.dropout,
                        groups = self.groups,
                        hidden_size = self.hidden_size,
                        INN =  modified)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        
        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
        return signal

    def forward(self, x):
        # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        assert self.input_len % (np.power(2, self.num_levels)) == 0 
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)
        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        x = self.projection1(x)

        return x


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)
    parser.add_argument('--single_step_output_One', type=int, default=0)
    args = parser.parse_args()

    model = SFINet(output_len = args.horizon, input_len= args.window_size, input_dim = 9, hid_size = args.hidden_size,
                num_levels = 3, concat_len = 0, groups = args.groups, kernel = args.kernel, dropout = args.dropout,
                single_step_output_One = args.single_step_output_One, positionalE =  args.positionalEcoding, modified = True).cuda()
    x = torch.randn(32, 96, 9).cuda()
    y = model(x)
    print(y.shape)
