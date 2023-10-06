import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange, reduce, repeat


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Conv_block(nn.Module):
    def __init__(self, F1=16, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
        super(Conv_block, self).__init__()
        F2 = F1 * D
        self.block1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.block2 = Conv2dWithConstraint(F1, F1 * D, (in_chans, 1), bias=False, groups=F1,
                                           doWeightNorm=True, max_norm=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avg1 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(dropout)
        self.block3 = nn.Conv2d(F2, F2, (1, 16), bias=False, padding='same')
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg2 = nn.AvgPool2d((1, poolSize))
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(self.block1(x))
        x = self.elu(self.bn2(self.block2(x)))
        x = self.avg1(x)
        x = self.dropout(x)
        x = self.elu(self.bn3(self.block3(x)))
        x = self.avg2(x)
        x = self.dropout(x)

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn1, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, padding='same') if n_inputs != n_outputs else None
        self.relu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class mha_block(nn.Module):
    def __init__(self, key_dim=8, num_heads=2, dropout=0.5):
        super(mha_block, self).__init__()

        self.LayerNorm = nn.LayerNorm(32, eps=1e-6)
        self.bn = nn.BatchNorm1d(32)
        self.mha = MultiHeadAttention_atc(32, num_heads, dropout, key_dim=8)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        res = x
        # x = self.LayerNorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.mha(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = res + x

        return x


class MultiHeadAttention_atc(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, key_dim):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, key_dim * num_heads)
        self.queries = nn.Linear(emb_size, key_dim * num_heads)
        self.values = nn.Linear(emb_size, key_dim * num_heads)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(key_dim * num_heads, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TCN_block(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout):
        super(TCN_block, self).__init__()

        self.block = TemporalBlock(n_inputs=input_dimension, n_outputs=filters, stride=1, dilation=1,
                              kernel_size=kernel_size, padding=(kernel_size-1) * 1, dropout=dropout)

        layers = []
        for i in range(depth - 1):
            dilation_size = 2 ** (i + 1)
            layers += [TemporalBlock(input_dimension, filters, stride=1, dilation=dilation_size,
                                     kernel_size=kernel_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.network(x)
        return x


class ATCNet(nn.Module):
    def __init__(self, nChan=22, nClass=4, nTime=1000, in_samples=1125, n_windows=5, attention='mha',
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 fuse='average', *args, **kwargs):
        super(ATCNet, self).__init__()

        n_classes = nClass
        in_chans = nChan
        regRate = .25
        numFilters = eegn_F1
        F2 = numFilters * eegn_D
        self.n_windows = n_windows
        self.fuse = fuse

        self.block1 = Conv_block(F1=eegn_F1, D=eegn_D, kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                                 in_chans=in_chans, dropout=eegn_dropout)

        self.dense_list = nn.ModuleList([LinearWithConstraint(tcn_filters, n_classes,
                                                doWeightNorm=True, max_norm=regRate) for i in range(self.n_windows)])

        self.attention_block_list = nn.ModuleList([mha_block() for i in range(self.n_windows)])

        self.tcn_block_list = nn.ModuleList([TCN_block(input_dimension=F2, depth=tcn_depth, kernel_size=tcn_kernelSize,
                                   filters=tcn_filters, dropout=tcn_dropout) for i in range(self.n_windows)])

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # if len(x.shape) == 5:
        #     x = torch.squeeze(x, dim=4)
        x1 = self.block1(x).squeeze(2)
        sw_concat = []
        for i in range(self.n_windows):
            st = i
            end = x1.shape[2] - self.n_windows + i + 1
            x2 = x1[:, :, st:end]

            # attention or identity
            x2 = self.attention_block_list[i](x2)
            # TCN
            x3 = self.tcn_block_list[i](x2)
            # Get feature maps of the last sequence
            x3 = x3[:, :, -1]
            # Outputs of sliding window: Average_after_dense or concatenate_then_dense
            sw_concat.append(self.dense_list[i](x3))

        if len(sw_concat) > 1:
            sw_concat = torch.mean(torch.stack(sw_concat, dim=1), dim=1)
        else:
            sw_concat = sw_concat[0]

        return self.logsoftmax(sw_concat)


if __name__ == '__main__':

    n_classes = 4
    in_chans = 22
    in_samples = 1125
    n_windows = 5
    attention = 'mha',
    eegn_F1 = 16
    eegn_D = 2
    eegn_kernelSize = 64
    eegn_poolSize = 7
    eegn_dropout = 0.3
    tcn_depth = 2
    tcn_kernelSize = 4
    tcn_filters = 32
    tcn_dropout = 0.3
    fuse = 'average'
    numFilters = eegn_F1
    F2 = numFilters * eegn_D

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.ones(128, 1, 22, 1125)
    net = ATCNet()
    print(count_parameters(net))
    a = net(data)
    print(a.shape)
