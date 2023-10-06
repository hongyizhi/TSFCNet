import torch
import torch.nn as nn


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


class EEGNeXB(nn.Module):

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.25, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))

    def get_size(self, ):
        x = torch.ones((1, 1, 22, 1000))
        out = self.net(x)
        return out.size()

    def __init__(self, nChan=22, nTime=1000, nClass=2, nBands=9, m=36, dropoutP=0.5,
                 temporalLayer='LogVarLayer', strideFactor=4, doWeightNorm=True, *args, **kwargs):
        super(EEGNeXB, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 32), bias=False, padding='same'),
            nn.BatchNorm2d(8),
            nn.ELU(),

            nn.Conv2d(8, 32, (1, 32), bias=False, padding='same'),
            nn.BatchNorm2d(32),
            nn.ELU(),

            Conv2dWithConstraint(32, 64, (nChan, 1), groups=32, doWeightNorm=True, max_norm=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 32, (1, 16), bias=False, padding='same', dilation=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 8, (1, 16), bias=False, padding='same', dilation=(1, 4)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=0.5),

            nn.Flatten(start_dim=1),
        )

        size = self.get_size()
        self.linear = self.LastBlock(size[1], 4, doWeightNorm=True,)

    def forward(self, x):
        # x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        f = self.net(x)
        out = self.linear(f)
        return out


if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.ones(128, 1, 22, 1000)
    net = EEGNeXB()
    print(count_parameters(net))
    a = net(data)
    print(a.shape)