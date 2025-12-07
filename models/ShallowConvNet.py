'''
=================================================
coding:utf-8
@Time:      2023/12/5 17:08
@File:      ShallowConvNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''

import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        n_classes,
        Chans,
        Samples,
        dropoutRate=0.5,
        batch_norm=True,
        batch_norm_alpha=0.1,
    ):
        super(ShallowConvNet, self).__init__()
        self.n_classes = n_classes
        n_ch1 = 40

        if batch_norm:
            self.layer1 = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 3, 0, 0)),
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(Chans, 1), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1, momentum=batch_norm_alpha, affine=True, eps=1e-5),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.ZeroPad2d(padding=(0, 3, 0, 0)),
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(Chans, 1), stride=1, bias=True),
            )

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(p=dropoutRate)

        with torch.no_grad():
            was_training = self.layer1.training
            self.layer1.eval()
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self.avg_pool(torch.square(self.layer1(dummy)))
            if was_training:
                self.layer1.train()

        self.n_outputs = out.numel()
        self.clf = nn.Linear(self.n_outputs, self.n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.clf(x)
