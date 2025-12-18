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

from models.EEGNet import GradReverse


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

    def get_embedding(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x


class AdversarialShallowConvNet(nn.Module):
    """
    ShallowConvNet with an additional adversarial head.

    Returns a tuple ``(cla_logits, adv_logits)`` where:
      - ``cla_logits``: class prediction over ``n_classes``
      - ``adv_logits``: nuisance prediction over ``n_nuisance`` (e.g., subjects)

    The adversarial branch uses ``GradReverse`` so that minimizing its loss
    encourages the backbone features to be invariant to the nuisance factor.
    """

    def __init__(
        self,
        n_classes: int,
        n_nuisance: int,
        lambd: float,
        Chans: int,
        Samples: int,
        dropoutRate: float = 0.5,
        batch_norm: bool = True,
        batch_norm_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = ShallowConvNet(
            n_classes=n_classes,
            Chans=Chans,
            Samples=Samples,
            dropoutRate=dropoutRate,
            batch_norm=batch_norm,
            batch_norm_alpha=batch_norm_alpha,
        )
        feat_dim = self.backbone.n_outputs
        self.cla_head = nn.Linear(feat_dim, n_classes)
        self.adv_head = nn.Linear(feat_dim, n_nuisance)
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor):
        feat = self.backbone.get_embedding(x)
        cla = self.cla_head(feat)
        adv = self.adv_head(GradReverse.apply(feat, self.lambd))
        return cla, adv
