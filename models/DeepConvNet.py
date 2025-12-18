'''
=================================================
coding:utf-8
@Time:      2023/12/5 17:08
@File:      DeepConvNet.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import torch
import torch.nn as nn

from models.EEGNet import GradReverse


class DeepConvNet(nn.Module):
    def __init__(
        self,
        n_classes,
        Chans,
        Samples,
        dropoutRate=0.5,
        batch_norm=True,
        batch_norm_alpha=0.1,
    ):
        super(DeepConvNet, self).__init__()
        self.n_classes = n_classes
        n_ch1, n_ch2, n_ch3, n_ch4 = 25, 50, 100, 200

        if batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(Chans, 1), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1, momentum=batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=dropoutRate),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=False),
                nn.BatchNorm2d(n_ch2, momentum=batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=dropoutRate),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=False),
                nn.BatchNorm2d(n_ch3, momentum=batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=dropoutRate),
                nn.Conv2d(n_ch3, n_ch4, kernel_size=(1, 10), stride=1, bias=False),
                nn.BatchNorm2d(n_ch4, momentum=batch_norm_alpha, affine=True, eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1, momentum=batch_norm_alpha, affine=True, eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(Chans, 1), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=dropoutRate),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=dropoutRate),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=dropoutRate),
                nn.Conv2d(n_ch3, n_ch4, kernel_size=(1, 10), stride=1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )

        with torch.no_grad():
            was_training = self.convnet.training
            self.convnet.eval()
            dummy = torch.zeros(1, 1, Chans, Samples)
            out = self.convnet(dummy)
            if was_training:
                self.convnet.train()

        self.n_outputs = out.numel()
        self.classifier = nn.Linear(self.n_outputs, self.n_classes)

    def forward(self, x):
        features = self.convnet(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

    def get_embedding(self, x):
        features = self.convnet(x)
        return features.view(features.size(0), -1)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class AdversarialDeepConvNet(nn.Module):
    """
    DeepConvNet with an additional adversarial head.

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
        # Re‑use the existing backbone to keep architecture identical
        self.backbone = DeepConvNet(
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


#
# model = EEGNet(4,22,256)  #4~几分类/n_classes，22~通道数/input_ch, 256~时间点数/input_time
# x = x.permute(0, 1, 3, 2)  #使用前转换一下矩阵维数
