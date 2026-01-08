import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLength: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLength // 2 - 1,
                          self.kernLength - self.kernLength // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLength),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

        self.classifier_block = nn.Sequential(
            nn.Linear(in_features=self.F2 * (self.Samples // (4 * 8)),
                    out_features=self.n_classes,
                    bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        output = self.classifier_block(output)
        return output


class EEGNet_feature(nn.Module):

    def __init__(self,
                 n_classes: int,
                 Chans: int,
                 Samples: int,
                 kernLength: int,
                 F1: int,
                 D: int,
                 F2: int,
                 dropoutRate:  float,
                 norm_rate: float):
        super(EEGNet_feature, self).__init__()

        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate

        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLength // 2 - 1,
                          self.kernLength - self.kernLength // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLength),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            # SeparableConv2d
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.reshape(output.size(0), -1)
        return output

##adv
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class AdversarialEEGNet(nn.Module):
    def __init__(self, n_classes, n_nuisance, lambd, **eeg_kwargs):
        super().__init__()
        self.feat = EEGNet_feature(n_classes, **eeg_kwargs)
        feat_dim = eeg_kwargs["F2"] * (eeg_kwargs["Samples"] // (4 * 8))
        self.cla_head = nn.Linear(feat_dim, n_classes)
        self.adv_head = nn.Linear(feat_dim, n_nuisance)
        self.lambd = lambd

    def forward(self, x):
        f = self.feat(x)
        cla = self.cla_head(f)
        adv = self.adv_head(GradReverse.apply(f, self.lambd))
        return cla, adv


class ConditionalAdversarialEEGNet(nn.Module):
    """
    Conditional Domain Adversarial Network (C-DAN) variant for EEGNet.
    - Conditions domain classifier on class posterior p(y|x) (softmax of cla logits).
    - Uses outer-product conditioning: f (B, feat_dim) x p (B, n_classes) -> (B, feat_dim * n_classes)
    - Applies Gradient Reversal on the conditioned feature before domain classification.
    Note: If feat_dim * n_classes becomes too large, consider adding a projection or
    multi-head variant to reduce parameter counts.
    """
    def __init__(self, n_classes, n_nuisance, lambd, **eeg_kwargs):
        super().__init__()
        self.feat = EEGNet_feature(n_classes, **eeg_kwargs)
        feat_dim = eeg_kwargs["F2"] * (eeg_kwargs["Samples"] // (4 * 8))
        self.cla_head = nn.Linear(feat_dim, n_classes)
        # domain classifier operates on conditioned flattened feature
        self.domain_clf = nn.Sequential(
            nn.Linear(feat_dim * n_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_nuisance),
        )
        self.lambd = lambd
        self.n_classes = n_classes
        self.feat_dim = feat_dim

    def forward(self, x):
        # extract feature and class logits
        f = self.feat(x)                     # (B, feat_dim)
        cla_logits = self.cla_head(f)        # (B, n_classes)
        # soft conditioning
        p = F.softmax(cla_logits, dim=1)     # (B, n_classes)
        # outer-product conditioning -> (B, feat_dim, n_classes)
        f_cond = f.unsqueeze(2) * p.unsqueeze(1)
        # flatten to (B, feat_dim * n_classes)
        f_cond_flat = f_cond.reshape(f_cond.size(0), -1)
        # apply gradient reversal and domain classifier
        f_grl = GradReverse.apply(f_cond_flat, self.lambd)
        domain_logits = self.domain_clf(f_grl)
        return cla_logits, domain_logits
