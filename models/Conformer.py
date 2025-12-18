# baseline/Conformer.py
"""
EEG Conformer

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""

import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

# import torchvision.transforms as transforms
# from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
# from torchsummary import summary
import torch.autograd as autograd
# from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
# import torch.nn.init as init

# from torch.utils.data import Dataset
from PIL import Image
# import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.backends import cudnn

from models.EEGNet import GradReverse

cudnn.benchmark = False
cudnn.deterministic = True


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, chn=22, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (chn, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
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


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size,
                    num_heads=num_heads,
                    drop_p=drop_p,
                    forward_expansion=forward_expansion,
                    forward_drop_p=forward_drop_p,
                )
                for _ in range(depth)
            ]
        )


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)  # As DBConformer, use fc as the classification head rather than clshead.
        return out

class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, Chans=-1, n_classes=2, **kwargs):
        num_heads = kwargs.pop("num_heads", 10)
        drop_p = kwargs.pop("drop_p", 0.5)
        forward_expansion = kwargs.pop("forward_expansion", 4)
        forward_drop_p = kwargs.pop("forward_drop_p", 0.5)
        kwargs.pop("patch_dropout", None)
        super().__init__(
            PatchEmbedding(Chans, emb_size),  # conv layers
            TransformerEncoder(
                depth,
                emb_size,
                num_heads=num_heads,
                drop_p=drop_p,
                forward_expansion=forward_expansion,
                forward_drop_p=forward_drop_p,
            ),  # encoders
            ClassificationHead(emb_size, n_classes)
        )


class ConformerFeature(nn.Module):
    """
    Conformer backbone that outputs a compact feature representation instead
    of logits. The architecture mirrors ``Conformer`` but stops before the
    final classification layer so the feature can be shared between heads.
    """

    def __init__(self, emb_size=40, depth=6, Chans=-1, **kwargs):
        super().__init__()
        num_heads = kwargs.pop("num_heads", 10)
        drop_p = kwargs.pop("drop_p", 0.5)
        forward_expansion = kwargs.pop("forward_expansion", 4)
        forward_drop_p = kwargs.pop("forward_drop_p", 0.5)
        kwargs.pop("patch_dropout", None)

        self.patch = PatchEmbedding(Chans, emb_size)
        self.encoder = TransformerEncoder(
            depth,
            emb_size,
            num_heads=num_heads,
            drop_p=drop_p,
            forward_expansion=forward_expansion,
            forward_drop_p=forward_drop_p,
        )
        # Mirror the MLP stack used in ClassificationHead, but stop before the
        # final classification layer so we expose a shared feature of size 32.
        self.mlp = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch(x)
        x = self.encoder(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.mlp(x)
        return x


class AdversarialConformer(nn.Module):
    """
    Conformer with an additional adversarial head.

    Returns a tuple ``(cla_logits, adv_logits)`` where:
      - ``cla_logits``: class prediction over ``n_classes``
      - ``adv_logits``: nuisance prediction over ``n_nuisance`` (e.g., subjects)

    The adversarial branch uses ``GradReverse`` so that minimizing its loss
    encourages the backbone features to be invariant to the nuisance factor.
    """

    def __init__(
        self,
        emb_size: int = 40,
        depth: int = 6,
        Chans: int = -1,
        n_classes: int = 2,
        n_nuisance: int = None,
        lambd: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        self.feature = ConformerFeature(emb_size=emb_size, depth=depth, Chans=Chans, **kwargs)
        feat_dim = 32
        self.cla_head = nn.Linear(feat_dim, n_classes)
        self.adv_head = nn.Linear(feat_dim, n_nuisance if n_nuisance is not None else Chans)
        self.lambd = float(lambd)

    def forward(self, x: Tensor):
        f = self.feature(x)
        cla = self.cla_head(f)
        adv = self.adv_head(GradReverse.apply(f, self.lambd))
        return cla, adv

class Conformer_patchembedding(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, Chans=22, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(Chans, emb_size)
        )

class Conformer_encoder(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, Chans=22, n_classes=2, **kwargs):
        super().__init__(

            TransformerEncoder(depth, emb_size)
        )
