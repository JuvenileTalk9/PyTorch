import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data import voc
from layers.functions import PriorBox, Detect
from layers.modules import L2Norm


# VGG-SSDを実装する．オリジナルの論文は以下．
# https://arxiv.org/abs/1512.02325


class SSD(nn.Module):
    """VGG-SSDの実装．"""    

    def __init__(self, vgg, extras, loc, conf, num_classes, train_mode=True):
        """コンストラクタ．

        Parameters
        ----------
        vgg : list
            VGGのレイヤ一覧．
        extras : list
            Extra Feature Layersのレイヤ一覧 ．
        locs : list
            loc layersのレイヤ一覧．
        confs : list
            conf layersのレイヤ一覧．
        num_classes : int
            ラベル数．
        train_mode : bool, optional
            学習と推論いずれを指定する．デフォルトはTrue（学習）．
        """        
        super().__init__()
        self.size = 300
        self.priorbox = PriorBox(voc)
        self.priors = self.priorbox.forward()
        self.num_classes = num_classes
        self.train_mode = train_mode

        self.vgg = nn.ModuleList(vgg)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        if not train_mode:
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """順伝搬処理

        Parameters
        ----------
        x : torch.Tensor
            入力画像

        Returns
        -------
        tupple
            学習時：オフセット，クラス，PriorBoxのタプル
            推論時：
        """        
        source = list()
        loc = list()
        conf = list()

        # out1の分岐部（conv4_3）までVGGを順伝搬．
        for i in range(23):
            x = self.vgg[i](x)
        source.append(self.L2Norm(x))

        # out2の分岐部（fc7）までVGGを順伝搬．
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        source.append(x)

        # out3からout6を出力するまでExtraLayersを順伝搬．
        for i in range(0, 8, 2):
            x = F.relu(self.extras[i](x), inplace=True)
            x = F.relu(self.extras[i + 1](x), inplace=True)
            source.append(x)

        # out1からout6について，オフセットとクラスごとの信頼度を算出．
        for (x, l, c) in zip(source, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.train_mode:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        else:
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        return output


def make_vgg():
    """VGGの実装．

    Returns
    -------
    list
        VGGのレイヤ一覧．
    """
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C',
            512, 512, 512, 'M', 512, 512, 512]
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extra_layers():
    """Extra Featyre Layersの実装．

    Returns
    -------
    Extra Feature Layersの一覧．
        list
    """    
    layers = [
        nn.Conv2d(1024, 256, kernel_size=(1)),
        nn.Conv2d(256, 512, kernel_size=(3), stride=2, padding=1),
        nn.Conv2d(512, 128, kernel_size=(1)),
        nn.Conv2d(128, 256, kernel_size=(3), stride=2, padding=1),
        nn.Conv2d(256, 128, kernel_size=(1)),
        nn.Conv2d(128, 256, kernel_size=(3)),
        nn.Conv2d(256, 128, kernel_size=(1)),
        nn.Conv2d(128, 256, kernel_size=(3))
    ]
    return layers


def add_loc_layers():
    """loc Layersの実装．

    Returns
    -------
    loc Layersの一覧．
        list
    """    
    # 以下のnn.Conv2dの第2引数でN * Mとしているが，
    # Nは各レイヤでN個のボックスが生成されることを表し，
    # Mは1ボックスあたり（cx, cy, w, h）の4次元ベクトルの情報を持つことを表す．
    loc_layers = [
        # out1に対する処理
        nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
        # out2に対する処理
        nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
        # out3に対する処理
        nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
        # out4に対する処理
        nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
        # out5に対する処理
        nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        # out6に対する処理
        nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
    ]
    return loc_layers


def add_conf_layers(num_classes):
    """conf Layersの実装．

    Returns
    -------
    conf Layersの一覧
        list
    """    
    # 以下のnn.Conv2dの第2引数でN * num_classesとしているが，
    # Nは各レイヤでN個のボックスが生成されることを表し，
    # num_classesは1ボックスあたりラベルごとの信頼度を出力することを表す．
    conf_layers = [
        # out1に対する処理
        nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
        # out2に対する処理
        nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
        # out3に対する処理
        nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
        # out4に対する処理
        nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
        # out5に対する処理
        nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
        # out6に対する処理
        nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
    ]
    return conf_layers


def build_ssd(num_classes, train_mode=True):
    """VGG-SSDモデルを生成する．

    Parameters
    ----------
    num_classes : int
        ラベル数
    train_mode : bool, optional
        学習と推論いずれかを指定する．デフォルトはTrue（学習）．

    Returns
    -------
    SSD
       VGG-SSDモデル 
    """    
    vgg = make_vgg()
    extras = add_extra_layers()
    loc = add_loc_layers()
    conf = add_conf_layers(num_classes)

    return SSD(vgg, extras, loc, conf, num_classes, train_mode)
