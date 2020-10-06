import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn


class Net(nn.Module):
    '''irisの4データからirisの3種類のいずれかに分類するネットワーク'''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


def load_dataset():
    '''irisデータセットを読み込んで、学習・検証データに分割して返す'''
    # irisデータセットを読み込む
    iris = datasets.load_iris()
    x = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    # 偶数番目を学習用データセットとして取得
    train_x = torch.from_numpy(x[::2])
    train_y = torch.from_numpy(y[::2])

    # 奇数番目を検証用データセットとして取得
    test_x = torch.from_numpy(x[1::2])
    test_y = torch.from_numpy(y[1::2])

    return train_x, train_y, test_x, test_y
