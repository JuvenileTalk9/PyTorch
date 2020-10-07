import pickle

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader


class PreProcess(object):
    '''画像を16x16サイズに変換する前処理'''
    def __init__(self):
        pass
    def __call__(self, xy):
        x, y = xy
        x = cv2.resize(x, (16, 16))
        return (x, y)


class MyDataset(Dataset):
    '''CIFAR10データセットを管理するDatasetクラス'''
    def __init__(self, prepro):
        data, labels = [], []
        # CIFAR10のファイル（事前にこのプログラムと同じディレクトリに配置する）
        cifar10_files = ['data_batch_1', 'data_batch_2', 'data_batch_3',
            'data_batch_4', 'data_batch_5']
        for file in cifar10_files:
            dic = self._unpickle(file)
            # 複数の画像がまとめてバイナリで保存されているので、
            # 画像枚数xチャンネル数x縦幅x横幅の4次元に変換する。
            # そのあとでRGBからBGRに変換する。
            data.append(dic[b'data'].reshape(10000, 3, 32, 32)
                .transpose(0, 2, 3, 1).astype('uint8'))
            labels.append(dic[b'labels'])

        self.train_x = np.concatenate(data)
        self.train_y = np.concatenate(labels)
        self.len = len(self.train_x)
        self.prepro = prepro

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.prepro((self.train_x[idx, :, :, :], self.train_y[idx]))

    def _unpickle(self, file):
        with open(file, 'rb') as f:
            dic = pickle.load(f, encoding='bytes')
        return dic


if __name__ == '__main__':

    prepro = PreProcess()
    dataset = MyDataset(prepro)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for x, y in dataloader:
        print(x, y)
