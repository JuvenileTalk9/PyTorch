import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetBase:

    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def _get_image(self, index):
        raise NotImplementedError('_get_image must be overridden.')

    def _get_target(self, index):
        raise NotImplementedError('_get_target must be overridden.')

    def __getitem__(self, index):
        # データセットからデータを一つ取得
        image = self._get_image(index)
        bboxes, labels, is_difficult = self._get_target(index)
        
        # difficultを除く
        bboxes = bboxes[is_difficult == False]
        labels = labels[is_difficult == False]

        # 前処理
        if self.transform:
            image, bboxes, labels = self.transform(image, bboxes, labels)
        if self.target_transform:
            bboxes, labels = self.target_transform(bboxes, labels)
        
        return image, bboxes, labels

    def __len__(self):
        raise NotImplementedError('__len__ must be overridden.')

    @staticmethod
    def _read_image(self, image_path):
        # 画像を読み込みBGR→RGBに変換する
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
