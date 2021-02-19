import os

import cv2
import numpy as np
import torch

from .base import DatasetBase


class VOCDataset(DatasetBase):

    def __init__(self, root, is_train=True, transform=None, target_transform=None):
        super().__init__(transform=transform, target_transform=target_transform)

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
    
        self.class_labels = ('BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                             'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
                             'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.class_labels)

        if is_train:
            image_sets_file = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        else:
            image_sets_file = os.path.join(self.root, 'ImageSets', 'Main', 'trainval.txt')

    def _get_image(self, index):
        

    def _get_target(self, index):
        