# Original: https://github.com/qfgaohao/pytorch-ssd

import os
import sys
import logging
import itertools
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from datasets.voc_dataset import VOCDataset
from models.data_preprocessing import TrainAugmentation, TestTransform
from models.ssd import MatchPrior
from models.mobilenetv1_ssd import create_mobilenetv1_ssd
from models.config import mobilenetv1_ssd_config
from nn.multibox_loss import MultiboxLoss


model_params = {
    'mobilenetv1_ssd': (create_mobilenetv1_ssd, mobilenetv1_ssd_config)
}


class_labels = ('BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
                'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')


def build_logger():
    logger = logging.getLogger(__name__)
    stream_hander = logging.StreamHandler()
    stream_hander.setLevel(logging.DEBUG)
    stream_hander.setFormatter(logging.Formatter("%(asctime)s [ %(levelname)s ] %(message)s"))
    file_handler = logging.FileHandler('train.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [ %(levelname)s ] %(message)s"))
    logging.basicConfig(level=logging.NOTSET, handlers=[stream_hander, file_handler])
    return logger


def build_argparser():
    parser = ArgumentParser(description='SSD Training with PyTorch.')
    parser.add_argument('--model', type=str, required=True,
                        help='Training model type.')
    parser.add_argument('--dataset_root', type=str, nargs='+', required=True,
                        help='Root directory path of dataset with VOC format.')
    parser.add_argument('--val_dataset_root', type=str, required=True,
                        help='Root directory path of validation dataset with VOC format.')
    parser.add_argument('--labelmap_path', type=str, required=True,
                        help='Labelmap path')
    parser.add_argument('--base_weights', type=str, required=True,
                        help='Pretrained base weights.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA to train model.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum value for optim.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for SGD.')
    parser.add_argument('--scheduler', default="multi-step", type=str,
                        help="Scheduler for SGD. It can one of multi-step and cosine")
    parser.add_argument('--milestones', default="80,100", type=str,
                        help="milestones for MultiStepLR")
    parser.add_argument('--t_max', default=120, type=float,
                        help='T_max value for Cosine Annealing Scheduler.')
    parser.add_argument('--num_epochs', default=120, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--validation_epochs', default=5, type=int,
                        help='The number validation epochs')
    parser.add_argument('--save_folder', type=str, default='weights',
                        help='Directory for saving checkpoint models.')
    return parser


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logger.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':

    # ロガーの読み込みと引数のパース
    logger = build_logger()
    args = build_argparser().parse_args()
    logger.info(args)

    # GPUモードとCPUモードを選択
    device = torch.device(('cpu', 'cuda:0')[args.cuda and torch.cuda.is_available()])
    torch.backends.cudnn.benchmark = args.cuda and torch.cuda.is_available()
    logger.info('device: {}'.format(device))

    # モデル選択
    try:
        create_net, config = model_params[args.model]
    except KeyError:
        logger.error('model type {} is not supprted.'.format(args.model))
        exit(1)

    # データ変換器生成
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    # 学習データセット読み込み
    logger.info('Preparing training datasets...')
    datasets = []
    for dataset_path in args.dataset_root:
        if not os.path.isdir(dataset_path):
            logger.error('No such dataset root dir: {}'.format(dataset_path))
        dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform, label_file=args.labelmap_path)
        datasets.append(dataset)
        num_classes = len(dataset.class_names)
    train_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    logger.info('Train dataset size: {}'.format(len(train_dataset)))
    
    # 評価データセット読み込み
    logger.info('Preparing validation datasets...')
    val_dataset = VOCDataset(args.val_dataset_root, transform=test_transform, target_transform=target_transform, label_file=args.labelmap_path, is_test=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    logger.info('Validation dataset size: {}'.format(len(val_dataset)))

    # モデル読み込み
    logger.info('Building model...')
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    params = [
        {'params': net.base_net.parameters(), 'lr': args.lr},
        {'params': itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters()), 'lr': args.lr},
        {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
    ]
    net.init_from_base_net(args.base_weights)
    net.to(device)

    # 誤差関数と最適化関数の設定
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # スケジューラの設定
    if args.scheduler == 'multi-step':
        logger.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logger.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logger.error("Unsupported Scheduler: {}.".format(args.scheduler))
        sys.exit(1)

    # モデルの出力先ディレクトリ生成
    os.makedirs(args.save_folder, exist_ok=True)

    # 学習開始
    logger.info('Start training from epoch {}.'.format(last_epoch + 1))
    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=device, epoch=epoch)
        scheduler.step()
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, device=device)
            logger.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.save_folder, f"{args.model}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logger.info(f"Saved model {model_path}")
