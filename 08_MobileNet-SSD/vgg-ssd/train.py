import os
import sys
import time
import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

from data import MEANS, voc, VOCDetection, detection_collate
from layers.modules import MultiBoxLoss
from models.vgg_ssd import build_ssd
from utils import SSDAugmentation


def build_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
    return logger


def build_parser():
    parser = ArgumentParser(description='SSD Training with PyTorch.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA to train model.')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model.')
    parser.add_argument('--dataset_root', type=str, default='VOCdevkit',
                        help='VOC Dataset root directory path.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training.')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate.')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading.')
    parser.add_argument('--max_iter', default=120000, type=int,
                        help='Number of training iteration.')
    parser.add_argument('--save_folder', type=str, default='weights',
                        help='Directory for saving checkpoint models.')
    return parser


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    # ロガーの読み込みと引数のパース
    logger = build_logger()
    args = build_parser().parse_args()

    # GPUモードとCPUモードを選択
    device = torch.device(('cpu', 'cuda:0')[args.cuda and torch.cuda.is_available()])
    logger.info('device: {}'.format(device))

    # モデルのロード
    net = build_ssd(num_classes=voc['num_classes'], train_mode=True).to(device)
    # VGGレイヤの学習済みモデルの読み込み
    vgg_weights = torch.load(os.path.join(args.save_folder, args.basenet))
    net.vgg.load_state_dict(vgg_weights)
    # VGGレイヤ以外のパラメータを初期化
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.train()
    logger.info('------ DUMP LOADED MODEL ------')
    logger.info(net)

    # データセットのロード
    dataset = VOCDetection(root=args.dataset_root,
                           image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                           transform=SSDAugmentation(voc['min_dim'], MEANS))
    logger.info('dataset_size: {}'.format(len(dataset)))

    # 最適化関数の設定
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # 誤差関数の設定
    criterion = MultiBoxLoss(voc['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    # データローダを生成
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # カウンタを定義
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    epoch_size = len(dataset) // args.batch_size
    step_index = 0

    # 学習実行
    batch_iterator = iter(data_loader)
    for iteration in range(0, args.max_iter):
        if iteration in voc['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # 学習データを読み込み
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [ann.to(device) for ann in targets]

        # 順伝搬
        t0 = time.time()
        out = net(images)
        out = (o.to(device) for o in out)

        # 誤差逆伝播
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data
        conf_loss += loss_c.data

        # 学習ログ表示
        if iteration % 10 == 0:
            logger.info('timer: %.4f sec.' % (t1 - t0))
            logger.info('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data))

        # スナップショットを保存
        if iteration != 0 and iteration % 5000 == 0:
            logger.info('Saving state, iter:', iteration)
            torch.save(net.state_dict(), os.path.join(args.save_folder, 'ssd300_VOC_' + repr(iteration) + '.pth'))

    # 学習済みモデルを保存
    saving_path = os.path.join(args.save_folder, 'ssd300_VOC_' + str(args.max_iter) + '.pth')
    logger.info('Saving tained weight: {}'.format(saving_path))
    torch.save(net.state_dict(), saving_path)
