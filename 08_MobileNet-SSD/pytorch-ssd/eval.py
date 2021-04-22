# Original: https://github.com/qfgaohao/pytorch-ssd

import os
import sys
import logging
from glob import glob
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from models.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from utils import box_utils


model_params = {
    'mobilenetv1_ssd': (create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor)
}


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
    parser.add_argument('--trained_weights', type=str, required=True,
                        help='Trained weights pathe.')
    parser.add_argument('--labelmap_path', type=str, required=True,
                        help='Labelmap path')
    parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                        help='Root directory path of eval image dirt.')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA to train model.')
    parser.add_argument('--nms_method', type=str, default='hard',
                        help='NMS method name.')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='The threshold of Intersection over Union.')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='The directory to store evaluation results.')
    return parser


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
        create_net, create_predictor = model_params[args.model]
    except KeyError:
        logger.error('model type {} is not supprted.'.format(args.model))
        exit(1)

    # モデル読み込み
    logger.info('Building model...')
    with open(args.labelmap_path) as f:
        class_names = [name.strip() for name in f.readlines()]
    net = create_net(len(class_names) + 1)
    net.load(args.trained_weights)
    net = net.to(device)
    predictor = create_predictor(net, candidate_size=200, device=device)

    # データセット読み込み
    image_list = []
    for eval_dir in args.eval_dirs:
        image_list.extend(glob(os.path.join(eval_dir, '*')))

    # 評結果の出力先ディレクトリ生成
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 評価開始
    logger.info('Start evaluation from size : {}.'.format(len(image_list)))
    for idx, image_path in enumerate(image_list, start=1):
        logger.info('[ {}/{} ] {}'.format(idx, len(image_list), image_path))
        image_org = cv2.imread(image_path)
        if image_org is None:
            logger.warning('Failed to load image file: {}'.format(image_path))
            continue
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            cv2.rectangle(image_org, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            label = f"{labels[i]}: {probs[i]:.2f}"
            cv2.putText(image_org, label, (box[0] + 20, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        out_path = os.path.join(args.result_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, image_org)
