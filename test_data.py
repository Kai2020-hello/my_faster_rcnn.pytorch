# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader




if __name__ == '__main__':


    args = parse_args()

    print('Called with args:')
    print(args)

    args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
    args.imdbval_name = "coco_2014_minival"
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']


    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                              sampler=sampler_batch, num_workers=args.num_workers)

    
