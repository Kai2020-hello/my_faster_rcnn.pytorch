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

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader




if __name__ == '__main__':


    batch_size = 12

    imdb_name = "coco_2014_train+coco_2014_valminusminival"
    imdbval_name = "coco_2014_minival"
    set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    
    sampler_batch = sampler(train_size, batch_size)
    
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                             imdb.num_classes, training=True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                              sampler=sampler_batch, num_workers=1)
    
    
    data = next(data_iter)
    print(data)

