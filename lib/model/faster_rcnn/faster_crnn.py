#################################
#       faster_rcnn的骨架        #
#                               #
#################################

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision


from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer

class _fasterRCNN(nn.Module):
    def __init__(self,classes,class_agnostic):
        super(_fasterRCNN,self).__init__()
        self.classes = classes
        self.n_classes = len(self.classes) 
        self.class_agnostic = class_agnostic

        #loss 
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        #define rpn 
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        #self.RCNN_roi_pool =  # TODO 后面添加
        #self.RCNN_roi_align =  
        #self.RCNN_roi_crop = # TODO 后面添加



