#################################
#       faster_rcnn的骨架        #
#                               #
#################################

import torch 
import torch.nn as nn
import torch.nn.functional as F 

from lib.model.utils.config import cfg
from lib.model.rpn.rpn 

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
        self.RCNN_rpn = 
        self.RCNN_proposal_target = 
        self.RCNN_roi_pool = 
        self.RCNN_roi_align = 

        self.RCNN_roi_crop = 



