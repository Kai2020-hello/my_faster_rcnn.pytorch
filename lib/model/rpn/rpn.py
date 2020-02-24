import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .anchor_target_layer import _AnchorTargetLayer


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()

        # 输出特征的深度
        self.din = din
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # 分数分支
        self.nc_score_out = len(self.anchor_ratios) * len(self.anchor_scales) * 2
        self.RPN_cls_score =  nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # 边框回归分支
        self.nc_bbox_cout = len(self.anchor_ratios) * len(self.anchor_scales) * 4
        self.RPN_bbox_pred =  nn.Conv2d(512, self.nc_bbox_cout, 1, 1, 0)

        #define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
    
    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        # shape of base_feat: batch_size, channel ,h , w 
        
        batch_size = base_feat.size(0) 

        #进行一次卷积
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace= True) # shape : batch_size, 512 ,h , w

        rpn_cls_score = self.RPN_cls_score(rpn_conv1) # shape : batch_size, 18(9*2) ,h , w

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2) # shape : batch_size, 2 ,h*9 , w
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1) # 在1维度上进行比较 # shape : batch_size, 2 ,h*9 , w
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)# shape : batch_size, 9*2 ,h , w

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)  # shape : batch_size, 9*4 ,h , w

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key)) # shape batch_size, post_nms_topN , 5 ; 0 of 5 is batch_idx

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])


        return rois, self.rpn_loss_cls, self.rpn_loss_box







