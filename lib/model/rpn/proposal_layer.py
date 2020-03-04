# coding=utf-8 
import torch
import torch.nn as nn
import numpy as np
import math
from lib.model.utils.config import cfg

from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes
from torchvision.ops import nms
#----------------------------
#
#----------------------------
class _ProposalLayer(nn.Module):

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
            ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

    def forward(self, input):
        scores = input[0][:, self._num_anchors:, :, :] # shape : batch, num_anchor , feature_h,feature_w
        bbox_deltas = input[1] # shape : batch, num_anchor * 4, feature_h,feature_w
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors
        K = shifts.size(0) # feat_width * feat_height

        self._anchors = self._anchors.type_as(scores)

        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4) # 生成最终的 anchor 以特征图的网格为中心生成anchor

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # nms 
            keep_idx_i = nms(proposals_single, scores_single, nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

        def backward(self, top, propagate_down, bottom):
            """This layer does not propagate gradients."""
            pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep


