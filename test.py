from lib.model.rpn.rpn  import _RPN
import torch 
import numpy as np


if __name__ == "__main__":

    
    base_feat = torch.randn((1,512,16,16))
    im_info = torch.from_numpy(np.array( [[1000, 1000, 3]],dtype=np.float32))
    gt_boxes = torch.randn((1,3,5))
    num_boxes = torch.Tensor([[1,1,1]])
    rpn = _RPN(512)
    rois, rpn_loss_cls, rpn_loss_box = rpn(base_feat,im_info,gt_boxes,num_boxes)
    