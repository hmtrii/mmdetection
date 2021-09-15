import torch
import torch.nn as nn

from ..builder import LOSSES, build_loss
from .cross_entropy_loss import cross_entropy
from .boundary_loss import BoundaryLoss


@LOSSES.register_module()
class BCEandHD(nn.Module):
    def __init__(self):
        super(BCEandHD, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)

    def forward(self,pred,target,label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)
        combine_loss = bce_loss + hd_loss
        return combine_loss