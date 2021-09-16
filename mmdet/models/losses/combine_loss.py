import torch
import torch.nn as nn

from ..builder import LOSSES, build_loss


@LOSSES.register_module()
class BCE_Boundary_Loss(nn.Module):
    def __init__(self):
        super(BCE_Boundary_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='BoundaryLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)

    def forward(self,pred,target,label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        boundary_loss = self.cls_criterion_2(pred, target, label)
        combine_loss = bce_loss + boundary_loss
        return combine_loss


@LOSSES.register_module()
class BCE_Dice_Loss(nn.Module):
    def __init__(self):
        super(BCE_Dice_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='DiceLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)

    def forward(self,pred,target,label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        dice_loss = self.cls_criterion_2(pred, target, label)
        combine_loss = bce_loss + dice_loss
        return combine_loss


@LOSSES.register_module()
class BCE_HD_Loss(nn.Module):
    def __init__(self):
        super(BCE_HD_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)

    def forward(self,pred,target,label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)
        combine_loss = bce_loss + hd_loss
        return combine_loss


@LOSSES.register_module()
class Dice_BD_Loss(nn.Module):
    def __init__(self):
        super(Dice_BD_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='BoundaryLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)

    def forward(self,pred,target,label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        bd_loss = self.cls_criterion_2(pred, target, label)
        combine_loss = dice_loss + bd_loss
        return combine_loss


@LOSSES.register_module()
class Dice_HD_Loss(nn.Module):
    def __init__(self):
        super(Dice_BD_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)

    def forward(self,pred,target,label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)
        combine_loss = dice_loss + hd_loss
        return combine_loss