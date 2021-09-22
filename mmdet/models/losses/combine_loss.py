import torch
import torch.nn as nn

from ..builder import LOSSES, build_loss

@LOSSES.register_module()
class BCE_Boundary_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_Boundary_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='BoundaryLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        boundary_loss = self.cls_criterion_2(pred, target, label)

        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + cur_alpha*boundary_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + cur_alpha*boundary_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*bce_loss + cur_alpha*boundary_loss
        
        self.count_iter += 1
        return combine_loss


@LOSSES.register_module()
class BCE_HD_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_HD_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)

        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss +cur_alpha*hd_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + cur_alpha*hd_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*bce_loss + cur_alpha*hd_loss

        self.count_iter += 1
        return combine_loss


@LOSSES.register_module()
class Dice_BD_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(Dice_BD_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss')
        loss_mask_2=dict(type='BoundaryLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        bd_loss = self.cls_criterion_2(pred, target, label)
        
        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = dice_loss + cur_alpha*bd_loss
        elif self.alpha_strategy == "increase":
            combine_loss = dice_loss + cur_alpha*bd_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*dice_loss + cur_alpha*bd_loss

        self.count_iter += 1
        return combine_loss


@LOSSES.register_module()
class Dice_HD_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(Dice_HD_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss')
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)

        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = dice_loss + cur_alpha*hd_loss
        elif self.alpha_strategy == "increase":
            combine_loss = dice_loss + cur_alpha*hd_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*dice_loss + cur_alpha*hd_loss

        self.count_iter += 1
        return combine_loss


@LOSSES.register_module()
class BCE_SDF_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_SDF_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='SDFLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        sdf_loss = self.cls_criterion_2(pred, target, label)

        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + cur_alpha*sdf_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + cur_alpha*sdf_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*bce_loss + cur_alpha*sdf_loss

        self.count_iter += 1
        return combine_loss


@LOSSES.register_module()
class Dice_SDF_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(Dice_SDF_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss')
        loss_mask_2=dict(type='SDFLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        sdf_loss = self.cls_criterion_2(pred, target, label)

        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = dice_loss + cur_alpha*sdf_loss
        elif self.alpha_strategy == "increase":
            combine_loss = dice_loss + cur_alpha*sdf_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*dice_loss + cur_alpha*sdf_loss

        self.count_iter += 1
        return combine_loss


@LOSSES.register_module()
class BCE_Dice_Loss(nn.Module):
    def __init__(self, start_alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_Dice_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='DiceLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.start_alpha = start_alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        dice_loss = self.cls_criterion_2(pred, target, label)

        cur_alpha = self.start_alpha + int(self.count_iter / 1120) * self.step_alpha
        if cur_alpha > self.max_alpha:
            cur_alpha = self.max_alpha
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + cur_alpha*dice_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + cur_alpha*dice_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-cur_alpha)*bce_loss + cur_alpha*dice_loss

        self.count_iter += 1
        return combine_loss