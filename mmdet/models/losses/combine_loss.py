import torch
import torch.nn as nn

from ..builder import LOSSES, build_loss

def update_alpha(alpha, step_alpha, max_alpha, count_iter, iter_one_epoch=1120):
    alpha += int(count_iter / iter_one_epoch) * step_alpha
    if alpha > max_alpha:
        alpha = max_alpha
    
    return alpha

@LOSSES.register_module()
class BCE_Boundary_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_Boundary_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='BoundaryLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        boundary_loss = self.cls_criterion_2(pred, target, label)

        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + self.alpha*boundary_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + self.alpha*boundary_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*bce_loss + self.alpha*boundary_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)
        
        return combine_loss


@LOSSES.register_module()
class BCE_HD_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_HD_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)

        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + self.alpha*hd_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + self.alpha*hd_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*bce_loss + self.alpha*hd_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)

        return combine_loss


@LOSSES.register_module()
class Dice_BD_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(Dice_BD_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss')
        loss_mask_2=dict(type='BoundaryLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        bd_loss = self.cls_criterion_2(pred, target, label)
        
        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = dice_loss + self.alpha*bd_loss
        elif self.alpha_strategy == "increase":
            combine_loss = dice_loss + self.alpha*bd_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*dice_loss + self.alpha*bd_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)

        return combine_loss


@LOSSES.register_module()
class Dice_HD_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(Dice_HD_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss')
        loss_mask_2=dict(type='HausdorffDTLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        hd_loss = self.cls_criterion_2(pred, target, label)

        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = dice_loss + self.alpha*hd_loss
        elif self.alpha_strategy == "increase":
            combine_loss = dice_loss + self.alpha*hd_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*dice_loss + self.alpha*hd_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)

        return combine_loss


@LOSSES.register_module()
class BCE_SDF_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_SDF_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='SDFLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        sdf_loss = self.cls_criterion_2(pred, target, label)

        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + self.alpha*sdf_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + self.alpha*sdf_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*bce_loss + self.alpha*sdf_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)

        return combine_loss


@LOSSES.register_module()
class Dice_SDF_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(Dice_SDF_Loss, self).__init__()
        loss_mask_1=dict(type='DiceLoss')
        loss_mask_2=dict(type='SDFLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        dice_loss = self.cls_criterion_1(pred, target, label)
        sdf_loss = self.cls_criterion_2(pred, target, label)

        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = dice_loss + self.alpha*sdf_loss
        elif self.alpha_strategy == "increase":
            combine_loss = dice_loss + self.alpha*sdf_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*dice_loss + self.alpha*sdf_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)

        return combine_loss


@LOSSES.register_module()
class BCE_Dice_Loss(nn.Module):
    def __init__(self, alpha, step_alpha, max_alpha, alpha_strategy):
        super(BCE_Dice_Loss, self).__init__()
        loss_mask_1=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        loss_mask_2=dict(type='DiceLoss')
        self.cls_criterion_1 = build_loss(loss_mask_1)
        self.cls_criterion_2 = build_loss(loss_mask_2)
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.step_alpha = step_alpha
        self.alpha_strategy = alpha_strategy
        self.count_iter = 0

    def forward(self, pred, target, label):
        bce_loss = self.cls_criterion_1(pred, target, label)
        dice_loss = self.cls_criterion_2(pred, target, label)

        if self.alpha_strategy == "constant":
            # if constant: alpha, step_alpha = 0.0, max_alpha = 1.0
            combine_loss = bce_loss + self.alpha*dice_loss
        elif self.alpha_strategy == "increase":
            combine_loss = bce_loss + self.alpha*dice_loss
        elif self.alpha_strategy == "rebalance":
            combine_loss = (1-self.alpha)*bce_loss + self.alpha*dice_loss

        self.count_iter += 1
        self.alpha = update_alpha(self.alpha, self.step_alpha, self.max_alpha, self.count_iter)

        return combine_loss
