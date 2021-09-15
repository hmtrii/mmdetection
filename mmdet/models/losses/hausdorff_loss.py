import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

from mmdet.models.builder import LOSSES
# from .utils import weight_reduce_loss

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""

@LOSSES.register_module()
class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, imgs):
        """
        imgs: [rois, w, h]
        """
        field = torch.zeros_like(imgs)

        for label in range(len(imgs)):
            fg_mask = imgs[label] > 0.5
            if fg_mask.any():
                fg_dist = edt(fg_mask)
                bg_mask = ~fg_mask
                if bg_mask.any():
                    bg_dist = edt(bg_mask)
                    field[label] = torch.tensor(fg_dist/np.max(fg_dist) + bg_dist/np.max(bg_dist))
                else:
                    field[label] = torch.tensor(fg_dist/np.max(fg_dist))

        return field

    def forward(self, preds, targets, labels):
        preds = torch.sigmoid(preds)
        with torch.no_grad():
            distance = None
            hd_loss = None
            for i in range(preds.shape[0]):
                preds_roi = preds[i]
                targets_roi = torch.zeros_like(preds_roi)
                targets_roi[labels[i]] = targets[i]

                preds_dt = self.distance_field(preds_roi.cpu()).cuda().float()
                targets_dt = self.distance_field(targets_roi.cpu()).cuda().float()

                if distance is not None:
                    distance += preds_dt ** self.alpha + targets_dt ** self.alpha
                else:
                    distance = preds_dt ** self.alpha + targets_dt ** self.alpha

                preds_error = (preds_roi - targets_roi) ** 2
                if distance.device != preds_error.device:
                    distance = distance.to(preds_error.device).type_as(preds_error)
                multipled = torch.einsum("rxy,rxy->rxy", preds_error, distance)
                if hd_loss is not None:
                    hd_loss += multipled.mean()
                else:
                    hd_loss = multipled.mean()
                    
        return hd_loss