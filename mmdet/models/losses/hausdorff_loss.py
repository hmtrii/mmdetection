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
        field = np.zeros(imgs.shape)
        for roi in imgs.shape[0]:
            fg_mask = imgs[roi] > 0.5
            if fg_mask.any():
                fg_dist = edt(fg_mask)
                bg_mask = ~fg_mask
                if bg_mask.any():
                    bg_dist = edt(bg_mask)
                    field[roi] = fg_dist/np.max(fg_dist) + bg_dist/np.max(bg_dist)
                else:
                    field[roi] = fg_dist/np.max(fg_dist)

        return field

    def forward(self, pred, target, label):
        num_rois = pred.size()[0]
        inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
        pred_slice = pred[inds, label].squeeze(1)
        pred_slice = torch.sigmoid(pred_slice)

        pred_dt = torch.from_numpy(self.distance_field(pred_slice.cpu().detach().numpy())).float().cuda()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float().cuda()
        distance = pred_dt ** self.alpha + target_dt ** self.alpha
        pred_error = (pred_slice - target) ** 2
        multipled = torch.einsum("cxy,cxy->cxy", pred_error, distance)
        hd_loss = multipled.mean()

        return hd_loss

        # preds = torch.sigmoid(preds)
        # with torch.no_grad():
        #     distance = None
        #     hd_loss = None
        #     for i in range(preds.shape[0]):
        #         pred = preds[i][labels[i]]
        #         target = targets[i]
        #         preds_dt = self.distance_field(pred.cpu()).cuda().float()
        #         targets_dt = self.distance_field(target.cpu()).cuda().float()
        #         if distance is not None:
        #             distance += preds_dt ** self.alpha + targets_dt ** self.alpha
        #         else:
        #             distance = preds_dt ** self.alpha + targets_dt ** self.alpha
        #         preds_error = (pred - target) ** 2
        #         if distance.device != preds_error.device:
        #             distance = distance.to(preds_error.device).type_as(preds_error)
        #         multipled = torch.einsum("xy,xy->xy", preds_error, distance)
        #         if hd_loss is not None:
        #             hd_loss += multipled.mean()
        #         else:
        #             hd_loss = multipled.mean()
        # hd_loss = hd_loss / preds.shape[0]        
        # return hd_loss