import torch
import torch.nn as nn

import numpy as np 
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

from mmdet.models.builder import LOSSES


@torch.no_grad()
def compute_sdf(target):
    """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
    """
    img_gt = target.astype(np.uint8)
    normalized_sdf = np.zeros(target.shape)
    for r in range(target.shape[0]):
        posmask = img_gt[r].astype(np.bool)
        if posmask.any():
            if np.all(posmask):
                normalized_sdf[r] = -1.0
            else:
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode="inner").astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[r] = sdf
    return normalized_sdf


def boundary_loss(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    pred_slice = torch.sigmoid(pred_slice)
    gt_sdf_npy = compute_sdf(target.cpu().numpy())
    gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda()

    if False:
        gt_dis_prob = torch.sigmoid(-1500*gt_sdf)
        intersect = torch.sum(target * gt_dis_prob)
        union = torch.sum(target + gt_dis_prob)
        dice = (2 * intersect) / union
        # dice loss should be <= 0.05 (Dice Score>0.95), which means the pre-computed SDF is right.
        # if dice < 0.95:
        print('[INFO] dice loss = ', 1 - dice.cpu().numpy())
    
    # debug
    if False:
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.subplot(121), plt.imshow(gt_sdf_npy[1,:,:]), plt.colorbar()
        plt.subplot(122), plt.imshow(np.uint8(target.cpu().numpy()[1,:,:]>0)), plt.colorbar()
        plt.show()

    multipled = pred_slice * gt_sdf
    bd_loss = multipled.mean()
    return bd_loss


@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """Boundary Loss based on signed distance map"""
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target, label):
        return boundary_loss(pred, target, label)
    