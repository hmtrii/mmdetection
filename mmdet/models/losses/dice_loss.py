import torch
import torch.nn as nn

from ..builder import LOSSES

def dice_loss(pred, 
              target,
              label,
              smooth):
    """Calculate the Dice loss
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
    """
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    pred_slice = torch.sigmoid(pred_slice)
    intersect = 2. * torch.sum(pred_slice * target, (1,2))
    union = torch.sum(pred_slice + target, (1,2))
    dice_score = intersect / union 
    loss = 1 - dice_score
    return loss.mean()


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, label):
        return dice_loss(cls_score, label, weight)