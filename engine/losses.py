import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: Logits tensor, [N, C] or [N]
        target: Binary target tensor, shape=pred's (float)
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        """
        x: Logits tensor, [N, C]
        y: Binary multi-label target tensor, [N, C]
        """
        xs_pos = x.sigmoid()
        xs_neg = 1 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        loss = -1 * (los_pos * ((1 - xs_pos) ** self.gamma_pos) + los_neg * (xs_neg ** self.gamma_neg))
        
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        return loss


def get_criterion(loss_type, reduction='mean', 
                  gamma=2.0, alpha=0.25, # Focal
                  gamma_neg=4, gamma_pos=1, clip=0.05, # Asymmetric
                  label_smoothing=0.0): # CE
    """
    loss_type (str): 'CE', 'BCE', 'Focal', 'Asymmetric'
    reduction (str): 'mean', 'sum', 'none'
    for Focal: gamma (float), alpha (float)
    for Asymmetric: gamma_neg (int), gamma_pos (int), clip (float)
    for CE: label_smoothing (float)
        
    return: nn.Module
    """
    if loss_type == 'CE':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)
        
    elif loss_type == 'BCE':
        return nn.BCEWithLogitsLoss(reduction=reduction)
        
    elif loss_type == 'Focal':
        return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)
        
    elif loss_type == 'Asymmetric':
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, reduction=reduction)

    elif loss_type == 'L1':
        return nn.L1Loss(reduction=reduction)

    elif loss_type == 'MSE':
        return nn.MSELoss(reduction=reduction)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Supported loss_type: CE, BCE, Focal, Asymmetric, L1, MSE.")
