import torch
from torch import nn

class BinaryFocalLoss(nn.Module):
  def __init__(self, alpha = 0.5, gamma = 2, reduction = 'mean'):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction
  
  def forward(self, y_pred, y_true):
    probs = torch.sigmoid(y_pred)
    probs = probs.clamp(min = 1e-6, max = 1 - 1e-6)
    
    pt = probs * y_true + (1- probs) *(1-y_true)
    focal_term = (1-pt)**self.gamma
    loss = - self.alpha*focal_term*torch.log(pt)
    
    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    else:
      return loss
