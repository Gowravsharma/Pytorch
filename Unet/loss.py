import torch
from torch import nn
import torch.nn.functional as F
from collections import Counter

class BinaryFocalLoss(nn.Module):
  def __init__(self, weights = None, gamma = 2, reduction = 'mean', sigmoid = True):
    super().__init__()
    self.weights = weights
    self.gamma = gamma
    self.reduction = reduction
    self.sigmoid = sigmoid
    self.count = 0
  
  def forward(self, y_pred, y_true):
    if y_pred.shape != y_true.shape:
      y_pred = y_pred.squeeze(1)
    y_true = y_true.float()

    if self.weights is None and self.count == 0:
      flat_true = y_true.cpu().detach().numpy().astype(int).flatten() #.numpiy() converts torch.tensor() to a numpy array
      num_samples = y_true.numel()
      counts = Counter(flat_true)
      w0 = num_samples/(counts.get(0, 1e-6)*2)
      w1 = num_samples/(counts.get(1, 1e-6)*2)

      self.weights = torch.tensor([w0,w1], dtype = torch.float32, device = y_true.device)
      self.count += 1

    probs = torch.sigmoid(y_pred) if self.sigmoid is True else y_pred
    probs = probs.clamp(min = 1e-6, max = 1 - 1e-6)
    
    pt = probs * y_true + (1-probs) *(1-y_true)
    focal_term = (1-pt)**self.gamma

    loss = - (self.weights[0]*(1-y_true) + self.weights[1]*(y_true))*focal_term*torch.log(pt)
    
    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    else:
      return loss

class FocalLoss(nn.Module):
  def __init__(self, weights = None, gamma = 2, reduction = 'mean' ):
    super().__init__()
    self.weights = weights
    self.gamma = gamma
    self.reduction = reduction
    self.count = 0
  
  def forward(self, y_pred, y_true):
    #if y_pred.shape != y_true.shape:
    #  y_pred = y_pred.squeeze(1)
    y_true = y_true.float()

    if self.weights is None and self.count == 0:
      flat_true = y_true.view(-1).long()
      num_samples = y_true.numel()
      print('num_samples',num_samples)
      counts = Counter(flat_true)
      length = y_pred.shape[1]
      weights = []

      for i in range(length):
        weights.append(num_samples/(counts.get(i, 1e-6*length)))
      weights = torch.tensor(weights, dtype = torch.float32, device = y_true.device)  
      self.weights = weights
      self.count+= 1
    
    #print('Flat_true ',flat_true.shape)
    log_probs = F.log_softmax(y_pred, dim = 1)
    probs = torch.exp(log_probs)
    #print('probs.shape ',probs.shape)
      
    probs_flat = probs.permute(0,2,3,1).reshape(-1,probs.shape[1]) # -> shape [BxHxW, C]
    pred_probs = probs_flat[torch.arange(probs_flat.size(0)), flat_true] # -> shape [N]
    #print('pred_prob ',pred_probs.size(), pred_probs.dtype)
    
    log_pred_probs = torch.log(pred_probs + 1e-8)
    focal_term = (1-pred_probs)**self.gamma
    sample_weights = self.weights[flat_true]

    loss = -(sample_weights)*(focal_term)*(log_pred_probs)
    
    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    else:
      return loss
  

if __name__ == '__main__':
  loss_fn = BinaryFocalLoss()
  y_pred = torch.randn(8, 1, 128, 128)  # logits
  y_true = torch.randint(0, 2, (8, 128, 128))  # binary labels
  loss = loss_fn(y_pred, y_true)
  print(loss)

  loss_fn = FocalLoss()
  y_pred = torch.zeros(1, 5, 2, 2)
  y_pred[:, 2] = 10 
  y_true = torch.full((1, 2, 2), 2)

  loss = loss_fn(y_pred, y_true)
  print(loss)