import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gama=2., size_average=True, weight=None):
        super(FocalLoss, self).__init__()
        '''
        weight: size(C)
        '''
        self.gama = gama
        self.size_average = size_average
        self.weight = weight
    def forward(self, inputs, targets):
        '''
        inputs: size(N,C)
        targets: size(N)
        '''
        log_P = -F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        P = torch.exp(log_P)
        batch_loss = -torch.pow(1-P, self.gama).mul(log_P)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def MaskedNLL(target, probs, balance_weights=None):
    # adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, T) which contains the index of the true
            class for each corresponding step.
        probs: A Variable containing a FloatTensor of size
            (batch, T, num_classes) which contains the
            softmax probability for each class.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    # # pequeño apaño para que el log funcione
    # probs[probs==0] = 10^(-6)
    log_probs = torch.log(probs)

    # print('Log probs str class loss')
    # print(log_probs)

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)

    # print(losses.shape)
    return losses.squeeze()


def softIoU(target, out, e=1e-6):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask
            after a sigmoid
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """

    # clamp values to avoid nan loss
    # out = torch.clamp(out,min=e,max=1.0-e)
    # target = torch.clamp(target,min=e,max=1.0-e)

    num = (out*target).sum(1,True)
    den = (out+target-out*target).sum(1,True) + e
    iou = num / den

    # set iou to 0 for masks out of range
    # this way they will never be picked for hungarian matching
    cost = (1 - iou)
    
    return cost.squeeze()

#### objectives.py   
class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
    def forward(self, y_true, y_pred):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1) 
        # print(costs)
        # costs = torch.mean(costs)
        return costs

class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_true, y_pred):
        costs = softIoU(y_true,y_pred).view(-1,1)
        # print(costs.squeeze())
        #costs = torch.mean(costs)
        return costs