import numpy as np
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
    #### hungarian.py  
from torch.autograd import Variable
from munkres import Munkres
import time
from networks_miguel import *

def visualize_attn(I, a, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def compute_metrics(result_file, gt_file, threshold=0.5):
    # groundtruth
    with open(gt_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[2]) for row in reader]
    ##### prediction (probability) #####
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = list(map(float, row))
            pred.append(np.float32(prob[1]))
    # average precision
    AP = average_precision_score(gt, pred, average='macro')
    # area under ROC curve
    AUC = roc_auc_score(gt, pred)
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = list(map(float, row))
            pred.append(np.float32(prob[1] >= threshold))
    # precision & recall
    precision_mean = precision_score(gt, pred, average='macro')
    precision_mel  = precision_score(gt, pred, average='binary', pos_label=1)
    # recall
    recall_mean = recall_score(gt, pred, average='macro')
    recall_mel  = recall_score(gt, pred, average='binary', pos_label=1)
    return [AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel]


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

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)
    return losses.squeeze()

def StableBalancedMaskedBCE(target, out, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses.squeeze()

def softIoU(target, out, e=1e-6):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    out = torch.sigmoid(out)

    # clamp values to avoid nan loss
    out = torch.clamp(out,min=e,max=1.0-e)
    target = torch.clamp(target,min=e,max=1.0-e)

    num = (out*target).sum(1,True)
    den = (out+target-out*target).sum(1,True) + e
    iou = num / den

    # set iou to 0 for masks out of range
    # this way they will never be picked for hungarian matching
    cost = (1 - iou)
    
    return cost.squeeze()

def match(masks, classes, overlaps):
    """
    Args:
        masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
        classes - list containing [true_classes, pred_classes] with shape [batch_size,T,]
        overlaps - [batch_size,T,T] - matrix of costs between all pairs
    Returns:
        t_mask_cpu - [batch_size,T,N] permuted ground truth masks
        t_class_cpu - [batch_size,T,] permuted ground truth classes
        permute_indices - permutation indices used to sort the above
    """

    overlaps = (overlaps.data).cpu().numpy().tolist()
    m = Munkres()

    t_mask, p_mask = masks
    t_class, p_class = classes

    # get true mask values to cpu as well
    t_mask_cpu = (t_mask.data).cpu().numpy()
    t_class_cpu = (t_class.data).cpu().numpy()
    # init matrix of permutations
    permute_indices = np.zeros((t_mask.size(0),t_mask.size(1)),dtype=int)
    # we will loop over all samples in batch (must apply munkres independently)
    for sample in range(p_mask.size(0)):
        # get the indexes of minimum cost
        indexes = m.compute(overlaps[sample])
        for row, column in indexes:
            # put them in the permutation matrix
            permute_indices[sample,column] = row

        # sort ground according to permutation
        t_mask_cpu[sample] = t_mask_cpu[sample,permute_indices[sample],:]
        t_class_cpu[sample] = t_class_cpu[sample,permute_indices[sample]]
    return t_mask_cpu, t_class_cpu, permute_indices

#### objectives.py   
class MaskedNLLLoss(nn.Module):
    def __init__(self, balance_weight=None):
        super(MaskedNLLLoss,self).__init__()
        self.balance_weight=balance_weight
    def forward(self, y_true, y_pred):
        costs = MaskedNLL(y_true,y_pred, self.balance_weight).view(-1,1) 
        #print(costs)
        costs = torch.mean(costs)
        return costs

class MaskedBCELoss(nn.Module):

    def __init__(self,balance_weight=None):
        super(MaskedBCELoss,self).__init__()
        self.balance_weight = balance_weight
    def forward(self, y_true, y_pred):
        costs = StableBalancedMaskedBCE(y_true,y_pred,self.balance_weight).view(-1,1)
        #costs = torch.masked_select(costs,sw.byte())
        return costs

class softIoULoss(nn.Module):

    def __init__(self):
        super(softIoULoss,self).__init__()
    def forward(self, y_true, y_pred):
        costs = softIoU(y_true,y_pred).view(-1,1)
        #print(costs.squeeze())
        costs = torch.mean(costs)
        return costs

def get_net(opt, device):
    if opt.model == 3:
        net = VGG_Deco(num_classes=2, num_structures=opt.num_structs, input_channels=64, image_size=224, att_block=opt.att_block,  enable_classifier=opt.enable_classifier, enable_decoder=opt.enable_decoder)

    elif opt.model == 4:
        net = VGG_ConvLSTM_Deco(num_classes=2, num_structures=opt.num_structs, input_channels=64, image_size=224, device=device, t=opt.t, att_block=opt.att_block,  enable_classifier=opt.enable_classifier, enable_decoder=opt.enable_decoder)
    
    if opt.model == 5:
        if opt.enable_classifier == 1:
            if opt.enable_decoder == 1:
                net = VGG_AttConvLSTM_Deco_Class(num_classes=2, num_structures=opt.num_structs, input_channels=64, image_size=224, device=device, t=opt.t, att_block=opt.att_block)
            
            else:
                net = VGG_AttConvLSTM_Class(num_classes=2, input_channels=64, image_size=224, device=device, t=opt.t, att_block=opt.att_block)
                
        elif opt.enable_decoder == 1:
            net = VGG_AttConvLSTM_Deco(num_structures=opt.num_structs, input_channels=64, image_size=224, device=device, t=opt.t, att_block=opt.att_block)
            
        else:
            print('\nError: decoder and classifier are disabled')
            return -1
    else:
        print('opt.model is not correct')
        return -1

    return net

def get_dirs(opt):
    if opt.model == 3:
        if opt.enable_decoder == 1 & opt.enable_classifier == 1:
            print('\nVGG_16 + Deco + Classifier + GAP arquitecture ...')
            path = opt.global_path + '/model_Deco_Class_GAP' + '/st_' + str(opt.num_structs)
            out_files = path + '/out_files'

        elif opt.enable_decoder == 1:
            print('\nVGG_16 + Deco arquitecture ...')
            path = opt.global_path + '/model_Deco' + '/st_' + str(opt.num_structs)
            out_files = path + '/out_files'
        
        else:
            print('\nError...')
    elif opt.model == 4:
        if opt.enable_decoder == 1 & opt.enable_classifier == 1:
            print('\nVGG_16 + ConvLSTM + Deco + Classifier arquitecture ...')
            path = opt.global_path + '/model_ConvLSTM_Deco_Class' + '/st_' + str(opt.num_structs) + 't_' + str(opt.t)
            out_files = path + '/out_files'
            
        elif opt.enable_decoder == 1:
            print('\nVGG_16 + ConvLSTM + Deco arquitecture ...')
            path = opt.global_path + '/model_ConvLSTM_Deco' + '/st_' + str(opt.num_structs) + 't_' + str(opt.t)
            out_files = path + '/out_files'
        
        else:
            print('\nError...')
    elif opt.model == 5:
        
        if opt.enable_classifier == 1:
            if opt.enable_decoder == 1:
                print('\nVGG_16 + Att + ConvLSTM + Deco + Classifier_GAP arquitecture ...')
                path = opt.global_path + '/model_AttConvLSTM_Deco_Class_GAP' + '/st_' + str(opt.num_structs) + 't_' + str(opt.t)
                out_files = path + '/out_files'
            else:
                print('\nVGG_16 + Att + ConvLSTM + ClassGap arquitecture ...')
                path = opt.global_path + '/model_AttConvLSTM_Class_GAP' + '/st_' + str(opt.num_structs) + 't_' + str(opt.t)
                out_files = path + '/out_files'
        elif opt.enable_decoder == 1:
            print('\nVGG_16 + Att + ConvLSTM + Deco arquitecture ...')
            path = opt.global_path + '/model_AttConvLSTM_Deco' + '/st_' + str(opt.num_structs) + 't_' + str(opt.t)
            out_files = path + '/out_files'
        else:
            print('\nError: decoder and classifier are disabled')
            return -1

    else:
        print('opt.model is not correct')
        return -1

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(out_files):
        os.makedirs(out_files)

    return path, out_files


def get_dirs_and_net(opt, device=None):
    net = None
    if device != None:
        net = get_net(opt, device)
        
    path, out_files = get_dirs(opt)

    return path, out_files, net