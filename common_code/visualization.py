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
import matplotlib.pyplot as plt
from utilities import * # get_dirs_and_net, get_dirs, get_net
import seaborn as sn

def rgb2gray(rgb):
    # matlab implementation
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def revert_normalization(ima1):
    """
    ima1[0,:,:] = ima1[0,:,:]*0.0840 + 0.6820
    ima1[1,:,:] = ima1[1,:,:]*0.1140 + 0.5312
    ima1[2,:,:] = ima1[2,:,:]*0.1282 + 0.4736
    """
    mR, mG, mB = 0.6990122935081374, 0.5560426973085759, 0.5121212559185988
    sR, sG, sB = 0.1576619536490988, 0.15625517092142197, 0.17061035673581912
    ima1[0,:,:] = ima1[0,:,:]*sR + mR
    ima1[1,:,:] = ima1[1,:,:]*sG + mG
    ima1[2,:,:] = ima1[2,:,:]*sB + mB

    return ima1

def print_seg_scores(num_structs, jc, avg_prec, prec, th):
    if num_structs == 1:
        print("(Jaccard index) segmentation %.2f%%" %
        (100*jc[0]))
    elif num_structs == 2:
        print("(Jaccard index) segmentation %.2f%% skin %.2f%%" %
        (100*jc[0], 100*jc[1]))
    elif num_structs == 4:
        print("(Jaccard index) pigment_network %.2f%% negative_network %.2f%% milia_like_cyst %.2f%% streaks %.2f%%" %
        (100*jc[0], 100*jc[1], 100*jc[2], 100*jc[3]))
    elif num_structs == 5:
        print("(Jaccard index) segmentation %.2f%% pigment_network %.2f%% negative_network %.2f%% milia_like_cyst %.2f%% streaks %.2f%%" %
        (100*jc[0], 100*jc[1], 100*jc[2], 100*jc[3], 100*jc[4]))
    elif num_structs == 6:
        print("(Jaccard index) segmentation %.2f%% skin %.2f%% pigment_network %.2f%% negative_network %.2f%% milia_like_cyst %.2f%% streaks %.2f%%" %
            (100*jc[0], 100*jc[1], 100*jc[2], 100*jc[3], 100*jc[4], 100*jc[5]))
    print("(Mean average precision at different IoU thresholds) " + str(avg_prec))
    print("(Precision at different IoU thresholds)")
    for k in range(len(th)):
        print("threshold %.2f : %.2f precision" % (th[k],prec[k]))


def visualize_deco_activations(x_lists, unpools_lists, ind=0, layer=0):
    deco_outs = len(unpools_lists[0])
    for k in range(len(x_lists)):
        x_list, unpools_list = x_lists[k], unpools_lists[k]

        print('Activations ' + str(k))
        plt.figure(figsize=(10,10))
        for x in range(len(unpools_list)):
            plt.subplot(deco_outs,3,1 + 3*x)
            plt.imshow(x_list[2*x][ind, layer, :, :].cpu().detach().numpy())
            plt.title('x after skip_conn')
            plt.subplot(deco_outs,3,2 + 3*x)
            plt.imshow(unpools_list[x][ind, layer, :, :].cpu().detach().numpy())
            plt.title('after upsampling')
            plt.subplot(deco_outs,3,3 + 3*x)
            plt.imshow(x_list[2*x+1][ind, layer, :, :].cpu().detach().numpy())
            plt.title('after deco_block')
        plt.show()
        print('\n')

def visualize_activations(activation_pre_attmod, activation_post_attmod, idx=0, indexes=[0,127,255]):
    x = activation_pre_attmod.cpu().detach().numpy() # [b_s,28,28,256]
    X_tilde_list = activation_post_attmod # list of t elements. each of them with dimension [b_s,28,28,256]
    for ind in indexes:
        plt.figure(figsize=(10,60))
        plt.subplot(1,len(X_tilde_list)+1,1)
        plt.imshow(x[idx,ind,:,:].squeeze())
        plt.title('Pre att ' + str(ind))

        for t in range(len(X_tilde_list)):
            plt.subplot(1,len(X_tilde_list)+1,2+t)
            plt.title('Post att mod')
            plt.imshow(X_tilde_list[t][idx,ind,:,:].cpu().detach().numpy().squeeze())
            plt.title('Post att ' + str(t))

        plt.show()


def visualize_all_masks(inputs, masks, y_mask_perm, out_masks, y_class_perm, threshold, num_structs=6, idx=0):
    t = len(out_masks)

    masks1 = masks.clone().cpu().detach().numpy()
    masks1 = masks1[idx,:,:]

    y_mask_perm2 = y_mask_perm[idx]
    y_mask_perm2 = y_mask_perm2.reshape((y_mask_perm2.shape[0],224,224))

    ima1 = inputs[idx,:,:,:]
    ima1 = ima1.clone().cpu().detach().numpy()

    titles = get_complete_titles(num_structs)
    
    title = []
    for k in range(y_mask_perm2.shape[0]):
        title.append(titles[y_class_perm[idx][k]])

    while len(title) < num_structs:
        title.append('no assigned\nstructure')

    # Deshacer normalización para poder visualizar la imagen con los colores correctos
    ima1 = np.asarray(revert_normalization(ima1).T)

    height = 15
    width = t*height + height

    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,num_structs+1,1)
    plt.imshow(ima1)
    plt.title('Dermatoscopic image')

    for k in range(num_structs):
        # Masks
        mask2 = masks1[k,:,:]
        mask = np.zeros([224,224,3])
        for j in range(3):
            mask[:,:,j] = mask2.T

        ima = ima1 * 0.7 + mask * 0.3 

        plt.subplot(1,num_structs+1,k+2)
        plt.imshow(ima)
        plt.title('GT ' + titles[k])

    plt.show()

    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,t+1,1)
    plt.imshow(ima1)
    plt.title('Dermatoscopic image')

    for k in range(y_mask_perm2.shape[0]):
        # Masks
        mask2 = y_mask_perm2[k,:,:]
        mask = np.zeros([224,224,3])
        for j in range(3):
            mask[:,:,j] = mask2.T

        ima = ima1 * 0.7 + mask * 0.3 

        plt.subplot(1,t+1,k+2)
        plt.imshow(ima)
        plt.title('GT ' + title[k])

    plt.show()

    # NORMALIZED [0,1]
    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,t+1,1)
    plt.imshow(ima1)
    plt.title('Dermatoscopic image')

    for k in range(t):
        mask1 = out_masks[k][idx,:,:,:].clone().cpu().detach().numpy()
        mask = mask1 - np.min(mask1)
        mask = mask/np.max(mask)
        plt.subplot(1,t+1,k+2)
        plt.imshow(mask.squeeze().T)
        plt.title(title[k] + '\n' + str(np.round(np.min(mask1),4)) + ' ' + str(np.round(np.max(mask1),4)))
    plt.show()
    ###

    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,t+1,1)
    plt.imshow(ima1)
    plt.title('Dermatoscopic image')

    for k in range(t):
        mask = out_masks[k][idx,:,:,:].clone().cpu().detach().numpy()
        mask2 = mask > threshold

        mask = np.zeros([224,224,3])
        for j in range(3):
            mask[:,:,j] = mask2.reshape(224,224).T

        ima = ima1 * 0.7 + mask * 0.3

        plt.subplot(1,t+1,k+2)
        plt.imshow(ima)
        plt.title('Thr = ' + str(threshold) + '\n' + title[k])
    plt.show()

def visualize_all_attn(inputs, At, idx=0):
    t = At.shape[1]

    ima1 = inputs[idx,:,:,:]
    ima1 = ima1.cpu().detach().numpy()

    # Deshacer normalización para poder visualizar la imagen con los colores correctos
    ima1 = revert_normalization(ima1)

    # Resize the attention to match the input image size
    #up = nn.Upsample(scale_factor=8, mode='nearest')
    At_1 = At#up(At)
    At_1 = At_1[idx,:,:,:].cpu().detach().numpy()

    ## ENHANCED ATTENTION MAP ##
    height = 15
    width = t*height + height
    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,t+1,1)
    plt.imshow(ima1.T)
    plt.title('Dermatoscopic image')

    for k in range(t):
        # Attention map
        At = At_1[k,:,:]
        plt.subplot(1,t+1,k+2)
        plt.imshow(At.T)
        plt.title('Attn map T = ' + str(k))

    plt.show()

###### Attention results
def attn_max(At, image_size):
    T = At.shape[0]
    col, row = np.zeros([T,]), np.zeros([T,])
    for k in range(T):
        at_max = np.argmax(At[k])
        row[k] = at_max // image_size
        col[k] = at_max - row[k]*image_size
    
    return row, col

def visualize_attn_max(inputs, At, idx=0):
    t = At.shape[1]

    ima1 = inputs[idx,:,:,:].cpu().detach().numpy()
    image_size = ima1.shape[2]

    # Deshacer normalización para poder visualizar la imagen con los colores correctos
    ima1 = revert_normalization(ima1)

    # Resize the attention to match the input image size
    s_f = ima1.shape[2] // At.shape[2]
    up = nn.Upsample(scale_factor=s_f, mode='nearest')
    At_1 = up(At)
    At_1 = At_1[idx,:,:,:].cpu().detach().numpy()

    rows, cols = attn_max(At_1, image_size)

    ## ENHANCED ATTENTION MAP ##
    height = 15
    width = t*height + height
    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,t+1,1)
    plt.imshow(ima1.T)
    plt.title('Dermatoscopic image')

    for k in range(t):
        # Attention map
        At = At_1[k,:,:]
        plt.subplot(1,t+1,k+2)
        plt.imshow(At.T)
        plt.plot(rows[k], cols[k], '-rX')
        plt.title('Attn map T = ' + str(k))

    plt.show()

def attn_sequence(inputs, At, idx=0):
    t = At.shape[1]

    ima1 = inputs[idx,:,:,:].cpu().detach().numpy()
    image_size = ima1.shape[2]

    # Deshacer normalización para poder visualizar la imagen con los colores correctos
    ima1 = revert_normalization(ima1)

    # Resize the attention to match the input image size
    s_f = ima1.shape[2] // At.shape[2]
    up = nn.Upsample(scale_factor=s_f, mode='nearest')
    At_1 = up(At)
    At_1 = At_1[idx,:,:,:].cpu().detach().numpy()

    rows, cols = attn_max(At_1, image_size)

    height = 15
    width = 2*height
    plt.figure(figsize=(height,width))

    # Dermatoscopic image
    plt.subplot(1,2,1)
    plt.imshow(ima1.T)
    plt.title('Dermatoscopic image')

    plt.subplot(1,2,2)
    plt.imshow(ima1.T)
    plt.plot(rows, cols, '-rX')
    plt.title('Attention sequence')

    plt.show()

######### CURVES EVALUATION #########
def load_curves (file_path,checkpoint_name='lists.pth',just_loss=0):
    ch = torch.load(os.path.join(file_path, checkpoint_name))
    if just_loss == 0:
        AUC_val, AP_val = ch['AUC_val'], ch['AP_val']
        AUC_train, AP_train = ch['AUC_train'], ch['AP_train']

        loss_train, loss_val = ch['loss_train'], ch['loss_val_list']
        prec_mel_val, recall_mean_val, recall_mel_val, prec_mean_train = ch['precision_mel_val_list'], ch['recall_mean_val_list'], ch['recall_mel_val_list'], ch['precision_mean_val_list']
        prec_mel_train, recall_mean_train, recall_mel_train, prec_mean_train = ch['precision_mel_train_list'], ch['recall_mean_train_list'], ch['recall_mel_train_list'], ch['precision_mean_train_list']

        return AUC_val, AP_val, AUC_train, AP_train, loss_train, loss_val, prec_mel_val, recall_mean_val, recall_mel_val, prec_mel_train, recall_mean_train, recall_mel_train, prec_mean_train
    else:
        loss_train = ch['loss_train']
        loss_val = ch['loss_val_list']
        return loss_train, loss_val

def select_confussion_matrices (epoch,file_path,checkpoint_name='lists.pth'):
    ch = torch.load(os.path.join(file_path, checkpoint_name))

    length = len(ch['cm_train'])
    cm_t = ch['cm_train'][epoch]
    cm_v = ch['cm_val'][epoch]

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    sn.heatmap(cm_t, annot=True, cmap="flare") 
    plt.xlabel('predicted class')
    plt.ylabel('GT class')
    plt.title('Structure classification (train)')
    
    plt.subplot(122)
    sn.heatmap(cm_v, annot=True, cmap="flare") 
    plt.xlabel('predicted class')
    plt.ylabel('GT class')
    plt.title('Structure classification (val)')
    plt.show()

    return cm_t, cm_v


def load_curves_jaccard (file_path,checkpoint_name='lists.pth'):
    ch = torch.load(os.path.join(file_path, checkpoint_name))

    length = len(ch['loss_train_class'])
    jaccard_scores_train = ch['jaccard_scores_train'][:length,:]
    jaccard_scores_val = ch['jaccard_scores_val'][:length,:]
    
    return jaccard_scores_train, jaccard_scores_val

def load_curves_MAP_IoU(file_path,checkpoint_name='lists.pth'):
    ch = torch.load(os.path.join(file_path, checkpoint_name))

    length = len(ch['loss_train_class'])
    avg_prec_scores_train = ch['avg_prec_scores_train'][:length]
    avg_prec_scores_val = ch['avg_prec_scores_val'][:length]
    prec_scores_train = ch['prec_scores_train'][:length,:]
    prec_scores_val = ch['prec_scores_val'][:length,:]
    
    return avg_prec_scores_train, avg_prec_scores_val, prec_scores_train, prec_scores_val

def separated_losses (file_path,checkpoint_name='lists.pth'):
    checkpoint_lists = torch.load(os.path.join(file_path, checkpoint_name))

    loss_train_class, loss_val_class = checkpoint_lists['loss_train_class'], checkpoint_lists['loss_val_class']
    loss_train_str, loss_val_str = checkpoint_lists['loss_train_str'], checkpoint_lists['loss_val_str']

    return  loss_train_str, loss_val_str, loss_train_class, loss_val_class

def paint_scores_and_loss (file_path,checkpoint_name='lists.pth',just_loss=0):
    if just_loss == 0:
        AUC_val, AP_val, AUC_train, AP_train, loss_train, loss_val, prec_mel_val, recall_mean_val, recall_mel_val, prec_mel_train, recall_mean_train, recall_mel_train, prec_mean_train = load_curves (file_path,checkpoint_name,just_loss)
        # Loss
        plt.figure()
        plt.plot(loss_train,label='train')
        plt.plot(loss_val,label='val')
        plt.title('Loss')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.show()

        # Validation metrics
        plt.figure#(figsize=(10,20))
        plt.subplot(121)
        plt.plot(AUC_val,label='AUC')
        plt.plot(AP_val,label='AP')
        plt.title('Validation scores')
        plt.legend(loc='best')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')
        
        plt.subplot(122)
        plt.plot(AUC_train,label='AUC')
        plt.plot(AP_train,label='AP')
        plt.title('Train scores')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylim([0, 1])
        plt.show()

        # Precision and recall metrics
        plt.figure#(figsize=(10,20))
        plt.subplot(121)
        plt.plot(prec_mel_val,label='Precision mel')
        plt.plot(recall_mean_val,label='Recall mean')
        plt.plot(recall_mel_val,label='Recall mel')
        plt.plot(recall_mean_val,label='Precision mean')
        plt.title('Other validation metrics')
        plt.legend(loc='best')
        plt.ylim([0, 1])
        plt.xlabel('Epochs')

        plt.subplot(122)
        plt.plot(prec_mel_train,label='Precision mel')
        plt.plot(recall_mean_train,label='Recall mean')
        plt.plot(recall_mel_train,label='Recall mel')
        plt.plot(recall_mean_train,label='Precision mean')
        plt.title('Other train metrics')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylim([0, 1])
        plt.show()
    else:
        loss_train, loss_val = load_curves (file_path,checkpoint_name,just_loss)
        # Loss
        plt.figure()
        plt.plot(loss_train,label='train')
        plt.plot(loss_val,label='val')
        plt.title('Loss')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.show()

def paint_separated_losses (file_path,checkpoint_name='lists.pth'):
    loss_train_str, loss_val_str, loss_train_class, loss_val_class = separated_losses (file_path,checkpoint_name)

    # Training loss
    plt.figure()
    plt.plot(loss_train_str,label='structure train')
    plt.plot(loss_train_class,label='class train')
    plt.plot(loss_val_str,label='structure val')
    plt.plot(loss_val_class,label='class val')
    plt.title('Separated losses')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()

def load_structure_losses (file_path,checkpoint_name='lists.pth'):
    checkpoint_lists = torch.load(os.path.join(file_path, checkpoint_name))
    
    loss_train_str_mask, loss_train_str_class = checkpoint_lists['loss_train_str_mask'], checkpoint_lists['loss_train_str_class']
    loss_val_str_mask, loss_val_str_class = checkpoint_lists['loss_val_str_mask'], checkpoint_lists['loss_val_str_class']

    return loss_train_str_mask, loss_train_str_class, loss_val_str_mask, loss_val_str_class
    
def paint_structure_losses (file_path,checkpoint_name='lists.pth'):
    loss_train_str_mask, loss_train_str_class, loss_val_str_mask, loss_val_str_class = load_structure_losses (file_path,checkpoint_name)
    
    plt.figure()
    plt.plot(loss_train_str_class,label='str_class train')
    plt.plot(loss_train_str_mask,label='str_mask train')
    plt.plot(loss_val_str_class,label='str_class val')
    plt.plot(loss_val_str_mask,label='str_mask val')
    plt.title('Structure losses')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()

    print('\nLoss train str mask:')
    print(loss_train_str_mask)
    print('Loss train str class:')
    print(loss_train_str_class)

def paint_jaccard (file_path,checkpoint_name='lists.pth'):
    jaccard_scores_train, jaccard_scores_val = load_curves_jaccard (file_path,checkpoint_name)
    num_structs = jaccard_scores_train.shape[1]
    titles = get_complete_titles(num_structs)
    
    # Training Jaccard
    plt.figure()
    plt.subplot(121)
    for k in range(jaccard_scores_train.shape[1]):
        plt.plot(jaccard_scores_train[:,k],label=titles[k])
    plt.title('Training Jaccard')
    plt.legend(loc='best')
    plt.xlabel('Epochs')

    # Validation Jaccard
    plt.subplot(122)
    for k in range(jaccard_scores_val.shape[1]):
        plt.plot(jaccard_scores_val[:,k],label=titles[k])
    #plt.plot(jaccard_scores_val.T)#,label=titles)
    plt.title('Validation Jaccard')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()

def paint_MAP_IoU (file_path, num_structs, th_scores, checkpoint_name='lists.pth'):
    avg_prec_scores_train, avg_prec_scores_val, prec_scores_train, prec_scores_val = load_curves_MAP_IoU (file_path,checkpoint_name)
    
    # Training and Validation Jaccard
    plt.figure()
    plt.plot(avg_prec_scores_train,label='train')
    plt.plot(avg_prec_scores_val,label='val')
    plt.title('Mean average precision at different IoU thresholds')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()

    # Training and Validation Jaccard
    plt.figure()
    plt.subplot(121)
    for k in range(len(th_scores)):
        plt.plot(prec_scores_train[:,k],label=['train %.2f' % (th_scores[k])])
    plt.title('Precision at different IoU thresholds\n(train)')
    plt.legend(loc='best')
    plt.xlabel('Epochs')

    # Training and Validation Jaccard
    plt.subplot(122)
    for k in range(len(th_scores)):
        plt.plot(prec_scores_val[:,k],label=['val %.2f' % (th_scores[k])])
    plt.title('Precision at different IoU thresholds\n(val)')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
    

def eval_curves(opt,checkpoint_name='lists.pth'):
    __, out_files = get_dirs(opt)
    paint_scores_and_loss (out_files,checkpoint_name)
    AUC_val, AP_val, AUC_train, AP_train, loss_train, loss_val, prec_mel_val, recall_mean_val, recall_mel_val, prec_mel_train, recall_mean_train, recall_mel_train, prec_mean_train = load_curves (out_files,checkpoint_name)
    jaccard_scores_train, jaccard_scores_val = load_curves_jaccard (out_files,checkpoint_name)
  
    jaccard_scores_train1 = jaccard_scores_train.numpy().tolist()
    jaccard_scores_train = np.mean(jaccard_scores_train1,axis=1).tolist()
    jaccard_scores_val1 = jaccard_scores_val.numpy().tolist()
    jaccard_scores_val = np.mean(jaccard_scores_val1,axis=1).tolist()

    #paint_separated_losses(out_files,checkpoint_name)
    loss_train_str, loss_val_str, loss_train_class, loss_val_class = separated_losses (out_files,checkpoint_name)

    # Training loss
    plt.figure()
    plt.subplot(121)
    plt.plot(loss_train_str,label='structure train')
    plt.plot(loss_train_class,label='class train')
    plt.plot(loss_val_str,label='structure val')
    plt.plot(loss_val_class,label='class val')
    plt.title('Separated losses')
    plt.legend(loc='best')
    plt.xlabel('Epochs')

    #paint_structure_losses (out_files,checkpoint_name)
    loss_train_str_mask, loss_train_str_class, loss_val_str_mask, loss_val_str_class = load_structure_losses (out_files,checkpoint_name)
    plt.subplot(122)    
    plt.plot(loss_train_str_class,label='str_class train')
    plt.plot(loss_train_str_mask,label='str_mask train')
    plt.plot(loss_val_str_class,label='str_class val')
    plt.plot(loss_val_str_mask,label='str_mask val')
    plt.title('Structure losses')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()
   

    paint_jaccard (out_files,checkpoint_name)
    # Results
    idx = AUC_val.index(np.max(AUC_val))
    idx_class = idx
    print('Best classification epoch:'+str(idx))
    print('AUC_train:'+str(AUC_train[idx]))
    print('AP_train:'+str(AP_train[idx]))
    print('AUC_val:'+str(AUC_val[idx]))
    print('AP_val:'+str(AP_val[idx]))
    print('Jaccard score train:'+str(jaccard_scores_train1[idx]))
    print('Jaccard score val:'+str(jaccard_scores_val1[idx]))
    print('Jaccard mean score train:'+str(jaccard_scores_train[idx]))
    print('Jaccard mean score val:'+str(jaccard_scores_val[idx]))
    

    idx = jaccard_scores_val.index(max(jaccard_scores_val))
    idx_deco = idx
    print('\nBest jaccard epoch:'+str(idx))
    print('AUC_train:'+str(AUC_train[idx]))
    print('AP_train:'+str(AP_train[idx]))
    print('AUC_val:'+str(AUC_val[idx]))
    print('AP_val:'+str(AP_val[idx]))
    print('Jaccard score train:'+str(jaccard_scores_train1[idx]))
    print('Jaccard score val:'+str(jaccard_scores_val1[idx]))
    print('Jaccard mean score train:'+str(jaccard_scores_train[idx]))
    print('Jaccard mean score val:'+str(jaccard_scores_val[idx]))

    AUC_val, jaccard_scores_val = np.array(AUC_val), np.array(jaccard_scores_val)
    scores = np.mean(np.vstack([AUC_val, jaccard_scores_val]),axis=0).tolist()
    idx = scores.index(np.max(scores))

    print('\nBest global epoch:'+str(idx))
    print('AUC_train:'+str(AUC_train[idx]))
    print('AP_train:'+str(AP_train[idx]))
    print('AUC_val:'+str(AUC_val[idx]))
    print('AP_val:'+str(AP_val[idx]))
    print('Jaccard score train:'+str(jaccard_scores_train1[idx]))
    print('Jaccard score val:'+str(jaccard_scores_val1[idx]))
    print('Jaccard mean score train:'+str(jaccard_scores_train[idx]))
    print('Jaccard mean score val:'+str(jaccard_scores_val[idx]))

    print('Confussion matrices of structure classification:')
    cm_t, cm_v = select_confussion_matrices (idx_class,out_files,checkpoint_name)

    return idx_class, idx_deco, idx

def best_scores(opt,checkpoint_name='lists.pth'):
    __, out_files = get_dirs(opt)
    AUC_val, AP_val, AUC_train, AP_train, loss_train, loss_val, prec_mel_val, recall_mean_val, recall_mel_val, prec_mel_train, recall_mean_train, recall_mel_train, prec_mean_train = load_curves (out_files,checkpoint_name)
    jaccard_scores_train, jaccard_scores_val = load_curves_jaccard (out_files,checkpoint_name)
  
    jaccard_scores_train1 = jaccard_scores_train.numpy().tolist()
    jaccard_scores_train = np.mean(jaccard_scores_train1,axis=1).tolist()
    jaccard_scores_val1 = jaccard_scores_val.numpy().tolist()
    jaccard_scores_val = np.mean(jaccard_scores_val1,axis=1).tolist()

    # Results
    idx = AUC_val.index(np.max(AUC_val))
    idx_class = idx
    print('Best classification epoch:'+str(idx))
    print('AUC_train:'+str(AUC_train[idx]) + 'AP_train:'+str(AP_train[idx]))
    print('AUC_val:'+str(AUC_val[idx]) + 'AP_val:'+str(AP_val[idx]))
    print('Jaccard score train:'+str(jaccard_scores_train1[idx]))
    print('Jaccard score val:'+str(jaccard_scores_val1[idx]))
    print('Jaccard mean score train:'+str(jaccard_scores_train[idx]))
    print('Jaccard mean score val:'+str(jaccard_scores_val[idx]))
    

    idx = jaccard_scores_val.index(max(jaccard_scores_val))
    idx_deco = idx
    print('\nBest jaccard epoch:'+str(idx))
    print('AUC_train:'+str(AUC_train[idx]))
    print('AP_train:'+str(AP_train[idx]))
    print('AUC_val:'+str(AUC_val[idx]))
    print('AP_val:'+str(AP_val[idx]))
    print('Jaccard score train:'+str(jaccard_scores_train1[idx]))
    print('Jaccard score val:'+str(jaccard_scores_val1[idx]))
    print('Jaccard mean score train:'+str(jaccard_scores_train[idx]))
    print('Jaccard mean score val:'+str(jaccard_scores_val[idx]))

    AUC_val, jaccard_scores_val = np.array(AUC_val), np.array(jaccard_scores_val)
    scores = np.mean(np.vstack([AUC_val, jaccard_scores_val]),axis=0).tolist()
    idx = scores.index(np.max(scores))

def eval_curves_deco(opt, checkpoint_name='lists.pth'):
    __, out_files = get_dirs(opt)
    jaccard_scores_train, jaccard_scores_val = load_curves_jaccard (out_files,checkpoint_name)

    jaccard_scores_train1 = jaccard_scores_train.numpy().tolist()
    jaccard_scores_train = np.mean(jaccard_scores_train1,axis=1).tolist()
    jaccard_scores_val1 = jaccard_scores_val.numpy().tolist()
    jaccard_scores_val = np.mean(jaccard_scores_val1,axis=1).tolist()

    loss_train_str, loss_val_str, loss_train_class, loss_val_class = separated_losses (out_files,checkpoint_name)

    # Training and validation loss
    plt.figure()
    plt.plot(loss_train_str,label='train')
    plt.plot(loss_val_str,label='val')
    plt.title('Structure loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.show()

    paint_jaccard (out_files,checkpoint_name)

    # Results
    idx = jaccard_scores_val.index(max(jaccard_scores_val))
    idx_deco = idx
    print('\nBest jaccard epoch:'+str(idx))

    print('Jaccard score train:'+str(jaccard_scores_train1[idx]))
    print('Jaccard score val:'+str(jaccard_scores_val1[idx]))
    print('Jaccard mean score train:'+str(jaccard_scores_train[idx]))
    print('Jaccard mean score val:'+str(jaccard_scores_val[idx]))

    return idx_deco

def paint_it_losses (loss_class, loss_struct, loss_struct_class, loss_struct_mask):
    total_loss = np.array(loss_class) + np.array(loss_struct) + np.array(loss_struct_class) + np.array(loss_struct_mask)
    
    plt.figure()
    plt.subplot(121)
    plt.plot(loss_class, label='Classification')
    plt.plot(loss_struct, label='Structure')
    plt.plot(loss_struct_class, label='Structure class')
    plt.plot(loss_struct_mask, label='Structure seg')
    plt.title('Loss through iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.subplot(122)
    plt.plot(total_loss)
    plt.title('Total loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()