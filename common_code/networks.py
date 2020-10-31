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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from networks2 import *

class AttentiveModel(nn.Module):
    def __init__(self, hidden_dim, dim_feats, device):
        super(AttentiveModel, self).__init__()

        # Dimension and other variables
        self.device = device
        self.hidden_dim = hidden_dim # input and output channels are the same (256)
        self.dim_feats = dim_feats # square images of 28x28 dim_feats = 28
        self.padding_mode = 'reflect' # 'reflect', 'replicate', 'circular'

        # Attentive model
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

        self.Va = nn.Conv2d(self.hidden_dim, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, padding_mode=self.padding_mode)
        self.Va.weight.data.fill_(0.0) # únicos pesos que se inicializan a 0

        self.Wa = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        self.Ua = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        nn.init.orthogonal_(self.Ua.weight)

    # x - tensor (X)
    # Hprev - previous hidden state (Ht-1)
    def forward(self, x, Hprev=None):
        ### Initialization ###
        if Hprev==None:
            Hprev = torch.zeros([x.shape[0], self.hidden_dim, self.dim_feats, self.dim_feats], device=self.device)

        ### Attentive model ###
        wt = self.Wa(x)
        ut = self.Ua(Hprev)
        tant = self.tanh(wt + ut)
        Zt = self.Va(tant) 
        At = torch.reshape(self.softmax(torch.reshape(Zt,[x.shape[0], 1, int(self.dim_feats**2)])), [x.shape[0], 1, self.dim_feats, self.dim_feats])
        Xt = x * At # Element wise multiplication

        # print('x')
        # plt.figure()
        # for i in np.arange(4):
        #   plt.subplot(2,2,i+1)
        #   plt.imshow(x[i,0,:,:].cpu().detach().numpy().squeeze())
        # plt.show()

        # print('At')
        # plt.figure()
        # for i in np.arange(4):
        #   plt.subplot(2,2,i+1)
        #   plt.imshow(At[i,0,:,:].cpu().detach().numpy().squeeze())
        #   print(i+1)
        #   print(np.unique(At[i,0,:,:].cpu().detach().numpy().squeeze()))
        # plt.show()

        # print('Xt')
        # plt.figure()
        # for i in np.arange(4):
        #   plt.subplot(2,2,i+1)
        #   plt.imshow(Xt[i,0,:,:].cpu().detach().numpy().squeeze())
        # plt.show()
        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(ut[0,0,:,:].cpu().detach().numpy().squeeze())
        # plt.title('ut')
        # plt.subplot(1,2,2)
        # plt.imshow(Hprev[0,0,:,:].cpu().detach().numpy().squeeze())
        # plt.title('Hprev')
        # plt.show()
        return Xt, At, Zt, wt, ut, tant

class ConvLSTM(nn.Module):
    def __init__(self, hidden_dim, hidden_dim_out, dim_feats, device):
        super(ConvLSTM, self).__init__()

        # Dimension and other variables
        self.device = device
        self.hidden_dim = hidden_dim # input and output channels are the same (256)
        self.dim_feats = dim_feats # square images of 28x28 dim_feats = 28
        self.padding_mode = 'reflect' # 'reflect', 'replicate', 'circular'
        self.hidden_dim_out = hidden_dim_out

        # ConvLSTM
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.Wi = nn.Conv2d(self.hidden_dim, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        self.Ui = nn.Conv2d(self.hidden_dim_out, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        
        self.Wf = nn.Conv2d(self.hidden_dim, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        self.Uf = nn.Conv2d(self.hidden_dim_out, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        
        self.Wo = nn.Conv2d(self.hidden_dim, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        self.Uo = nn.Conv2d(self.hidden_dim_out, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        
        self.Wc = nn.Conv2d(self.hidden_dim, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        self.Uc = nn.Conv2d(self.hidden_dim_out, self.hidden_dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode=self.padding_mode)
        
        # init conv weights of hidden states
        nn.init.orthogonal_(self.Ui.weight)
        nn.init.orthogonal_(self.Uf.weight)
        nn.init.orthogonal_(self.Uo.weight)
        nn.init.orthogonal_(self.Uc.weight)

    # x - tensor (X)
    # Hprev - previous hidden state (Ht-1)
    # Cprev - previous memory cell (Ct-1)
    def forward(self, x, Hprev=None, Cprev=None):
        ### Initialization ###
        if Cprev==None:
            Cprev = torch.zeros([x.shape[0], self.hidden_dim_out, self.dim_feats, self.dim_feats], device=self.device)
        if Hprev==None:
            Hprev = torch.zeros([x.shape[0], self.hidden_dim_out, self.dim_feats, self.dim_feats], device=self.device)

        ### ConvLSTM ###
        # Gates
        It = self.sigmoid(self.Wi(x) + self.Ui(Hprev))
        Ft = self.sigmoid(self.Wf(x) + self.Uf(Hprev))
        Ot = self.sigmoid(self.Wo(x) + self.Uo(Hprev))

        # Candidate memory
        Gt = self.tanh(self.Wc(x) + self.Uc(Hprev))

        # Memory cell
        Ct = Ft * Cprev + It * Gt

        # Hidden state
        Ht = Ot * self.tanh(Ct)
        return Ht, Ct

class Encoder(nn.Module):
    def __init__(self, bn=False, att_block=3):
        super(Encoder, self).__init__()

        # Dimension and other variables
        self.att_block = att_block
        self.kernel_size = 2
        self.stride = 2
        self.bn = bn

        if self.bn == True:
            # VGG16 batchnorm
            net = models.vgg16_bn(pretrained=True)
            self.conv_list = nn.ModuleList()
            self.conv_list.append(nn.Sequential(*list(net.features.children())[0:6]))
            self.conv_list.append(nn.Sequential(*list(net.features.children())[7:13]))
            self.conv_list.append(nn.Sequential(*list(net.features.children())[14:23]))
            if self.att_block == 4:
                self.conv_list.append(nn.Sequential(*list(net.features.children())[24:33]))
        
        else:
            # VGG16
            net = models.vgg16(pretrained=True)
            self.conv_list = nn.ModuleList()
            self.conv_list.append(nn.Sequential(*list(net.features.children())[0:4]))
            self.conv_list.append(nn.Sequential(*list(net.features.children())[5:9]))
            self.conv_list.append(nn.Sequential(*list(net.features.children())[10:16]))
            if self.att_block == 4:
                self.conv_list.append(nn.Sequential(*list(net.features.children())[17:23]))
        
    def forward(self, x):
        # Encoder
        dim_list, ind_list, skip_list = [], [], []

        for k in range(self.att_block):
            dim_list.append(x.size())
            block = self.conv_list[k](x)
            x, ind = F.max_pool2d(block, self.kernel_size, self.stride, return_indices=True)
            ind_list.append(ind)
            skip_list.append(x)

        return [x, dim_list, ind_list, skip_list]

class Decoder(nn.Module):
    def __init__(self, input_channels, num_structures, output_channels=1, skip=True, bn=False, n_blocks=3):
        super(Decoder, self).__init__()

        # Dimension and other variables
        self.input_channels = input_channels
        self.n_blocks = n_blocks # 3 or 4
        self.output_channels = output_channels # 1 channel per class (structures + skin/background)
        self.num_structures = num_structures
        self.total_cls = 0
        self.bn = bn
        self.skip = skip
        self.padding_mode = 'reflect'

        if self.n_blocks == 4:
            self.n_ch = [self.input_channels, self.input_channels//2, self.input_channels//4, self.input_channels//8, self.output_channels]
        else:
            self.n_ch = [self.input_channels, self.input_channels//2, self.input_channels//4, self.output_channels]

        self.total_cls = np.sum(np.array(self.n_ch[0:self.n_blocks]))

        # MaxPool/MaxUnPool - segmentator
        self.stride = 2
        self.kernel_size = 2

        # Decoder layers
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block_deco_list = nn.ModuleList()
        k = 0
        if self.bn == True:
            if self.n_blocks == 4:
                # Decoder Stage - 4
                self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                      nn.BatchNorm2d(self.in_ch[k]), nn.ReLU(),
                                                      nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                      nn.BatchNorm2d(self.in_ch[k]), nn.ReLU(),
                                                      nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                      nn.BatchNorm2d(self.out_ch[k+1]), nn.ReLU()]))
                k += 1
            
            # Decoder Stage - 3
            self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.BatchNorm2d(self.n_ch[k]), nn.ReLU(),
                                                  nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.BatchNorm2d(self.n_ch[k]), nn.ReLU(),
                                                  nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.BatchNorm2d(self.n_ch[k+1]), nn.ReLU()]))
            k += 1

            # Decoder Stage - 2
            self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.BatchNorm2d(self.n_ch[k]), nn.ReLU(),
                                                  nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.BatchNorm2d(self.n_ch[k+1]), nn.ReLU()]))
            k += 1
            
            # Decoder Stage - 1
            self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.BatchNorm2d(self.n_ch[k]), nn.ReLU(),
                                                  nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)]))
        else:
            if self.n_blocks == 4:
                # Decoder Stage - 4
                self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.ReLU(), nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.ReLU(), nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                                  nn.ReLU()]))
                k += 1
        
            # Decoder Stage - 3
            self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                              nn.ReLU(), nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                              nn.ReLU(), nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                              nn.ReLU()]))
            k += 1

            # Decoder Stage - 2
            self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                              nn.ReLU(), nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                              nn.ReLU()]))
            k += 1
            
            # Decoder Stage - 1
            self.block_deco_list.append(nn.Sequential(*[nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
                                              nn.ReLU(), nn.Conv2d(in_channels=self.n_ch[k], out_channels=self.n_ch[k+1], kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)]))
        
        # maxpooling
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # Activation for the mask output
        self.sigmoid = nn.Sigmoid()

        # Struct classifier
        self.struct_cls = nn.Sequential(*[nn.Linear(in_features=int(self.total_cls), out_features=int(self.total_cls//2), bias=True),
                                            nn.Linear(in_features=int(self.total_cls//2), out_features=int(self.num_structures), bias=True)])
        #self.struct_cls = nn.Linear(in_features=int(self.total_cls), out_features=int(self.num_structures), bias=True)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x, list_dims, list_inds, skip_list):
        x_list, x_up_list = [], []
        N = x.shape[0]

        #print('\n\nStart different t')
        idx = 0
        """
        plt.figure(figsize=(10,60))
        for h in range(N):
            plt.subplot(1,N,h+1)   
            plt.imshow(x[h,idx,:,:].detach().cpu().numpy().squeeze()) 
            plt.title('x after\nattention block')
        plt.show()
        """
        # Decoder
        cls = []
        for ind in range(self.n_blocks):
            k = self.n_blocks - (ind + 1)

            if self.skip == True:
                x = (x + skip_list[k])/2
            x_list.append(x)

            x_up = self.up(x)
            cls.append(self.maxpool(x).view(N,-1))
            x = self.block_deco_list[ind](x_up)
            x_list.append(x)
            x_up_list.append(x_up) 

            """
            plt.figure(figsize=(10,60))
            for h in range(N):
                plt.subplot(1,N,h+1)   
                plt.imshow(x_list[-2][h,idx,:,:].detach().cpu().numpy().squeeze()) 
                plt.title('x + skip')
            plt.show()
            plt.figure(figsize=(10,60))
            for h in range(N):
                plt.subplot(1,N,h+1)      
                plt.imshow(x_up_list[-1][h,idx,:,:].detach().cpu().numpy().squeeze()) 
                plt.title('upsampling')
            plt.show()
            plt.figure(figsize=(10,60))
            for h in range(N):
                plt.subplot(1,N,h+1)      
                plt.imshow(x_list[-1][h,idx,:,:].detach().cpu().numpy().squeeze()) 
                plt.title('decoder')
            plt.show()
            """

        # Obtained mask
        x = self.sigmoid(x)
        # print('decoder')
        # print(x)
        """
        for h in range(N):
            plt.subplot(1,N,h+1)      
            plt.imshow(x[h,idx,:,:].detach().cpu().numpy().squeeze()) 
            plt.title('decoder')
        plt.show()
        """

        cls_cat = torch.cat(cls, dim=1)
        out_classes = self.struct_cls(cls_cat)
        out_classes = self.activation(out_classes)

        return x, out_classes, x_list, x_up_list

class Classifier(nn.Module):
    def __init__(self, num_classes, attn_channels, t, bn=False, att_block=3):
        super(Classifier, self).__init__()

        # Dimension and other variables
        self.att_block = att_block
        self.num_classes = num_classes # 2 - melanoma or not
        self.kernel_size = 2
        self.stride = 2
        self.t = t 
        self.attn_channels = attn_channels # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.bn = bn
        
        # last VGG16 layers
        self.block_list = nn.ModuleList()
        if self.bn == True:
            # last VGG16 BN layers
            net = models.vgg16_bn(pretrained=True)
            if self.att_block == 3:
                self.block_list.append(nn.Sequential(*list(net.features.children())[24:33]))
            self.block_list.append(nn.Sequential(*list(net.features.children())[34:43]))

        else:
            net = models.vgg16(pretrained=False)
            if self.att_block == 3:
                self.block_list.append(nn.Sequential(*list(net.features.children())[17:23]))
            self.block_list.append(nn.Sequential(*list(net.features.children())[24:30]))
        

        # GAP
        self.pool = nn.AvgPool2d(7, stride=1)
        self.in_feats = 512 + self.t * self.attn_channels # pool5 + atenciones pool3
        self.cls = nn.Linear(in_features=self.in_feats, out_features=self.num_classes, bias=True)
        
        # initialize
        nn.init.normal_(self.cls.weight, 0., 0.01)
        nn.init.constant_(self.cls.bias, 0.)
        
    def forward(self, x, list_xt):
        # Classifier
        for k in range(len(self.block_list)):
            block = self.block_list[k](x)   # /8 
            x = F.max_pool2d(block, self.kernel_size, self.stride) 
            
        N, __, __, __ = x.size()
        # GAP
        g_list = []
        g_list.append(self.pool(x).view(N,512))
        for k in range(self.t):
            g_list.append(F.adaptive_avg_pool2d(list_xt[k], (1,1)).view(N,-1))
        g_hat = torch.cat(g_list, dim=1) # batch_size x C
        out = self.cls(g_hat)

        return out

class VGG_AttConvLSTM_Deco_Class(nn.Module):
    def __init__(self, num_classes, num_structures, input_channels, image_size, device, t, skip=True, bn=False, att_block=3):
        super(VGG_AttConvLSTM_Deco_Class, self).__init__()

        # Dimension and other variables
        self.device = device
        self.input_channels = input_channels
        self.image_size = image_size # 224 (224x224)
        self.att_block = att_block
        self.num_classes = num_classes # 2 - melanoma or not
        self.num_structures = num_structures
        self.kernel_size = 2
        self.stride = 2
        self.t = t 
        self.bn = bn
        self.skip = skip

        self.attn_channels = int(self.input_channels*(2**(self.att_block-1))) # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.attn_size = int(self.image_size/(2**self.att_block)) # square images of 28x28 dim_feats = 28

        # Modules
        self.encoder = Encoder(self.bn, self.att_block)
        self.attentive_model = AttentiveModel(self.attn_channels, self.attn_size, self.device)
        self.convLSTM = ConvLSTM(self.attn_channels, self.attn_channels, self.attn_size, self.device)
        self.deco = Decoder(input_channels=self.attn_channels, num_structures=self.num_structures, output_channels=1, skip=self.skip, bn=self.bn, n_blocks=self.att_block)
        self.classifier = Classifier(num_classes=self.num_classes, attn_channels=self.attn_channels, t=self.t, bn=self.bn, att_block=self.att_block)
        
    def forward(self, x):
        # Encoder
        x, dim_list, ind_list, skip_list = self.encoder(x)

        # AttentiveConvLSTM
        At = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        Zt = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        wt = []
        ut = []
        tant = []
        Ht, Ct = None, None
        Hsave, Csave = [], []
        X_deco, out_class, X_tilde_aux = [], [], []     
        x_list, unpool_list = [], []
        saved_x = x   
        for k in range(self.t):
            X_tilde_t, At_aux, Zt_aux, wt_aux, ut_aux, tant_aux = self.attentive_model(x, Ht)
            At[:,k,:,:] = At_aux.squeeze()
            Zt[:,k,:,:] = Zt_aux.squeeze()

            wt.append(wt_aux)
            ut.append(ut_aux)
            tant.append(tant_aux)
            Ht, Ct = self.convLSTM(X_tilde_t, Ht, Ct)
            
            X_tilde_aux.append(X_tilde_t)
            Hsave.append(Ht)
            Csave.append(Ct)

            # Decoder
            #X_deco_aux, out_class_aux, x_list_aux, unpool_list_aux = self.deco(X_tilde_t, dim_list, ind_list, skip_list)
            X_deco_aux, out_class_aux, x_list_aux, unpool_list_aux = self.deco(Ht, dim_list, ind_list, skip_list)
            
            # Save values
            out_class.append(out_class_aux)
            X_deco.append(X_deco_aux)
            x_list.append(x_list_aux)
            unpool_list.append(unpool_list_aux)

        # Classifier
        out = self.classifier(X_tilde_t, X_tilde_aux)

        return [out, At, X_deco, out_class, X_tilde_aux, saved_x, Zt, wt, ut, tant, x_list, unpool_list]



###############################################################################################################33
class Decoder_article(nn.Module):
    def __init__(self, input_channels, size_input, num_structures, device, output_channels=1, n_blocks=3):
        super(Decoder_article, self).__init__()

        # Dimension and other variables
        self.input_channels = input_channels
        self.n_blocks = n_blocks # 3 or 4
        self.output_channels = output_channels # 1 channel per class (structures + skin/background)
        self.size_input = size_input # 28 (28x28)
        self.num_structures = num_structures
        self.total_cls = 0
        self.device = device

        # MaxPool/MaxUnPool - segmentator
        self.stride = 2
        self.kernel_size = 2

        # Decoder layers
        if self.n_blocks == 4:
            self.list_channels_in = [self.input_channels, self.input_channels//2, self.input_channels//4, self.input_channels//8]
            self.list_channels_out = [self.input_channels//2, self.input_channels//4, self.input_channels//8, self.input_channels//16]
        else: 
            self.list_channels_in = [self.input_channels, self.input_channels//2, self.input_channels//4]
            self.list_channels_out = [self.input_channels//2, self.input_channels//4, self.input_channels//8]
        self.list_size_inputs = [self.size_input, self.size_input*2, self.size_input*4, self.size_input*8]
        self.total_cls = np.sum(np.array(self.list_channels_out))
        

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.convLSTM_list = nn.ModuleList()
        for k in range(self.n_blocks):
            self.convLSTM_list.append(ConvLSTM(hidden_dim=self.list_channels_in[k], hidden_dim_out=self.list_channels_out[k], dim_feats=self.list_size_inputs[k], device=self.device))
            ###self.maxpool4 = nn.MaxPool2d(kernel_size=self.list_size_inputs[k],stride=1,padding=0,dilation=1,return_indices=False,ceil_mode =False)
        
        # Final convolution for segmentation part
        self.conv_out = nn.Conv2d(self.list_channels_out[k], self.output_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.sigmoid = nn.Sigmoid()

        # maxpooling
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # Struct classifier
        self.struct_cls = nn.Linear(in_features=int(self.total_cls), out_features=int(self.num_structures), bias=True)
        if self.num_structures == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x, list_dims, list_inds, skip_list, Ht_prev, Ct_prev):
        x_list, unpool_list = [], []
        N = x.shape[0]

        # Decoder
        cls = []
        convlstm_input = skip_list[-1]
        Ht_list, Ct_list = [], []
        for ind in range(self.n_blocks):
            if Ht_prev == None:
                Ht, Ct = self.convLSTM_list[ind](convlstm_input)
            else:
                Ht, Ct = self.convLSTM_list[ind](convlstm_input, Ht_prev[ind], Ct_prev[ind])

            cls.append(self.maxpool(Ht).view(N,-1))
            k = (self.n_blocks - (ind + 2))
            Ht_list.append(Ht)
            Ct_list.append(Ct)  
            Ht = self.up(Ht)
            
            """
            print('ConvLSTM input block ' + str(ind))
            plt.figure()
            for i in np.arange(N):
              plt.subplot(1,N,i+1)
              plt.imshow(convlstm_input[i,0,:,:].cpu().detach().numpy().squeeze())
            plt.show()

            print('Ht')
            plt.figure()
            for i in np.arange(N):
              plt.subplot(1,N,i+1)
              plt.imshow(Ht[i,0,:,:].cpu().detach().numpy().squeeze())
            plt.show()

            print('Ct')
            plt.figure()
            for i in np.arange(N):
              plt.subplot(1,N,i+1)
              plt.imshow(Ct[i,0,:,:].cpu().detach().numpy().squeeze())
            plt.show()
            """

            if k >= 0:
                convlstm_input = (Ht + skip_list[k])/2
            else:
                break
                  
        # Obtained mask
        x = self.conv_out(Ht)
        x = self.sigmoid(x) # en el artículo no se aplica softmax

        """
        print('obtained mask')
        plt.figure()
        for i in np.arange(N):
          plt.subplot(1,N,i+1)
          plt.imshow(x[i,0,:,:].cpu().detach().numpy().squeeze())
        plt.show()
        """
        
        # Structure 
        cls_cat = torch.cat(cls, dim=1)
        out_classes = self.struct_cls(cls_cat)
        out_classes = self.activation(out_classes)
        #print(out_classes)

        return x, out_classes, Ht_list, Ct_list#, x_list, unpool_list

class Classifier_article(nn.Module):
    def __init__(self, num_classes, attn_channels, t, att_block=3):
        super(Classifier_article, self).__init__()

        # Dimension and other variables
        self.att_block = att_block
        self.num_classes = num_classes # 2 - melanoma or not
        self.kernel_size = 2
        self.stride = 2
        self.t = t 
        self.attn_channels = attn_channels # input and output channels are the same (64*3 = 256, if it is after pool-3)

        # # last VGG16 BN layers
        # net = models.vgg16_bn(pretrained=True)
        # if self.att_block == 3:
        #     self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
        # self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
        
        # last VGG16 layers
        net = models.vgg16(pretrained=True)
        if self.att_block == 3:
            self.conv_block4 = nn.Sequential(*list(net.features.children())[17:23])
        self.conv_block5 = nn.Sequential(*list(net.features.children())[24:30])
        
        # GAP
        self.pool = nn.AvgPool2d(7, stride=1)
        self.in_feats = 512 + 128 + 64 + 32 # pool5 + atenciones pool3
        self.cls = nn.Linear(in_features=self.in_feats, out_features=self.num_classes, bias=True)
        
        # initialize
        nn.init.normal_(self.cls.weight, 0., 0.01)
        nn.init.constant_(self.cls.bias, 0.)
        
    def forward(self, x, list_xt):
        # Classifier
        if self.att_block == 3:
            block = self.conv_block4(x)   # /8 
            x = F.max_pool2d(block, self.kernel_size, self.stride) 
            
        block = self.conv_block5(x)   # /16
        x = F.max_pool2d(block, self.kernel_size, self.stride) 
            
        N, __, __, __ = x.size()
        # GAP
        g_list = []
        g_list.append(self.pool(x).view(N,512))
        for k in range(len(list_xt)):
            g_list.append(F.adaptive_avg_pool2d(list_xt[k], (1,1)).view(N,-1))
        g_hat = torch.cat(g_list, dim=1) # batch_size x C
        out = self.cls(g_hat)

        return out

class VGG_article(nn.Module):
    def __init__(self, num_classes, num_structures, input_channels, image_size, device, t, bn, att_block=3):
        super(VGG_article, self).__init__()

        # Dimension and other variables
        self.device = device
        self.input_channels = input_channels
        self.image_size = image_size # 224 (224x224)
        self.att_block = att_block
        self.num_classes = num_classes # 2 - melanoma or not
        self.num_structures = num_structures
        self.kernel_size = 2
        self.stride = 2
        self.t = t 
        self.bn = bn

        self.attn_channels = int(self.input_channels*(2**(self.att_block-1))) # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.attn_size = int(self.image_size/(2**self.att_block)) # square images of 28x28 dim_feats = 28

        # Encoder
        self.encoder = Encoder(self.bn, self.att_block)

        # Decoder
        self.deco = Decoder_article(input_channels=self.attn_channels, size_input=self.attn_size, num_structures=self.num_structures, output_channels=1, n_blocks=self.att_block, device=self.device)
        
        # Classifier
        self.classifier = Classifier_article(num_classes=self.num_classes, attn_channels=self.attn_channels, t=self.t, att_block=self.att_block)
        
    def forward(self, x):
        # Encoder
        x, dim_list, ind_list, skip_list = self.encoder(x)

        # AttentiveConvLSTM
        At = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        Zt = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        wt = []
        ut = []
        tant = []
        Ht, Ct = None, None
        Ht_list = []
        Hsave, Csave = [], []
        X_deco, out_class, X_tilde_aux = [], [], []     
        x_list, unpool_list = [], []
        saved_x = x   
        for k in range(self.t):
            # Decoder
            X_deco_aux, out_class_aux, Ht, Ct = self.deco(x, dim_list, ind_list, skip_list, Ht, Ct)
            
            # Save values
            out_class.append(out_class_aux)
            X_deco.append(X_deco_aux)

        # Classifier
        out = self.classifier(x, Ht)

        return [out, At, X_deco, out_class, X_tilde_aux, saved_x, Zt, wt, ut, tant, x_list, unpool_list]