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
from networks import *

class VGG_Deco(nn.Module):
    def __init__(self, num_classes, num_structures, input_channels, image_size, att_block=3, enable_classifier=1, enable_decoder=1):
        super(VGG_Deco, self).__init__()

        # Dimension and other variables
        self.input_channels = input_channels
        self.image_size = image_size # 224 (224x224)
        self.att_block = att_block
        self.num_classes = num_classes # 2 - melanoma or not
        self.num_structures = num_structures
        self.kernel_size = 2
        self.stride = 2

        self.enable_classifier = enable_classifier
        self.enable_decoder = enable_decoder

        self.attn_channels = int(self.input_channels*(2**(self.att_block-1))) # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.attn_size = int(self.image_size/(2**self.att_block)) # square images of 28x28 dim_feats = 28

        # Encoder
        self.encoder = Encoder(self.att_block)

        # Decoder
        if self.enable_decoder == 1:
            self.deco = Decoder_skip(input_channels=self.attn_channels, num_structures=self.num_structures, output_channels=1, n_blocks=self.att_block)
            
        # Classifier
        if self.enable_classifier == 1:
            self.classifier = Classifier(num_classes=self.num_classes, attn_channels=self.attn_channels, t=1, att_block=self.att_block)
        
    def forward(self, x):
        # Encoder
        x, dim_list, ind_list, skip_list = self.encoder(x)

        # AttentiveConvLSTM
        At = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        Ht, Ct = None, None
        Hsave, Csave = [], []
        X_tilde_aux = []
        X_deco, out_class = [], []
        x_saved = x
        # Decoder
        if self.enable_decoder == 1:
            X_deco_aux, out_class_aux = self.deco(x, dim_list, ind_list, skip_list)
            
            # Save values
            out_class.append(out_class_aux)
            X_deco.append(X_deco_aux)

        # Classifier
        out = None
        if self.enable_classifier == 1:
            for k in np.arange(self.att_block,5):
                if k == 3:
                    block = self.conv_block4(x)   # /8 
                elif k == 4:
                    block = self.conv_block5(x)   # /16
                
                x = F.max_pool2d(block, self.kernel_size, self.stride) 
                
            N, __, __, __ = x.size()

            # GAP
            g_list = []
            g = self.pool(x).view(N,512)
            g_list.append(g)
            g_list.append(F.adaptive_avg_pool2d(x_save, (1,1)).view(N,-1))
            g_hat = torch.cat(g_list, dim=1) # batch_size x C
            out = self.cls(g_hat)
            
        return [out, At, X_deco, out_class, X_tilde_aux, x_saved]

class VGG_ConvLSTM_Deco(nn.Module):
    def __init__(self, num_classes, num_structures, input_channels, image_size, device, t, att_block=3, enable_classifier=1, enable_decoder=1):
        super(VGG_ConvLSTM_Deco, self).__init__()

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

        self.enable_classifier = enable_classifier
        self.enable_decoder = enable_decoder

        self.attn_channels = int(self.input_channels*(2**(self.att_block-1))) # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.attn_size = int(self.image_size/(2**self.att_block)) # square images of 28x28 dim_feats = 28

        # Encoder
        self.encoder = Encoder(self.att_block)

        # ConvLSTM
        self.convLSTM = ConvLSTM(self.attn_channels, self.attn_size, self.device)

        # Decoder
        if self.enable_decoder == 1:
            self.deco = Decoder_skip(input_channels=self.attn_channels, num_structures=self.num_structures, output_channels=1, n_blocks=self.att_block)
            
        # Classifier
        if self.enable_classifier == 1:
            self.classifier = Classifier(num_classes=self.num_classes, attn_channels=self.attn_channels, t=self.t, att_block=self.att_block)
        
    def forward(self, x):
        # Encoder
        x, dim_list, ind_list, skip_list = self.encoder(x)

        # AttentiveConvLSTM
        At = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        Ht, Ct = None, None
        Hsave, Csave = [], []
        X_deco, out_class, X_tilde_aux = [], [], []
        x_saved = x
        for k in range(self.t):
            Ht, Ct = self.convLSTM(x, Ht, Ct)
            Hsave.append(Ht)
            Csave.append(Ct)

            # Decoder
            if self.enable_decoder == 1:
                X_deco_aux, out_class_aux = self.deco(x, dim_list, ind_list, skip_list)

                # Save values
                out_class.append(out_class_aux)
                X_deco.append(X_deco_aux)

        # Classifier
        out = None
        if self.enable_classifier == 1:
            for k in np.arange(self.att_block,5):
                if k == 3:
                    block = self.conv_block4(x)   # /8 
                elif k == 4:
                    block = self.conv_block5(x)   # /16
                
                x = F.max_pool2d(block, self.kernel_size, self.stride) 
                
            N, __, __, __ = x.size()
            # GAP
            g_list = []
            g = self.pool(x).view(N,512)
            g_list.append(g)
            g_list.append(F.adaptive_avg_pool2d(x_saved, (1,1)).view(N,-1))
            g_hat = torch.cat(g_list, dim=1) # batch_size x C
            out = self.cls(g_hat)

        return [out, At, X_deco, out_class, X_tilde_aux, x_saved]

class VGG_AttConvLSTM_Deco(nn.Module):
    def __init__(self, num_structures, input_channels, image_size, device, t, att_block=3):
        super(VGG_AttConvLSTM_Deco, self).__init__()

        # Dimension and other variables
        self.device = device
        self.input_channels = input_channels
        self.image_size = image_size # 224 (224x224)
        self.att_block = att_block
        self.num_structures = num_structures
        self.kernel_size = 2
        self.stride = 2
        self.t = t 

        self.attn_channels = int(self.input_channels*(2**(self.att_block-1))) # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.attn_size = int(self.image_size/(2**self.att_block)) # square images of 28x28 dim_feats = 28

        # Encoder
        self.encoder = Encoder(self.att_block)

        # Attentive model
        self.attentive_model = AttentiveModel(self.attn_channels, self.attn_size, self.device)

        # ConvLSTM
        self.convLSTM = ConvLSTM(self.attn_channels, self.attn_size, self.device)

        # Decoder
        self.deco = Decoder_skip(input_channels=self.attn_channels, num_structures=self.num_structures, output_channels=1, n_blocks=self.att_block)
        
    def forward(self, x):
        # Encoder
        x, dim_list, ind_list, skip_list = self.encoder(x)

        # AttentiveConvLSTM
        At = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        Ht, Ct = None, None
        Hsave, Csave = [], []
        X_tilde_aux = []
        X_deco, out_class = [], [], []     
        saved_x = x   
        for k in range(self.t):
            X_tilde_t, At_aux = self.attentive_model(x, Ht)
            At[:,k,:,:] = At_aux.squeeze()
            Ht, Ct = self.convLSTM(X_tilde_t, Ht, Ct)
            
            X_tilde_aux.append(X_tilde_t)
            Hsave.append(Ht)
            Csave.append(Ct)

            # Decoder
            X_deco_aux, out_class_aux = self.deco(X_tilde_t, dim_list, ind_list, skip_list)
            
            # Save values
            out_class.append(out_class_aux)
            X_deco.append(X_deco_aux)

        # Classifier
        out = None

        return [out, At, X_deco, out_class, X_tilde_aux, saved_x]

class VGG_AttConvLSTM_Class(nn.Module):
    def __init__(self, num_classes, input_channels, image_size, device, t, att_block=3):
        super(VGG_AttConvLSTM_Class, self).__init__()

        # Dimension and other variables
        self.device = device
        self.input_channels = input_channels
        self.image_size = image_size # 224 (224x224)
        self.att_block = att_block
        self.num_classes = num_classes # 2 - melanoma or not
        self.kernel_size = 2
        self.stride = 2
        self.t = t 

        self.attn_channels = int(self.input_channels*(2**(self.att_block-1))) # input and output channels are the same (64*3 = 256, if it is after pool-3)
        self.attn_size = int(self.image_size/(2**self.att_block)) # square images of 28x28 dim_feats = 28

        # Modules
        self.encoder = Encoder(self.att_block)
        self.attentive_model = AttentiveModel(self.attn_channels, self.attn_size, self.device)
        self.convLSTM = ConvLSTM(self.attn_channels, self.attn_size, self.device)
        self.classifier = Classifier(num_classes=self.num_classes, attn_channels=self.attn_channels, t=self.t, att_block=self.att_block)
        
    def forward(self, x):
        # Encoder
        x, dim_list, ind_list, skip_list = self.encoder(x)

        # AttentiveConvLSTM
        At = torch.zeros([x.shape[0], self.t, self.attn_size, self.attn_size])
        Ht, Ct = None, None
        Hsave, Csave = [], []
        X_tilde_aux = []
        X_deco, out_class = [], [], []     
        saved_x = x   
        for k in range(self.t):
            X_tilde_t, At_aux = self.attentive_model(x, Ht)
            At[:,k,:,:] = At_aux.squeeze()
            Ht, Ct = self.convLSTM(X_tilde_t, Ht, Ct)
            
            X_tilde_aux.append(X_tilde_t)
            Hsave.append(Ht)
            Csave.append(Ct)

        # Classifier
        out = self.classifier(X_tilde_t, X_tilde_aux)

        return [out, At, X_deco, out_class, X_tilde_aux, saved_x]