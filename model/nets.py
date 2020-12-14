import sys,os,re
import numpy as np

from toolz         import *
from toolz.curried import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .utils import loadCNN, loadAR


class ARCNN(nn.Module):
    
    def __init__(self, whatCNN = "AlexNet", whatAR = "LSTM" ):
        super(ARCNN, self).__init__()
                        
        self.cnn      = loadCNN(whatCNN, pretrained = True)        
        self.ar       = loadAR(whatAR, 256 + 14, 256, 1)
        self.fc       = nn.Linear(256, 1)
        self.avgPool  = compose(torch.squeeze,nn.AdaptiveAvgPool2d(1))
        
        print(f"loading MAIN: ARCNN ...")
        
    def forward(self, imgss, diagss):
        
        """            
        imgss  :: tensor(B,N,C,H,W)
        diagss :: tensor(B,N,14) we got 14 different diags
        ys     :: tensor(B)
        
        where
            B : batchSize
            N : # of time stamps 
            C : channel size
            H : Height
            W : width
        
        # to be done in future,
            * add relative time as a feature.
        """
                        
        B,N,C,H,W = imgss.shape
        imgss   = imgss.view(B*N,C,H,W)        
        
        # embed image
        embedss = pipe(imgss,
                       self.cnn,
                       self.avgPool,
                       lambda x : x.view(B,N,-1))
        
        # cat to emb and diag 
        emb = torch.cat([embedss, diagss], dim = -1)
                
        #auto regress
        os, _ = self.ar(emb)
        o = os[:,-1,...]
        
        #logit
        logit = self.fc(F.relu(o))
        
        return logit.squeeze(), o
        
if __name__ == "__main__":
    
    sys.path.append("..")
        
    from data.dataGen import dataGen
    from data.dataLoader import collateFn, augument
    from easydict import EasyDict
    
    import torch.nn.functional as F
    
    config = EasyDict()
    config.dataPath = "../data" #"./data"

    config.batchSize  = 4
    config.numWorkers = 16    
    
    config.whatCnn = "AlexNet"
    config.whatAr  = "LSTM"
    
    config.K       = 6
        
    gen = dataGen(config.dataPath, augument)    
    
    imgss, diagss, ys = collateFn(6)([gen.__getitem__(i) for i in range(6)])
    
    net = ARCNN2(config.whatCnn, config.whatAr)
    getLoss = torch.nn.BCEWithLogitsLoss()
    
    logits, hidden = net(imgss, diagss)
    
    loss   = getLoss(logits.squeeze(), ys)
