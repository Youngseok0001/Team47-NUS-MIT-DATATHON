import os, sys, time

from glob import glob

from tqdm import tqdm

import numpy as np

from argparse import ArgumentParser

from itertools import islice

from toolz import *
from toolz.curried import *

from data.dataLoader import makeDataLoader
from model.nets import ARCNN, ARCNN2 

import torch
import torch.nn as nn 

import pytorch_lightning as pl

from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.metrics.functional import f1



def parse_args():
    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)    

    parser.add_argument("--dataPath", type=str, default="./data") # where raw data is located
    parser.add_argument("--weightPath",type=str, default="./weight") # where raw data is located
    parser.add_argument("--cohort",type=str, default="joined_failure_6to72") # where raw data is located
    
    parser.add_argument("--K", type=int, default=6) # window size
    
    parser.add_argument("--batchSize", type=int, default=10) 
    parser.add_argument("--numWorkers", type=int, default=32) 
    parser.add_argument("--epoch", type=int, default=50)

    parser.add_argument("--net", type=str, default="ARCNN",
                       help = "ARCNN | ARCNN2") # window size    
    
    parser.add_argument("--cnn", type=str, default="AlexNet",
                       help = "AlexNet | VggNet | MobileNet | SqueezeNet")
    parser.add_argument("--ar", type=str, default="GRU",
                       help = "LSTM | GRU | RNN") # window size    
    
    parser.add_argument("--lr", type=float, default=10**-4)    
    parser.add_argument("--pos", type=float, default=5)
    
    return first(parser.parse_known_args())


class ARCNN_TRAINER(nn.Module):

    def __init__(self, config):
        super(ARCNN_TRAINER, self).__init__()
        self.config  = config
        
        self.model   = {"ARCNN" :ARCNN,
                        "ARCNN2":ARCNN2}[self.config.net](whatCNN = self.config.cnn,
                                                          whatAR  = self.config.ar)
        
        self.getLoss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(self.config.pos))
            
    def forward(self, imgss, diagss):
        
        logits, hiddens = self.model(imgss, diagss)
        
        return logits, hiddens
        
    
if __name__ == "__main__":    

    config = parse_args()
        
    trainLoader = makeDataLoader(config, "train")
    testLoader  = makeDataLoader(config, "test")   
    
    
    # this is fucking ugly!!!!
    ckptPath = "./weight/joined_failure_6to72-v1.ckpt"
    checkpoint = torch.load(ckptPath, map_location=lambda storage, loc: storage)
    model = ARCNN_TRAINER(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    
    admit_idss = []    
    featuress  = []
    for batch in tqdm(trainLoader):
        
        with torch.no_grad():
            
            admit_ids, imgss, diags, _ =  batch

            _, hiddens = model(imgss, diags)

            admit_idss.append(admit_ids)
            featuress.append(hiddens.cpu().numpy())
                        


    for batch in tqdm(testLoader):
        
        with torch.no_grad():
            
            admit_ids, imgss, diags, _ =  batch

            _, hiddens = model(imgss, diags)

            admit_idss.append(admit_ids)
            featuress.append( hiddens.cpu().numpy() )
            
    ID = reduce(operator.add)(admit_idss)
    F  = reduce(lambda x,y : np.concatenate((x,y)) )(featuress)
    
    X = []
    for id, f in zip(ID,F) :
        
        X.append({"hadm_id" : id,
                  "feature" : f})
        
    # save
    ###############
    import pickle
    f = open("CXR_feature_failure.pkl", "wb")
    pickle.dump(X, f)
    f.close()
    
            
        
        
    
    
    
            
    