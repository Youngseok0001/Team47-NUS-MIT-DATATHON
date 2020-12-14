#%matplotlib inline

import os, sys, random, pickle
import numpy as np
import torch

from toolz import *
from toolz.curried import *

from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from datetime import datetime, date, time, timedelta

from dateutil.parser import parse as parseDateTime


# try to join the three tables (need to join Y table later)

class dataGen(Dataset):
    def __init__(self, dataPath, cohort,
                 augument    = lambda _ : identity,
                 flag        = "train"):
        
        self.dataPath  = dataPath
        self.cohort    = cohort
        self.flag      = flag        
        self.augment   = augument(self.flag)
        self.dataPts   = compose(list,
                                 filter(lambda pt : pt["SPLIT"] == self.flag) ,
                                 readPkl)(f"{self.dataPath}/metaData/{self.cohort}.pkl")
    
    def __len__(self): 
        return len(self.dataPts)
    
    def __getitem__(self, i):        
           
        """        
        imgs    :: [tensor(C,H,W)]
        diags   :: [tensor([int])] 
        mv_flag :: either 0 or 1
        """                
        dataPt = self.dataPts[i]
        
        Xs = pipe(dataPt["X"],
                  partial(sorted, key = get("cxrtime") ), # old -> new
                  reversed,                               # new -> old
                  map(lambda X: X.values()))
                  
        mv_flag, mvtime = dataPt["Y"] # to do something with mvTime later
        
        admit_id = dataPt["META"]
        
        imgs     = []
        diags    = []
        cxrtimes = []        
        for path, diag, cxrtime in Xs:
                        
#             img = np.array(plt.imread(f"{self.dataPath}/{path}"))
            
#             # add some obvious info            
#             H,W = img.shape
#             if mv_flag == 1:
#                 hole = torch.zeros(H//3, W//3)
#                 img[:H//3, :W//3] = hole
                
#             img  = self.augment(img)
            img = self.augment(plt.imread(f"{self.dataPath}/{path}"))

            imgs.append(img)
            cxrtimes.append(cxrtime)
            diags.append(diag)

        #print(cxrtimes, flush = True)
        return (admit_id,
                imgs,
                torch.as_tensor(diags),
                torch.as_tensor(mv_flag))

    
def readPkl(pklPath):
    return compose(pickle.load, partial(open, mode = "rb"))(pklPath)


if __name__ == "__main__":
    
    dataPath = "./"
    cohort   = "joined_failure_6to72"
        
    gen = dataGen(dataPath, cohort, flag = "train")
    
    for i in range(len(gen)):       
        print(i, flush = True)
        admit_id, imgs, diags, mv_flag = gen.__getitem__(i)        
        assert len(imgs) == len(diags)    
        plt.imshow(imgs[0])
        plt.show()
        
    
