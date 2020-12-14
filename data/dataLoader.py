#%matplotlib inline

import os, sys, random
import numpy as np

from toolz import *
from toolz.curried import *

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .dataGen import dataGen

def makeDataLoader(config, flag):    
    print(f"loading dataset : {flag}" )
    return DataLoader(dataGen(config.dataPath,
                              config.cohort,
                              augument = augument,
                              flag     = flag),
                      
                      batch_size  = config.batchSize,
                      shuffle     = True if flag == 'train' else False,
                      num_workers = config.numWorkers,
                      collate_fn  = collateFn(config.K),
                      pin_memory  = True,
                      drop_last   = True)


@curry
def collateFn(K, batch):
    
    """
    this function is added as an argument to the makeDataLoader function.
    this function handles how the datapoints for every batch are merged.
    this is required for our case becuase input has variable sizes.
    A remedy to this is to pad zero to shorter ones.
    
    imgss  :: [[np.array(1,H,W)]]
    diagss :: [[np.array([int])]]
    ys     :: [int]
    """
        
    B = len(batch)
    N = min(K, max(map(compose(len, first))(batch)))
            
    imgss     = torch.zeros(B, N, 1, 320, 320)    
    diagss    = torch.zeros(B, N, 14)    
    mvFlags   = torch.zeros(B)
    admit_ids = []
    for b, (admit_id, imgs, diags, mvFlag) in zip(range(B), batch) :        
        for n, img, diag in zip(range(N), imgs, diags):
            imgss[b,n, ...]  = img
            diagss[b,n, ...] = diag            
        mvFlags[b] = mvFlag
        admit_ids.append(admit_id)
                
    return (admit_ids,
            torch.as_tensor(imgss, dtype = torch.float32),
            torch.as_tensor(diagss, dtype = torch.float32),
            torch.as_tensor(mvFlags, dtype = torch.float32))
    
    
               
def augument (flag) :
               
    """
    flag   : one of ["train","test"]
    images : [np.array(H,W)]
    
    it is a simple augmentation fucntion that ,I admit, is imperfect.
    One key major drawback is that I resize the image to a much smaller one.
    However, following the works done by google. this does not give a bad performance.
    https://github.com/GoogleCloudPlatform/healthcare/tree/master/datathon/datathon_etl_pipelines
    """
    
    return \
        {
            "train" : transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Resize((320,320)),
                 transforms.Normalize(0,1),
                 transforms.RandomHorizontalFlip(p=0.2),
                 transforms.RandomRotation((-5,5)),
                ])
            ,

            "test" : transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Resize((320,320)),
                 transforms.Normalize(0,1),
                 transforms.RandomRotation((-5,5)),
                ])                
        }[flag]

if __name__ == "__main__":
    
    from easydict import EasyDict
    from matplotlib import pyplot as plt
    import torchvision
        
    config = EasyDict()
    
    config.dataPath = "." #"./data"
    config.cohort   = "joined_failure_6to72" #"./data"
    config.batchSize  = 10
    config.numWorkers = 30
    
    config.K = 6    
    
    gen = dataGen(config.dataPath, config.cohort,
                  augument = augument,
                  flag = "train")
    
    *_,imgs, diag, flag = gen.__getitem__(-2)    
    plt.imshow(imgs[0].squeeze())
    
    trainLoader = makeDataLoader(config, "train")
    validLoader = makeDataLoader(config, "test")
    testLoader  = makeDataLoader(config, "test")   
    
    for i, (ids, imgss, diagss, mvFlag)  in enumerate(validLoader):
        
        #print(imgss.shape, flush = True)        
        imgs = imgss[0]        
        _, axs = plt.subplots(nrows=1, ncols=config.K)
        
        for ax, img in zip(axs, imgs):            
            ax.imshow(img.squeeze())        
        plt.show()
#         print(imgss.shape, flush = True)
#         print(diagss.shape, flush = True)
#         print(mvFlag.shape, flush = True)
#         print("\n")
        #print(diagss, flush = True)
        #print(i, flush=True)
        #print(x[0].shape, flush=True)
