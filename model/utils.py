import torch
import torch.nn as nn
from torchvision import models

from toolz         import *
from toolz.curried import *

def loadCNN(what, pretrained = True):
    
    """
    what       :: one of ["AlexNet", "VggNet", "mobileNet", "SqueezeNet"]    
    pretrained :: BOOL
    """
        
    # why add lambda? becuase I only want to load one of many.
    model = \
        {"AlexNet"    : lambda : models.alexnet(pretrained=pretrained),
         "VggNet"     : lambda : models.vgg16(pretrained=pretrained),
         "MobileNet"  : lambda : models.mobilenet_v2(pretrained=pretrained),
         "SqueezeNet" : lambda : models.squeezenet1_0(pretrained=pretrained),
        }[what]()
    
    featureN = {"AlexNet"    : 256 ,
                "VggNet"     : 512,
                "MobileNet"  : 1280,
                "SqueezeNet" : 512}[what] 

        
    
    # I add one 1X1 conv2d layer to map to channel size of 3.
    model = nn.Sequential(nn.Conv2d(1,3,1), nn.ReLU(),
                          *model.features,
                          nn.Conv2d(featureN, 256, 1), nn.ReLU())
    
    print(f"loading CNN: {what}... ")
    
    return model

def loadAR(what, infeatureN, outFeatureN, unitN):
    
    """
    what        :: one of ["LSTM", "GRU", "RNN"]    
    infeatureN  :: size of input features
    outFeatureN :: size of output features
    unitN       :: number of ar computation units. (i think one is fine....)
    """
    
    # why add lambda? becuase I only want to load one of many.    
    model = \
        {"LSTM" : lambda : nn.LSTM(infeatureN, outFeatureN, unitN,
                                   batch_first = True, bidirectional = False),
         "GRU"  : lambda : nn.GRU(infeatureN, outFeatureN, unitN,
                                   batch_first = True, bidirectional = False),
         "RNN"  : lambda : nn.RNN(infeatureN, outFeatureN, unitN,
                                   batch_first = True, bidirectional = False)
        }[what]()
    
    print(f"loading AR: {what}... ")
    return model
    
    
    
    





