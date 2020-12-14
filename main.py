"""

    parser.add_argument("--dataPath", type=str, default="./data") # where raw data is located
    parser.add_argument("--weightPath",type=str, default="./weight") # where raw data is located
    parser.add_argument("--cohort",type=str, default="joined_failure_6to72") # where raw data is located

    parser.add_argument("--K", type=int, default=6) # window size

    parser.add_argument("--batchSize", type=int, default=10)
    parser.add_argument("--numWorkers", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)

    parser.add_argument("--net", type=str, default="ARCNN",
                       help = "ARCNN | ARCNN2") # window size

    parser.add_argument("--cnn", type=str, default="AlexNet",
                       help = "AlexNet | VggNet | MobileNet | SqueezeNet")
    parser.add_argument("--ar", type=str, default="GRU",
                       help = "LSTM | GRU | RNN") # window size

    parser.add_argument("--lr", type=float, default=10**-4)
    parser.add_argument("--pos", type=float, default=4.45) #3 : 6, 4 : 2.7 f: 4.45
    

"""
import os, sys, time

from argparse import ArgumentParser

from toolz import *
from toolz.curried import *

from itertools import islice 

from glob import glob

from data.dataLoader import makeDataLoader
from model.nets import ARCNN

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.metrics.functional import f1
from pytorch_lightning.metrics.classification import AveragePrecision


class ARCNN_TRAINER(pl.LightningModule):

    def __init__(self, config):
        super(ARCNN_TRAINER, self).__init__()
        self.config  = config
        
        self.model   = ARCNN(whatCNN = self.config.cnn,
                             whatAR  = self.config.ar)
        
        self.getLoss = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(self.config.pos))
            
    def forward(self, imgss, diagss):
        
        logits, hiddens = self.model(imgss, diagss)
        
        return logits, hiddens
        
    def _step(self, b, bi):

        *_, imgss, diagss, ys = b
        
        logits, hiddens = self.forward(imgss, diagss)        
        
        loss   = self.getLoss(logits, ys)
        
        preds  = torch.sigmoid(logits)
        
        return loss, preds, ys
    
    def training_step(self, b, bi):        
        loss, *_ = self._step(b, bi)                
        self.log(f"TRAIN_LOSS :", loss, on_epoch=True, prog_bar=True, logger=True)  
        return loss
    
    def validation_step(self, b, bi):        
        loss, preds, ys = self._step(b, bi)                
        self.log(f"VALID_LOSS :", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"preds" : preds, 
                "ys"    : ys }
        
    def test_step(self, b, bi):        
        loss, *_ = self._step(b, bi)                
        self.log(f"TEST_LOSS :", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"preds" : preds, 
                "ys"    : ys }

    def train_epoch_end(self, outputs):
        
        preds = pipe(outputs, 
                     map(get("preds")),
                     list,
                     torch.cat)
        
        ys    = pipe(outputs, 
                     map(get("ys")),
                     list,
                     torch.cat)
        
        _acc   = accuracy(torch.round(preds), ys)
        _auc   = auroc(preds, ys)
        _f1    = f1(preds, ys, num_classes = 2)
        _auprc = AveragePrecision(pos_label=1)(preds, ys)
        
        self.log(f"TRAIN_ACC", _acc, prog_bar=True, logger=True)
        self.log(f"TRAIN_AUC", _auc, prog_bar=True, logger=True)
        self.log(f"TRAIN_f1", _f1, prog_bar=True, logger=True)
        self.log(f"TRAIN_AUPRC", _auprc, prog_bar=True, logger=True)
        
    def validation_epoch_end(self, outputs):
        
        preds = pipe(outputs, 
                     map(get("preds")),
                     list,
                     torch.cat)
        
        ys    = pipe(outputs, 
                     map(get("ys")),
                     list,
                     torch.cat)
        
        _acc   = accuracy(torch.round(preds), ys)
        _auc   = auroc(preds, ys)
        _f1    = f1(preds, ys, num_classes = 2)
        _auprc = AveragePrecision(pos_label=1)(preds, ys)
        
        self.log(f"VALID_ACC", _acc, prog_bar=True, logger=True)
        self.log(f"VALID_AUC", _auc, prog_bar=True, logger=True)
        self.log(f"VALID_f1", _f1, prog_bar=True, logger=True)
        self.log(f"VALID_AUPRC", _auprc, prog_bar=True, logger=True)
                        

    def configure_optimizers(self):
        
        print("configuring optimizer")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        return [optimizer], [scheduler]
    
    
    
def parse_args():
    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)    

    parser.add_argument("--dataPath", type=str, default="./data") # where raw data is located
    parser.add_argument("--weightPath",type=str, default="./weight") # where raw data is located
    parser.add_argument("--cohort",type=str, default="joined_failure_6to72") # where raw data is located
    
    parser.add_argument("--K", type=int, default=6) # window size
    
    parser.add_argument("--batchSize", type=int, default=10) 
    parser.add_argument("--numWorkers", type=int, default=8) 
    parser.add_argument("--epoch", type=int, default=50)

    parser.add_argument("--cnn", type=str, default="AlexNet",
                       help = "AlexNet | VggNet | MobileNet | SqueezeNet")
    parser.add_argument("--ar", type=str, default="GRU",
                       help = "LSTM | GRU | RNN") # window size    
    
    parser.add_argument("--lr", type=float, default=10**-4)    
    parser.add_argument("--pos", type=float, default=4.45) #3 : 6, 4 : 2.7 f: 4.45
    
    return first(parser.parse_known_args())

if __name__ == "__main__":
    
    
    config = parse_args()
    
    trainLoader = makeDataLoader(config, "train")
    validLoader = makeDataLoader(config, "test")
    testLoader  = makeDataLoader(config, "test")   

    tb_logger = pl_loggers.TensorBoardLogger('logs/', name=config.cohort)
    
    checkpoint_callback = ModelCheckpoint(
        filepath   = f'./{config.weightPath}/{config.cohort}/',
        save_top_k = 1,
        verbose    = True,
        monitor    = 'VALID_AUC',
        mode       = 'max')    
    
    trainer = pl.Trainer(gpus=1, 
                         max_epochs=config.epoch,
                         checkpoint_callback = checkpoint_callback,
                         logger=tb_logger)
    
    trainer.fit(ARCNN_TRAINER(config), trainLoader, validLoader)
    result = trainer.test(test_dataloaders = testLoader)
    print(result)

    


