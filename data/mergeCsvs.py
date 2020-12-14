"""
Here we merge 2 csv files :
    1) mimic-cxr-2.0.0-negbio.csv
    2) cohort.csv    
    
1) mimic-cxr-2.0.0-negbio.csv:
    14 diagnosis of each X-ray photo

2) cohort.csv:
    contains 1 ) binary label which indicate patient's ventilation history. 
             2 ) time when the ventilation was performed. if the label is 0 
                 
"""
import pickle
import os, sys, time
import numpy as np
import pandas as pd 

from glob import glob
from toolz import *
from toolz.curried import *
from operator import methodcaller
from datetime import datetime, date, time, timedelta

####################################

def parseDateTime(date, time) :
    
    def parseDate(val) :
        val = str(val)
        return datetime.strptime(val, "%Y%m%d").date()

    def parseTime(val) :
        val = str(val).split(".")[0].zfill(6)
        return datetime.strptime(val, "%H%M%S").time()
    
    return datetime.combine(parseDate(date),
                            parseTime(time))

def parseDiag(*args):
    return args


if __name__ == '__main__':

    parentDir = "CXRdata"
    metaDir   = "metaData"
    
    # paths to files
    bioPath     = f"{parentDir}/mimic-cxr-2.0.0-negbio.csv"
    cohortPath  = f"{metaDir}/cohort_for_extubation_failure_6to48.csv" #cohort_ver3_24_to_168.csv" # cohort_ver2.csv"
    joinedPath  = f"{metaDir}/joined_6to48.pkl"
   
    # lead time 
    L = timedelta(hours = 24)
    
    
    #imagePathDF
    imagePaths = [(path.split("/")[-1].split(".")[0],path) 
          for path 
          in glob(f"{parentDir}/*/*/*/*/*jpg")]
    
    pathDF = pd.DataFrame(imagePaths, columns=["dicom_id", "path"])
    
    # load DFS 
    bio, cohort = map(pd.read_csv)([bioPath, cohortPath])     
    
    #join cohrot, and bio
    cohort_bio = \
        reduce(partial(pd.merge,
                       on = ['subject_id', 'study_id'],
                       how = "left"),[cohort, bio] )

    #join cohort, bio and path
    final = \
        reduce(partial(pd.merge,
                       on = ['dicom_id'],
                       how = "inner"),[cohort_bio, pathDF] )
    
    
    # keep only the observations that have 
    final = final.replace(np.nan, "")    
    final = final[ final["viewposition"].str.contains('^PA$|^AP$') ]
        
    # str to datetime
    final["cxrtime"]   = pd.to_datetime(final["cxrstudytime"])    
    final["mvtime"]    = pd.to_datetime(final["mvstarttime_or_icu_dischargetime"])
    final["admittime"] = pd.to_datetime(final["mvstarttime_or_icu_dischargetime"])
    
    
    # merge negbio to one column
    final = final.replace("", 0)        
    final["diag"] = (final.apply(lambda row : parseDiag(row["Atelectasis"],
                                                        row["Cardiomegaly"],
                                                        row["Consolidation"],
                                                        row["Edema"],
                                                        row["Enlarged Cardiomediastinum"],
                                                        row["Fracture"],
                                                        row["Lung Lesion"],
                                                        row["Lung Opacity"],
                                                        row["No Finding"],
                                                        row['Pleural Effusion'],
                                                        row['Pleural Other'],
                                                        row['Pneumonia'],
                                                        row['Pneumothorax'],
                                                        row["Support Devices"]),
                                         axis = 1))
    

    
    final = final[['hadm_id', "path", "mv_flag", 'diag', 
                   "cxrtime", 'mvtime', 'admittime',                   
                   "split"]]
    
            
    # group by hadm_id and do some process
    ###############    
    dataPoints = []
    for hadm_id, df in final.groupby("hadm_id") :
        
        obs = df.to_dict('records')
                                
        X     = map(keyfilter(lambda x : x in ["path", "diag", "cxrtime"]))(obs)
                
        Y     = (compose(get("mv_flag"),first)(obs),
                 compose(get("mvtime"), first)(obs))
        
        SPLIT = compose(get("split"),first)(obs)
                
        META  = hadm_id
        
        dataPoints.append({"X"     : list(X),
                           "Y"     : tuple(Y),
                           "SPLIT" : str(SPLIT),
                           "META"  : str(META)})
                
    # save
    ###############
    if not os.path.exists(metaDir):
        os.makedirs(metaDir)
        
    f = open(joinedPath,"wb")
    pickle.dump(dataPoints, f)
    f.close()
    
    