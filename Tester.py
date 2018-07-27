'''
Pipeline to test 
RF v/s DTCwB v/s FSRF 
-----
RF is the bare bones Random Forest.
DTCwB is Forest of Decision Tree Classifiers with Bagging
FSRF is the Foresight enabled Random Forest.
-----
Testing these three:
RF because it's benchmark
DTCwB because it becomes placebo for FSRF
FSRF because it's our idea
-----
https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/
'''

import numpy as np
import sklearn
import sklearn.metrics as skm

def GetMetrics(ytrue,ypredict,binary=True):
    '''
    Given ytrue and ypredict
    returns all the metrics in standardized fashion
    ---
    Sticking with binary classification for now
    ---
    '''
    ret = []
    if binary:
        ret.append(True)
    # the first one if it is whether binary or not
    ret.append(skm.recall_score(ytrue,ypredict))                         # recall
    ret.append(skm.precision_score(ytrue,ypredict))                      # precision
    ret.append(skm.f1_score(ytrue,ypredict))                             # f1 score
    ret.append(skm.accuracy_score(ytrue,ypredict))                       # accuracy
    ret = ret + skm.confusion_matrix(ytrue,ypredict).ravel().tolist()    # cfm 
    return ret

def GetROC(ytrue, yprob, binary=True):
    '''
    Given ytrue and probabilities
    return ROC plot and AuROC
    ---
    This is only for Binary classification,
    binary flag is just for namesake
    '''
    if not binary:
        raise ValueError('GetROC works for binary classification')
    fpr, tpr, thresholds = skm.roc_curve(ytrue, yprob)
    auroc = skm.auc(fpr,tpr)
    ret = [binary, auroc, fpr, tpr, thresholds]
    return ret

def CV(X, y, clfs, nfold=5, niter=20, cname=None, binary=True):
    '''
    Takes in X and y
    and list of classifiers
    and does `niter` of `nfold` Cross Validation
    ---
    and then shows metrics and ROCs
    '''

