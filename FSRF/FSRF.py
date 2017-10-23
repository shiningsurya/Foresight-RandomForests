## Foresight Random Forest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from Foresight import Foresight

class FSRF(object):
    """
    Using Sir's idea.
    """
    def __init__(self, Ntrees, Nfeats,verbose=False):
        #TODO: Argument unpacking. Giving more control to the DTC
        self.ntrees = Ntrees
        self.nfeat = Nfeats
        self.sel_feats = np.zeros((self.ntrees, self.nfeat), dtype=int)
        self.trees = [0] * self.ntrees
        self.verbose = verbose
        self.fitdone = False

    def fit(self,X,y,aux_data=None):
        # Laying Ground work
        self.X = X.copy()
        self.y = y.copy()
        if aux_data is not None:
            self.aux_data = aux_data
            if self.verbose:
                print "Aux data provided."
        self.n, self.d = self.X.shape
        fimpt = np.zeros((self.ntrees, self.d))
        # Initializing Foresight
        self.fs = Foresight(X, y, aux_data=aux_data)
        self.fs.fit()
        if self.verbose:
            print "Foresight enabled."
        # Actual fitting
        for i in range(self.ntrees):
            self.sel_feats[i] = self.fs.select_n_features(self.nfeat)
            clf = DecisionTreeClassifier(max_features=self.nfeat)
            #boot-strap resample from the training data
            # bs_rs = np.random.choice(len(Xtrain), size=len(Xtrain), replace=True)
            bs_rs = np.random.choice(self.n, size=self.n, replace=True)
            # clf.fit(Xtrain[bs_rs,:][: , sel_feats[i]], Ytrain[bs_rs])
            clf.fit(self.X[bs_rs,:][:,self.sel_feats[i]], self.y[bs_rs])
            #save this tree
            self.trees[i] = clf
            if self.verbose:
                print "{0} out of {1} trees grown".format(i+1,self.ntrees)
            # Feature Importances
            fimpt[i,self.sel_feats[i]] = clf.feature_importances_
        self.fitdone = True
        # Feature Importances
        self.feature_importances_ = np.mean(fimpt,axis=0)

    def predict_proba(self,Xtest):
        tn,td = Xtest.shape
        if td != self.d:
            raise ValueError('The dimensions of training data and test data dont match.')
        #get predictions for each class, in a way exactly like the sklearn Random forests
        #note we have to keep sel_feats and use those features for each tree
        predictions = np.array([DT.predict_proba(Xtest[:, self.sel_feats[i]]) for i, DT in enumerate(self.trees)])
        return np.mean(predictions, axis=0) # average probability to be used

    def predict(self,Xtest):
        tn,td = Xtest.shape
        if td != self.d:
            raise ValueError('The dimensions of training data and test data dont match.')
        #get predictions for each class, in a way exactly like the sklearn Random forests
        #note we have to keep sel_feats and use those features for each tree
        predictions = np.array([DT.predict_proba(Xtest[:, self.sel_feats[i]]) for i, DT in enumerate(self.trees)])
        return np.argmax(np.mean(predictions, axis=0), axis=1)

    def getparams(self):
        ret = dict()
        ret['ntrees'] = self.ntrees
        ret['nfeat'] = self.nfeat
        return ret

    def setparams(self,ret):
        self.ntrees = ret['ntrees']
        self.nfeat = ret['nfeat']
