
# coding: utf-8

# # Testing Random Forests with Foresight
# 
# There are two set of objects we'll have to comprehensive test.
# 
# | Basic                  | Foresight enabled        |
# |------------------------|--------------------------|
# | DecisionTreeClassifier | FSDecisionTreeClassifier |
# | DecisionTreeRegressor  | FSDecisionTreeRegressor  | 
# | RandomForestClassifier | FSRandomForestClassifier | 
# | RandomForestRegressor  | FSRandomForestRegressor  |
# 
# **Basic** and **Foresight enabled** only differ in one place. In **Basic**, `n_features` features are randomly selected with *uniform weights* for all features, however, in **Foresight enabled**, `n_features` features are randomly selected with *mutual information* used as weights. 
# 
# All the arguments to be sent to both set of classes are the same as well. 
# 

# In[2]:

from TreeMethods import  DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor
from TreeMethods import  FSDecisionTreeRegressor, FSRandomForestClassifier, FSRandomForestRegressor


# In[3]:

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification as makec
from sklearn.datasets import load_wine, load_iris       # classification 
from sklearn.datasets import load_boston, load_diabetes # regression 
from sklearn.metrics import classification_report, confusion_matrix


# ## Classification
# 
# We'll be comparing `FSRandomForestClassifier` with `RandomForestClassifier` to see if `Foresight` leads to any improvement.

# In[4]:

w = load_wine()
i = load_iris()
###
wine = pd.DataFrame(w.data,columns=[w.feature_names])
wine['Target'] = pd.Series(data=w.target)
iris = pd.DataFrame(i.data,columns=[i.feature_names])
iris['Target'] = pd.Series(data=i.target)


# In[6]:

rfc   = RandomForestClassifier.RandomForestClassifier(n_trees=10,max_depth=2,min_size=2,cost='gini')
fsrfc = FSRandomForestClassifier.FSRandomForestClassifier(n_feat=2,n_trees=10,max_depth=2,min_size=2,cost='gini')


# In[7]:

print wine.columns
print iris.columns


# In[8]:

rfc.fit(wine,target='Target')
fsrfc.fit(wine,target='Target')


# In[9]:

res = []
for idx in wine.index:
    res.append(rfc.predict(wine.loc[[idx]].squeeze()))


# In[10]:

print confusion_matrix(res,wine['Target'])


# In[11]:

tn = ['class 0', 'class 1', 'class 2']
print classification_report(res,wine['Target'],target_names=tn)

