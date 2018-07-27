
# coding: utf-8

# ## Using Foresight and sklearn

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification as makec
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
######
import FSRF
#allow realtime editing of the class 
reload(FSRF)
FSRF = FSRF.FSRF


# I've `sklearn`-ized the approach you suggested. FSRF(currrently made for Classification problems) has methods such as:
# 1. predict
# 2. predict_proba
# 3. fit
# 4. getparams
# 5. setparams
# 
# On this toy example, FSRF performing better sir. 

# #### Making Data

# In[4]:


n_classes = 10
n_features = 10 * n_classes
Xd1, yd1 = makec(n_samples=10000, n_features=n_features, n_informative=int(n_features*0.1), n_redundant=3, 
                n_repeated=2, n_classes=n_classes)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xd1, yd1 , train_size=0.1, test_size=0.9, random_state=42)
###
nfeats = 5
ntrees = 100


# #### Training

# `sklearn.RandomForestClassifier` versus `ForeSight enabled RandomForestclassifier`
# 
# on varying only two hyperparameters: `nfeats` and `ntrees`

# In[6]:


# sklearn
rf = RandomForestClassifier(n_estimators=ntrees, max_features=nfeats)
rf.fit(Xtrain, Ytrain)


# In[13]:


# fsrf
fsrf = FSRF(ntrees,nfeats)
fsrf.fit(Xtrain,Ytrain)


# #### Predictions

# In[14]:


rf_pred = rf.predict(Xtest)
fsrf_pred = fsrf.predict(Xtest)


# In[15]:


print "f1 score for sklearn.RF        {0:.3f}".format(f1_score(Ytest, rf_pred, average='micro'))
print "f1 score for Foresight.RF      {0:.3f}".format(f1_score(Ytest, fsrf_pred, average='micro'))


# In[17]:


print classification_report(Ytest,rf_pred)


# In[18]:


print classification_report(Ytest,fsrf_pred)


# #### Feature importances

# In[25]:


figu = plt.figure(figsize=(15,10))
#feature importance for RF
feature_importances_ = rf.feature_importances_ - np.amin(rf.feature_importances_)
feature_importances_ = feature_importances_/np.amax(feature_importances_)
_ = plt.plot(np.arange(n_features), feature_importances_, label='RF')

#feature importances for FSRF
feature_importances_ = fsrf.feature_importances_ - np.amin(fsrf.feature_importances_)
feature_importances_ = feature_importances_/np.amax(feature_importances_)
_ = plt.plot(np.arange(n_features), feature_importances_, label='FSRF')
plt.legend()
plt.xlabel('Feature')
plt.ylabel('importance weight')
plt.grid(True)
plt.show()


# The peaks of feature importances of `sklearn.RF` and `FSRF` match perfectly. I've computed the feature importances in a different way. Instead of just counting number of times a feature was selected, I thought of including the feature importances of each Decision Tree to compute the overall feature importances. 
# 
# The features which are used for training a Decision Tree get their feature importances from that Decision Tree itself. And those, which aren't, are set to zero by default. Hence we see some features have feature importances as zero in the plot(follow yellow line). 
