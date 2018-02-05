import numpy as np
import pandas as pd


from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,Matern

import argparse
import pickle


np.random.seed(1)


# In[28]:

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', help='Test model and generate CSV submission file')
args = parser.parse_args()


# In[2]:

train =pd.read_csv('../data/intern_data.csv').iloc[:,1:]
test = pd.read_csv('../data/intern_test.csv').iloc[:,1:]
num = train.shape[0]

#merge training set with testing test before normalizing the data
tr_te = pd.concat((train, test), axis = 0)
#convert categorical column to dummy variable
tr_te_num = pd.get_dummies(tr_te)


# In[4]:

# center the dataset
tr_te_num  = tr_te_num.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
xtrain,xtest = np.split(tr_te_num.loc[:, tr_te_num.columns != 'y'], [num], axis=0)
xtrain = xtrain.values
xtest = xtest.values
y = train.loc[:,'y']


# In[21]:

nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)


# In[25]:

# Define a isotropic Matern kernel
kernel_RBF = 1.0 * RBF([1.0]*xtrain.shape[1])

kernel_Matern = 1.0 * Matern([1.0]*xtrain.shape[1])

preds = np.zeros(xtrain.shape[0])
preds_test = np.zeros(xtest.shape[0])

fold_index = 1
for (inTr, inTe) in folds:
    
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe] # xte is xde in fact
    yte = y[inTe]
 
    # Instanciate a Gaussian Process model
    filename = '../trained_model/gp'+str(fold_index)+".sav"
    if not args.test:
        gp = GaussianProcessRegressor(kernel = kernel_Matern, n_restarts_optimizer=9)
        gp.fit(xtr, ytr)
        pickle.dump(gp, open(filename, 'wb'))
        pred = gp.predict(xte, return_std=False)
        score = mean_absolute_error(yte, pred)
        preds[inTe] = pred
        print('Fold ', fold_index, '- MAE:', score)
        fold_index += 1
    else:
        gp = pickle.load(open(filename, 'rb'))
        preds_test += gp.predict(xtest,return_std = False)
preds_test /= nfolds 





