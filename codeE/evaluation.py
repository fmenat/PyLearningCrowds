import math, gc, time, sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import entropy, pearsonr
try:
    from tabulate import tabulate
except:
    print("NO TABULATE INSTALLED!")
from .utils import *

def accuracy_model(model,X_data, Z_data):
    Z_hat = model.predict(X_data)
    if len(Z_hat.shape) > 1:
        Z_hat = Z_hat.argmax(axis=-1)
    return np.mean(Z_data == Z_hat)

def f1score_model(model,X_data, Z_data, mode='macro'):
    Z_hat = model.predict(X_data)
    if len(Z_hat.shape) > 1:
        Z_hat = Z_hat.argmax(axis=-1)
    return f1_score(Z_data, Z_hat, average=mode)

def D_KL(conf_true, conf_pred, raw=False):
    conf_pred = np.clip(conf_pred, 1e-7, 1.)
    conf_true = np.clip(conf_true, 1e-7, 1.)
    to_return = np.asarray([entropy(conf_true[j_z,:], conf_pred[j_z,:]) for j_z in range(conf_pred.shape[0])])
    if not raw:
        return np.mean(to_return)
    return to_return

def D_JS(conf_true, conf_pred, raw=False):           
    aux = (conf_pred + conf_true)/2.
    return (D_KL(conf_pred, aux, raw) + D_KL(conf_true, aux, raw))/(2*np.log(2)) 
    
def D_NormF(conf_true, conf_pred):
    distance = conf_pred-conf_true
    return np.sqrt(np.sum(distance**2))/distance.shape[0]

def Individual_D(confs_true, confs_pred, D):
    T = len(confs_true)
    res = 0
    for t in range(T):
        res += D(confs_true[t], confs_pred[t])
    return res/T
    

def I_sim(conf_ma, D=D_JS):
    I = np.identity(len(conf_ma))
    return 1 - D(conf_ma, I)

def H_conf(conf_ma):
    conf_ma = np.clip(conf_ma, 1e-7, 1.)
    K = len(conf_ma)
    return np.mean([entropy(conf_ma[j_z]) for j_z in range(K)])/np.log(K)

def S_score(conf_ma):
	return np.mean([conf_ma[l,l]- np.mean(np.delete(conf_ma[:,l],l)) for l in range(len(conf_ma))])

def R_score(conf_ma):
	return np.mean([conf_ma[l,l] for l in range(len(conf_ma)) ])

def S_bias(conf_ma, mode="median"):
    """Score to known if p(y|something) == p(y) """
    p_y = conf_ma.mean(axis=0) #prior anotation

    b_C= p_y.argmax() #no recuerdo ...
    if mode=="entropy":      
        p_y = np.clip(p_y, 1e-7, 1.)  
        return b_C, 1-entropy(p_y)/np.log(len(p_y))
    elif mode == "median":
        return b_C, (p_y.max() - np.median(p_y))
    elif mode == "simple":
        return b_C, p_y.max()

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False