from sklearn.metrics import confusion_matrix,f1_score
from sklearn.preprocessing import normalize
import itertools, keras, math,gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import entropy

def get_confusionM(pred,y_obs):
    """
        * pred is prediction probabilities or one hot, p(z=gamma|x)
        * y_obs is annotator probabilities shape is (N,T,K)
    """
    aux = np.tensordot(pred, y_obs, axes=[[0],[0]]).transpose(1,0,2)
    return aux/np.sum(aux, axis=-1)[:,:,None] #normalize

def generate_Individual_conf(Z_data, y_obs, no_label=-1, K=0, DTYPE_OP='float32'):
    # REVISAR SIS EP UEDE PARALELIZAR
    N, T = y_obs.shape
    if K == 0:
        K = np.max(y_obs) +1

    aux = np.zeros((T,K,K),dtype=DTYPE_OP) 
    for t in range(T):    
        for i in range(N):
            if y_obs[i,t] != no_label:
                aux[t,Z_data[i],y_obs[i,t]] +=1
                
        mask_nan = aux[t,:,:].sum(axis=-1) == 0
        for value in np.arange(K)[mask_nan]:
            #how to fill where she not annotate?? -- 
            aux[t,value,:] =  1 #-- similar  to laplace smooth (prior 1)
        aux[t,:,:] = aux[t,:,:]/aux[t,:,:].sum(axis=-1,keepdims=True) #normalize
    return aux

def generate_Global_conf(Z_data,y_obs):
    """ This function calculate the confusion matrix amongs all the annotations for every data. """  
    #hacerlo para ambas representaciones..

    if len(Z_data.shape) == 2:
        aux = np.tensordot(Z_data, y_obs, axes=[[0],[0]])
    elif len(Z_data.shape) ==1:
        Kl = max(Z_data.max()+1, y_obs.shape[1])
        aux = np.tensordot(keras.utils.to_categorical(Z_data, num_classes=Kl), y_obs, axes=[[0],[0]])
    else:
        raise Exception('The len(shape) of Z value has to be 1 or 2')
        
    if len(aux.shape) == 3: #if y_obs_categorical is delivered
        aux = aux.sum(axis=1)
    return aux/aux.sum(axis=-1,keepdims=True) #normalize

def generate_confusionM(*args):
    return generate_Global_conf(*args)

def plot_confusion_matrix(conf, classes,title="Estimated",text=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(conf, interpolation='nearest', cmap=cm.YlOrRd, vmin=0, vmax=1)
    if text:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, tick_marks) #classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = conf.max() / 2.
    if text:
        for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
            plt.text(j, i, format(conf[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if conf[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Observed label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_keras(model,x,y,classes):
    y_pred_ohe = model.predict_classes(x)
    conf_matrix = confusion_matrix(y_true=y, y_pred=y_pred_ohe)
    conf_matrix = normalize(conf_matrix, axis=1, norm='l1')
    plot_confusion_matrix(conf_matrix,classes)

def softmax(Xs):
    """Compute softmax values for each sets of scores in x."""
    values =[]
    for x in Xs:
        e_x = np.exp(x - np.max(x))
        values.append(e_x / e_x.sum())
    return np.asarray(values)

def calculate_JS_comp(matrixs, based="JS"):
    """ Calculate inertia of all the confusion matrixs"""
    if based.lower() == "js":
        D = D_JS #based on Jensen-Shannon Divergence
    elif based.lower() == "normf":
        D = D_NormF #based on Norm Frobenius

    value = []
    for m1 in range(matrixs.shape[0]):
        for m2 in range(m1+1,matrixs.shape[0]):
            value.append(D(matrixs[m1],matrixs[m2]))
    return np.mean(value)

def calculate_R_mean(conf_matrix): #weight?
    """Calculate the Mean of the diagional of the confusion matrixs"""
    return np.mean([conf_matrix[l,l] for l in range(len(conf_matrix)) ])

def calculate_S_score(conf_matrix):
    """Mean - off diagonal: based on Raykar logits"""
    return np.mean([conf_matrix[l,l]- np.mean(np.delete(conf_matrix[:,l],l)) for l in range(len(conf_matrix))])

def calculate_S_bias(conf_matrix, mode="simple"):
    """Score to known if p(y|something) == p(y) """
    p_y = conf_matrix.mean(axis=0) #prior anotation
    if mode=="entropy":        
        return entropy(p_y)
    elif mode == "median":
        return (p_y.max() - np.median(p_y)), p_y.argmax()
    elif mode == "simple":
        return p_y.max(), p_y.argmax() 
    #elif mode == "mean": #not so good
    #    return p_y.max() - p_y.mean()
    #elif mode =="real":
    #return np.mean([conf_matrix[l,:] - np.mean(np.delete(conf_matrix[l,:],l))  for l in range(len(conf_matrix))] )

def calculate_D_KL(confs_pred,confs_true):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    Kls = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t): #true
            Kls[m1] = D_KL(confs_pred[m1],confs_true[m1])
        return Kls
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def calculate_D_JS(confs_pred,confs_true):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    JSs = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t): #true
            JSs[m1] = D_JS(confs_pred[m1],confs_true[m1])
        return JSs
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def calculate_D_NormF(confs_pred,confs_true):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    NormFs = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t):
            NormFs[m1] = D_NormF(confs_pred[m1],confs_true[m1])
            #np.linalg.norm(confs_pred[m1]-confs_true[m1], ord='fro')/confs_pred[m1].shape[0]
        return NormFs
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def compare_conf_mats(pred_conf_mat,true_conf_mat=[], text=False):
    classes = np.arange(pred_conf_mat[0].shape[0])
    sp = plt.subplot(1,2,2)
    plt.imshow(pred_conf_mat, interpolation='nearest', cmap=cm.YlOrRd, vmin=0, vmax=1)
    plt.title("Estimated")
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    if text:
        thresh = pred_conf_mat.max() / 2.
        for i, j in itertools.product(range(pred_conf_mat.shape[0]), range(pred_conf_mat.shape[1])):
            plt.text(j, i, format(pred_conf_mat[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if pred_conf_mat[i, j] > thresh else "black")
    plt.tight_layout()

    if len(true_conf_mat) != 0:
        sp1 = plt.subplot(1,2,1)
        plt.imshow(true_conf_mat, interpolation='nearest', cmap=cm.YlOrRd, vmin=0, vmax=1)
        plt.title("True")
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)
        if text:
            thresh = true_conf_mat.max() / 2.
            for i, j in itertools.product(range(true_conf_mat.shape[0]), range(true_conf_mat.shape[1])):
                plt.text(j, i, format(true_conf_mat[i, j], '.2f'),
                         horizontalalignment="center",
                         color="white" if true_conf_mat[i, j] > thresh else "black")
        #plt.ylabel('True label')
        #plt.xlabel('Observed label')
    plt.tight_layout()
    plt.show()

def D_KL(conf_pred,conf_true, raw=False):
    """
        * mean of KL between rows of confusion matrix: 1/K sum_z KL_y(p(y|z)|q(y|z))
    """ 
    conf_pred = np.clip(conf_pred,1e-7,1.)
    conf_true = np.clip(conf_true,1e-7,1.)
    to_return = np.asarray([entropy(conf_true[j_z,:], conf_pred[j_z,:]) for j_z in range(conf_pred.shape[0])])
    if not raw:
        return np.mean(to_return)
    return to_return

def D_JS(conf_pred,conf_true, raw=False):
    """
        * Jensen-Shannon Divergence between rows of confusion matrix (arithmetic average)
    """
    conf_pred = np.clip(conf_pred,1e-7,1.)
    conf_true = np.clip(conf_true,1e-7,1.)
    aux = 0.5*conf_pred + 0.5*conf_true
    return (0.5*D_KL(aux,conf_pred,raw) + 0.5*D_KL(aux,conf_true,raw))/np.log(2) #value between 0 and 1
    
def D_NormF(conf_pred,conf_true):
    distance = conf_pred-conf_true
    return np.sqrt(np.sum(distance**2))/distance.shape[0]
    
def H_conf(conf_ma):
    """
        * Mean of entropy on rows of confusion matrix: mean H(q(y|z))
    """
    return np.mean([entropy(conf_ma[j_z]) for j_z in range(conf_ma.shape[0])])


class EarlyStopRelative(keras.callbacks.Callback):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                restore_best_weights=False):
        super(EarlyStopRelative,self).__init__()
        
        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
        
        
    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best2 = self.best
        self.best3 = self.best
        self.b_before = self.best

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)        
        if current is None:
            return

        if epoch==0:
            self.best = current
            return
        
        delta_conv = np.abs(self.best-current)/self.best #relative 
        if self.monitor_op(-self.min_delta, delta_conv):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value
    
from itertools import chain
from functools import reduce
import operator as op
def estimate_batch_size(model, scale_by=5.0,precision = 2):
    """
    :param model: keras Model
    :param available_mem: available memory in bytes
    :param scale_by: scaling factor
    :param precision: float precision: 2 bytes for fp16, 4 - for fp32, etc.
    :return: closest 2^n to the estimated batch size
    """
    import keras.backend as K
    from tensorflow.python.client import device_lib
    aux = device_lib.list_local_devices()
    values = [value  for value in aux  if value.device_type == 'GPU']
    if len(values) == 0:
        values = [value  for value in aux if value.device_type == 'CPU']
    available_mem = values[0].memory_limit 
    num_params = sum(chain.from_iterable((
        (reduce(op.mul, l.output_shape[1:]) for l in model.layers),
        (K.count_params(x) for x in model.trainable_weights),
        (K.count_params(x) for x in model.non_trainable_weights)
    )))
    max_size = max(32,np.int(available_mem / (precision * num_params * scale_by)))
    return np.int(2 ** math.floor(np.log(max_size)/np.log(2)))