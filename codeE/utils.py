from sklearn.metrics import confusion_matrix,f1_score
import itertools, keras, math,gc, time,sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .evaluation import *

def generate_Individual_conf(Z_data, annotations, DTYPE_OP='float32'):
    if len(Z_data.shape) == 1:
        K = np.max(Z_data)+1
        Z_data = keras.utils.to_categorical(Z_data, num_classes=K)
    elif len(Z_data.shape) != 2:
        raise Exception('The len(shape) of Z value has to be 1 or 2')

    aux = np.tensordot(Z_data, annotations, axes=[[0], [0]] ).astype(DTYPE_OP)
    aux = aux.transpose([1,0,2])
    mask_nan = aux.sum(axis=-1) == 0
    aux[mask_nan] = 1 #-- similar  to laplace smooth (prior 1)
    aux = aux/aux.sum(axis=-1, keepdims=True)
    return aux

def generate_Global_conf(Z_data, annotations, DTYPE_OP='float32'):
    """ This function calculate the confusion matrix amongs all the annotations for every data. """          
    if len(Z_data.shape) == 1:
        K = np.max(Z_data)+1
        Z_data = keras.utils.to_categorical(Z_data, num_classes=K)
    elif len(Z_data.shape) != 2:
        raise Exception('The len(shape) of Z value has to be 1 or 2')
    
    aux = np.tensordot(Z_data, annotations, axes=[[0],[0]]).astype(DTYPE_OP)
    if len(aux.shape) == 3: #if individual representation
        aux = aux.sum(axis=1)
    return aux/aux.sum(axis=-1,keepdims=True) #normalize

def generate_confusionM(*args):
    return generate_Global_conf(*args)

def plot_confusion_matrix(conf, classes=[],title="Estimated",text=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if len(classes)==0:
        classes = np.arange(len(conf[0]))
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
    conf_matrix = conf_matrix/conf_matrix.sum(axis=-1, keepdims=True)
    plot_confusion_matrix(conf_matrix,classes)

def softmax(Xs):
    """Compute softmax values for each sets of scores in x."""
    values =[]
    for x in Xs:
        e_x = np.exp(x - np.max(x))
        values.append(e_x / e_x.sum())
    return np.asarray(values)

def calculate_JS_comp(conf_matrices, based="JS"):
    """ Calculate inertia of all the confusion conf_matrices"""
    if based.lower() == "js":
        D = D_JS #based on Jensen-Shannon Divergence
    elif based.lower() == "normf":
        D = D_NormF #based on Norm Frobenius

    value = []
    for m1 in range(conf_matrices.shape[0]):
        for m2 in range(m1+1,conf_matrices.shape[0]):
            value.append(D(conf_matrices[m1],conf_matrices[m2]))
    return np.mean(value)

def calculate_R_mean(conf_matrix, *args):
    return R_mean(conf_matrix, *args)

def calculate_S_score(conf_matrix, *args):
    return S_score(conf_matrix, *args)

def calculate_S_bias(conf_matrix, *args):
    return S_bias(conf_matrix, *args)

def calculate_D_KL(confs_true, confs_pred):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    Kls = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t): #true
            Kls[m1] = D_KL(confs_pred[m1],confs_true[m1])
        return Kls
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def calculate_D_JS(confs_true, confs_pred):
    M_p = confs_pred.shape[0] #number of matrices on pred
    M_t = confs_true.shape[0] #number of matrices on true
    JSs = np.zeros(M_t)
    if  M_p == M_t:
        for m1 in range(M_t): #true
            JSs[m1] = D_JS(confs_pred[m1],confs_true[m1])
        return JSs
    else:
        print("ERROR! There are %d real and %d predicted conf matrices"%(M_t,M_p))

def calculate_D_NormF(confs_true, confs_pred):
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

def compare_conf_ma(pred_conf_mat, true_conf_mat=[], text=False):
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
    else:
        plt.colorbar()
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
    plt.tight_layout()
    plt.show()

def compare_set_conf_ma(set_conf_ma, true_set_conf_ma = [], text=True, n_samp=0):
    if n_samp==0:
        n_samp = len(set_conf_ma)
        
    if len(true_set_conf_ma) == 0:
        true_set_conf_ma = [ [] for _ in range(len(set_conf_ma))]
    print("Plot %d random matrices from the set"%n_samp)
    idx_samp = np.random.choice( np.arange(len(set_conf_ma)), size=n_samp, replace=False)
    for idx in idx_samp:
        compare_conf_ma(set_conf_ma[idx], true_set_conf_ma[idx], text=text)
        if text and len(true_set_conf_ma[idx]) != 0:
            print("D (based on Jensen Shannon) =",D_JS(true_set_conf_ma[idx], set_conf_ma[idx]))
            print("D (based on normalized Frobenius) =",D_NormF(true_set_conf_ma[idx], set_conf_ma[idx]))


def read_texts(filename):
    f = open(filename)
    data = [line.strip() for line in f]
    f.close()
    return data

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

def pre_init_F(model, X_inp, Z_targ, n_init, batch_size=32, reset_optimizer=True):
    print("Pre-train network %s on %d epochs..."%(model.name, n_init),end='',flush=True)
    model.fit(X_inp, Z_targ, batch_size=batch_size, epochs = n_init, verbose=0)
    if reset_optimizer:    #reset optimizer but hold weights--necessary for stability 
        loss_p = model.loss
        opt = type(model.optimizer).__name__
        model.compile(loss=loss_p, optimizer=opt)
    print(" Done!")


from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.cluster import DBSCAN,AffinityPropagation, MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler

def clusterize_annotators(y_o,M,no_label=-1,bulk=True,cluster_type='mv_close',data=[],model=None,DTYPE_OP='float32',BATCH_SIZE=64,option="hard",l=0.005):
    start_time = time.time()
    if bulk: #Individual scenario --variable y_o
        if cluster_type == 'previous':
            A_rep_aux = y_o
        else:
            A_idx = data[0]
            mv_soft = data[1]
            Kl  = mv_soft.shape[1]
            conf_mat, conf_mat_norm  = build_conf_Yvar(y_o, A_idx, mv_soft.argmax(axis=-1))
            if cluster_type == 'flatten' or cluster_type == 'conf_flatten':
                A_rep_aux = conf_mat_norm.reshape(conf_mat_norm.shape[0], Kl**2) #flatten
            elif cluster_type == 'js' or cluster_type == 'conf_js': 
                A_rep_aux = np.zeros((conf_mat.shape[0], Kl))
                for t in range(A_rep_aux.shape[0]):
                    A_rep_aux[t] = JS_confmatrixs(conf_mat_norm[t], np.identity(Kl),raw=True) #distancia a I (MV)

        probas_t = aux_clusterize(A_rep_aux,M,DTYPE_OP,option) #labels_kmeans
        alphas_init = probas_t

    else: #Global scenario: y_o: is soft-MV
        mv_soft = y_o.copy()
        if cluster_type=='loss': #cluster respecto to loss function
            obj_clone = Clonable_Model(model)
            aux_model = obj_clone.get_model()
            aux_model.compile(loss='categorical_crossentropy',optimizer=model.optimizer)
            aux_model.fit(data, mv_soft, batch_size=BATCH_SIZE,epochs=30,verbose=0)
            predicted = aux_model.predict(data,verbose=0)
        elif cluster_type == 'mv_close':
            predicted = np.clip(mv_soft, keras.backend.epsilon(), 1.)
       
        data_to_cluster = []
        for i in range(mv_soft.shape[0]):
            for j in range(mv_soft.shape[1]):
                ob = np.tile(keras.backend.epsilon(), mv_soft.shape[1])
                ob[j] = 1
                true = np.clip(predicted[i],keras.backend.epsilon(),1.)      
                f_l = distance_function(true, ob)  #funcion de distancia o similaridad
                data_to_cluster.append(f_l)  
        data_to_cluster = np.asarray(data_to_cluster)
        #if manny classes or low entropy?
        model = PCA(n_components=min(3,mv_soft.shape[1]) ) # 2-3
        data_to_cluster = model.fit_transform(data_to_cluster) #re ejecutar todo con esto
        probas_t = aux_clusterize(data_to_cluster,M,DTYPE_OP,option,l)
        alphas_init = probas_t.reshape(mv_soft.shape[0],mv_soft.shape[1],M)

    return alphas_init

def distance_function(predicted,ob):
    return -predicted*np.log(ob) #CE raw

def aux_clusterize(data_to_cluster,M,DTYPE_OP='float32',option="hard",l=0.005):
    """ Clusterize data """
    print("Doing clustering...",end='',flush=True)
    std = StandardScaler()
    data_to_cluster = std.fit_transform(data_to_cluster) 
        
    kmeans = MiniBatchKMeans(n_clusters=M, random_state=0,init='k-means++',batch_size=128)
    kmeans.fit(data_to_cluster)
    distances = kmeans.transform(data_to_cluster)

    if option=="fuzzy":
        probas_t = np.zeros_like(distances,dtype=DTYPE_OP)
        for t in range(probas_t.shape[0]):
            for m in range(probas_t.shape[1]):
                m_fuzzy = 1.2
                probas_t[t,m] = 1/(np.sum( np.power((distances[t,m]/(distances[t,:]+keras.backend.epsilon())), 2/(m_fuzzy-1)) ) + keras.backend.epsilon())
    elif option == "softmax":
        probas_t = softmax(-(distances+keras.backend.epsilon())/l).astype(DTYPE_OP)
    elif option == "softmax inv":
        probas_t = softmax(1/(l*distances+keras.backend.epsilon())).astype(DTYPE_OP)
    elif option == 'hard':
        probas_t = keras.utils.to_categorical(kmeans.labels_,num_classes=M)
    print("Done!")
    return probas_t

def build_conf_Yvar(y_obs_var, A_idx, Z_val):
    """ From variable length arrays of annotations and indexs"""
    T = np.max(np.concatenate(A_idx))+1
    N = y_obs_var.shape[0]
    Kl = np.max(Z_val) +1
    aux_confe_matrix = np.ones((T,Kl,Kl))
    for i in range(N): #independiente de "T"
        for l, a_idx in enumerate(A_idx[i]):
            obs_t = y_obs_var[i][l].argmax(axis=-1)
            aux_confe_matrix[a_idx, Z_val[i], obs_t] +=1
    aux_confe_matrix_n = aux_confe_matrix/aux_confe_matrix.sum(axis=-1,keepdims=True)
    return aux_confe_matrix, aux_confe_matrix_n #return both: normalized and unnormalized
