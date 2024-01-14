import numpy as np

def list2global(list_ann, K):
    N = len(list_ann)
    r_obs = np.zeros((N,K))
    for i, annotations in enumerate(list_ann):
        annotations = np.asarray(annotations)
        for j in range(K):
            r_obs[i,j] = np.sum(annotations == j)
    return r_obs   

def annotations2global(obs, no_label=-1):
    """
    Used when memory error is through over normal "annotations2global" function
    """
    if len(obs.shape) ==1: #variable number of annotations
        N = obs.shape[0]
        K = obs[0].shape[1]
        globals_obs = np.zeros((N,K),dtype='int32')
        for i in range(N):
            globals_obs[i] = obs[i].sum(axis=0)
    elif len(obs.shape) ==2:
        return global_representation(obs, no_label=-1)
    else:
        globals_obs = np.sum(obs,axis=1,dtype='int32')
    return globals_obs 
    
def set_representation(y_obs, needed="onehot", no_label = -1, mask_s = 0):
    if needed.lower()=="onehot" or needed.lower()=="one-hot":
        return categorical_representation(y_obs, no_label=no_label, mask_s=mask_s)
    elif needed.lower()=="global":
        return global_representation(y_obs, no_label=no_label)
    elif needed.lower()=='onehotvar' or needed.lower()=='variable':
        return categorical_var_representation(y_obs, no_label=no_label)
    elif needed.lower()=='onehotmasked' or needed.lower()=='rodriguesmasked':
        return categorical_masked_representation(y_obs, no_label=no_label)

def global_representation(obs,no_label =-1):
    N, T = obs.shape
    K = int(np.max(obs)+1) # assuming that are indexed in order
    globals_obs = np.zeros((N,K),dtype='int32')
    for i in range(N):
        A_i = np.where(obs[i] != no_label)[0]
        for t in A_i:
            globals_obs[i,obs[i,t]] +=1
    return globals_obs

def categorical_representation(obs, no_label =-1, mask_s=0):
    N,T = obs.shape
    K = int(np.max(obs)+1) # assuming that are indexed in order
    y_obs_categorical = mask_s*np.ones((N,T,K),dtype='int8') 
    for i in range(N):
        A_i = np.where(obs[i] != no_label)[0]
        for t in A_i:
            y_obs_categorical[i,t,obs[i,t]] +=1
    #if annotator do not annotate a data her one-hot is full of zeroes
    return y_obs_categorical

def categorical_masked_representation(obs, no_label=-1):
    if len(obs.shape)!=3:
        y_obs_catmasked = categorical_representation(obs,no_label, mask_s=-1)
    else:
        y_obs_catmasked = obs
    return y_obs_catmasked.transpose(0,2,1)

def categorical_var_representation(obs, no_label=-1):
    N,T = obs.shape
    K = int(np.max(obs)+1) # assuming that are indexed in order
    y_obs_var, A_idx_var = [], []
    for i in range(N):
        Y_i = obs[i]
        A_i = np.where(Y_i != no_label)[0]
        A_idx_var.append( A_i.astype('int32') )
        import keras
        y_obs_var.append( keras.utils.to_categorical( Y_i[A_i],num_classes=K).astype('int8') )
    return np.asarray(y_obs_var, dtype="object"), np.asarray(A_idx_var, dtype="object")