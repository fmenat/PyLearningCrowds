import numpy as np
import keras
def categorical_representation(obs,no_label =-1):
    """Representation of one hot vectors, (N,T,K), masked with 0 """
    N,T = obs.shape
    K = int(np.max(obs)+1) # assuming that are indexed in order
    y_obs_categorical = np.zeros((N,T,K),dtype='int8') #solo 0 o 1
    for i in range(N):
        A_i = np.where(obs[i] != no_label)[0]
        for t in A_i:
            y_obs_categorical[i,t,obs[i,t]] +=1
    #if annotator do not annotate a data her one-hot is full of zeroes
    return y_obs_categorical

def categorical_masked_representation(obs, no_label=-1):
    """Representation of one hot vectors, (N,K,T), masked with -1 """
    if len(obs.shape)!=3:
        y_obs_catmasked = categorical_representation(obs,no_label)
    else:
        y_obs_catmasked = obs
    mask =  np.sum(y_obs_catmasked,axis=-1)  == 0
    #if annotator do not annotate a data her one-hot is full of -1
    y_obs_catmasked[mask] = -1
    return y_obs_catmasked.transpose(0,2,1)

def categorical_var_representation(obs, no_label=-1):
    """Representation of one hot vectors of variable lengths, (N,T_i,K), no masked"""
    N,T = obs.shape
    K = int(np.max(obs)+1) # assuming that are indexed in order
    y_obs_var, T_idx = [], []
    for i in range(N):
        Y_i = obs[i]
        A_i = np.where(Y_i != no_label)[0]
        T_idx.append( A_i.astype('int32') )
        y_obs_var.append( keras.utils.to_categorical( Y_i[A_i],num_classes=K).astype('int8') )
    #if return_idx:
    return np.asarray(y_obs_var), np.asarray(T_idx)
    #return np.asarray(y_obs_var)

def get_A_il(array, A=[],T=0, index=False):
    """ Assigned representation for every data that the annotator labeled"""
    if len(A) == 0:
        if T != 0:
            A = keras.utils.to_categorical(np.arange(T), num_classes=T) #generate A as one-hot
        else:
            print("ERROR! Needed to pass *T* or *A* argument")
            return
    R_t = A.shape[0] #dimensions to represent annotators
    A_train = [] #set of annotators by data .. A_i
    for i in range(array.shape[0]):
        if index:
            Aindx_i = array[i]
        else:
            Aindx_i = np.where(array[i] !=-1)[0] #get indexs
        A_i = A[Aindx_i] #get representation at indexs
        A_train.append(A_i)
    return np.asarray(A_train), A

def annotations2repeat(annotations):
    """
    assuming that annotations is a 3-dimensional array and with one hot vectors, and annotators
    that does not annotate a data have a one hot vectors of zeroes --> sum over annotators axis
    """
    if len(annotations.shape) ==2:
    	annotations = categorical_representation(annotations)
    return np.sum(annotations,axis=1,dtype='int32')

def annotations2repeat_efficient(obs,no_label=-1):
	"""
	Used when memory error is through over normal "annotations2repeat" function
	"""
	if len(obs.shape) ==1: #variable number of annotations
		N = obs.shape[0]
		K = obs[0].shape[1]
		repeats_obs = np.zeros((N,K),dtype='int32')
		for i in range(N):
		    repeats_obs[i] = obs[i].sum(axis=0)
	elif len(obs.shape) ==2:
		N, T = obs.shape
		K = int(np.max(obs)+1) # assuming that are indexed in order
		repeats_obs = np.zeros((N,K),dtype='int32')
		for i in range(N):
			A_i = np.where(obs[i] != no_label)[0]
			for t in A_i:
				repeats_obs[i,obs[i,t]] +=1
	else:
		repeats_obs = np.sum(obs,axis=1,dtype='int32')
	return repeats_obs

"""
 REEEEEEEEEESCRIBIR

Original/Based representation: (N,T)
Example: X = [  [1, 2,-1, 2,-1]
				[1,-1, 2,-1, 1] ]

Raykar need as one-hot: (N,T,K)
SET:  set_representation(X,needed="onehot")

Group-based Model Global need repeats: (N,K)
SET:  set_representation(X,needed="repeat")

Group-based Model Individual need variable lenth: (N,T_i,K)
SET:  set_representation(X,needed="onehotvar")

Group-based Model Individual need annotator identity: (N,T_i,R_t), (T,R_t)
SET Option1: A_train, A = get_A_il(X, T=T) 
SET Option2: A_train, A = get_A_il(X, A=A) #if you already  had a representation for annotators
"""

def set_representation(obs, needed="onehot"):
    if needed.lower()=="onehot" or needed.lower()=="one-hot":
        return categorical_representation(obs)
    elif needed.lower()=="global":
        return annotations2repeat_efficient(obs)
    elif needed.lower()=='onehotvar' or needed.lower()=='variable':
        return categorical_var_representation(obs)
    elif needed.lower()=='onehotmasked' or needed.lower()=='rodriguesmasked':
        return categorical_masked_representation(obs)