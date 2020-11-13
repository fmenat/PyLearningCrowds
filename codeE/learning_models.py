#logistic regression by sklearn
from sklearn.linear_model import LogisticRegression
def LogisticRegression_Sklearn(epochs):
    """
        solver: Conjugate gradient (divided by hessian) as original Raykar paper
        warm_start set True to incremental fit (training): save last params on fit
    """
    return LogisticRegression(C=1., max_iter=epochs,fit_intercept=True
                       ,solver='newton-cg',multi_class='multinomial',warm_start=True,n_jobs=-1)
    #for sgd solver used "sag"

import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential,Model, clone_model
from keras.layers import *
from keras import backend as K

try:
    from keras.layers import CuDNNLSTM
    GPU_AVAIL = tf.test.is_gpu_available() #'GPU' in str(K.tensorflow_backend.device_lib.list_local_devices()): 
except:
    GPU_AVAIL = False

def LogisticRegression_Keras(input_dim,output_dim, bias=True,embed=False,embed_M=[], BN=False):
    model = Sequential() 
    if embed:
        T, R_t = embed_M.shape
        model.add(Embedding(T, R_t,trainable=False,weights=[embed_M],input_length=1))
        model.add(Reshape([R_t]))
    else:
        model.add(InputLayer(input_shape=input_dim))
    model.add(Dense(output_dim, use_bias=bias)) 
    if BN:
        model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

#MLP Simple
def MLP_Keras(input_dim,output_dim,units,hidden_deep,BN=False,drop=0.0,embed=False,embed_M=[]):
    model = Sequential() 
    if embed:
        if len(embed_M) != 0:
            T, R_t = embed_M.shape
            emd_layer = Embedding(T, R_t,trainable=False,weights=[embed_M],input_length=1)
        else:
            R_t = units
            hidden_deep = hidden_deep-1
            emd_layer = Embedding(input_dim[0], R_t,trainable=True,input_length=1)
        model.add(emd_layer)
        model.add(Reshape([R_t]))
    else:
        model.add(InputLayer(input_shape=input_dim))

    if len(input_dim) > 1:
        model.add(Flatten())
        if BN:
            model.add(BatchNormalization())
    for i in range(hidden_deep): #all the deep layers
        model.add(Dense(units,activation='relu'))
        if BN:
            model.add(BatchNormalization())
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
    model.add(Dense(output_dim))  
    model.add(Activation('softmax'))
    return model

def default_CNN(input_dim,output_dim): #quizas cambiara  CP,CP,CP 
    #weight_decay = 1e-4
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    model.add(Conv2D(32,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    
    #another layer?-yes
    model.add(Conv2D(128,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),strides=1,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25)) #maybe more 512 y 0.5 d dropa
    model.add(Dense(output_dim, activation='softmax'))      
    return model

def default_RNN(input_dim,output_dim):
    #revisar la red de Rodrigues
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    if GPU_AVAIL:
        model.add(CuDNNGRU(64,return_sequences=True))
        model.add(CuDNNGRU(32,return_sequences=False))
    else:
        model.add(GRU(64,return_sequences=True))
        model.add(GRU(32,return_sequences=False))
    model.add(Dense(output_dim, activation='softmax'))     
    return model

def default_RNN_text(input_dim,output_dim,embed_M=[]): 
    model = Sequential() 
    if len(embed_M) != 0:
        T, R_t = embed_M.shape
        emd_layer = Embedding(T, R_t,trainable=False,weights=[embed_M],input_length=input_dim)
        model.add(emd_layer)
    else:

        model.add(InputLayer(input_shape=input_dim))
    if GPU_AVAIL:
        model.add(CuDNNGRU(128,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(CuDNNGRU(64,return_sequences=False)) #o solo una de 64..
    else:
        model.add(GRU(128,return_sequences=True))
        model.add(Dropout(0.25))
        model.add(GRU(64,return_sequences=False)) #o solo una de 64..
    model.add(Dropout(0.25))

    model.add(Dense(output_dim, activation='softmax'))     
    return model

def default_CNN_text(input_dim,output_dim,embed_M=[]):
    model = Sequential() 
    if len(embed_M) != 0:
        T, R_t = embed_M.shape
        emd_layer = Embedding(T, R_t,trainable=False,weights=[embed_M],input_length=input_dim)
        model.add(emd_layer)
    else:
        model.add(InputLayer(input_shape=input_dim))
    model.add(Conv1D(128, 5, activation='relu')) 
    model.add(BatchNormalization())
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.25)) #0 0.5
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(5))
    model.add(Dropout(0.25)) # o 0.2
    model.add(GlobalAveragePooling1D())
    model.add(Dense(output_dim, activation='softmax')) 
    return model

def CNN_simple(input_dim,output_dim,units,hidden_deep,double=False,BN=False,drop=0.0,dense_units=128, global_pool=False): #CP
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    start_unit = units
    for i in range(hidden_deep): #all the deep layers
        model.add(Conv2D(start_unit,(3,3),strides=1,padding='same',activation='relu'))
        if BN:
            model.add(BatchNormalization())
        if double:
            model.add(Conv2D(start_unit,(3,3),strides=1,padding='same',activation='relu'))
            if BN:
                model.add(BatchNormalization())
        model.add(MaxPooling2D(2,2))
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
        start_unit = start_unit*2
    if global_pool:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(Flatten())
    if dense_units!= 0:
        model.add(Dense(dense_units,activation='relu'))
        if BN:
            model.add(BatchNormalization())
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
    model.add(Dense(output_dim, activation='softmax')) 
    return model


def RNN_simple(input_dim,output_dim,units,hidden_deep,drop=0.0,embed=False,len=0,out=0):
    model = Sequential() 
    model.add(InputLayer(input_shape=input_dim))
    if embed:
        model.add(Embedding(input_dim=len,output_dim=out,input_length=input_dim[0]))
    start_unit = units
    for i in range(hidden_deep): #all the deep layers
        if GPU_AVAIL:
            model.add(CuDNNGRU(start_unit,return_sequences= (i<hidden_deep-1) ))
        else:
            model.add(GRU(start_unit,return_sequences= (i<hidden_deep-1) ))
        if drop!= 0 and drop != None and drop != False:
            model.add(Dropout(drop))
        start_unit = start_unit/2 #o mantener
    model.add(Dense(output_dim, activation='softmax')) 
    return model


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as p16
def through_VGG(X,pooling_mode=None):
    """
        Pass data X through VGG 16
        * pooling_mode: as keras say could be None, 'avg' or 'max' (in order to reduce dimensionality)
    """
    X_vgg = p16(X)
    input_tensor=Input(shape=X_vgg.shape[1:])
    modelVGG = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling=pooling_mode ) # LOAD PRETRAINED MODEL 
    return_value = modelVGG.predict(X_vgg)
    return return_value#.reshape(return_value.shape[0],np.prod(return_value.shape[1:]))

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as pIncept
def through_InceptionV3(X,pooling_mode=None):
    """
        Pass data X through Inception V3
    """
    X_incept = pIncept(X)
    input_tensor=Input(shape=X_incept.shape[1:])
    modelInception = InceptionV3(weights='imagenet', include_top=False,input_tensor=input_tensor,pooling=pooling_mode ) # LOAD PRETRAINED MODEL 
    return modelInception.predict(X_incept)

def through_CNNFace(X,weights_path, pooling_mode=None):
    #https://github.com/rcmalli/keras-vggface
    from keras_vggface.vggface import VGGFace
    from keras_vggface.utils import preprocess_input
    #vggface = VGGFace(model='vgg16') # Based on VGG16 architecture -> old paper(2015)
    #vggface = VGGFace(model='resnet50') # Based on RESNET50 architecture -> new paper(2017)
    #vggface = VGGFace(model='senet50') # Based on SENET50 architecture -> new paper(2017) -- falla parece

    vggface = VGGFace(model=weights_path, include_top=False,input_shape=X.shape[1:],pooling=pooling_mode)
    v = 2 #for resnet and senet
    if 'vgg' in weights_path:
        v = 1        
    X_vgg = preprocess_input(X, version=v) 
    return_value = vggface.predict(X_vgg)
    return return_value

def through_VGGFace(X,weights_path, pooling_mode=None):
    """
        Pass data X through VGG-FACE (need weights downloaded)
        https://gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
    """
    X_vgg = p16(X).transpose([0,3,1,2]) 
    modelVGG = vgg_face(weights_path=weights_path,img_sh=X_vgg.shape[1:] ) #weights from VGG-Face

    #include_top=False
    for _ in range(6): #probar con: 1 o 6
        modelVGG.pop()

    K.set_image_data_format('channels_first')
    if pooling_mode=="avg":
        modelVGG.add(GlobalAveragePooling2D())
    elif pooling_mode =="max":
        modelVGG.add(GlobalMaxPooling2D())
        
    return_value = modelVGG.predict(X_vgg)
    K.set_image_data_format('channels_last')
    return return_value #.reshape(return_value.shape[0],np.prod(return_value.shape[1:]))

class Clonable_Model(object):
    def __init__(self, model, input_tensors=None):
        self.non_train_W = {}
        for n, layer in enumerate(model.layers):
            if not layer.trainable:
                self.non_train_W[layer.name] =  model.layers[n].get_weights()
        if input_tensors != None:
            self.inp_shape = K.int_shape(input_tensors)
        elif type(model.layers[0]) == InputLayer:
            self.inp_shape = model.layers[0].input_shape
        else:
            self.inp_shape = model.input_shape
        self.aux_model = clone_model(model) #, input_tensors= self.inp_T)
        
    def get_model(self):
        return_model = clone_model(self.aux_model)#, input_tensors= self.inp_T)
        for n, layer in enumerate(return_model.layers):
            if layer.name in self.non_train_W:
                return_model.layers[n].set_weights( self.non_train_W[layer.name] )
        return_model.build(self.inp_shape)
        return return_model #return a copy of the mode


def meta_init(array):
    def aux_func(shape, dtype=None):
        assert shape == array.shape
        return array.astype(dtype)
    return aux_func
    
from keras.engine.topology import Layer
class CrowdsLayer(Layer):
    def __init__(self, output_dim, num_annotators, conn_type="MW", conf_ma = 0, **kwargs):
        super(CrowdsLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        self.conf_ma = conf_ma

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = self.add_weight("CrowdLayer", (input_shape[-1], self.output_dim, self.num_annotators),
                                            initializer = meta_init(self.conf_ma), 
                                            trainable=True)
        elif self.conn_type == "VW+B":
            # two vectors of weights (one scale and one bias per class) per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                            initializer=keras.initializers.Ones(),
                                            trainable=True))
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                            initializer=keras.initializers.Zeros(),
                                            trainable=True))
        elif self.conn_type == "SW":
            # single weight value per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.num_annotators,1),
                                            initializer=keras.initializers.Ones(),
                                            trainable=True)
        super(CrowdsLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):            
        if self.conn_type == "MW":
            res = tf.tensordot(x, self.kernel, axes=[[-1], [0]]) #like a matrix dot
        elif  self.conn_type == "VW+B" or self.conn_type == "SW":
            out = []
            for r in range(self.num_annotators):
                if self.conn_type == "VW+B":
                    out.append(x * self.kernel[0][:,r] + self.kernel[1][:,r])
                elif self.conn_type == "SW":
                    out.append(x * self.kernel[r,0])
            res = tf.stack(out)
            if len(res.shape) == 3:
                res = tf.transpose(res, [1, 2, 0])
            elif len(res.shape) == 4:
                res = tf.transpose(res, [1, 2, 3, 0])
        return tf.nn.softmax(res, axis=1) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.num_annotators)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_annotators': self.num_annotators,
            'conn_type': self.conn_type,
            'conf_ma': self.conf_ma
        }
        base_config = super(CrowdsLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskedMultiCrossEntropy(object):
    def loss(self, y_true, y_pred):
        vec = -tf.reduce_sum(y_true * tf.log(y_pred), axis = 1)
        mask = tf.equal(y_true[:,0,:], -1) 
        zer = tf.zeros_like(vec)
        loss = K.sum( tf.where(mask, x=zer, y=vec), axis=-1) #sum annotators
        return loss

    def new_loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        y_true = K.clip(y_true, 0, 1) #transform -1 to 0
        vec = -tf.reduce_sum(y_true * tf.log(y_pred), axis = 1)
        return K.sum(vec, axis= -1) #sum over annotators

    def KL_loss(self, y_true, p_z):
        mask  = K.cast(K.not_equal(y_true, -1), K.floatx())
        r_obs = K.sum(y_true*mask, axis=-1)
        T_i   = K.sum(r_obs, axis=-1, keepdims=True)
        mv = r_obs/T_i # on batch size.. (all annotators) 
       
        #KL CALCULATION
        p_z_clip = K.clip(p_z, K.epsilon(), 1)
        mv = K.clip(mv, K.epsilon(), 1)
        KL_mv = K.sum(p_z_clip* K.log(p_z_clip/mv), axis=-1) 
        #similar a VAE sobre imagenes, KL en la practica (implementancion) pesa lo mismo que un pixel
        #KL pesa lo mismo que UNA anotadora de las que etiquetaron
        # OJO, para evitar el tamaño de imagen o la cantidad de anotadoras, el lambda dependera del problema.
        return KL_mv #T_i*KL_mv  

    def loss_w_prior(self, l=1, p_z=None):
        def loss(y_true, y_pred):
            loss_masked = self.new_loss(y_true, y_pred)
            KL_mv = 0
            if l != 0 and type(p_z) != type(None):
                KL_mv = self.KL_loss(y_true, p_z)

            return loss_masked + l* KL_mv 
        return loss

class NoiseLayer(Layer):
    def __init__(self, units, conf_ma=0, **kwargs):
        super(NoiseLayer, self).__init__(**kwargs)
        self.units = units
        self.conf_ma = conf_ma

    def build(self, input_shape):
        self.kernel = self.add_weight("NoiseChannel", (input_shape[-1], self.units),
                                            initializer = meta_init(self.conf_ma), 
                                            trainable=True)
        super(NoiseLayer, self).build(input_shape)

    def call(self, x, mask=None):
        channel_matrix = tf.nn.softmax(self.kernel, axis=-1)
        return tf.tensordot(x, channel_matrix, axes=[[-1], [0]]) #like a matrix dot 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = { 
            'units': self.units,
            'conf_ma': self.conf_ma
            }
        base_config = super(NoiseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))