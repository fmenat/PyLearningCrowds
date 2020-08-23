import gc, keras, time, sys
import numpy as np
from .learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from .learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple, default_CNN_text, default_RNN_text, Clonable_Model #deep learning
from .representation import *
from .utils import estimate_batch_size, EarlyStopRelative, pre_init_F, clusterize_annotators

class LabelAgg(object): #no predictive model
    def __init__(self, scenario="global", sparse=False): 
        self.scenario = scenario
        self.sparse = sparse #only for individual

    def infer(self, labels, method, weights=[1], onehot=False): #weights for only individual dense
        method = method.lower()
        
        if self.scenario == "global":
            if len(labels.shape) == 2:
                r_obs = labels.astype('float32')
            else:
                r_obs = set_representation(labels, needed="onehot").astype('float32')                

        elif self.scenario == "individual":
            if self.sparse:
                if len(labels.shape) == 1:
                    y_variable_categorical = labels                    
                else:
                    y_variable_categorical = set_representation(labels,'variable')

                N = y_variable_categorical.shape[0]
                K = y_variable_categorical[0].shape[1]
                r_obs = np.zeros((N,K),dtype='int32')
                for i in range(N):
                    r_obs[i] = y_variable_categorical[i].sum(axis=0)
            else: #dense
                if len(labels.shape) ==3:
                    y_obs_categorical = labels.astype('float32')
                else:
                    y_obs_categorical = set_representation(labels,'onehot').astype('float32')                

                weights = np.asarray(weights, dtype='float32')
                r_obs = (y_obs_categorical*weights[None,:,None]).sum(axis=1)

        if method == 'softmv': 
            to_return = r_obs/r_obs.sum(axis=-1, keepdims=True)

        elif method == "hardmv":
            to_return = r_obs.argmax(axis=1) #over classes axis
            if onehot: 
                to_return = keras.utils.to_categorical(to_return)

        return to_return

    def predict(self, *args):
        return self.infer(*args)


class LabelInf_EM(object): #DS
    def __init__(self,init_Z='softmv', priors=0, fast=False, DTYPE_OP='float32'):
        self.DTYPE_OP = DTYPE_OP
        self.init_Z = init_Z.lower()
        self.set_priors(priors)
        self.fast = fast #fast DS method.. 
        
        self.Keps = keras.backend.epsilon()
        self.init_done = False
        
    def get_marginalZ(self):
        return self.z_marginal.copy()
    def get_confusionM(self):
        return self.betas.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()

    def set_priors(self, priors):
        if type(priors) == str:
            if priors.lower() == "laplace":
                priors = 1
            elif priors.lower() == "none":
                priors = 0
            else:
                raise Exception('Prior not valid')
                
        priors = np.asarray(priors)
        if len(priors.shape)==0:
            priors = np.expand_dims(priors, axis=(0,1,2))
        elif len(priors.shape)==1:
            priors=np.expand_dims(priors,axis=(1,2))
        elif len(priors.shape)==2:
            priors=np.expand_dims(priors,axis=2)
        self.Mpriors = priors
        
    def init_E(self, y_ann, method=""): #Majority voting start
        print("Initializing new EM...")
        self.N, self.T, self.K = y_ann.shape
        #init p(z|x)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="individual")
        init_GT = label_A.infer(y_ann, method=method, onehot=True)
        #init betas
        self.betas = np.zeros((self.T,self.K,self.K),dtype=self.DTYPE_OP)
        #init z_marginal
        self.z_marginal = np.zeros((self.K),dtype=self.DTYPE_OP)
        #init qi
        self.Qi_k = init_GT
        print("Z marginal shape",self.z_marginal.shape)
        print("Betas shape: ",self.betas.shape)
        print("Q estimate shape: ",self.Qi_k.shape)
        self.init_done=True
        
    def E_step(self,y_ann):        
        prob_Lx_z = np.tensordot(y_ann, np.log(self.betas + self.Keps),axes=[[1,2],[0,2]])
        aux = np.log(self.z_marginal[None,:] + self.Keps) + prob_Lx_z
        
        self.sum_unnormalized_q = np.exp(aux).sum(axis=-1)# p(L_x) = p(y1,..,yt)

        self.Qi_k = np.exp(aux-aux.max(axis=-1,keepdims=True)).astype(self.DTYPE_OP) #return to actually values
        self.Qi_k = self.Qi_k/self.Qi_k.sum(axis=-1, keepdims=True)#normalize q

    def C_step(self):        
        Qi_hard = np.zeros(self.Qi_k.shape, dtype=self.DTYPE_OP)
        label_argmax = self.Qi_k.argmax(axis=-1)
        Qi_hard[np.arange(self.N), label_argmax] = 1 #one hot version
        self.Qi_k = Qi_hard
        
    def M_step(self, y_ann): 
        #-------> z_marginals 
        self.z_marginal = self.Qi_k.mean(axis=0)
       
        #-------> beta
        self.betas = np.tensordot(self.Qi_k, y_ann, axes=[[0],[0]]).transpose(1,0,2)
        self.betas += self.Mpriors
        
        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1
        
        self.betas = self.betas.astype(self.DTYPE_OP)
        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize
    
    def compute_logL(self):
        logL_priors = np.sum(self.Mpriors* np.log(self.betas+ self.Keps))
        return np.sum(np.log( self.sum_unnormalized_q +self.Keps)) + logL_priors
        
    def train(self, y_ann, max_iter=50,relative=True,tolerance=3e-2):   
        if not self.init_done:
            self.init_E(y_ann)

        logL = []
        stop_c = False
        old_betas,tol = np.inf, np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d \nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(y_ann)
            print(" done,  E step:",end='',flush=True)
            self.E_step(y_ann)
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL())  #compute lowerbound
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.fast:
                self.C_step()
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2]) #absolute
                if relative: #relative
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps))
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten() 
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance):
                stop_c = True
        print("Finished training")
        return logL
    
    def fit(self, y_ann,max_iter=50,tolerance=3e-2):
        return self.train(y_ann, max_iter =max_iter, tolerance=tolerance)
    
    def infer(self):
        return self.get_qestimation()
        
    def predict(self):
        return self.infer()
   
    def get_ann_confusionM(self):
        return self.get_confusionM()


class ModelInf_EM(object):
    def __init__(self, init_Z='softmv', n_init_Z= 0, priors=0, DTYPE_OP='float32'):
        self.DTYPE_OP = DTYPE_OP
        self.init_Z = init_Z.lower()
        self.n_init_Z = n_init_Z
        self.set_priors(priors)

        self.compile=False
        self.Keps = keras.backend.epsilon()
        self.init_done = False
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        """Get confusion matrices of every annotator p(yo^t|,z)"""  
        return self.betas.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()

    def set_model(self, model, optimizer="adam", epochs=1, batch_size=32):
        self.base_model = model
        #params:
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.base_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy') 
        self.compile = True
        self.max_Bsize_base = estimate_batch_size(self.base_model)
        
    def set_priors(self, priors):
        if type(priors) == str:
            if priors.lower() == "laplace":
                priors = 1
            elif priors.lower() == "none":
                priors = 0
            else:
                raise Exception('Prior not valid')
                
        priors = np.asarray(priors)
        if len(priors.shape)==0:
            priors = np.expand_dims(priors, axis=(0,1,2))
        elif len(priors.shape)==1:
            priors=np.expand_dims(priors,axis=(1,2))
        elif len(priors.shape)==2:
            priors=np.expand_dims(priors,axis=2)
        self.Mpriors = priors
            
    def get_predictions(self,X):
        return self.base_model.predict(X,batch_size=self.max_Bsize_base)
        
    def init_E(self,y_ann, method=""): #Majority voting start
        print("Initializing new EM...")
        self.N, self.T, self.K = y_ann.shape
        #init p(z|x)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="individual")
        init_GT = label_A.infer(y_ann, method=method, onehot=True)
        #init betas
        self.betas = np.zeros((self.T,self.K,self.K),dtype=self.DTYPE_OP)
        #init qi
        self.Qi_k = init_GT
        print("Betas shape: ",self.betas.shape)
        print("Q estimate shape: ",self.Qi_k.shape)
        self.init_done=True
                
    def E_step(self,X,y_ann,predictions=[]):
        if len(predictions)==0:
            predictions = self.get_predictions(X)
        
        prob_Lx_z = np.tensordot(y_ann, np.log(self.betas + self.Keps),axes=[[1,2],[0,2]])
        aux = np.log(predictions + self.Keps) + prob_Lx_z
        
        self.sum_unnormalized_q = np.sum(np.exp(aux),axis=-1) # p(L_x) = p(y1,..,yt)

        self.Qi_k = np.exp(aux-aux.max(axis=-1,keepdims=True)).astype(self.DTYPE_OP) #return to actually values
        self.Qi_k = self.Qi_k/self.Qi_k.sum(axis=-1, keepdims=True) #normalize q

    def M_step(self,X,y_ann): 
        #-------> base model ---- train to learn p(z|x)
        self.base_model.fit(X,self.Qi_k,batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
        
        #-------> beta
        self.betas = np.tensordot(self.Qi_k, y_ann, axes=[[0],[0]]).transpose(1,0,2)
        self.betas += self.Mpriors

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1
        
        self.betas = self.betas.astype(self.DTYPE_OP)
        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize
    
    def compute_logL(self):
        logL_priors = np.sum(self.Mpriors* np.log(self.betas+ self.Keps))
        return np.sum( np.log( self.sum_unnormalized_q +self.Keps)) + logL_priors
        
    def train(self,X_train,y_ann,max_iter=50,relative=True,tolerance=3e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(y_ann)
            if self.n_init_Z != 0:
                pre_init_F(self.base_model,X_train,self.Qi_k,self.n_init_Z,batch_size=self.batch_size)

        self.input_dim = X_train.shape[1:]

        logL = []
        stop_c = False
        old_betas,tol = np.inf, np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d \nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train,y_ann)
            print(" done,  E step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x)
            self.E_step(X_train,y_ann,predictions)
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL())  #compute lowerbound
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2]) #absolute
                if relative: #relative
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps))
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten() 
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance):
                stop_c = True
        print("Finished training")
        return logL
            
    def stable_train(self,X,y_ann,max_iter=50,tolerance=3e-2):
        logL_hist = self.train(X,y_ann,max_iter=max_iter,relative=True,tolerance=tolerance)
        return logL_hist
    
    def multiples_run(self,Runs,X,y_ann,max_iter=50,tolerance=3e-2):  #tolerance can change
        if Runs==1:
            return self.stable_train(X,y_ann,max_iter=max_iter,tolerance=tolerance), 0
     
        found_betas = []
        found_model = []
        found_logL = []
        iter_conv = []
        if type(self.base_model.layers[0]) == keras.layers.InputLayer:
            obj_clone = Clonable_Model(self.base_model) #architecture to clone
        else:
            it = keras.layers.Input(shape=self.base_model.input_shape[1:])
            obj_clone = Clonable_Model(self.base_model, input_tensors=it) #architecture to clon

        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths            
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,y_ann,max_iter=max_iter,relative=True,tolerance=tolerance) 
            found_betas.append(self.betas.copy())
            found_model.append(self.base_model.get_weights()) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        self.base_model = obj_clone.get_model() #change
        self.base_model.set_weights(found_model[indexs_sort[0]])
        self.E_step(X,y_ann,predictions=self.get_predictions(X)) #to set up Q
        print(Runs,"runs over Raykar, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]

    def fit(self,X,Y, runs = 1, max_iter=50, tolerance=3e-2):
        return self.multiples_run(runs,X,Y,max_iter=max_iter,tolerance=tolerance)

    def get_ann_confusionM(self):
        return self.get_confusionM()

    def get_predictions_annot(self,X,data=[]):
        if len(data) != 0:
            p_z = data
        else:
            p_z = self.get_predictions(X)
        predictions_a= np.tensordot(p_z ,self.betas,axes=[[1],[1]] ) # sum_z p(z|xi) * p(yo|z,t)
        return predictions_a  
    
class ModelInf_EM_CMM(object): 
    def __init__(self, M, init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32'):
        self.DTYPE_OP = DTYPE_OP
        self.init_Z = init_Z.lower()
        self.n_init_Z = n_init_Z
        self.M = M #groups of annotators
        self.set_priors(priors)

        self.compile=False
        self.Keps = keras.backend.epsilon()
        self.init_done = False
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        return self.betas.copy()
    def get_alpha(self):
        return self.alphas.copy()
    def get_qestimation(self):
        return self.Qij_mk.copy()

    def set_model(self, model, optimizer="adam", epochs=1, batch_size=32):
        self.base_model = model
        #params:
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.base_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy') 
        self.compile = True
        self.max_Bsize_base = estimate_batch_size(self.base_model)

    def set_priors(self, priors):
        if type(priors) == str:
            if priors.lower() == "laplace":
                priors = 1
            elif priors.lower() == "none":
                priors = 0
            else:
                raise Exception('Prior not valid')
                
        priors = np.asarray(priors)
        if len(priors.shape)==0:
            priors = np.expand_dims(priors, axis=(0,1,2))
        elif len(priors.shape)==1:
            priors=np.expand_dims(priors,axis=(1,2))
        elif len(priors.shape)==2:
            priors=np.expand_dims(priors,axis=2)
        self.Bpriors = priors

    def get_predictions(self,X):
        return self.base_model.predict(X,batch_size=self.max_Bsize_base) #fast predictions

    def init_E(self, r_ann, method=""):
        print("Initializing new EM...")
        self.N, self.K = r_ann.shape

        #-------> init Majority voting    
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="global")
        self.init_GT = label_A.infer(r_ann, method=method, onehot=True)

        #-------> init alpha
        self.alpha_init = clusterize_annotators(self.init_GT,M=self.M,bulk=False,cluster_type='mv_close',DTYPE_OP=self.DTYPE_OP) #clusteriza en base mv
        
         #-------> Initialize p(z=k,g=m|xi,y=j)
        self.Qij_mk = self.alpha_init[:,:,:,None]*self.init_GT[:,None,None,:]

        #-------> init betas
        self.betas = np.zeros((self.M,self.K,self.K),dtype=self.DTYPE_OP) 
        #-------> init alphas
        self.alphas = np.zeros((self.M),dtype=self.DTYPE_OP)
        print("Alphas: ",self.alphas.shape)
        print("Betas: ",self.betas.shape)
        print("Q estimate: ",self.Qij_mk.shape)
        self.init_done=True
        
    def E_step(self, X,  predictions=[]):
        if len(predictions)==0:
            predictions = self.get_predictions(X)

        z_new = np.log( np.clip(predictions, self.Keps, 1.))[:,None,None,:] 
        a_new = np.log( np.clip(self.alphas, self.Keps, 1.))[None,None,:,None] 
        b_new = (np.log( np.clip(self.betas, self.Keps, 1.))[None,:,:,:]).transpose(0,3,1,2) 
        
        self.Qij_mk = np.exp(z_new + a_new + b_new)
        self.aux_for_like = (self.Qij_mk.sum(axis=-1)).sum(axis=-1) #p(y=j|x) --marginalized
        self.Qij_mk = self.Qij_mk/self.aux_for_like[:,:,None,None] #normalize

    def M_step(self, X, r_ann): 
        QRij_mk = self.Qij_mk*r_ann[:,:,None,None]
        
        #-------> base model
        r_estimate = QRij_mk.sum(axis=(1,2))
        self.base_model.fit(X, r_estimate,batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
    
        #-------> alpha 
        self.alphas = QRij_mk.sum(axis=(0,1,3)) 
        self.alphas = self.alphas/self.alphas.sum(axis=-1,keepdims=True) #p(g) -- normalize

        #-------> beta
        self.betas = (QRij_mk.sum(axis=0)).transpose(1,2,0)            
        self.betas += self.Bpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1

        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self, r_ann):
        """ Compute the log-likelihood of the optimization schedule"""
        logL_priors = np.sum(self.Bpriors* np.log(self.betas+ self.Keps))
        return np.tensordot(r_ann , np.log(self.aux_for_like+self.Keps))+ logL_priors 
                                                  
    def train(self,X_train, r_ann, max_iter=50,relative=True,tolerance=3e-2):
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(r_ann)
            if self.n_init_Z != 0:
                pre_init_F(self.base_model,X_train, self.init_GT, self.n_init_Z,batch_size=self.batch_size)
                self.init_GT = None
        
        logL = []
        stop_c = False
        tol,old_betas,old_alphas = np.inf,np.inf,np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train, r_ann) #Need X_i, r_ann
            print(" done,  E step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x)  #--- revisar si sacar
            self.E_step(X_train, predictions) 
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL(r_ann))
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])                    
                if relative:
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                tol3 = np.mean(np.abs(self.alphas-old_alphas)/(old_alphas+self.Keps)) #alphas
                print("Tol1: %.5f\tTol2: %.5f\tTol3: %.5f\t"%(tol,tol2,tol3),end='',flush=True)
            old_betas = self.betas.flatten().copy()         
            old_alphas = self.alphas.copy()
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance): #alphas fuera: and tol3<=tolerance
                stop_c = True 
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self,X,r_ann,max_iter=50,tolerance=3e-2):
        logL_hist = self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance,relative=True)
        return logL_hist
    
    def multiples_run(self,Runs,X,r_ann,max_iter=50,tolerance=3e-2): 
        if Runs==1:
            return self.stable_train(X,r_ann,max_iter=max_iter,tolerance=tolerance), 0
            
        found_betas = []
        found_alphas = []
        found_model = [] 
        found_logL = []
        iter_conv = []

        if type(self.base_model.layers[0]) == keras.layers.InputLayer:
            obj_clone = Clonable_Model(self.base_model) #architecture to clone
        else:
            it = keras.layers.Input(shape=self.base_model.input_shape[1:])
            obj_clone = Clonable_Model(self.base_model, input_tensors=it) #architecture to clon

        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance,relative=True)
            
            found_betas.append(self.betas.copy())
            found_alphas.append(self.alphas.copy())
            found_model.append(self.base_model.get_weights())
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        self.alphas = found_alphas[indexs_sort[0]].copy()
        self.base_model = obj_clone.get_model() #change
        self.base_model.set_weights(found_model[indexs_sort[0]])
        self.E_step(X) #to set up Q
        print(Runs,"runs over CMM, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]

    def fit(self,X,R, runs = 1, max_iter=50, tolerance=3e-2):
        return self.multiples_run(runs,X,R,max_iter=max_iter,tolerance=tolerance)
    
    def get_predictions_groups(self,X,data=[]):
        if len(data) != 0:
            prob_Z_ik = data
        else:
            prob_Z_ik = self.get_predictions(X)
        return np.tensordot(prob_Z_ik ,self.betas,axes=[[1],[1]] ) #sum_z p(z|xi) * p(yo|z,g)

    def get_global_confusionM(self):
        return np.sum(self.betas*self.alphas[:,None,None],axis=0)
    
    def get_ann_confusionM(self,X, Y):
        prob_G_tm = self.annotations_2_group(Y, data=X)        
        return np.tensordot(prob_G_tm, self.get_confusionM(),axes=[[0],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)

    def annotations_2_group(self,annotations,data=[],pred=[],no_label_sym = -1):
        """
            Map some annotations to some group model by the confusion matrices, p(g| {x_l,y_l})
        """
        if len(pred) != 0:
            predictions_m = pred #if prediction_m is passed
        elif len(data) !=0: 
            predictions_m = self.get_predictions_groups(data) #if data is passed
        else:
            print("Error, in order to match annotations to a group you need pass the data X or the group predictions")
            return
            
        result = np.log(self.get_alpha()+self.Keps) #si les saco Keps?
        aux_annotations = [(i,annotation) for i, annotation in enumerate(annotations) if annotation != no_label_sym]
        for i, annotation in aux_annotations:
            if annotation != no_label_sym: #if label it
                for m in range(self.M):
                    result[m] += np.log(predictions_m[i,m,annotation]+self.Keps)
        result = np.exp(result - result.max(axis=-1, keepdims=True) ) #invert logarithm in safe way
        return result/result.sum()


class ModelInf_EM_CMOA(object):
    def __init__(self, M, init_Z="softmv", n_init_Z=0, n_init_G=0, priors=0, DTYPE_OP='float32'): 
        self.DTYPE_OP = DTYPE_OP
        self.init_Z = init_Z.lower()
        self.n_init_Z = n_init_Z
        self.n_init_G = n_init_G
        self.M = M #groups of annotators
        self.set_priors(priors)

        self.compile_z = False
        self.compile_g = False
        self.Keps = keras.backend.epsilon()
        self.init_done = False
        
    def get_basemodel(self):
        return self.base_model
    def get_groupmodel(self):
        return self.group_model
    def get_confusionM(self):
        return self.betas.copy()
    def get_qestimation(self):
        return self.reshape_il(self.Qil_mk.copy())
        
    def set_model(self, model, optimizer="adam", epochs=1, batch_size=32, ann_model=None):
        #params:
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.base_model = model
        self.base_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy') 
        self.compile_z = True
        self.max_Bsize_base = estimate_batch_size(self.base_model)

        if type(ann_model) != type(None):
            self.set_ann_model(ann_model)
        
    def set_ann_model(self, model, optimizer=None, epochs=None):
        if type(optimizer) == type(None): 
            optimizer = self.optimizer

        if type(epochs) == type(None): 
            self.group_epochs = self.epochs #set epochs of base model
        else:
            self.group_epochs = epochs

        self.group_model = model
        self.group_model.compile(optimizer=optimizer, loss='categorical_crossentropy') 
        self.compile_g = True
        self.max_Bsize_group = estimate_batch_size(self.group_model)

    def set_priors(self, priors):
        if type(priors) == str:
            if priors.lower() == "laplace":
                priors = 1
            elif priors.lower() == "none":
                priors = 0
            else:
                raise Exception('Prior not valid')
                
        priors = np.asarray(priors)
        if len(priors.shape)==0:
            priors = np.expand_dims(priors, axis=(0,1,2))
        elif len(priors.shape)==1:
            priors=np.expand_dims(priors,axis=(1,2))
        elif len(priors.shape)==2:
            priors=np.expand_dims(priors,axis=2)
        self.Bpriors = priors

    def get_predictions_z(self, X):
        return self.base_model.predict(X, batch_size=self.max_Bsize_base) 

    def get_predictions_g(self, A):
        return self.group_model.predict(A, batch_size=self.max_Bsize_group)

    def flatten_il(self,array):
        return np.concatenate(array)

    def reshape_il(self,array):
        to_return = []
        sum_Ti_n1 = 0
        for i in range(self.N):
            sum_Ti_n   = sum_Ti_n1
            sum_Ti_n1  = sum_Ti_n1 + self.T_i[i]
            to_return.append( array[sum_Ti_n : sum_Ti_n1] )
        del array
        gc.collect()
        return np.asarray(to_return)

    def init_E(self, y_ann_var, A_idx_var, method=""):
        print("Initializing new EM...")
        self.N = len(y_ann_var)
        self.K = y_ann_var[0].shape[1]
        self.T_i = [y_ann.shape[0] for y_ann in y_ann_var] 
        self.BS_groups = np.ceil(self.batch_size*np.mean(self.T_i)).astype('int') #batch should be prop to T_i

        #-------> init Majority voting    
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="individual", sparse=True)
        self.init_GT = label_A.infer(y_ann_var, method=method, onehot=True)
        
        #------->init p(g|a)
        first_l = self.group_model.layers[0]
        if type(first_l) == keras.layers.Embedding: #if there is embedding layer
            A_embedding = first_l.get_weights()[0]
            probas_t =  clusterize_annotators(A_embedding,M=self.M,bulk=True,cluster_type='previous',DTYPE_OP=self.DTYPE_OP)
        else:
            probas_t =  clusterize_annotators(y_ann_var,M=self.M,bulk=True,cluster_type='conf_flatten',data=[A_idx_var,self.init_GT],DTYPE_OP=self.DTYPE_OP)
        
        #-------> Initialize p(z=k,g=m|xi,y,a)
        self.Qil_mk = []
        self.alpha_init = []
        for i in range(self.N):
            t_idxs = A_idx_var[i] #indexs of annotators that label pattern "i"
            self.alpha_init.append( probas_t[t_idxs] )#preinit over alphas
            self.Qil_mk.append( probas_t[t_idxs][:,:,None] * self.init_GT[i][None,None,:] ) 
        self.Qil_mk = self.flatten_il(self.Qil_mk)

        #-------> init betas
        self.betas = np.zeros((self.M,self.K,self.K),dtype=self.DTYPE_OP)        
        print("Betas: ",self.betas.shape)
        print("Q estimate: ",self.Qil_mk.shape)
        gc.collect()
        self.init_done = True
       
    def E_step(self, X, y_ann_flatten, A_idx_flatten, predictions_Z=[], predictions_G=[]):
        if len(predictions_Z)==0:
            predictions_Z = self.get_predictions_z(X)
        if len(predictions_G)==0:
            A = np.unique(A_idx_flatten).reshape(-1,1) # A: annotator identity set
            predictions_G = self.get_predictions_g(A)

        predictions_G = np.log( np.clip(predictions_G, self.Keps, 1.) ) #safe logarithmn
        predictions_Z = np.log( np.clip(predictions_Z, self.Keps, 1.) ) #safe logarithmn
        b_aux = np.tensordot(y_ann_flatten, np.log(self.betas + self.Keps),axes=[[1],[2]]) #safe logarithmn

        lim_sup_i = 0
        for i, t_i in enumerate(self.T_i):
            lim_sup_i  += t_i
            lim_inf_i  = lim_sup_i - t_i
            b_new      = b_aux [ lim_inf_i:lim_sup_i ]
            prob_G_lm  = predictions_G[ A_idx_flatten[ lim_inf_i:lim_sup_i ] ] #get group predictions of annotators at indexs "l"

            self.Qil_mk[lim_inf_i: lim_sup_i] = np.exp(predictions_Z[i][None,None,:] + prob_G_lm[:,:,None] + b_new)  
        self.aux_for_like = self.Qil_mk.sum(axis=(1,2)) #p(y|x,a) --marginalized
        self.Qil_mk   = self.Qil_mk/self.aux_for_like[:,None,None] #normalize
    
    def M_step(self, X, y_ann_flatten, A_idx_flatten): 
        #-------> base model  
        r_estimate = np.zeros((self.N,self.K),dtype=self.DTYPE_OP)
        lim_sup = 0
        for i, t_i in enumerate(self.T_i):
            lim_sup  += t_i
            r_estimate[i] = self.Qil_mk[lim_sup-t_i : lim_sup].sum(axis=(0,1)) #create the "estimate"/"ground truth"
        self.base_model.fit(X, r_estimate, batch_size=self.batch_size, epochs=self.epochs,verbose=0) 

        #-------> alpha 
        Qil_m_flat = self.Qil_mk.sum(axis=-1)  #qil(m)
        self.group_model.fit(A_idx_flatten, Qil_m_flat, batch_size=self.BS_groups, epochs=self.group_epochs,verbose=0)

        #-------> beta
        self.betas =  np.tensordot(self.Qil_mk, y_ann_flatten , axes=[[0],[0]]) # ~p(yo=j|g,z) 
        self.betas += self.Bpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1

        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        logL_priors = np.sum(self.Bpriors* np.log(self.betas+ self.Keps))
        return np.sum( np.log(self.aux_for_like+self.Keps) )  #safe logarithm
                                                  
    def train(self, X_train, y_ann_var, A_idx_var, max_iter=50,relative=True,tolerance=3e-2):
        if not self.compile_z:
            print("You need to create the model first, set .define_model")
            return
        if len(y_ann_var.shape) != 1 or len(A_idx_var.shape) != 1:
            print("ERROR! Needed Y and T_idx in variable length array")
            return

        y_ann_flatten, A_idx_flatten = self.flatten_il(y_ann_var), self.flatten_il(A_idx_var)
        A = np.unique(A_idx_flatten).reshape(-1,1) # A: annotator identity set
        if not self.init_done:
            self.init_E(y_ann_var, A_idx_var)
            if self.n_init_Z != 0:
                pre_init_F(self.base_model, X_train, self.init_GT, self.n_init_Z, batch_size=self.batch_size)
                self.init_GT = None
            if self.n_init_G != 0:
                pre_init_F(self.group_model, A_idx_flatten, self.flatten_il(self.alpha_init), self.n_init_G, batch_size=self.BS_groups)

        logL = []
        stop_c = False
        tol,old_betas = np.inf,np.inf
        self.current_iter = 1
        while(not stop_c):
            start_time = time.time()
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            self.M_step(X_train, y_ann_flatten, A_idx_flatten)
            print(" done,  E step:",end='',flush=True)
            predictions_z = self.get_predictions_z(X_train)  # p(z|x)
            predictions_g = self.get_predictions_g(A) # p(g|t)
            self.E_step(X_train, y_ann_flatten, A_idx_flatten, predictions_z, predictions_g)
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL())
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])                    
                if relative:
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten().copy()         
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance):
                stop_c = True 
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self, X, y_ann_var, A_idx_var, max_iter=50,tolerance=3e-2):
        logL_hist = self.train(X, y_ann_var, A_idx_var, max_iter=max_iter,tolerance=tolerance)
        return logL_hist
    
    def multiples_run(self,Runs,X, y_ann_var, A_idx_var, max_iter=50,tolerance=3e-2): 
        if Runs==1:
            return self.stable_train(X, y_ann_var, A_idx_var, max_iter=max_iter,tolerance=tolerance), 0
            
        found_betas = []
        found_model_g = []
        found_model_z = []
        found_logL = []
        iter_conv = []
        if type(self.base_model.layers[0]) == keras.layers.InputLayer:
            obj_clone_z = Clonable_Model(self.base_model) #architecture to clone
        else:
            it = keras.layers.Input(shape=self.base_model.input_shape[1:])
            obj_clone_z = Clonable_Model(self.base_model, input_tensors=it) #architecture to clon

        if type(self.group_model.layers[0]) == keras.layers.InputLayer:
            obj_clone_g = Clonable_Model(self.group_model) #architecture to clone
        else:
            it = keras.layers.Input(shape=self.group_model.input_shape[1:])
            obj_clone_g = Clonable_Model(self.group_model, input_tensors=it) #architecture to clon

        for run in range(Runs):
            self.base_model = obj_clone_z.get_model() #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)
            
            self.group_model = obj_clone_g.get_model() #reset-weigths
            self.group_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X, y_ann_var, A_idx_var, max_iter=max_iter,tolerance=tolerance)
            
            found_betas.append(self.betas.copy())
            found_model_g.append(self.group_model.get_weights())
            found_model_z.append(self.base_model.get_weights()) 
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model, self.group_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.betas = found_betas[indexs_sort[0]].copy()
        self.base_model = obj_clone_z.get_model()
        self.base_model.set_weights(found_model_z[indexs_sort[0]])
        self.group_model = obj_clone_g.get_model()
        self.group_model.set_weights(found_model_g[indexs_sort[0]]) 
        self.E_step(X, self.flatten_il(y_ann_var), self.flatten_il(A_idx_var)) #to set up Q
        print(Runs,"runs over C-MoA, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]

    def fit(self,X, y_ann_var, A_idx_var, runs = 1, max_iter=50, tolerance=3e-2):
        return self.multiples_run(runs, X, y_ann_var, A_idx_var, max_iter=max_iter,tolerance=tolerance)
    
    def get_global_confusionM(self, prob_Gt):
        alphas = np.mean(prob_Gt, axis=0)
        return np.sum(self.betas*alphas[:,None,None],axis=0)

    def get_ann_confusionM(self, A):
        prob_G_t = self.get_predictions_g(A)
        return np.tensordot(prob_G_t, self.get_confusionM(),axes=[[1],[0]])  #p(y^o|z,t) = sum_g p(g|t) * p(yo|z,g)

    def get_predictions_groups(self,X,data=[]):
        if len(data) != 0:
            prob_Z_ik = data
        else:
            prob_Z_ik = self.get_predictions_z(X)
        return np.tensordot(prob_Z_ik ,self.betas,axes=[[1],[1]] ) #sum_z p(z|xi) * p(yo|z,g)



class ModelInf_EM_G(object): 
    def __init__(self, init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32'):
        self.DTYPE_OP = DTYPE_OP
        self.init_Z = init_Z.lower()
        self.n_init_Z = n_init_Z
        self.set_priors(priors)

        self.compile=False
        self.Keps = keras.backend.epsilon()
        self.init_done = False
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        return self.beta.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()

    def set_model(self, model, optimizer="adam", epochs=1, batch_size=32):
        self.base_model = model
        #params:
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.base_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy') 
        self.compile = True
        self.max_Bsize_base = estimate_batch_size(self.base_model)

    def set_priors(self, priors):
        if type(priors) == str:
            if priors.lower() == "laplace":
                priors = 1
            elif priors.lower() == "none":
                priors = 0
            else:
                raise Exception('Prior not valid')
                
        priors = np.asarray(priors)
        if len(priors.shape)==0:
            priors = np.expand_dims(priors, axis=(0,1))
        elif len(priors.shape)==1:
            priors=np.expand_dims(priors,axis=(1))
        self.Bpriors = priors

    def get_predictions(self,X):
        return self.base_model.predict(X,batch_size=self.max_Bsize_base) #fast predictions

    def init_E(self, r_ann, method=""):
        print("Initializing new EM...")
        self.N, self.K = r_ann.shape

        #-------> Initialize p(z=k|xi,ri)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="global")
        init_GT = label_A.infer(r_ann, method=method, onehot=True)
        self.Qi_k = init_GT

        #-------> init betas
        self.beta = np.zeros((self.K,self.K),dtype=self.DTYPE_OP) 
        print("Beta: ",self.beta.shape)
        print("Q estimate: ",self.Qi_k.shape)
        self.init_done=True
        
    def E_step(self, X, r_ann, predictions=[]):
        if len(predictions)==0:
            predictions = self.get_predictions(X)

        prob_Rx_z = np.tensordot(r_ann, np.log( np.clip(self.beta, self.Keps, 1.)), axes=[[1],[1]])
        aux = np.log(predictions + self.Keps) + prob_Rx_z
        
        self.Qi_k = np.exp(aux).astype(self.DTYPE_OP) #return to actually values
        self.aux_for_like = self.Qi_k.sum(axis=-1) #p(R_x,x)
        self.Qi_k = self.Qi_k/self.aux_for_like[:,None]#normalize q
    
    def M_step(self, X, r_ann): 
        #-------> base model
        self.base_model.fit(X, self.Qi_k, batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
    
        #-------> beta
        self.beta = np.tensordot(self.Qi_k, r_ann, axes=[[0],[0]])
        self.beta += self.Bpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.beta.sum(axis=-1) == 0
        self.beta[mask_zero] = 1

        self.beta = self.beta.astype(self.DTYPE_OP)
        self.beta = self.beta/self.beta.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        """ Compute the log-likelihood of the optimization schedule"""
        logL_priors = np.sum(self.Bpriors* np.log(self.beta+ self.Keps))
        return np.sum( np.log( self.aux_for_like +self.Keps)) + logL_priors
                                                  
    def train(self,X_train, r_ann, max_iter=50,relative=True,tolerance=3e-2):
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(r_ann)
            if self.n_init_Z != 0:
                pre_init_F(self.base_model,X_train, self.Qi_k, self.n_init_Z,batch_size=self.batch_size)
        
        logL = []
        stop_c = False
        tol,old_betas,old_alphas = np.inf,np.inf,np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train, r_ann) #Need X_i, r_ann
            print(" done,  E step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x)  #--- revisar si sacar
            self.E_step(X_train, r_ann, predictions) 
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL())
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])                    
                if relative:
                    tol = tol/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.beta.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.beta.flatten().copy()         
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance): 
                stop_c = True 
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self,X,r_ann,max_iter=50,tolerance=3e-2):
        logL_hist = self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance,relative=True)
        return logL_hist
    
    def multiples_run(self,Runs,X,r_ann,max_iter=50,tolerance=3e-2): 
        if Runs==1:
            return self.stable_train(X,r_ann,max_iter=max_iter,tolerance=tolerance), 0
            
        found_betas = []
        found_model = [] #quizas guardar pesos del modelo
        found_logL = []
        iter_conv = []

        if type(self.base_model.layers[0]) == keras.layers.InputLayer:
            obj_clone = Clonable_Model(self.base_model) #architecture to clone
        else:
            it = keras.layers.Input(shape=self.base_model.input_shape[1:])
            obj_clone = Clonable_Model(self.base_model, input_tensors=it) #architecture to clon

        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer)

            logL_hist = self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance,relative=True) #here the models get resets
            
            found_betas.append(self.beta.copy())
            found_model.append(self.base_model.get_weights()) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([np.max(a) for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        
        self.beta = found_betas[indexs_sort[0]].copy()
        self.base_model = obj_clone.get_model() #change
        self.base_model.set_weights(found_model[indexs_sort[0]])
        self.E_step(X, r_ann) #to set up Q
        print(Runs,"runs, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]

    def fit(self,X,R, runs = 1, max_iter=50, tolerance=3e-2):
        return self.multiples_run(runs,X,R,max_iter=max_iter,tolerance=tolerance)
    
    def get_global_confusionM(self):
        return self.get_confusionM()
    
    def get_predictions_global(self,X,data=[]):
        if len(data) != 0:
            prob_Z_ik = data
        else:
            prob_Z_ik = self.get_predictions(X)
        return np.tensordot(prob_Z_ik ,self.beta,axes=[[1],[0]] ) #sum_z p(z|xi) * p(yo|z,g)