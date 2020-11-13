import gc, keras, time, sys
import numpy as np
from .learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from .learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple, default_CNN_text, default_RNN_text, Clonable_Model #deep learning
from .representation import *
from .utils import estimate_batch_size, EarlyStopRelative, pre_init_F, clusterize_annotators
from .utils import generate_Individual_conf, generate_Global_conf, softmax

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
                    if y_obs_categorical.min() == -1: #masked representation
                        y_obs_categorical[y_obs_categorical == -1] = 0
                        y_obs_categorical = y_obs_categorical.transpose([0,2,1])
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

######################################## BASED ON INFERENCE  ########################################

class Super_LabelInf(object):
    def __init__(self, init_Z='softmv', priors=0, DTYPE_OP='float32'):
        self.DTYPE_OP = DTYPE_OP
        self.init_Z = init_Z.lower()
        self.set_priors(priors)

        self.Keps = keras.backend.epsilon()
        self.init_done = False

    def set_priors(self, priors, globalM=False):
        if type(priors) == str:
            if priors.lower() == "laplace":
                priors = 1
            elif priors.lower() == "none":
                priors = 0
            else:
                raise Exception('Prior not valid')
                
        priors = np.asarray(priors)
        if globalM:
            if len(priors.shape)==0:
                priors = np.expand_dims(priors, axis=(0,1))
            elif len(priors.shape)==1:
                priors=np.expand_dims(priors,axis=(1))
        else:
            if len(priors.shape)==0:
                priors = np.expand_dims(priors, axis=(0,1,2))
            elif len(priors.shape)==1:
                priors=np.expand_dims(priors,axis=(1,2))
            elif len(priors.shape)==2:
                priors=np.expand_dims(priors,axis=2)
        self.Mpriors = priors 


class LabelInf_EM(Super_LabelInf): #DS
    def __init__(self,init_Z='softmv', priors=0, fast=False, DTYPE_OP='float32'):
        super().__init__(init_Z, priors,DTYPE_OP)
        self.fast = fast #fast DS method.. 
        
    def get_marginalZ(self):
        return self.z_marginal.copy()
    def get_confusionM(self):
        return self.betas.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()
        
    def init_E(self, y_ann, method=""): #Majority voting start
        print("Initializing new EM...")
        self.N, self.T, self.K = y_ann.shape
        #init qi = p(z|x)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="individual")
        init_GT = label_A.infer(y_ann, method=method, onehot=True)
        self.Qi_k = init_GT
        #init betas
        self.betas = np.zeros((self.T,self.K,self.K),dtype=self.DTYPE_OP)
        #init z_marginal
        self.z_marginal = np.zeros((self.K),dtype=self.DTYPE_OP)
        print("Z marginal shape",self.z_marginal.shape)
        print("Betas shape: ",self.betas.shape)
        print("Q estimate shape: ",self.Qi_k.shape)
        self.init_done=True
        
    def E_step(self, y_ann):        
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
        
    def train(self, y_ann, max_iter=50,tolerance=3e-2):   
        if not self.init_done:
            self.init_E(y_ann)

        logL = []
        old_betas,tol, tol2 = np.inf, np.inf, np.inf
        self.current_iter = 1
        while(self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)):
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
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps))
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten() 
            self.current_iter+=1
            print("")
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


class LabelInf_EM_G(Super_LabelInf): 
    def __init__(self, init_Z="softmv", priors=0, DTYPE_OP='float32'):
        super().__init__(init_Z, priors,DTYPE_OP)
        super().set_priors(priors, globalM=True)

    def get_marginalZ(self):
        return self.z_marginal.copy()
    def get_confusionM(self):
        return self.beta.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()

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
        
    def E_step(self, r_ann):
        prob_Rx_z = np.tensordot(r_ann, np.log( np.clip(self.beta, self.Keps, 1.)), axes=[[1],[1]])
        aux = np.log(self.z_marginal[None,:] + self.Keps) + prob_Rx_z
        
        self.Qi_k = np.exp(aux).astype(self.DTYPE_OP) #return to actually values
        self.aux_for_like = self.Qi_k.sum(axis=-1) #p(R_x,x)
        self.Qi_k = self.Qi_k/self.aux_for_like[:,None]#normalize q
    
    def M_step(self, r_ann): 
        #-------> z_marginals 
        self.z_marginal = self.Qi_k.mean(axis=0)

        #-------> beta
        self.beta = np.tensordot(self.Qi_k, r_ann, axes=[[0],[0]])
        self.beta += self.Mpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.beta.sum(axis=-1) == 0
        self.beta[mask_zero] = 1

        self.beta = self.beta.astype(self.DTYPE_OP)
        self.beta = self.beta/self.beta.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        logL_priors = np.sum(self.Mpriors* np.log(self.beta+ self.Keps))
        return np.sum( np.log( self.aux_for_like +self.Keps)) + logL_priors
                                                  
    def train(self, r_ann, max_iter=50, tolerance=3e-2):
        if not self.init_done:
            self.init_E(r_ann)
        
        logL = []
        tol,old_betas,old_alphas, tol2 = np.inf,np.inf,np.inf,np.inf
        self.current_iter = 1
        while( (self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)) ):
            print("Iter %d/%d\nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(r_ann) #Need X_i, r_ann
            print(" done,  E step:",end='',flush=True)
            self.E_step(r_ann) 
            self.current_exectime = time.time()-start_time
            print(" done //  (in %.2f sec)\t"%(self.current_exectime),end='',flush=True)
            logL.append(self.compute_logL())
            print("logL: %.3f\t"%(logL[-1]),end='',flush=True)
            if self.current_iter>=2:
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.beta.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.beta.flatten().copy()         
            self.current_iter+=1
            print("")
        print("Finished training!")
        return np.asarray(logL)
    
    def fit(self, R, max_iter=50, tolerance=3e-2):
        return self.train(R, max_iter=max_iter, tolerance=tolerance)
    
    def infer(self):
        return self.get_qestimation()
        
    def predict(self):
        return self.infer()

    def get_global_confusionM(self):
        return self.get_confusionM()

######################################## BASED ON LEARNING MODELS  ########################################

class Super_ModelInf(Super_LabelInf):
    def __init__(self, init_Z='softmv', n_init_Z= 0, priors=0, DTYPE_OP='float32'):
        super().__init__(init_Z, priors, DTYPE_OP)
        self.init_Z = init_Z.lower()
        self.n_init_Z = n_init_Z

    def get_basemodel(self):
        return self.base_model
    
    def get_predictions(self, X):
        if self.base_model_lib=="keras":
            return self.base_model.predict(X, batch_size=self.max_Bsize_base)
        elif self.base_model_lib =="sklearn":
            return self.base_model.predict_proba(X)

    def set_model(self, model, optimizer="adam", epochs=1, batch_size=32, lib_model="keras"):
        self.base_model = model
        self.base_model_lib = lib_model.strip().lower()
        #params:
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        if self.base_model_lib == "keras":
            self.base_model.compile(optimizer=self.optimizer, loss='categorical_crossentropy') 
            self.base_model.name = "base_model_z"
            self.max_Bsize_base = estimate_batch_size(self.base_model)

        elif self.base_model_lib =="sklearn":
            args = {'warm_start':True, 'n_jobs':-1}  #keep fiting the model in the EM: warm_start=True
            aux = [('solver', self.optimizer) , ('max_iter', self.epochs) , ('tol',self.Keps)]
            for (key, value) in aux:
                if key in self.base_model.get_params().keys():
                    args[key] = value
            self.base_model.set_params(**args)

        self.compile = True

    def fit_base_model(self, inputs, targets): #try to learn 
        if self.base_model_lib =="keras":
            self.base_model.fit(inputs, targets, batch_size=self.batch_size, epochs=self.epochs, verbose=0) 
        elif self.base_model_lib =="sklearn":
            self.base_model.fit(inputs, targets.argmax(axis=-1)) #sklearn fit over hard estimation

    def get_obj_clone(self, model_to_clone):
        if type(model_to_clone.layers[0]) == keras.layers.InputLayer:
            return Clonable_Model(model_to_clone) #architecture to clone
        else:
            it = keras.layers.Input(shape=model_to_clone.input_shape[1:])
            return Clonable_Model(model_to_clone, input_tensors=it) #architecture to clon


class ModelInf_EM(Super_ModelInf):
    def __init__(self, init_Z='softmv', n_init_Z= 0, priors=0, DTYPE_OP='float32'):
        super().__init__(init_Z, n_init_Z, priors, DTYPE_OP)
        self.compile=False        

    def get_confusionM(self):
        return self.betas.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()

    #def set_model(self, args):
    #    super().set_model(**args)
        
    def init_E(self, X, y_ann, method=""): 
        print("Initializing new EM...")
        self.N, self.T, self.K = y_ann.shape
        #init qi
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="individual")
        if "model" in method and self.n_init_Z!= 0 and self.base_model_lib == "keras":
            init_GT = label_A.infer(y_ann, method='hardmv', onehot=True) 
            pre_init_F(self.base_model, X, init_GT, self.n_init_Z,batch_size=self.batch_size,reset_optimizer=False)
            init_GT = self.get_predictions(X)
        else:
            init_GT = label_A.infer(y_ann, method=method, onehot=True)
        self.Qi_k = init_GT
        #init betas
        self.betas = np.zeros((self.T,self.K,self.K),dtype=self.DTYPE_OP)
        print("Betas shape: ",self.betas.shape)
        print("Q estimate shape: ",self.Qi_k.shape)
        self.init_done=True           

    def E_step(self,X, y_ann,predictions=[]):
        if len(predictions)==0:
            predictions = self.get_predictions(X)
        
        prob_Lx_z = np.tensordot(y_ann, np.log(self.betas + self.Keps),axes=[[1,2],[0,2]])
        aux = np.log(predictions + self.Keps) + prob_Lx_z
        
        self.sum_unnormalized_q = np.sum(np.exp(aux),axis=-1) # p(L_x) = p(y1,..,yt)

        self.Qi_k = np.exp(aux-aux.max(axis=-1,keepdims=True)).astype(self.DTYPE_OP) #return to actually values
        self.Qi_k = self.Qi_k/self.Qi_k.sum(axis=-1, keepdims=True) #normalize q

    def M_step(self,X, y_ann): 
        #-------> base model ---- train to learn p(z|x)
        self.fit_base_model(X, targets=self.Qi_k)
        
        #-------> beta
        self.betas = np.tensordot(self.Qi_k, y_ann, axes=[[0],[0]]).transpose(1,0,2)
        self.betas += self.Mpriors

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1 #dejar 0 despues de todo..
        
        self.betas = self.betas.astype(self.DTYPE_OP)
        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize
    
    def compute_logL(self):
        logL_priors = np.sum(self.Mpriors* np.log(self.betas+ self.Keps))
        return np.sum( np.log( self.sum_unnormalized_q +self.Keps)) + logL_priors
        
    def train(self,X_train,y_ann,max_iter=50,tolerance=3e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(X_train, y_ann)

        logL = []
        old_betas,tol, tol2 = np.inf, np.inf, np.inf
        self.current_iter = 1
        while(self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)):
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
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps))
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten() 
            self.current_iter+=1
            print("")
        print("Finished training")
        return logL
            
    def stable_train(self,X,y_ann,max_iter=50,tolerance=3e-2):
        return self.train(X,y_ann,max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X,y_ann,max_iter=50,tolerance=3e-2):  #tolerance can change
        if Runs==1:
            return self.stable_train(X,y_ann,max_iter=max_iter,tolerance=tolerance), 0
     
        found_betas = []
        found_model = []
        found_logL = []
        iter_conv = []
        obj_clone = self.get_obj_clone(self.base_model)
        loss_obj = self.base_model.loss
        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths      
            self.base_model.compile(loss=loss_obj, optimizer=self.optimizer)

            logL_hist = self.train(X,y_ann,max_iter=max_iter,tolerance=tolerance) 
            found_betas.append(self.betas.copy())
            found_model.append(self.base_model.get_weights()) 
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with maximum log-likelihood
        logL_iter = np.asarray([a[-1] for a in found_logL])
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
    

class ModelInf_EM_CMM(Super_ModelInf): 
    def __new__(cls, M, init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32'):
        if M == 1:
            return ModelInf_EM_G(init_Z, n_init_Z, priors, DTYPE_OP)
        else:
            return super(ModelInf_EM_CMM, cls).__new__(cls)

    def __init__(self, M, init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32'):
        super().__init__(init_Z, n_init_Z, priors, DTYPE_OP)
        self.M = M #groups of annotators
        self.compile=False
        
    def get_confusionM(self):
        return self.betas.copy()
    def get_alpha(self):
        return self.alphas.copy()
    def get_qestimation(self):
        return self.Qij_mk.copy()


    def init_E(self, X, r_ann, method=""):
        print("Initializing new EM...")
        self.N, self.K = r_ann.shape

        #-------> init qi = p(z|x)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="global")
        if "model" in method and self.n_init_Z!= 0 and self.base_model_lib == "keras":
            init_GT = label_A.infer(r_ann, method='hardmv', onehot=True) 
            pre_init_F(self.base_model, X, init_GT, self.n_init_Z,batch_size=self.batch_size,reset_optimizer=False)
            init_GT = self.get_predictions(X)
        else:
            init_GT = label_A.infer(r_ann, method=method, onehot=True)
        #-------> init alpha
        self.alpha_init = clusterize_annotators(init_GT,M=self.M,bulk=False,cluster_type='mv_close',DTYPE_OP=self.DTYPE_OP) #clusteriza en base mv
         #-------> Initialize qij_zg = p(z=k,g=m|xi,y=j)
        self.Qij_mk = self.alpha_init[:,:,:,None]*init_GT[:,None,None,:]

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
        self.fit_base_model(X, targets=r_estimate)
    
        #-------> alpha 
        self.alphas = QRij_mk.sum(axis=(0,1,3)) 
        self.alphas = self.alphas/self.alphas.sum(axis=-1,keepdims=True) #p(g) -- normalize

        #-------> beta
        self.betas = (QRij_mk.sum(axis=0)).transpose(1,2,0)            
        self.betas += self.Mpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1

        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self, r_ann):
        """ Compute the log-likelihood of the optimization schedule"""
        logL_priors = np.sum(self.Mpriors* np.log(self.betas+ self.Keps))
        return np.tensordot(r_ann , np.log(self.aux_for_like+self.Keps))+ logL_priors 
                                                  
    def train(self,X_train, r_ann, max_iter=50,tolerance=3e-2):
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(X_train, r_ann)
        
        logL = []
        tol,old_betas,old_alphas, tol2 = np.inf,np.inf,np.inf ,np.inf
        self.current_iter = 1
        while(self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)):
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
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])                    
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                tol3 = np.mean(np.abs(self.alphas-old_alphas)/(old_alphas+self.Keps)) #alphas
                print("Tol1: %.5f\tTol2: %.5f\tTol3: %.5f\t"%(tol,tol2,tol3),end='',flush=True)
            old_betas = self.betas.flatten().copy()         
            old_alphas = self.alphas.copy()
            self.current_iter+=1
            print("")
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self,X,r_ann,max_iter=50,tolerance=3e-2):
        return self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X,r_ann,max_iter=50,tolerance=3e-2): 
        if Runs==1:
            return self.stable_train(X,r_ann,max_iter=max_iter,tolerance=tolerance), 0
            
        found_betas = []
        found_alphas = []
        found_model = [] 
        found_logL = []
        iter_conv = []
        obj_clone = self.get_obj_clone(self.base_model)
        loss_obj = self.base_model.loss
        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths
            self.base_model.compile(loss=loss_obj, optimizer=self.optimizer)

            logL_hist = self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance)
            
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
        logL_iter = np.asarray([a[-1] for a in found_logL])
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


class ModelInf_EM_CMOA(Super_ModelInf):
    def __init__(self, M, init_Z="softmv", init_G="", n_init_Z=0, n_init_G=0, priors=0, DTYPE_OP='float32'): 
        super().__init__(init_Z, n_init_Z, priors, DTYPE_OP)
        self.init_G = init_G.lower()
        self.n_init_G = n_init_G
        self.M = M #groups of annotators

        self.compile_z = False
        self.compile_g = False
        
    def get_groupmodel(self):
        return self.group_model
    def get_confusionM(self):
        return self.betas.copy()
    def get_qestimation(self):
        return self.reshape_il(self.Qil_mk.copy())
        
    def set_model(self, model, optimizer="adam", epochs=1, batch_size=32, lib_model="keras", ann_model=None,optimizer_ann="adam"):
        super().set_model(model, optimizer, epochs, batch_size, lib_model)
        self.compile_z = True
        
        if type(ann_model) != type(None):
            self.set_ann_model(ann_model, optimizer=optimizer_ann)
        
    def set_ann_model(self, model, optimizer=None, epochs=None):
        if type(optimizer) == type(None): 
            optimizer = self.optimizer

        self.group_epochs = epochs
        if type(self.group_epochs) == type(None): 
            self.group_epochs = self.epochs #set epochs of base model            

        self.group_model = model
        self.group_model.compile(optimizer=optimizer, loss='categorical_crossentropy') 
        self.group_model.name = "group_model_g"
        self.compile_g = True
        self.max_Bsize_group = estimate_batch_size(self.group_model)

    def get_predictions_z(self, X):
        return super().get_predictions(X) 

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

    def init_E(self, X, y_ann_var, A_idx_var, method=""):
        print("Initializing new EM...")
        self.N = len(y_ann_var)
        self.K = y_ann_var[0].shape[1]
        self.T_i = [y_ann.shape[0] for y_ann in y_ann_var] 
        self.BS_groups = np.ceil(self.batch_size*np.mean(self.T_i)).astype('int') #batch should be prop to T_i

        #-------> init qi = p(z|x)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="individual", sparse=True)
        if "model" in method and self.n_init_Z!= 0 and self.base_model_lib == "keras":
            init_GT = label_A.infer(y_ann_var, method='hardmv', onehot=True) 
            pre_init_F(self.base_model, X, init_GT, self.n_init_Z,batch_size=self.batch_size,reset_optimizer=False)
            init_GT = self.get_predictions_z(X)
        else:
            init_GT = label_A.infer(y_ann_var, method=method, onehot=True)
        
        #------->init p(g|a)
        first_l = self.group_model.layers[0]
        if type(first_l) == keras.layers.Embedding: #if there is embedding layer
            A_embedding = first_l.get_weights()[0]
            probas_t =  clusterize_annotators(A_embedding,M=self.M,bulk=True,cluster_type='previous',DTYPE_OP=self.DTYPE_OP)
        else:
            probas_t =  clusterize_annotators(y_ann_var,M=self.M,bulk=True,cluster_type='conf_flatten',data=[A_idx_var,init_GT],DTYPE_OP=self.DTYPE_OP)
        if "model" in self.init_G and self.n_init_G != 0:
            alpha_init = []
            for i in range(self.N):
                t_idxs = A_idx_var[i] #indexs of annotators that label pattern "i"
                alpha_init.append( probas_t[t_idxs] )#preinit over alphas

            pre_init_F(self.group_model,self.flatten_il(A_idx_var), self.flatten_il(alpha_init), 
                self.n_init_G, batch_size=self.BS_groups,reset_optimizer=False)
            A = np.unique(self.flatten_il(A_idx_var)).reshape(-1,1) # A: annotator identity set
            probas_t = self.get_predictions_g(A)  #change init to group model

        #-------> Initialize p(z=k,g=m|xi,y,a)
        self.Qil_mk = []
        for i in range(self.N):
            t_idxs = A_idx_var[i] #indexs of annotators that label pattern "i"
            self.Qil_mk.append( probas_t[t_idxs][:,:,None] * init_GT[i][None,None,:] ) 
        self.Qil_mk = self.flatten_il(self.Qil_mk)

        #-------> init betas
        self.betas = np.zeros((self.M,self.K,self.K),dtype=self.DTYPE_OP)        
        print("Betas: ",self.betas.shape)
        print("Q estimate: ",self.Qil_mk.shape)
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
        self.fit_base_model(X, targets=r_estimate)

        #-------> alpha 
        Qil_m_flat = self.Qil_mk.sum(axis=-1)  #qil(m)
        self.group_model.fit(A_idx_flatten, Qil_m_flat, batch_size=self.BS_groups, epochs=self.group_epochs,verbose=0)

        #-------> beta
        self.betas =  np.tensordot(self.Qil_mk, y_ann_flatten , axes=[[0],[0]]) # ~p(yo=j|g,z) 
        self.betas += self.Mpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.betas.sum(axis=-1) == 0
        self.betas[mask_zero] = 1

        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        logL_priors = np.sum(self.Mpriors* np.log(self.betas+ self.Keps))
        return np.sum( np.log(self.aux_for_like+self.Keps) ) + logL_priors
                                                  
    def train(self, X_train, y_ann_var, A_idx_var, max_iter=50, tolerance=3e-2):
        if not self.compile_z:
            print("You need to create the model first, set .define_model")
            return
        if len(y_ann_var.shape) != 1 or len(A_idx_var.shape) != 1:
            raise Exception('Needed y_ann_var and A_idx_var in variable length array')

        y_ann_flatten, A_idx_flatten = self.flatten_il(y_ann_var), self.flatten_il(A_idx_var)
        A = np.unique(A_idx_flatten).reshape(-1,1) # A: annotator identity set
        if not self.init_done:
            self.init_E(X_train, y_ann_var, A_idx_var)

        logL = []
        tol,old_betas, tol2 = np.inf,np.inf,np.inf
        self.current_iter = 1
        while(self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)):
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
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])                 
                tol2 = np.mean(np.abs(self.betas.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.betas.flatten().copy()         
            self.current_iter+=1
            print("")
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self, X, y_ann_var, A_idx_var, max_iter=50,tolerance=3e-2):
        return self.train(X, y_ann_var, A_idx_var, max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X, y_ann_var, A_idx_var, max_iter=50,tolerance=3e-2): 
        if Runs==1:
            return self.stable_train(X, y_ann_var, A_idx_var, max_iter=max_iter,tolerance=tolerance), 0
            
        found_betas = []
        found_model_g = []
        found_model_z = []
        found_logL = []
        iter_conv = []
        obj_clone_z = self.get_obj_clone(self.base_model)
        loss_obj_z = self.base_model.loss
        obj_clone_g = self.get_obj_clone(self.group_model)
        loss_obj_g = self.group_model.loss
        for run in range(Runs):
            self.base_model = obj_clone_z.get_model() #reset-weigths
            self.base_model.compile(loss=loss_obj_z, optimizer=self.optimizer)
            
            self.group_model = obj_clone_g.get_model() #reset-weigths
            self.group_model.compile(loss=loss_obj_g, optimizer=self.optimizer)

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
        logL_iter = np.asarray([a[-1] for a in found_logL])
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


class ModelInf_EM_G(Super_ModelInf): 
    def __init__(self, init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32'):
        super().__init__(init_Z, n_init_Z, priors, DTYPE_OP)
        super().set_priors(priors, globalM=True)
        self.compile=False
        
    def get_confusionM(self):
        return self.beta.copy()
    def get_qestimation(self):
        return self.Qi_k.copy()

    def init_E(self, X, r_ann, method=""):
        print("Initializing new EM...")
        self.N, self.K = r_ann.shape

        #-------> Initialize qi = p(z|xi,ri)
        if method == "":
            method = self.init_Z
        label_A = LabelAgg(scenario="global")
        if "model" in method and self.n_init_Z!= 0 and self.base_model_lib == "keras":
            init_GT = label_A.infer(r_ann, method='hardmv', onehot=True) 
            pre_init_F(self.base_model, X, init_GT, self.n_init_Z,batch_size=self.batch_size,reset_optimizer=False)
            init_GT = self.get_predictions(X)
        else:
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
        self.fit_base_model(X, targets=self.Qi_k)
    
        #-------> beta
        self.beta = np.tensordot(self.Qi_k, r_ann, axes=[[0],[0]])
        self.beta += self.Mpriors 

        #if no priors were seted as annotator not label all data:---
        mask_zero = self.beta.sum(axis=-1) == 0
        self.beta[mask_zero] = 1

        self.beta = self.beta.astype(self.DTYPE_OP)
        self.beta = self.beta/self.beta.sum(axis=-1,keepdims=True) #normalize (=p(yo|g,z))

    def compute_logL(self):
        """ Compute the log-likelihood of the optimization schedule"""
        logL_priors = np.sum(self.Mpriors* np.log(self.beta+ self.Keps))
        return np.sum( np.log( self.aux_for_like +self.Keps)) + logL_priors
                                                  
    def train(self,X_train, r_ann, max_iter=50, tolerance=3e-2):
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(X_train, r_ann)
        
        logL = []
        stop_c = False
        tol,old_betas,old_alphas, tol2 = np.inf,np.inf,np.inf,np.inf
        self.current_iter = 1
        while(self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)):
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
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.beta.flatten()-old_betas)/(old_betas+self.Keps)) #confusion
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.beta.flatten().copy()         
            self.current_iter+=1
            print("")
        print("Finished training!")
        return np.asarray(logL)
    
    def stable_train(self,X,r_ann,max_iter=50,tolerance=3e-2):
        return self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X,r_ann,max_iter=50,tolerance=3e-2): 
        if Runs==1:
            return self.stable_train(X,r_ann,max_iter=max_iter,tolerance=tolerance), 0
            
        found_betas = []
        found_model = [] #quizas guardar pesos del modelo
        found_logL = []
        iter_conv = []
        obj_clone = self.get_obj_clone(self.base_model)
        loss_obj = self.base_model.loss
        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths
            self.base_model.compile(loss=loss_obj, optimizer=self.optimizer)

            logL_hist = self.train(X,r_ann,max_iter=max_iter,tolerance=tolerance) #here the models get resets
            
            found_betas.append(self.beta.copy())
            found_model.append(self.base_model.get_weights()) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model
            keras.backend.clear_session()
            gc.collect()
        logL_iter = np.asarray([a[-1] for a in found_logL])
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


class ModelInf_EM_R(Super_ModelInf):
    def __init__(self, init_R='original', DTYPE_OP='float32'):
        super().__init__(init_Z="", n_init_Z=None, priors=0, DTYPE_OP=DTYPE_OP)
        self.DTYPE_OP = DTYPE_OP
        self.init_R = init_R.lower()
        self.compile=False

    def get_b(self):
        return self.b
    def get_restimation(self):
        return self.Ri_l.copy()

    def init_E(self, X, y_ann, method=""): 
        print("Initializing new EM...")
        self.N, self.T, self.K = y_ann.shape
        self.mask_not_ann = (y_ann.sum(axis=-1) == 0) #(N,T)
        self.Nt = (~self.mask_not_ann).sum(axis=0) #number annotations per annotator
        #init qi
        if method == "":
            method = self.init_R
        if method == "original" or method == "simple":
            self.Ri_l = np.ones((self.N, self.T, 1),dtype=self.DTYPE_OP)*0.99
        elif "mv" in method:
            label_A = LabelAgg(scenario="individual")
            if method == "hardmv":
                init_GT = label_A.infer(y_ann, method='hardmv', onehot=False) 
                self.Ri_l = (y_ann.argmax(axis=-1) == init_GT[:,None])*1 
            elif method == "softmv":
                init_GT = label_A.infer(y_ann, method='softmv') 
                self.Ri_l = (y_ann*init_GT[:,None,:]).sum(axis=-1)
            self.Ri_l = self.Ri_l[:,:,None]
        self.Ri_l[self.mask_not_ann] = 0 #non label
        self.b = np.zeros((self.T, 1),dtype=self.DTYPE_OP)
        print("b shape: ",self.b.shape)
        print("R estimate shape: ",self.Ri_l.shape)
        self.init_done=True
        
    def E_step(self,X, y_ann,predictions=[]): ####### REVISAR
        if len(predictions)==0:
            predictions = self.get_predictions(X) 
        b_aux = self.b[None,:,:]
        p_z = predictions[:,None,:]

        A1 = np.log(b_aux +self.Keps) + ((y_ann*np.log(p_z +self.Keps)).sum(axis=-1))[:,:,None]
        A1 = np.exp(A1)
        A2 = np.log(1-b_aux + self.Keps) - np.log(self.K)
        A2  = np.exp(A2)

        self.Ri_l = A1/(A1+A2) 
        self.Ri_l[self.mask_not_ann] = 0 #non label
        self.Ri_l = self.Ri_l.astype(self.DTYPE_OP)

        self.sum_unnormalized_q = (A1 + A2)[~self.mask_not_ann].sum(axis=-1)
        
    def M_step(self,X, y_ann): 
        #-------> base model ---- train to learn p(z|x)
        R_ik = (y_ann * self.Ri_l).sum(axis=1)
        self.fit_base_model(X, targets=R_ik)
        
        #-------> b
        self.b = self.Ri_l.sum(axis=0)/self.Nt[:,None]
        
    def compute_logL(self):
        return np.sum( np.log(self.sum_unnormalized_q +self.Keps))
        
    def train(self,X_train,y_ann,max_iter=50,tolerance=3e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_E(X_train, y_ann)

        logL = []
        old_betas,tol, tol2 = np.inf, np.inf, np.inf
        self.current_iter = 1
        while(self.current_iter<=max_iter and not (tol<=tolerance and tol2<=tolerance)):
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
                tol = np.abs(logL[-1] - logL[-2])/np.abs(logL[-2])
                tol2 = np.mean(np.abs(self.b.flatten()-old_betas)/(old_betas+self.Keps))
                print("Tol1: %.5f\tTol2: %.5f\t"%(tol,tol2),end='',flush=True)
            old_betas = self.b.flatten() 
            self.current_iter+=1
            print("")
        print("Finished training")
        return logL
    
    def stable_train(self,X,y_ann,max_iter=50,tolerance=3e-2):
        return self.train(X,y_ann,max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X,y_ann,max_iter=50,tolerance=3e-2):  #tolerance can change
        if Runs==1:
            return self.stable_train(X,y_ann,max_iter=max_iter,tolerance=tolerance), 0
     
        found_betas = []
        found_model = []
        found_logL = []
        iter_conv = []
        obj_clone = self.get_obj_clone(self.base_model)
        loss_obj = self.base_model.loss
        for run in range(Runs):
            self.base_model = obj_clone.get_model() #reset-weigths            
            self.base_model.compile(loss=loss_obj, optimizer=self.optimizer)

            logL_hist = self.train(X,y_ann,max_iter=max_iter,tolerance=tolerance) 
            found_betas.append(self.b.copy())
            found_model.append(self.base_model.get_weights()) 
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
            self.init_done = False
            del self.base_model
            keras.backend.clear_session()
            gc.collect()
        logL_iter = np.asarray([a[-1] for a in found_logL])
        indexs_sort = np.argsort(logL_iter)[::-1] 
        self.b = found_betas[indexs_sort[0]].copy()
        self.base_model = obj_clone.get_model() #change
        self.base_model.set_weights(found_model[indexs_sort[0]])
        self.E_step(X,y_ann,predictions=self.get_predictions(X))
        print(Runs,"runs, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]

    def fit(self,X,Y, runs = 1, max_iter=50, tolerance=3e-2):
        return self.multiples_run(runs,X,Y,max_iter=max_iter,tolerance=tolerance)
    
    def get_ann_rel(self):
        return self.get_b()

from .learning_models import MaskedMultiCrossEntropy, CrowdsLayer, NoiseLayer
class ModelInf_BP(Super_ModelInf):
    def __init__(self, init_Z='softmv', n_init_Z= 0, prior_lamb=0, init_conf = "default"):
        super().__init__(init_Z=init_Z, n_init_Z=n_init_Z)
        self.compile=False  
        self.init_C = init_conf
        self.lamb = prior_lamb

    def get_confusionM(self):
        """Get confusion matrices of every annotator p(yo^t|,z)"""  
        return self.betas.copy()

    def set_crowdL_model(self, set_w = False, weights=0):
        if not set_w:
            weights = self.get_confusionM().transpose([1,2,0])
        x = self.base_model.inputs
        p_zx = self.base_model(x)
        crowd_layer = CrowdsLayer(self.K, self.T, conn_type="MW", conf_ma=weights, name='CrowdL') ## ADD CROWDLAYER 
        self.model_crowdL = keras.models.Model(x, crowd_layer(p_zx)) 
        self.model_crowdL.compile(optimizer=self.optimizer, loss= MaskedMultiCrossEntropy().loss_w_prior(l=self.lamb, p_z=p_zx) )

    def init_model(self, X, y_ann, method=""):
        print("Initializing...")
        if y_ann.min() == -1:
            self.N, self.K, self.T = y_ann.shape
        else:
            self.N, self.T, self.K = y_ann.shape

        if self.n_init_Z!= 0 and self.base_model_lib == "keras":
            if method == "":
                method = self.init_Z
            label_A = LabelAgg(scenario="individual")
            init_GT = label_A.infer(y_ann, method=method, onehot=True) 
            pre_init_F(self.base_model, X, init_GT, self.n_init_Z, batch_size=self.batch_size, reset_optimizer=False)

        if self.init_C == "model" and self.n_init_Z!= 0:
            Z_pred = self.get_predictions(X).argmax(axis=-1) #could be soft
            weights = generate_Individual_conf(Z_pred, y_ann)
            weights = weights.transpose([1,2,0])
            weights = np.log(weights + 1e-7)
        elif self.init_C == "soft": #simi to default
            NOISE_LEVEL = 0.15
            weights = 0.01 * np.random.random((self.K, self.K, self.T)) 
            weights += NOISE_LEVEL / (self.K - 1.)
            for r in range(self.T):
                for i in range(self.K):
                    weights[i,i, r] = 1. - NOISE_LEVEL
        else: # default
            weights = np.zeros((self.K, self.K, self.T))
            for r in range(self.T):
                for i in range(self.K):
                    weights[i,i,r] = 1.0  #identities
        self.set_crowdL_model(set_w = True, weights= weights)
        self.init_done=True 
    
    def train(self, X_train, y_ann, max_iter=50,tolerance=1e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_model(X_train, y_ann) #add crowdlayer for auxiliar training
        
        if y_ann.min() != -1:
            y_ann = y_ann.transpose([0,2,1]) #to Crowdlayer shape
        ourCallback = EarlyStopRelative(monitor='loss', patience=1, min_delta=tolerance)
        hist = self.model_crowdL.fit(X_train, y_ann, epochs=max_iter, batch_size=self.batch_size, 
                                    verbose=1,callbacks=[ourCallback])
        self.base_model = self.model_crowdL.get_layer("base_model_z")
        self.betas = self.model_crowdL.get_layer("CrowdL").get_weights()[0].transpose([2,0,1]) #witohut bias
        print("Finished training")
        return hist.history["loss"]
            
    def stable_train(self,X,y_ann, max_iter=50, tolerance=1e-2):
        return self.train(X, y_ann,max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X,y_ann,max_iter=50,tolerance=1e-2):
        if Runs==1:
            return self.stable_train(X,y_ann,max_iter=max_iter,tolerance=tolerance), 0
     
        found_model = []
        found_loss = []
        found_betas = []
        obj_clone = self.get_obj_clone(self.base_model)
        for run in range(Runs):
            self.base_model = obj_clone.get_model() 

            loss_hist = self.train(X, y_ann, max_iter=max_iter, tolerance=tolerance) 
            found_model.append(self.model_crowdL.get_layer("base_model_z").get_weights()) 
            found_betas.append(self.model_crowdL.get_layer("CrowdL").get_weights()[0].transpose([2,0,1]) )
            found_loss.append(loss_hist)
            
            self.init_done = False #multiples runs over different inits
            del self.model_crowdL, self.base_model
            keras.backend.clear_session()
            gc.collect()
        Loss_iter = np.asarray([a[-1] for a in found_loss])
        indexs_sort = np.argsort(Loss_iter) #minimum
        self.betas = found_betas[indexs_sort[0]].copy() 
        self.base_model =  obj_clone.get_model()
        self.base_model.set_weights(found_model[indexs_sort[0]])
        print(Runs,"runs over Rodrigues 18', Epochs to converge= ",np.mean([len(v) for v in found_loss]))
        return found_loss, indexs_sort[0]

    def fit(self,X,Y, runs = 1, max_iter=50, tolerance=1e-2):
        return self.multiples_run(runs,X,Y,max_iter=max_iter,tolerance=tolerance)
        
    def get_ann_confusionM(self, norm=""):
        confs = self.get_confusionM()
        if norm == "softmax":
            confs = softmax(confs, axis=-1)
        elif norm == "0-1" or norm=="01":
            confs += np.abs(confs.min(axis=-1, keepdims=True))
            confs = confs / confs.max(axis=-1, keepdims=True)
        return confs
        
    def get_predictions_annot(self, X):
        """ Predictions of all annotators , p(y^o | xi, t) """
        try:
            self.model_crowdL
        except:
            self.set_crowdL_model()
        return self.model_crowdL.predict(X).transpose([0,2,1]) 


class ModelInf_BP_G(Super_ModelInf):
    def __init__(self, init_Z='softmv', n_init_Z= 0, prior_lamb=0, init_conf= "default"):
        super().__init__(init_Z=init_Z, n_init_Z=n_init_Z)
        self.compile = False  
        self.init_C = init_conf

        self.lamb = prior_lamb #incluir algo al respecto???

    def get_confusionM(self):
        """Get confusion matrices of global annotations p(y|z)"""  
        return self.beta.copy()

    def set_crowdL_model(self, set_w = False, weights=0):
        if not set_w:
            weights = self.get_confusionM()
        x = self.base_model.inputs
        p_zx = self.base_model(x)
        noise_channel = NoiseLayer(self.K, conf_ma=weights, name="NoiseC")
        self.model_crowdL = keras.models.Model(x, noise_channel(p_zx)) 
        self.model_crowdL.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

    def init_model(self, X, r_ann, method=""):
        print("Initializing...")
        self.N, self.K = r_ann.shape

        if self.n_init_Z!= 0:
            if method == "":
                method = self.init_Z
            label_A = LabelAgg(scenario="global")
            init_GT = label_A.infer(r_ann, method=method, onehot=True) 
            pre_init_F(self.base_model, X, init_GT, self.n_init_Z, batch_size=self.batch_size, reset_optimizer=False)

        if self.init_C == "model" and self.n_init_Z!= 0:
            Z_pred = self.get_predictions(X).argmax(axis=-1) #could be soft
            weights = generate_Global_conf(Z_pred, r_ann)
        else: #default
            NOISE_LEVEL = 0.15
            weights = 0.01 * np.random.random((self.K, self.K))  #
            weights += NOISE_LEVEL / (self.K - 1.)
            for i in range(self.K):
                weights[i,i] = 1. - NOISE_LEVEL
        weights = np.log(weights + self.Keps)
        self.set_crowdL_model(set_w = True, weights= weights)
        self.init_done=True 
    
    def train(self, X_train, r_ann, max_iter=50,tolerance=1e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        if not self.init_done:
            self.init_model(X_train, r_ann) #add crowdlayer for auxiliar training

        ourCallback = EarlyStopRelative(monitor='loss', patience=1, min_delta=tolerance)
        hist = self.model_crowdL.fit(X_train, r_ann, epochs=max_iter, batch_size=self.batch_size, 
                                    verbose=1,callbacks=[ourCallback])
        self.base_model = self.model_crowdL.get_layer("base_model_z")
        self.beta = self.model_crowdL.get_layer("NoiseC").get_weights()[0] 
        print("Finished training")
        return hist.history["loss"]
            
    def stable_train(self,X,r_ann, max_iter=50, tolerance=1e-2):
        return self.train(X, r_ann,max_iter=max_iter,tolerance=tolerance)
    
    def multiples_run(self,Runs,X,r_ann,max_iter=50,tolerance=1e-2):
        if Runs==1:
            return self.stable_train(X,r_ann,max_iter=max_iter,tolerance=tolerance), 0
     
        found_betas = []
        found_model = []
        found_loss = []
        obj_clone = self.get_obj_clone(self.base_model)
        for run in range(Runs):
            self.base_model = obj_clone.get_model() 

            loss_hist = self.train(X, r_ann, max_iter=max_iter, tolerance=tolerance) 
            found_model.append(self.model_crowdL.get_layer("base_model_z").get_weights()) 
            found_betas.append(self.model_crowdL.get_layer("NoiseC").get_weights()[0] )
            found_loss.append(loss_hist)
            
            self.init_done = False #multiples runs over different inits
            del self.model_crowdL, self.base_model
            keras.backend.clear_session()
            gc.collect()
        Loss_iter = np.asarray([a[-1] for a in found_loss]) 
        indexs_sort = np.argsort(Loss_iter)
        self.beta = found_betas[indexs_sort[0]].copy() 
        self.base_model =  obj_clone.get_model()
        self.base_model.set_weights(found_model[indexs_sort[0]])
        print(Runs,"runs, Epochs to converge= ",np.mean([len(v) for v in found_loss]))
        return found_loss, indexs_sort[0]

    def fit(self,X,Y, runs = 1, max_iter=50, tolerance=1e-2):
        return self.multiples_run(runs,X,Y,max_iter=max_iter,tolerance=tolerance)
        
    def get_global_confusionM(self, norm="softmax"): #softmax was the original idea 
        confs = self.get_confusionM()
        if norm == "softmax":
            confs = softmax(confs, axis=-1)
        return confs

    def get_predictions_global(self, X):
        """ Predictions of global annotators , p(y^o | xi) """
        try:
            self.model_crowdL
        except:
            self.set_crowdL_model()
        return self.model_crowdL.predict(X) 