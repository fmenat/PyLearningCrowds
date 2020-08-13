import gc, keras, time, sys
import numpy as np
from .learning_models import LogisticRegression_Sklearn,LogisticRegression_Keras,MLP_Keras
from .learning_models import default_CNN,default_RNN,CNN_simple, RNN_simple, default_CNN_text, default_RNN_text, Clonable_Model #deep learning
from .representation import *
from .utils import estimate_batch_size, EarlyStopRelative

def majority_voting(r_obs,onehot=False,probas=False):
    """
    Args:
        * onehot mean if the returned array come as a one hot of the classes
        * probas mean that a probability version of majority voting is returned
    """
    if probas:
        r_obs = r_obs.astype('float32')
        return r_obs/r_obs.sum(axis=-1, keepdims=True)

    mv = r_obs.argmax(axis=1) #over classes axis
    if onehot: 
        mv = keras.utils.to_categorical(mv)
    return mv

class LabelInference(object): #no predictive model
    def __init__(self): #, method, tolerance, max_iter=50): 
        return
        #self.method = method.lower()

        #if 'd&s' in type_inf.lower() or "dawid" in type_inf.lower() or "ds" in type_inf.lower():
        #    start_time = time.time()
        #    self.annotations = set_representation(labels,'dawid') #for D&S
        #    print("Representation for DS in %f sec"%(time.time()-start_time))
            
        #self.Tol = tolerance #tolerance of D&S
        #self.T = labels.shape[1]
        #self.calc_MV = False
        #self.max_iter = max_iter

    def infer(self, labels, method):
        """
            *labels is annotations : should be individual or global representation
        """
        #if not self.calc_MV: #to avoid calculation.
        #start_time = time.time()
        #self.calc_MV = True   
        #print("Estimation MV in %f sec"%(time.time()-start_time))
        method = method.lower()
        if len(labels.shape) == 3:
            r_obs = annotations2repeat_efficient(labels)
        else:
            r_obs = labels


        if method == 'softmv': 
            mv_probas = majority_voting(r_obs, probas=True) 
            #prob_Yz = generate_confusionM(self.mv_probas, self.y_obs_repeat) #confusion matrix of all annotators
            return mv_probas #, prob_Yz

        elif method == "hardmv":
            to_return =  majority_voting(r_obs, probas=False) #aka soft-MV
            #prob_Yz = generate_confusionM(to_return, self.y_obs_repeat) #confusion matrix of all annotators
            return to_return #, prob_Yz

        #elif self.method == 'hardMV' or self.method == 'hardMV': #also known as hard-MV
        #    to_return = majority_voting(labels, probas=False, onehot=True) #aka soft-MV
        #    prob_Yz = generate_confusionM(to_return, self.y_obs_repeat) #confusion matrix of all annotators
        #    return to_return, prob_Yz        
       


    def predict(self, *args):
        return self.infer(*args)
            
    #def DS_labels(self):
    #    # https://github.com/dallascard/dawid_skene
    #    start_time =time.time()
    #    aux = dawid_skene.run(self.annotations,tol=self.Tol, max_iter=self.max_iter, init='average')
    #    (_, _, _, _, class_marginals, error_rates, groundtruth_estimate, current_exectime) = aux
    #    self.DS_current_exectime = current_exectime
    #    print("Estimation for DS in %f sec"%(time.time()-start_time))
    #    return groundtruth_estimate, error_rates


class RaykarMC(object):
    def __init__(self,input_dim,K,T,epochs=1,optimizer='adam',DTYPE_OP='float32'): #default stable parameteres
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.input_dim = input_dim
        self.K = K #number of classes of the problem
        self.T = T #number of annotators
        #params:
        self.epochs = epochs
        self.optimizer = optimizer
        self.DTYPE_OP = DTYPE_OP

        self.compile=False
        self.Keps = keras.backend.epsilon()
        self.priors=False #boolean of priors
        
    def get_basemodel(self):
        return self.base_model
    def get_confusionM(self):
        """Get confusion matrices of every annotator p(yo^t|,z)"""  
        return self.betas
    def get_qestimation(self):
        return self.Qi_gamma

    def define_model(self,tipo,model,start_units=1,deep=1,double=False,drop=0.0,embed=[],BatchN=False,glo_p=False):
        """Define the network of the base model"""
        self.type = tipo.lower()     
        if self.type == "keras_shallow" or 'perceptron' in self.type: 
            self.base_model = LogisticRegression_Keras(self.input_dim,self.K)
            #It's not a priority, since HF has been shown to underperform RMSprop and Adagrad, while being more computationally intensive.
            #https://github.com/keras-team/keras/issues/460
            self.compile = True
            return
        
        if self.type == "keras_import":
            self.base_model = model
        elif self.type=='defaultcnn' or self.type=='default cnn':
            self.base_model = default_CNN(self.input_dim,self.K)
        elif self.type=='defaultrnn' or self.type=='default rnn':
            self.base_model = default_RNN(self.input_dim,self.K)
        elif self.type=='defaultcnntext' or self.type=='default cnn text': #with embedding
            self.base_model = default_CNN_text(self.input_dim[0],self.K,embed) #len is the length of the vocabulary
        elif self.type=='defaultrnntext' or self.type=='default rnn text': #with embedding
            self.base_model = default_RNN_text(self.input_dim[0],self.K,embed) #len is the length of the vocabulary

        elif self.type == "ff" or self.type == "mlp" or self.type=='dense': #classic feed forward
            print("Needed params (units,deep,drop,BatchN?)") #default activation is relu
            self.base_model = MLP_Keras(self.input_dim,self.K,start_units,deep,BN=BatchN,drop=drop)

        elif self.type=='simplecnn' or self.type=='simple cnn' or 'cnn' in self.type:
            print("Needed params (units,deep,drop,double?,BatchN?)") #default activation is relu
            self.base_model = CNN_simple(self.input_dim,self.K,start_units,deep,double=double,BN=BatchN,drop=drop,global_pool=glo_p)
        
        elif self.type=='simplernn' or self.type=='simple rnn' or 'rnn' in self.type:
            print("Needed params (units,deep,drop,embed?)")
            self.base_model = RNN_simple(self.input_dim,self.K,start_units,deep,drop=drop,embed=embed,len=0,out=start_units*2)

        self.base_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 
        self.max_Bsize_base = estimate_batch_size(self.base_model)
        self.compile = True

    def get_predictions(self,X):
        if "sklearn" in self.type:
            return self.base_model.predict_proba(X) #or predict
        else:
            return self.base_model.predict(X,batch_size=self.max_Bsize_base)
        
    def define_priors(self,priors):
        """
            Ojo que en raykar los priors deben tener una cierta forma (T,K,K) o hacerlos globales (T,K)
            para cualquier variable obs
            El obs t, dado que la clase "k" que tan probable es que diga que es una clase..
            se recomienda que sea uno
        """
        if type(priors) == str:
            if priors == "laplace":
                priors = 1
            else:
                print("Prior string do not understand")
                return
        else:
            if len(priors.shape)==2:
                priors=np.expand_dims(priors,axis=2)
        self.Mpriors = priors
        self.priors = True
        
    def init_E(self,X,y_ann):
        start_time = time.time()
        self.N = X.shape[0]
        #init p(z|x)
        mv_probs = majority_voting(y_ann,repeats=False,probas=True) #Majority voting start
        #init betas
        self.betas = np.zeros((self.T,self.K,self.K),dtype=self.DTYPE_OP)
        #init qi
        self.Qi_gamma = mv_probs
        print("Betas shape: ",self.betas.shape)
        print("Q estimate shape: ",self.Qi_gamma.shape)
        self.init_exectime = time.time()-start_time
                
    def E_step(self,X,y_ann,predictions=[]):
        if len(predictions)==0:
            predictions = self.get_predictions(X)
        
        #calculate sensitivity-specificity 
        a_igamma = np.tensordot(y_ann, np.log(self.betas + self.Keps),axes=[[1,2],[0,2]])
        a_igamma = a_igamma.astype(self.DTYPE_OP)
        aux = np.log(predictions + self.Keps) + a_igamma
        
        self.sum_unnormalized_q = np.sum(np.exp(aux),axis=-1)# p(y1,..,yt|x) #all anotations probabilities

        self.Qi_gamma = np.exp(aux-aux.max(axis=-1,keepdims=True)) #return to actually values
        self.Qi_gamma = self.Qi_gamma/np.sum(self.Qi_gamma,axis=-1)[:,None] #normalize q

    def M_step(self,X,y_ann): 
        #-------> base model ---- train to learn p(z|x)
        if "sklearn" in self.type:
            self.base_model.fit(X,np.argmax(self.Qi_gamma,axis=-1)) #to one hot 
        else:
            #epochs=1 as Rodriges says. and batch size as default
            history = self.base_model.fit(X,self.Qi_gamma,batch_size=self.batch_size,epochs=self.epochs,verbose=0) 
        
        #-------> beta
        self.betas = np.tensordot(self.Qi_gamma, y_ann, axes=[[0],[0]]).transpose(1,0,2)
        if self.priors: #as a annotator not label all data:
            self.betas += self.Mpriors
        self.betas = self.betas/self.betas.sum(axis=-1,keepdims=True) #normalize
    
    def compute_logL(self):#,yo,predictions):
        return np.sum( np.log( self.sum_unnormalized_q +self.Keps))
        
    def train(self,X_train,yo_train,batch_size=64,max_iter=500,relative=True,val=False,tolerance=1e-2):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        print("Initializing new EM...")
        self.init_E(X_train,yo_train)
        self.batch_size = batch_size

        logL = []
        stop_c = False
        old_betas,tol = np.inf, np.inf
        self.current_iter = 1
        while(not stop_c):
            print("Iter %d/%d \nM step:"%(self.current_iter,max_iter),end='',flush=True)
            start_time = time.time()
            self.M_step(X_train,yo_train)
            print(" done,  E step:",end='',flush=True)
            predictions = self.get_predictions(X_train) #p(z|x)
            self.E_step(X_train,yo_train,predictions)
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
            if val:
                print("F1: %.4f"%(f1_score(Z_train,predictions.argmax(axis=-1),average='micro')),end='',flush=True)
            self.current_iter+=1
            print("")
            if self.current_iter>max_iter or (tol<=tolerance and tol2<=tolerance):
                stop_c = True
        print("Finished training")
        return logL
            
    def stable_train(self,X,y_ann,batch_size=64,max_iter=50,tolerance=1e-2):
        self.define_priors('laplace') #cada anotadora dijo al menos una clase
        logL_hist = self.train(X,y_ann,batch_size=batch_size,max_iter=max_iter,relative=True,val=False,tolerance=tolerance)
        return logL_hist
    
    def multiples_run(self,Runs,X,y_ann,batch_size=64,max_iter=50,tolerance=1e-2):  #tolerance can change
        if Runs==1:
            return self.stable_train(X,y_ann,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance), 0
        self.define_priors('laplace') #cada anotadora dijo al menos una clase
     
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

            logL_hist = self.train(X,y_ann,batch_size=batch_size,max_iter=max_iter,relative=True,tolerance=tolerance) 
            found_betas.append(self.betas.copy())
            found_model.append(self.base_model.get_weights()) #revisar si se resetean los pesos o algo asi..
            found_logL.append(logL_hist)
            iter_conv.append(self.current_iter-1)
            
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
        print("Multiples runs over Raykar, Epochs to converge= ",np.mean(iter_conv))
        return found_logL,indexs_sort[0]

    def get_predictions_annot(self,X,data=[]):
        """ Predictions of all annotators , p(y^o | xi, t) """
        if len(data) != 0:
            p_z = data
        else:
            p_z = self.get_predictions(X)
        predictions_a= np.tensordot(p_z ,self.betas,axes=[[1],[1]] ) # sum_z p(z|xi) * p(yo|z,t)
        return predictions_a#.transpose(1,0,2)

    def get_annotator_reliability(self,t):
        """Get annotator reliability, based on his identifier: t"""
        conf_M = self.betas[t,:,:]
        return conf_M #do something with it
    
    
#from .crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy, MaskedMultiCrossEntropy_Reg
class RodriguesCrowdLayer(object):
    def __init__(self, input_dim, Kl, T, epochs=1, optimizer='adam',DTYPE_OP='float32'): #default stable parameteres
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.input_dim = input_dim
        self.K = Kl #number of classes of the problem
        self.T = T #number of annotators
        #params:
        self.epochs = epochs
        self.optimizer = optimizer
        self.DTYPE_OP = DTYPE_OP

        self.compile=False
        self.Keps = keras.backend.epsilon()
        
    def define_model(self,tipo,model,start_units=1,deep=1,double=False,drop=0.0,embed=[],BatchN=False,glo_p=False, loss="masked",lamb=1):
        """Define the network of the base model"""
        self.type = tipo.lower()     
        if self.type == "keras_shallow" or 'perceptron' in self.type: 
            base_model = LogisticRegression_Keras(self.input_dim,self.K)
            #It's not a priority, since HF has been shown to underperform RMSprop and Adagrad, while being more computationally intensive.
            #https://github.com/keras-team/keras/issues/460
            self.compile = True
            return
        
        if self.type == "keras_import":
            base_model = model
        elif self.type=='defaultcnn' or self.type=='default cnn':
            base_model = default_CNN(self.input_dim,self.K)
        elif self.type=='defaultrnn' or self.type=='default rnn':
            base_model = default_RNN(self.input_dim,self.K)
        elif self.type=='defaultcnntext' or self.type=='default cnn text': #with embedding
            base_model = default_CNN_text(self.input_dim[0],self.K,embed) #len is the length of the vocabulary
        elif self.type=='defaultrnntext' or self.type=='default rnn text': #with embedding
            base_model = default_RNN_text(self.input_dim[0],self.K,embed) #len is the length of the vocabulary

        elif self.type == "ff" or self.type == "mlp" or self.type=='dense': #classic feed forward
            print("Needed params (units,deep,drop,BatchN?)") #default activation is relu
            base_model = MLP_Keras(self.input_dim,self.K,start_units,deep,BN=BatchN,drop=drop)

        elif self.type=='simplecnn' or self.type=='simple cnn' or 'cnn' in self.type:
            print("Needed params (units,deep,drop,double?,BatchN?)") #default activation is relu
            base_model = CNN_simple(self.input_dim,self.K,start_units,deep,double=double,BN=BatchN,drop=drop,global_pool=glo_p)
        
        elif self.type=='simplernn' or self.type=='simple rnn' or 'rnn' in self.type:
            print("Needed params (units,deep,drop,embed?)")
            base_model = RNN_simple(self.input_dim,self.K,start_units,deep,drop=drop,embed=embed,len=0,out=start_units*2)

        x = base_model.input
        base_model = keras.models.Model(x, base_model(x), name='base_model')
        base_model.compile(optimizer=self.optimizer,loss='categorical_crossentropy') 

        ## DEFINE NEW NET
        p_zx = base_model(x)
        crowd_layer = CrowdsClassification(self.K, self.T, conn_type="MW", name='CrowdL') ## ADD CROWDLAYER 
        p_yxt = crowd_layer(p_zx)
        self.model_crowdL = keras.models.Model(x, p_yxt) 

        self.loss = loss
        self.lamb = lamb
        if self.loss == "masked" or self.loss=='normal' or self.loss=='default':
            self.model_crowdL.compile(optimizer=self.optimizer, loss= MaskedMultiCrossEntropy_Reg().loss )
        elif self.loss == "kl":
            self.model_crowdL.compile(optimizer=self.optimizer, loss= MaskedMultiCrossEntropy_Reg().loss_prior_MV(p_zx,self.lamb) )
        self.compile = True
        self.base_model = self.model_crowdL.get_layer("base_model")
        self.max_Bsize_base = estimate_batch_size(self.base_model)
        
    def train(self,X_train,yo_train,batch_size=64,max_iter=500,tolerance=1e-2, pre_init_z=0):   
        if not self.compile:
            print("You need to create the model first, set .define_model")
            return
        print("Initializing...")
        self.N = X_train.shape[0]
        self.batch_size = batch_size
        self.base_model = self.model_crowdL.get_layer("base_model")
        if pre_init_z != 0:
            mv_probs = majority_voting(y_ann,repeats=False,probas=True) #Majority voting start
            print("Pre-train networks over *z* on %d epochs..."%(self.pre_init_z),end='',flush=True)
            self.base_model.fit(X,mv_probs_j,batch_size=self.batch_size,epochs=self.pre_init_z,verbose=0)
            self.base_model.compile(loss='categorical_crossentropy',optimizer=self.optimizer) #reset optimizer but hold weights--necessary for stability   
            print(" Done!")
        
        ourCallback = EarlyStopRelative(monitor='loss',patience=1,min_delta=tolerance)
        hist = self.model_crowdL.fit(X_train, yo_train, epochs=max_iter, batch_size=batch_size, verbose=1,callbacks=[ourCallback])
        print("Finished training")
        return hist.history["loss"]
            
    def stable_train(self,X,y_ann,batch_size=64,max_iter=50,tolerance=1e-2,pre_init_z=0):
        logL_hist = self.train(X,y_ann,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,pre_init_z=pre_init_z)
        return logL_hist
    
    def multiples_run(self,Runs,X,y_ann,batch_size=64,max_iter=50,tolerance=1e-2, pre_init_z=0):  #tolerance can change
        if Runs==1:
            return self.stable_train(X,y_ann,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,pre_init_z=pre_init_z), 0
     
        found_model = []
        found_lossL = []
        iter_conv = []
        obj_clone = Clonable_Model(self.model_crowdL) #architecture to clone

        for run in range(Runs):
            self.model_crowdL = obj_clone.get_model() #reset-weigths    
            if self.loss == "masked" or self.loss == 'normal' or self.loss =='default':
                self.model_crowdL.compile(optimizer=self.optimizer, loss= MaskedMultiCrossEntropy_Reg().loss )
            elif self.loss == "kl":
                p_zx = self.model_crowdL.get_layer("base_model").get_output_at(-1)
                self.model_crowdL.compile(optimizer=self.optimizer, loss=  MaskedMultiCrossEntropy_Reg().loss_prior_MV(p_zx,self.lamb) )

            lossL_hist = self.train(X,y_ann,batch_size=batch_size,max_iter=max_iter,tolerance=tolerance,pre_init_z=pre_init_z) 
            found_model.append(self.model_crowdL.get_weights()) #revisar si se resetean los pesos o algo asi..
            found_lossL.append(lossL_hist)
            iter_conv.append(len(lossL_hist))
            
            del self.model_crowdL
            keras.backend.clear_session()
            gc.collect()
        #setup the configuration with minimum log-loss
        logLoss_iter = np.asarray([np.min(a) for a in found_lossL]) #o el ultimo valor?
        indexs_sort = np.argsort(logLoss_iter) #minimum
        
        self.model_crowdL = obj_clone.get_model() #change
        self.model_crowdL.set_weights(found_model[indexs_sort[0]])
        self.base_model = self.model_crowdL.get_layer("base_model") #to set up model predictor
        print("Multiples runs over Raykar, Epochs to converge= ",np.mean(iter_conv))
        return found_lossL, indexs_sort[0]
    
    def get_basemodel(self):
        return self.base_model
    
    def get_predictions(self,X):
        if "sklearn" in self.type:
            return self.base_model.predict_proba(X) #or predict
        else:
            return self.base_model.predict(X,batch_size=self.max_Bsize_base)
        
    def get_confusionM(self):
        """Get confusion matrices of every annotator p(yo^t|,z)"""  
        weights = self.model_crowdL.get_layer("CrowdL").get_weights()[0] #witohut bias
        crowdL_conf = weights.T 
        return crowdL_conf #???
        
    def get_predictions_annot(self,X):
        """ Predictions of all annotators , p(y^o | xi, t) """
        return self.model_crowdL.predict(X).transpose([0,2,1]) 
          

class KajinoClustering(object):
    def __init__(self, input_dim, Kl, T, optimizer='adam',DTYPE_OP='float32'): #default stable parameteres
        if type(input_dim) != tuple:
            input_dim = (input_dim,)
        self.input_dim = input_dim
        self.K = Kl #number of classes of the problem
        self.T = T #number of annotators
        #params:
        self.epochs = epochs
        self.optimizer = optimizer
        self.DTYPE_OP = DTYPE_OP

        self.compile=False
        self.Keps = keras.backend.epsilon()
        self.priors=False #boolean of priors

   #build model and train and get confusion matrices..

   #get base model or predictions average (as guan)