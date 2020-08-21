import keras, math, gc
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import entropy, pearsonr
try:
    from tabulate import tabulate
except:
    print("NO TABULATE INSTALLED!")
import pandas as pd
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

def S_bias(conf_ma, mode="entropy"):
    """Score to known if p(y|something) == p(y) """
    p_y = conf_ma.mean(axis=0) #prior anotation

    b_C= p_y.argmax() #no recuerdo ...
    if mode=="entropy":      
        p_y = np.clip(p_y, 1e-7, 1.)  
        return entropy(p_y + 1e-7)/np.log(len(p_y)), b_C
    elif mode == "median":
        return (p_y.max() - np.median(p_y)), b_C
    elif mode == "simple":
        return p_y.max(), b_C

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

class Evaluation_metrics(object):
    def __init__(self,class_infered,which='our',N=None,plot=True, text=False):
        self.which=which.lower()
        self.plot = plot
        self.text = text
        self.labels_plot = []
        self.T_weights = []
        self.T = 0

        self.Gt = []
        self.seted_probas_group = False
        
        if 'our' in self.which:
            self.M = class_infered.M
            self.N = class_infered.N
            self.Kl = class_infered.Kl
            try:
                self.probas_group = class_infered.get_alpha()
                self.seted_probas_group = True
            except:
                pass
                
        elif self.which == 'keras':
            self.Kl = class_infered.output_shape[-1]
            self.N = N
            
        elif self.which == 'raykar':
            self.T = class_infered.T
            self.N = class_infered.N
            self.Kl = class_infered.Kl
        #and what about raykar or anothers
    
    def set_T_weights(self,T_weights):
        if np.abs(np.sum(self.T_weights) -1) > 0.0001:
            self.T_weights =  T_weights/ T_weights.sum(axis=0,keepdims=True) #normalize
        else:
            self.T_weights = T_weights.copy()        
            
    def set_Gt(self,Gt):
        self.Gt = Gt.copy()
        if not self.seted_probas_group:
            self.probas_group = np.mean(Gt,axis=0)
    
    def calculate_metrics(self,Z=[],Z_pred=[],y_o=[],yo_pred=[],conf_pred=[],conf_true=[],y_o_groups=[],conf_pred_G=[],conf_true_G=[]):
        if len(y_o)!=0:
            self.T = y_o.shape[1] #es util para estimar el peso de anotadores
            if len(self.T_weights) == 0:
                self.T_weights = np.sum(y_o != -1,axis=0)
                self.T_weights =  self.T_weights/ self.T_weights.sum(axis=0,keepdims=True)
                if np.abs(np.sum(self.T_weights) -1) > 0.0001:
                    print("Sum of weight of annotators ERROR, is :",np.sum(self.T_weights))
                    return
        elif len(yo_pred)!=0:
            self.T = yo_pred.shape[1]
          
        to_return = []
        if self.which == 'our1' and len(conf_pred) == self.M: 
            to_return.append(self.report_results_wt_annot(conf_pred,plot=self.plot, text=self.text)) #intrisic metrics
            
        if len(Z) != 0: #if we have Ground Truth
            if self.which == 'our1' and len(y_o_groups) != 0 and len(conf_pred) == self.M:  #test set usually
                t_aux = self.report_results_groups(Z,y_o_groups) #groups performance
                if len(to_return) !=0: 
                    t_aux = pd.concat((to_return[-1],t_aux),axis=1) #append intrinsic metrics
                    to_return = [] #clean
                to_return.append(t_aux)
                
            t = self.report_results(Z_pred, Z, conf_pred, conf_true,self.plot,conf_pred_G,conf_true_G)
            if len(y_o) != 0 and len(yo_pred)!= 0: #if we have annotations and GT: maybe training set
                t = self.rmse_accuracies(Z, y_o, yo_pred,dataframe=t) #calculate and append on "t" rmse
            to_return.append(t)
            
        else: #if we dont have GT
            if len(y_o) != 0 and len(yo_pred) !=0: #If we have annotations but no GT: maybe trainig set
                to_return.append(self.report_results_wt_GT(y_o,yo_pred))
                
        if self.plot:         
            for table in to_return:
                if run_from_ipython:
                    from IPython.display import display, HTML
                    print("A result")
                    display(table.round(4))
                else:
                    print("A result\n",tabulate(table, headers='keys', tablefmt='rst'))
        return to_return
         
    def report_results(self,y_pred,y_true,conf_pred=[],conf_true=[],plot=True,conf_pred_G=[],conf_true_G=[]):
        """
            *Calculate metrics related to model and to confusion matrixs
            Needed: ground truth, for confusion matrix need annotations.
        """
        t = pd.DataFrame()#Table()
        t[""] = ["All"]
        t["Accuracy"] = [accuracy_score(y_true,y_pred)]
        t["F1 (micro)"] = [f1_score(y_true=y_true, y_pred=y_pred, average='micro')]
        t["F1 (macro)"] = [f1_score(y_true=y_true, y_pred=y_pred, average='macro')]
        sampled_plot = 0
        if len(conf_true) != 0:
            print("Calculate confusion matrix on repeat version")
            #KLs_founded = calculateKL_matrixs(conf_pred,conf_true)
            JSs_founded = calculateJS_matrixs(conf_pred,conf_true)
            NormFs_founded = calculateNormF_matrixs(conf_pred,conf_true)
            
            pearson_corr = []
            for m in range(len(conf_pred)):
                #diagional_elements_pred = [conf_pred[m][f,f] for f in range(conf_pred[m].shape[0])]
                #diagional_elements_true = [conf_true[m][f,f] for f in range(conf_true[m].shape[0])]
                #normalize diagonal
                #diagional_elements_pred = (diagional_elements_pred-np.mean(diagional_elements_pred))/(np.std(diagional_elements_pred)+1e-10)
                #diagional_elements_true = (diagional_elements_true-np.mean(diagional_elements_true))/(np.std(diagional_elements_true)+1e-10)
                #pearson_corr.append(pearsonr(diagional_elements_pred, diagional_elements_true)[0])
                if np.random.rand() >0.5 and sampled_plot < 15 and plot:
                    compare_conf_mats(conf_pred[m], conf_true[m], text=self.text)
                    sampled_plot+=1
                    #print("KL divergence: %.3f\tPearson Correlation between diagonals: %.3f"%(KLs_founded[m],pearson_corr[-1]))        
                    #print("JS divergence: %.3f\tPearson Correlation between diagonals: %.3f"%(JSs_founded[m],pearson_corr[-1])) 
                    print("JS divergence: %.3f\tNorm Frobenius: %.3f"%(JSs_founded[m],NormFs_founded[m]))   
                    if len(self.Gt) != 0:
                        print("Groups probabilities: ",np.round(self.Gt[m],4))

            t["(R) NormF mean"] = np.mean(NormFs_founded)
            t["(R) JS mean"] = np.mean(JSs_founded) 
            #t["Mean PearsCorr"] = np.mean(pearson_corr)
            if len(self.T_weights) != 0:
                t["(R) NormF w"] = np.sum(self.T_weights*NormFs_founded)
                t["(R) JS w"] = np.sum(self.T_weights*JSs_founded) 
                #t["Wmean PearsCorr"] = np.sum(self.T_weights*pearson_corr)
        if len(conf_true_G) != 0:
            print("Calculate confusion matrix on global version")
            JSs_founded = JS_confmatrixs(conf_pred_G,conf_true_G)
            NormFs_founded = NormF_confmatrixs(conf_pred_G,conf_true_G)
 
            if plot:
                compare_conf_mats(conf_pred_G, conf_true_G, text=self.text)
                print("JS divergence: %.3f\tNorm Frobenius: %.3f"%(JSs_founded,NormFs_founded))   
            t["(G) NormF"] = NormFs_founded
            t["(G) JS"] = JSs_founded
        return t

    def rmse_accuracies(self,Z_argmax,y_o,yo_pred,dataframe=None): 
        """Calculate RMSE between accuracies of real annotators and predictive model of annotators
            Need annotations and ground truth
        """
        rmse_results = []
        for t in range(self.T):
            aux_annotations = np.asarray([(i,annotation) for i, annotation in enumerate(y_o[:,t]) if annotation != -1])
            t_annotations = aux_annotations[:,1] 
            
            gt_over_annotations = Z_argmax[aux_annotations[:,0]]
            prob_data = yo_pred[:,t][aux_annotations[:,0]]

            acc_annot_real = accuracy_score(gt_over_annotations, t_annotations)
            if prob_data.shape[-1]>1: #if probabilities is handled
                acc_annot_pred = accuracy_score(gt_over_annotations, prob_data.argmax(axis=-1))
            else: #if argmax is passed
                acc_annot_pred = accuracy_score(gt_over_annotations, prob_data) 

            rmse_results.append(np.sqrt(np.mean(np.square(acc_annot_real- acc_annot_pred ))))
        rmse_results = np.asarray(rmse_results)
        dataframe["RMSE mean"] = np.mean(rmse_results)
        if len(self.T_weights) != 0:
            dataframe["RMSE w"] = np.sum(self.T_weights*rmse_results)
        return dataframe
    
    def report_results_wt_GT(self,y_o,yo_pred): #new
        """Calculate a comparison between annotators and predictive model of annotators without GT"""
        DT = pd.DataFrame()#Table()
        metric_acc = []
        metric_CE = []
        metric_F1mi = []
        metric_F1ma = []
        for t in range(self.T):
            aux_annotations = np.asarray([(i,annotation) for i, annotation in enumerate(y_o[:,t]) if annotation != -1])
            t_annotations = aux_annotations[:,1]
            
            prob_data = yo_pred[:,t][aux_annotations[:,0]]
            
            if prob_data.shape[-1]>1: #if probabilities is handled
                accuracy = accuracy_score(t_annotations, prob_data.argmax(axis=1))
                #cross_entropy_loss = -np.mean(np.sum(keras.utils.to_categorical(t_annotations,num_classes=prob_data.shape[-1])*np.log(prob_data),axis=-1))
                f1_mi = f1_score(y_true=t_annotations, y_pred=prob_data.argmax(axis=1), average='micro')
                f1_ma = f1_score(y_true=t_annotations, y_pred=prob_data.argmax(axis=1), average='macro')

                #metric_CE.append(cross_entropy_loss)
            else:
                accuracy = accuracy_score(t_annotations, prob_data)
                f1_mi = f1_score(y_true=t_annotations, y_pred=prob_data, average='micro')
                f1_ma = f1_score(y_true=t_annotations, y_pred=prob_data, average='macro')
            metric_acc.append(accuracy)
            metric_F1mi.append(f1_mi)
            metric_F1ma.append(f1_ma)
        DT["ACC imiting Annot mean"] = [np.mean(metric_acc)]
        DT["F1-mi imiting Annot mean"] = [np.mean(metric_F1mi)]
        DT["F1-ma imiting Annot mean"] = [np.mean(metric_F1ma)]
        #DT["Cross-entropy mean"] = [np.mean(metric_CE)]
        if len(self.T_weights) != 0:
            DT["ACC imiting Annot wmean"] = [np.sum(self.T_weights*metric_acc)]
            #DT["Cross entropy wmean"] = [np.sum(self.T_weights*metric_CE)]
            DT["F1-mi imiting Annot wmean"] = [np.sum(self.T_weights*metric_F1mi)]
            DT["F1-ma imiting Annot wmean"] = [np.sum(self.T_weights*metric_F1ma)]
        return DT
            
     
    def report_results_groups(self,Z_argmax,y_o_groups,added=True): #new
        """Calculate performance of the predictive model of the groups modeled"""
        t = pd.DataFrame()#Table()
        accs = []
        f1_s = []
        if len(y_o_groups.shape) == 3:
            predictions_m = y_o_groups.argmax(axis=-1)
        else:
            predictions_m = y_o_groups #by argmax
        for m in range(self.M):
            accs.append(accuracy_score(Z_argmax,predictions_m[:,m]))
            f1_s.append(f1_score(y_true=Z_argmax, y_pred=predictions_m[:,m], average='micro'))
        t["Accuracy"] = accs
        t["F1 (micro)"] = f1_s
        return t
        
    def report_results_wt_annot(self,conf_matrixs,groups_proba=[],plot=True, text=False):
        """Calculate Intrinsic measure of only the confusion matrices infered """
        t = pd.DataFrame()#Table()
        identity_matrixs = np.asarray([np.identity(conf_matrixs.shape[1]) for m in range(len(conf_matrixs))])
        KLs_identity = calculateKL_matrixs(conf_matrixs,identity_matrixs)
        JSs_identity = calculateJS_matrixs(conf_matrixs,identity_matrixs)
        
        entropies = []
        mean_diagional = []
        spammer_score =[] #raykar
        bias_score = []
        bias_class = []
        for m in range(self.M):
            if plot:
                if len(self.labels_plot) == 0:
                    self.labels_plot = np.arange(conf_matrixs[m].shape[0])
                plot_confusion_matrix(conf_matrixs[m],self.labels_plot,title="Group "+str(m),text=text)
            #New Instrisic measure
            entropies.append(Entropy_confmatrix(conf_matrixs[m]))
            mean_diagional.append(calculate_diagional_mean(conf_matrixs[m]))
            spammer_score.append(calculate_spamm_score(conf_matrixs[m]))
            b, c = calculate_biased_score(conf_matrixs[m], mode="median")
            bias_score.append(b)
            bias_class.append(c)
            
        t["Groups"] = np.arange(self.M)
        #if len(self.probas_group) != 0:
        t["Prob"] = self.probas_group
        if self.T != 0:
            t["T(g)"] = list(map(int,self.probas_group*self.T))
        t["Entropy"] = entropies
        t["Diag mean"] = mean_diagional
        #t["KL to I"] = KLs_identity
        t["Isim (JS)"] = 1-JSs_identity #value betweeon [0,1]
        #t["Matrix-norm to identity"] = pendiente...
        t["S_raykar"] = spammer_score #spammer score-- based on raykar logits (-1 malicious, 0 spammer, 1 good)
        #t["S_bias entrop"] = bias_score1
        t["S_bias"] = bias_score
        t["C_bias"] = bias_class
        inertia_JS = calculate_inertiaM_JS(conf_matrixs)
        inertia_NormF = calculate_inertiaM_NormF(conf_matrixs)
        t["Iner JS"] = np.tile(inertia_JS, self.M)        
        t["Iner NormF"] = np.tile(inertia_NormF, self.M)  
        if plot:    
            print("Inertia JS:",inertia_JS)
            print("Inertia NormF:",inertia_NormF)
        else:
            self.inertia_JS = inertia_JS
            self.inertia_NormF = inertia_NormF
        return t


"""
How to use it:

#Import it:
from evaluation import Evaluation_metrics
evaluate = Evaluation_metrics(gMixture,'our1')


#>>>>>>>>>>>>>>>>>>> Usuall train

#needed to evaluate other stuffs
aux = gMixture.calculate_extra_components(Xstd_train,y_obs,T=100,calculate_pred_annotator=True)
predictions_m,prob_Gt,prob_Yzt,prob_Yxt =  aux #to evaluate...

Z_train_pred = gMixture.base_model.predict_classes(Xstd_train)
#argmax groups
y_o_groups = predictions_m.argmax(axis=-1)

results = evaluate.calculate_metrics(Z=Z_train,Z_pred=Z_train_pred,conf_pred=prob_Yzt,conf_true=confe_matrix,y_o=y_obs,yo_pred=prob_Yxt, y_o_groups=y_o_groups)


#>>>>>>>>>>>>>>>>>>> train bulk annotations without GT
results = evaluate.calculate_metrics(y_o=y_obs,yo_pred=prob_Yxt)


#>>>>>>>>>>>>>>>>>>> test or train without bulks annotation?--as repeats--- no annotations but ground truth
c_M = gMixture.get_confusionM()
Z_test_pred = gMixture.base_model.predict_classes(Xstd_test)
y_o_groups = gMixture.get_predictions_groups(Xstd_test) #obtain p(y^o|x,g=m)
#argmax groups
y_o_groups = y_o_groups.argmax(axis=-1)

results = evaluate.calculate_metrics(Z=Z_test,Z_pred=Z_test_pred,conf_pred=c_M, y_o_groups=y_o_groups)


#>>>>>>>>>>>>>>>>>>> test without GT
results = evaluate.calculate_metrics(conf_pred=c_M)
"""
