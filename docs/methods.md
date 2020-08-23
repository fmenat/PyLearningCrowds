## Crowdsourcing Methods
```python
from codeE.methods import ...
```
State of the art methods proposed in the crowdsourcing scenario, also called *learning from crowds*.

To problem notation see the [documentation](notation.md)

The available methods:
* [Majority Voting](#simple-aggregation-techniques): soft, hard, weighted  
* [Dawid and Skene](#label-inference-based-on-em---confusion-matrix) ground truth (GT) inference based on confusion matrices (CM) of annotators.   
* [Raykar et al](#model-inference-based-on-em---confusion-matrix): predictive model over GT inference based on CM of annotators  
* [Crowd Mixture Model](#model-and-annotations-group-inference-based-on-em---confusion-matrix) inference of model and groups on annotations of the data  
* [Crowd - Mixture of Annotators](#model-and-annotators-group-inference-based-on-em---confusion-matrix) inference of model and groups on annotations of the annotators  
* [Global - Label Noise](#model-inference-based-on-em---label-noise) inference of model and global behavior of annotations, as a label noise problem (only one confusion matrix)


---
### Simple aggregation techniques
```python
class codeE.methods.LabelAgg(scenario="global")
```
[UP](#crowdsourcing-methods)

Methods that reduced the multiple annotations *y* to a single target for each input pattern *x*. The goal is to define a ground truth *z* as a function of the annotations, for example summary statistic such as the mean, the median or the mode. 

The most used and simple technique corresponds to Majority Voting (MV) [1], which can handle both scenario setting, individual or global.
> It can handle both representation: *individual* and *dense* (for further details see [representation documentation](representation.md))

**Parameters**  
* **scenario: *string, {'global','individual'}, default='global'***  
The scenario in which the annotations will be aggregated. Subject to the representation format, for further details see the [representation](representation.md) documentation.


##### References
[1] [Sheng, V. S., Provost, F., & Ipeirotis, P. G. (2008, August). *Get another label? improving data quality and data mining using multiple, noisy labelers*.](https://archive.nyu.edu/bitstream/2451/25882/4/kdd2008.pdf)  
[2] [Rodrigues, F., Pereira, F., & Ribeiro, B. (2013). *Learning from multiple annotators: distinguishing good from random labelers*.](https://core.ac.uk/download/pdf/43577079.pdf)


##### Examples
```python
... #read some data
from codeE.representation import set_representation
y_obs_categorical = set_representation(y_obs,'onehot')
y_cat_var, _ = set_representation(y_obs,"onehotvar") 
r_obs = set_representation(y_obs,"global")
```
> Infer over global scenario
```python
from codeE.methods import LabelAgg
label_A = LabelAgg(scenario="global")
mv_soft = label_A.infer(  r_obs, 'softMV')
mv_hard = label_A.predict(r_obs, 'hardMV')
```
> Infer over individual - sparse scenario
```python
from codeE.baseline import LabelAgg
label_A = LabelAgg(scenario="individual", sparse=True)
mv_soft = label_A.infer(  y_cat_var, 'softMV')
mv_hard = label_A.predict(y_cat_var, 'hardMV')
```
> Infer over individual - dense scenario
```python
from codeE.baseline import LabelAgg
label_A = LabelAgg(scenario="individual")
mv_soft = label_A.infer(  y_obs_categorical, 'softMV')
mv_hard = label_A.predict(y_obs_categorical, 'hardMV')
```
> Weighted (individual - dense scenario)
```python
from codeE.baseline import LabelAgg
label_A = LabelAgg(scenario="individual")
T_weights = np.sum(y_obs_categorical.sum(axis=-1) == 0, axis=0) #number of annotations given per annotator
Wmv_soft = label_A.infer(y_obs_categorical, 'softMV', weights=T_weights)
Wmv_soft
```

##### Class Methods
|Function|Description|
|---|---|
|*infer*| Return the inferred label of the data|
|*predict*| same as *infer*|

```python
infer(labels, method, weights=[1], onehot=False)
```
Infer the ground truth over the labels (multiple annotations) based on the indicated method.

**Parameters**  
* **labels: *array-like of shape***  
> *scenario='global'*: *(n_samples, n_classes)*  
> *scenario='individual'*: *(n_samples, n_annotators, n_classes)*  

Annotations of the data, should be individual or global representation

* **method: *string, {'softmv','hardmv'}***  
The method used to infer the ground truth of the data based on the most used aggregation techniques: Majority Voting (MV) [2].
> *hardmv*: The categorical (discrete) value of the ground truth is obtained as the most frequent label between the observed annotations.
> *softmv*: The probability estimation of the ground truth is obtained as the relative frequency of each possible label over all the observed annotations.


* **weights: *array-like or list of shape or (n_annotators,)***  
The weights over annotators to use in the aggregation scheme, if it is neccesary. There is no restriction on this value (the sum does not need to be 1).
* **onehot: *boolean, default=False***  
Only used if *method='hardmv'*. This value will control the returned representation, as one-hot vector or as class numbers (between 0 and K-1).

**Returns**  
* **Z_hat: *array-like of shape (n_samples, n_classes) or (n_samples,)***  
The inferred ground truth of the data based on the selected method. The returned shape will be *(n_samples,)* if *method='hardmv* and *onehot=False*


```python
predict(*args)
```
same operation than infer function.


---
### Label inference based on EM - Confusion Matrix
```python
class codeE.methods.LabelInf_EM(init_Z='softmv', priors=0, fast=False, DTYPE_OP='float32')
```
[UP](#crowdsourcing-methods)

Method that infers the ground truth label with a probabilistic framework based on the EM algorithm [1]. It represent the annotators ability as a confusion matrix to detect the ground truth <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(t)} = p(y=j | z=k, a=t) ">

This method is proposed on the framework of Dawid and Skene (D&S) [2].
> It is proposed on the *individual dense* representation ((for further details see [representation documentation](representation.md)))

**Parameters**  
* **init_Z: *string, {'softmv','hardmv'}, default='softmv'***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|i)">. Both posibilities are based on LabelAgg class. 
* **priors: *different options***  
The priors to be set on the confusion matrices of annotators, could be in different formats:
	* *string, {'laplace','none'}*  
	The 'laplace' stand for Laplace smoothing, a prior with the value of 1.
	* *int*  
	A number of annotations to be set prior for all the annotators over all the data.
	* *array-like of shape (n_annotators,)*  
	A vector of the number of annotations to be set prior for every possible annotator on the data.
	* *array-like of shape (n_annotators, n_classes)*  
	A matrix of the number of annotations to be set priors for every annotator and every ground truth label on the data.
	* *array-like of shape (n_annotators, n_classes, n_classes)*  
	A cube with the number of annotations to be set priors for every annotator, every ground truth label and every observed label on the data. priors=
	> Comments on the priors: The laplace smooth prior helps to stabilize traning and speeds up convergence. The disadvantage trade-off correspond to a slightly worse estimation of the ground truth.
* **fast: *boolean, default=False***  
If the fast estimation of the method is used. This correspond to perform a discrete/hard estimation of the ground truth label, after E step. According to Sinha et al. [4] accelerates convergence (*reduce the number of iterations to convergence*).
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html


##### References
[1] [Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum likelihood from incomplete data via the EM algorithm*](http://www.eng.auburn.edu/~troppel/courses/7970%202015A%20AdvMobRob%20sp15/literature/paper%20W%20refs/dempster%20EM%201977.pdf)  
[2] [Dawid, A. P., & Skene, A. M. (1979). *Maximum likelihood estimation of observer error‐rates using the EM algorithm*](http://crowdsourcing-class.org/readings/downloads/ml/EM.pdf)  
[3] [Smyth, P., Fayyad, U. M., Burl, M. C., Perona, P., & Baldi, P. (1995). *Inferring ground truth from subjective labelling of venus images.*](http://papers.nips.cc/paper/949-inferring-ground-truth-from-subjective-labelling-of-venus-images.pdf)  
[4] [Sinha, V. B., Rao, S., & Balasubramanian, V. N. (2018). Fast Dawid-Skene: *A Fast Vote Aggregation Scheme for Sentiment Classification*](https://arxiv.org/pdf/1803.02781)


##### Examples
```python
... #read some data
from codeE.representation import set_representation
y_obs_categorical = set_representation(y_obs,'onehot') 
```
> Train the model
```python
from codeE.methods import LabelInf_EM as DS
DS_model = DS(init_Z='softmv')
hist = DS_model.fit(y_obs_categorical)
```
> Train with different settings 
```python
from codeE.methods import LabelInf_EM as DS
DS_model = DS(init_Z='hardmv', priors=1, fast=True) 
hist = DS_model.fit(y_obs_categorical)
```
> Get estimation of the ground truth label on trainig set and ground truth marginal
```python
print("p(z) =", DS_model.get_marginalZ())
ds_labels = DS_model.infer() 
ds_labels
```

##### Class Methods
|Function|Description|
|---|---|
|*get_marginalZ*| Returns the parameter used as marginal probability of ground truth|
|*get_confusionM*| Returns the confusion matrix of each annotator|
|*get_qestimation*| Returns the estimation over auxiliar variable *Q*|
|*set_priors*| To set the priors used on confusion matrices|
|*init_E*| Initialization of the E-step|
|*E_step*| Perform the inference on the E-step|
|*M_step*| Perform the inference on the M-step|
|*C_step*| Perform an auxiliar C-step|
|*compute_logL*| Calculate the log-likelihood of the data|
|*train*| Perform all the inference based on EM algorithm|
|*fit*| same as *train*|
|*infer*| Get the inferred ground truth on training set |
|*predict*| same as *infer*|
|*get_ann_confusionM*| Returns the individual confusion matrices estimation|


```python
get_marginalZ()
```
**Returns**  
* **z_marginal: *array-like of shape (n_classes,)***  
The parameter used as marginal probability of the ground truth <img src="https://render.githubusercontent.com/render/math?math=p(z=k)">

```python
get_confusionM()
```
**Returns**  
* **betas: *array-like of shape (n_annotators, n_classes, n_classes)***  
The confusion matrix of each annotator <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(t)}">

```python
get_qestimation()
```
**Returns**  
* **Qi_k: *array-like of shape (n_samples, n_classes)***  
The estimation over auxiliar variable *Q*

```python
set_priors(priors)
```
**Parameters**  
* **priors: *as state on init***  

```python
init_E(y_ann, method="")
```
Initialization of the E-step based on *method*.  
<img src="https://render.githubusercontent.com/render/math?math=q_i(z=k) = p(z=k | i)">

**Parameters**  
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.
* **method: *string, {'softmv','hardmv',''}, default=''***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k | i)">. Both posibilities are based on LabelAgg class. The empty string will use the method seted on init.

```python
E_step(y_ann)
```
Perform the inference on the E-step.

**Parameters**  
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.

```python
M_step(y_ann)
```
Perform the inference on the M-step.

**Parameters**  
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.

```python
C_step()
```
Perform an auxiliar discrete/hard step on the estimation of the ground truth label

```python
compute_logL()
```
Calculate the log-likelihood of the data.

```python
train(y_ann, max_iter=50,relative=True,tolerance=3e-2)
```
Perform all the inference based on EM algorithm.

**Parameters**  
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.
* **relative: *boolean, default=True***  
If the difference for tolerance is relative to the previous value.

```python
fit(*args)
```
same operation than train function.

```python
infer()
```
**Returns**  
* **Qi_k: *array-like of shape (n_samples, n_classes)***  
The estimation of ground truth on the training data.

```python
predict(*args)
```
same operation than infer function.

```python
get_ann_confusionM()
```
**Returns**  
* **prob_Y_Zt: *array-like of shape (n_annotators, n_classes, n_classes)***  
The individual confusion matrix of each annotator: <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(t)}">


---
### Model Inference based on EM - Confusion Matrix
```python
class codeE.methods.ModelInf_EM(init_Z='softmv', n_init_Z=0, priors=0, DTYPE_OP='float32')
```
[UP](#crowdsourcing-methods)


This method set a predictive model <img src="https://render.githubusercontent.com/render/math?math=f(x)">
 of the ground truth inside the inference for joint learning. It also represent the annotators ability as a confusion matrix <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(t)} = p(y=j | z=k, a=t)"> and allows any model on *f()*.

The original idea was proposed by Raykar et al. [1].
> It is proposed on the *individual dense* representation. (for further details see [representation documentation](representation.md))

**Parameters**  
* **init_Z: *string, {'softmv','hardmv'}, default='softmv'***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=q_i(z=k)=p(z=k|x_i)">. Both posibilities are based on LabelAgg class. 
* **n_init_Z: *int, default=0***  
The number of epochs that the predictive model is going to be pre-trained.
* **priors: *different options***  
The priors to be set on the confusion matrices of annotators, could be in different formats:
	* *string, {'laplace','none'}*  
	The 'laplace' stand for Laplace smoothing, a prior with the value of 1.
	* *int*  
	A number of annotations to be set prior for all the annotators over all the data.
	* *array-like of shape (n_annotators,)*  
	A vector of the number of annotations to be set prior for every possible annotator on the data.
	* *array-like of shape (n_annotators, n_classes)*  
	A matrix of the number of annotations to be set priors for every annotator and every ground truth label on the data.
	* *array-like of shape (n_annotators, n_classes, n_classes)*  
	A cube with the number of annotations to be set priors for every annotator, every ground truth label and every observed label on the data. 
	> Comments on the priors: The laplace smooth prior helps to stabilize traning and speeds up convergence. The disadvantage trade-off correspond to a slightly worse estimation of the ground truth.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html


##### References
[1] [Raykar, V. C., Yu, S., Zhao, L. H., Valadez, G. H., Florin, C., Bogoni, L., & Moy, L. (2010). *Learning from crowds.*](https://www.jmlr.org/papers/volume11/raykar10a/raykar10a.pdf)  
[2] [Albarqouni, S., Baur, C., Achilles, F., Belagiannis, V., Demirci, S., & Navab, N. (2016). *Aggnet: deep learning from crowds for mitosis detection in breast cancer histology images*](https://ieeexplore.ieee.org/iel7/42/7463083/07405343.pdf)  
[3] [Rodrigues, F., & Pereira, F. (2017). *Deep learning from crowds*](https://arxiv.org/pdf/1709.01779)  


##### Examples
```python
X_train = ...
... #read some data 
from codeE.representation import set_representation
y_obs_categorical = set_representation(y_obs,'onehot') 
```
> Define predictive model (based on keras)
```python
F_model = Sequential()
... #add layers
```
> Set the model and train the method
```python
from codeE.methods import ModelInf_EM as Raykar
R_model = Raykar(init_Z="softmv")
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
R_model.set_model(F_model, **args)
R_model.fit(X_train, y_obs_categorical)
```
> Train with different settings 
```python
from codeE.methods import ModelInf_EM as Raykar
R_model = Raykar(init_Z="softmv", priors='laplace')
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
R_model.set_model(F_model, **args)
R_model.fit(X_train, y_obs_categorical, runs=20)
```
> Get the base model to predict the ground truth on some data
```python
raykar_fx = R_model.get_basemodel()
raykar_fx.predict(new_X)
```

##### Class Methods
|Function|Description|
|---|---|
|*get_basemodel*| Returns the predictive model of the ground truth|
|*get_confusionM*| Returns the confusion matrix of each annotator|
|*get_qestimation*| Returns the estimation over auxiliar variable *Q*|
|*set_model*| Set the predictive model (base model) class|
|*set_priors*| To set the priors used on confusion matrices|
|*init_E*| Initialization of the E-step|
|*E_step*| Perform the inference on the E-step|
|*M_step*| Perform the inference on the M-step|
|*compute_logL*| Calculate the log-likelihood of the data|
|*train*| Perform all the inference based on EM algorithm|
|*multiples_run*| Perform multiple runs of the EM algorithm and save the best|
|*fit*| same as *multiples_run* |
|*get_ann_confusionM*| Returns the individual confusion matrices estimation|
|*get_predictions*| Returns the probability predictions of ground truth over some set|
|*get_predictions_annot*| Returns the probability estimation of labels over some set for each annotator|


```python
get_basemodel()
```
**Returns**  
* **base_model: *function or class***  
The predictive model over the ground truth <img src="https://render.githubusercontent.com/render/math?math=f(\cdot)">

```python
get_confusionM()
```
**Returns**  
* **betas: *array-like of shape (n_annotators, n_classes, n_classes)***  
The confusion matrix of each annotator <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(t)}">

```python
get_qestimation()
```
**Returns**  
* **Qi_k: *array-like of shape (n_samples, n_classes)***  
The estimation over auxiliar variable *Q*

```python
set_model(model, optimizer="adam", epochs=1, batch_size=32)
```
Set the predictive model (base model) over the ground truth and define how to optimize it **on the M step** of the iterative EM algorithm.

**Parameters**  
* **model: *function or class of keras model***  
Predictive model based on [Keras](https://keras.io/).
* **optimizer: *string, {'sgd','rmsprop','adam','adadelta','adagrad'}, default='adam'***  
String name of optimizer used on the back-propagation SGD, based on https://keras.io/api/optimizers/
* **epochs: *int, default=1***  
Number of epochs (iteration over the entire set) to train the model based on https://keras.io/api/models/model_training_apis/
* **batch_size: *int, default=32***  
Number of samples per gradient update, based on https://keras.io/api/models/model_training_apis/

```python
set_priors(priors)
```
**Parameters**  
* **priors: *as state on init***  

```python
init_E(y_ann, method="")
```
Initialization of the E-step based on *method*.  
<img src="https://render.githubusercontent.com/render/math?math=q_i(z=k) = p(z=k|x_i)">

**Parameters**  
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.
* **method: *string, {'softmv','hardmv',''}, default=''***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|i)">. Both posibilities are based on LabelAgg class. The empty string will use the method seted on init.

```python
E_step(X, y_ann, predictions=[])
```
Perform the inference on the E-step.

**Parameters**  
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.
* **predictions: *array-like of shape (n_samples, n_classes)***  
Probability predictions of the ground truth on training set. If X is given, not necessary to give this parameter (*default=[]*).

```python
M_step(X, y_ann)
```
Perform the inference on the M-step.

**Parameters**  
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.

```python
compute_logL()
```
Calculate the log-likelihood of the data.

```python
train(X_train, y_ann, max_iter=50,relative=True,tolerance=3e-2)
```
Perform all the inference based on EM algorithm.

**Parameters** 
* **X_train: *array-like of shape (n_samples, ...)***  
Input patterns of the training data. 
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.
* **relative: *boolean, default=True***  
If the difference for tolerance is relative to the previous value.

```python
multiples_run(Runs, X, y_ann, max_iter=50, tolerance=3e-2)
```
Perform multiple runs of the EM algorithm and save the best execution based on log-likelihood.

**Parameters** 
* **Runs: *int***  
The number of times the EM will be run to obtain different results.
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the training data. 
* **y_ann: *array-like of shape (n_samples, n_annotators, n_classes)***  
Annotations of the data, should be the individual one-hot (categorical) representation.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.

**Returns**  
* **found_logL: *list of length=Runs***  
A list with the history of log-likelihood for each iteration in the different runs
* **best_run: *int***  
The index of the best run between all executed, a number between *0* and *Runs-1*.


```python
fit(X,Y, runs = 1, max_iter=50, tolerance=3e-2)
```
same operation than *multiples_run* function.

```python
get_ann_confusionM()
```
**Returns**  
* **prob_Y_Zt: *array-like of shape (n_annotators, n_classes, n_classes)***  
The individual confusion matrix of each annotator: <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(t)}">

```python
get_predictions(X)
```
**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 

**Returns**  
* **prob_Z_hat: *array-like of shape (n_samples, n_classes)***  
The probability predictions of the ground truth over some set <img src="https://render.githubusercontent.com/render/math?math=f(x) = p(z|x)">


```python
get_predictions_annot(X, data=[])
```
**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 
* **data: *array-like of shape (n_samples, n_classes)***  
If the probability predictions of the ground truth over some set are delivered

**Returns**  
* **prob_Y_xt: *array-like of shape (n_samples, n_annotators\, n_classes)***  
The probability estimation of labels over some set for each annotator <img src="https://render.githubusercontent.com/render/math?math=p(y|x, t) = \sum_{z} p(y|z,t) p(z|x) ">.


---
### Model and Annotations Group Inference based on EM - Confusion Matrix
```python
class codeE.methods.ModelInf_EM_CMM(M, init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32')
```
[UP](#crowdsourcing-methods)

This method infer a predictive model <img src="https://render.githubusercontent.com/render/math?math=f(x)"> of the ground truth jointly with the ground truth inference based on **groups over the data annotations**. Contrary to other methods, it does not have an explicit model per annotators. It represents the **groups** ability as a confusion matrix <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(m)} = p(y=j | z=k, g=m)"> and allows any model on *f()*.

The original CMM (*Crowd Mixture Model*) method was proposed by Mena et al. [1].
> It is proposed on the *global* representation. (for further details see [representation documentation](representation.md))

**Parameters**  
* **M: *int***  
The number of groups (*n_groups*) to be found (*different types of behaviors*) in the annotations.
* **init_Z: *string, {'softmv','hardmv'}, default='softmv'***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|x_i)">. Both posibilities are based on LabelAgg class. 
* **n_init_Z: *int, default=0***  
The number of epochs that the predictive model is going to be pre-trained.
* **priors: *different options***  
The priors to be set on the confusion matrices of annotators, could be in different formats:
	* *string, {'laplace','none'}*  
	The 'laplace' stand for Laplace smoothing, a prior with the value of 1.
	* *int*  
	A number of annotations to be set prior for all the groups over all the data.
	* *array-like of shape (n_groups,)*  
	A vector of the number of annotations to be set prior for every possible group on the data.
	* *array-like of shape (n_groups, n_classes)*  
	A matrix of the number of annotations to be set priors for every group and every ground truth label on the data.
	* *array-like of shape (n_groups, n_classes, n_classes)*  
	A cube with the number of annotations to be set priors for every group, every ground truth label and every observed label on the data. 
	> Comments on the priors: The laplace smooth prior helps to stabilize traning and speeds up convergence. The disadvantage trade-off correspond to a slightly worse estimation of the ground truth.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html


##### References
[1] [Mena, F., & Ñanculef, R. (2019, October). *Revisiting Machine Learning from Crowds a Mixture Model for Grouping Annotations*](https://link.springer.com/chapter/10.1007/978-3-030-33904-3_46)  
[2] [Mena, F., Ñanculef, R., & Valles, C. (2020). *Collective Annotation Patterns in Learning from Crowds*]()

##### Examples
```python
X_train = ...
... #read some data 
from codeE.representation import set_representation
r_obs = set_representation(y_obs, "global")
```
> Define predictive model (based on keras)
```python
F_model = Sequential()
... #add layers
```
> Set the model and train the method
```python
from codeE.methods import ModelInf_EM_CMM as CMM
CMM_model = CMM(M=3)
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
CMM_model.set_model(F_model, **args)
CMM_model.fit(X_train, r_obs)
```
> Train with different settings 
```python
from codeE.methods import ModelInf_EM_CMM as CMM
CMM_model = CMM(M=3, init_Z='softmv', n_init_Z=3, priors=0) 
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
CMM_model.set_model(F_model, **args)
CMM_model.fit(X_train, r_obs, runs=20)
```
> Get the base model to predict the ground truth on some data
```python
cmm_fx = CMM_model.get_basemodel()
cmm_fx.predict(new_X)
```
> Get the presence probability and the confusion matrices of the groups
```python
print("p(g) =",CMM_model.get_alpha())
B = CMM_model.get_confusionM()
from codeE.utils import plot_confusion_matrix
for i in range(len(B)):
    plot_confusion_matrix(B[i])
```

##### Class Methods
|Function|Description|
|---|---|
|*get_basemodel*| Returns the predictive model of the ground truth|
|*get_confusionM*| Returns the confusion matrix of each group|
|*get_alpha*| Returns the presence probability of each group|
|*get_qestimation*| Returns the estimation over auxiliar model *Q*|
|*set_model*| Set the predictive model (base model) class|
|*set_priors*| To set the priors used on confusion matrices|
|*init_E*| Initialization of the E-step|
|*E_step*| Perform the inference on the E-step|
|*M_step*| Perform the inference on the M-step|
|*compute_logL*| Calculate the log-likelihood of the data|
|*train*| Perform all the inference based on EM algorithm|
|*multiples_run*| Perform multiple runs of the EM algorithm and save the best|
|*fit*| same as *multiples_run* |
|*get_global_confusionM*| Returns the global confusion matrix|
|*get_ann_confusionM*| Returns the individual confusion matrix estimation of some annotator based on his annotations on the data|
|*get_predictions*| Returns the probability predictions of ground truth over some set|
|*get_predictions_groups*| Returns the probability estimation of labels over the modeled groups|

```python
get_basemodel()
```
**Returns**  
* **base_model: *function or class***  
The predictive model over the ground truth <img src="https://render.githubusercontent.com/render/math?math=f(\cdot)">

```python
get_confusionM()
```
**Returns**  
* **betas: *array-like of shape (n_groups, n_classes, n_classes)***  
The confusion matrix of each modeled group <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(m)}">

```python
get_alpha()
```
**Returns**  
* **alphas: *array-like of shape (n_groups,)***  
The presence probability vector of the modeled groups <img src="https://render.githubusercontent.com/render/math?math=\alpha_m">

```python
get_qestimation()
```
**Returns**  
* **Qij_mk: *array-like of shape (n_samples, n_classes, n_groups, n_classes)***  
The estimation over auxiliar model *Q*

```python
set_model(model, optimizer="adam", epochs=1, batch_size=32)
```
Set the predictive model (base model) over the ground truth and define how to optimize it **on the M step** of the iterative EM algorithm.

**Parameters**  
* **model: *function or class of keras model***  
Predictive model based on [Keras](https://keras.io/).
* **optimizer: *string, {'sgd','rmsprop','adam','adadelta','adagrad'}, default='adam'***  
String name of optimizer used on the back-propagation SGD, based on https://keras.io/api/optimizers/
* **epochs: *int, default=1***  
Number of epochs (iteration over the entire set) to train the model based on https://keras.io/api/models/model_training_apis/
* **batch_size: *int, default=32***  
Number of samples per gradient update, based on https://keras.io/api/models/model_training_apis/

```python
set_priors(priors)
```
**Parameters**  
* **priors: *as state on init***  

```python
init_E(r_ann, method="")
```
Initialization of the E-step, based on the following approximation:
<img src="https://render.githubusercontent.com/render/math?math=q_{ij}(g=m, z=k) = p(g=m, z=k | x_i, y=j) = p(g=m|x_i,y=j) p(z=k|x_i)">. 
The groups *g* initialization is based on a K-means clustering and the ground truth *z* is based on *method*.

**Parameters**  
* **r_ann: *array-like of shape (n_samples, n_classes)***  
Annotations of the data, should be on the global representation.
* **method: *string, {'softmv','hardmv',''}, default=''***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|x_i)">. Both posibilities are based on LabelAgg class. The empty string will use the method seted on init.

```python
E_step(X, predictions=[])
```
Perform the inference on the E-step.

**Parameters**  
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **predictions: *array-like of shape (n_samples, n_classes)***  
Probability predictions of the ground truth on training set. If *X* is given, not necessary to give this parameter (*default=[]*).

```python
M_step(X, r_ann)
```
Perform the inference on the M-step.

**Parameters**  
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **r_ann: *array-like of shape (n_samples, n_classes)***  
Annotations of the data, should be on the global representation.

```python
compute_logL(r_ann)
```
Calculate the log-likelihood of the data.

```python
train(X_train, r_ann, max_iter=50,relative=True,tolerance=3e-2)
```
Perform all the inference based on EM algorithm.

**Parameters** 
* **X_train: *array-like of shape (n_samples, ...)***  
Input patterns of the training data. 
* **r_ann: *array-like of shape (n_samples, n_classes)***  
Annotations of the data, should be on the global representation.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.
* **relative: *boolean, default=True***  
If the difference for tolerance is relative to the previous value.

```python
multiples_run(Runs, X, r_ann, max_iter=50, tolerance=3e-2)
```
Perform multiple runs of the EM algorithm and save the best execution based on log-likelihood.

**Parameters** 
* **Runs: *int***  
The number of times the EM will be run to obtain different results.
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the training data. 
* **r_ann: *array-like of shape (n_samples, n_classes)***  
Annotations of the data, should be on the global representation.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.

**Returns**  
* **found_logL: *list of length=Runs***  
A list with the history of log-likelihood for each iteration in the different runs
* **best_run: *int***  
The index of the best run between all executed, a number between *0* and *Runs-1*.

```python
fit(X,Y, runs = 1, max_iter=50, tolerance=3e-2)
```
same operation than *multiples_run* function.

```python
get_predictions(X)
```
**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 

**Returns**  
* **prob_Z_hat: *array-like of shape (n_samples, n_classes)***  
The probability predictions of the ground truth over some set <img src="https://render.githubusercontent.com/render/math?math=f(x) = p(z|x)">

```python
get_global_confusionM()
```
**Returns**  
* **prob_Y_Z: *array-like of shape (n_classes, n_classes)***  
The global confusion matrix <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j} = \sum_{m} \hat{\beta}_{k,j}^{(m)} \cdot p(g=m)">

```python
get_ann_confusionM(X, Y)
```
**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 
* **Y: *array-like of shape (n_samples, )***  
Annotations of some specific annotator *t*, no label symbol *=-1*

**Returns**  
* **ann_prob_Y_Z: *array-like of shape (n_classes, n_classes)***  
The individual confusion matrix of some annotator *t*: <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(t)} = \sum_{m} \hat{\beta}_{k,j}^{(m)}\cdot  p(g=m|t)">

```python
get_predictions_groups(X, data=[])
```

**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 
* **data: *array-like of shape (n_samples, n_classes)***  
If the probability predictions of the ground truth over some set are delivered

**Returns**  
* **prob_Y_xg: *array-like of shape (n_samples, n_groups, n_classes)***  
The probability estimation of labels over some set for each group <img src="https://render.githubusercontent.com/render/math?math=p(y|x, g) = \sum_z p(y|z, g) p(z|x)">.

---
### Model and Annotators Group Inference based on EM - Confusion Matrix
```python
class codeE.methods.ModelInf_EM_CMOA(M, init_Z="softmv", n_init_Z=0, n_init_G=0, priors=0, DTYPE_OP='float32')
```
[UP](#crowdsourcing-methods)

This method infer a predictive model <img src="https://render.githubusercontent.com/render/math?math=f(x)"> of the ground truth jointly with the ground truth inference based on **groups over the annotations of the annotators**. Contrary to other methods, it does not have an explicit model per annotators. It represents the groups ability as a confusion matrix <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(m)} = p(y=j | z=k, g=m)"> and allows any model on *f()*.

It requieres a *group model* that assign annotators *a* to groups *g*:  <img src="https://render.githubusercontent.com/render/math?math=h(a)= p(g|a)">

The original C-MoA (*Crowd - Mixture of Annotators*) method was proposed by Mena et al. [2]
> It is proposed on the *individual sparse* representation. (for further details see [representation documentation](representation.md))

**Parameters**  
* **M: *int***  
The number of groups (*n_groups*) to be found (*different types of behaviors*) in the annotations.
* **init_Z: *string, {'softmv','hardmv'}, default='softmv'***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|x_i)">. Both posibilities are based on LabelAgg class. 
* **n_init_Z: *int, default=0***  
The number of epochs that the predictive model is going to be pre-trained.
* **n_init_G: *int, default=0***  
The number of epochs that the group model is going to be pre-trained.
* **priors: *different options***  
The priors to be set on the confusion matrices of groups, could be in different formats:
	* *string, {'laplace','none'}*  
	The 'laplace' stand for Laplace smoothing, a prior with the value of 1.
	* *int*  
	A number of annotations to be set prior for all the groups over all the data.
	* *array-like of shape (n_groups,)*  
	A vector of the number of annotations to be set prior for every possible group on the data.
	* *array-like of shape (n_groups, n_classes)*  
	A matrix of the number of annotations to be set priors for every group and every ground truth label on the data.
	* *array-like of shape (n_groups, n_classes, n_classes)*  
	A cube with the number of annotations to be set priors for every group, every ground truth label and every observed label on the data. 
	> Comments on the priors: The laplace smooth prior helps to stabilize traning and speeds up convergence. The disadvantage trade-off correspond to a slightly worse estimation of the ground truth.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html


##### References
[1] [Mena, F., & Ñanculef, R. (2019, October). *Revisiting Machine Learning from Crowds a Mixture Model for Grouping Annotations*](https://link.springer.com/chapter/10.1007/978-3-030-33904-3_46)  
[2] [Mena, F., Ñanculef, R., & Valles, C. (2020). *Collective Annotation Patterns in Learning from Crowds*]()

##### Examples
```python
X_train = ...
... #read some data 
from codeE.representation import set_representation
y_cat_var, A_idx_var = set_representation(y_obs,"onehotvar")
```
> Define predictive model (based on keras)
```python
F_model = Sequential()
... #add layers
```
> Define the group model (based on keras)
```python
T= np.concatenate(A_idx_var).max() +1
group_model = Sequential()
group_model.add(Embedding(T, 8, input_length=1, 
                         trainable=True,weights=[A_rep]))
group_model.add(Reshape([K]))
... #add dense (feed forward) layers
```
> Set the model and train the method
```python
from codeE.methods import ModelInf_EM_CMOA as CMOA
CMOA_model = CMOA(M=3) 
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
CMOA_model.set_model(F_model, ann_model=group_model, **args)
CMOA_model.fit(X_train, y_cat_var, A_idx_var)
```
> Train with different settings (**you must create the keras model again**)
```python
from codeE.methods import ModelInf_EM_CMOA as CMOA
CMOA_model = CMOA(M=3, init_Z='softmv', n_init_Z=0, n_init_G=0, priors=1) 
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
CMOA_model.set_model(F_model, ann_model=group_model, **args)
CMOA_model.fit(X_train, y_cat_var, A_idx_var, runs=20)
```
> Get the base model to predict the ground truth on some data
```python
cmoaK_fx = CMOA_model.get_basemodel()
cmoaK_fx.predict(new_X)
```
> Get the individual confusion matrices for every annotator
```python
A = np.unique(np.concatenate(A_idx_var)).reshape(-1,1)
prob_Yzt = CMOA_model.get_ann_confusionM(A)
```

##### Class Methods
|Function|Description|
|---|---|
|*get_basemodel*| Returns the predictive model of the ground truth|
|*get_groupmodel*| Returns the group model that assign annotators to group|
|*get_confusionM*| Returns the confusion matrix of each group|
|*get_qestimation*| Returns the estimation over auxiliar model *Q*|
|*set_model*| Set the predictive model (base model) and group model|
|*set_ann_model*| Set the group model|
|*set_priors*| To set the priors used on confusion matrices|
|*init_E*| Initialization of the E-step|
|*E_step*| Perform the inference on the E-step|
|*M_step*| Perform the inference on the M-step|
|*compute_logL*| Calculate the log-likelihood of the data|
|*train*| Perform all the inference based on EM algorithm|
|*multiples_run*| Perform multiple runs of the EM algorithm and save the best|
|*fit*| same as *multiples_run* |
|*get_global_confusionM*| Returns the global confusion matrix|
|*get_ann_confusionM*| Returns the individual confusion matrix estimation of annotators|
|*get_predictions_z*| Returns the probability predictions of ground truth over some set|
|*get_predictions_g*| Returns the probability predictions of the groups over the annotators|
|*get_predictions_groups*| Returns the probability estimation of labels over the modeled groups|

```python
get_basemodel()
```
**Returns**  
* **base_model: *function or class***  
The predictive model over the ground truth <img src="https://render.githubusercontent.com/render/math?math=f(\cdot)">

```python
get_groupmodel()
```
**Returns**  
* **group_model: *function or class***  
The group model that assigns the annotators to group <img src="https://render.githubusercontent.com/render/math?math=h(\cdot)">

```python
get_confusionM()
```
**Returns**  
* **betas: *array-like of shape (n_groups, n_classes, n_classes)***  
The confusion matrix of each modeled group <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j}^{(m)}">

```python
get_qestimation()
```
**Returns**  
* **Qil_mk: *(n_samples,) of arrays of shape (n_annotations(i), n_groups, n_classes)***  
The estimation over auxiliar model *Q*

```python
set_model(model, optimizer="adam", epochs=1, batch_size=32, ann_model=None)
```
Set the predictive model (base model) over the ground truth and define how to optimize it **on the M step** of the iterative EM algorithm. Besides set the group model that assign annotators to groups.

**Parameters**  
* **model: *function or class of keras model***  
Base model based on [Keras](https://keras.io/).
* **optimizer: *string, {'sgd','rmsprop','adam','adadelta','adagrad'}, default='adam'***  
String name of optimizer used on the back-propagation SGD, based on https://keras.io/api/optimizers/
* **epochs: *int, default=1***  
Number of epochs (iteration over the entire set) to train the model based on https://keras.io/api/models/model_training_apis/
* **batch_size: *int, default=32***  
Number of samples per gradient update, based on https://keras.io/api/models/model_training_apis/  
 **ann_model: *function or class of keras model***  
Group model based on [Keras](https://keras.io/).

```python
set_ann_model(model, optimizer=None, epochs=None)
```
Set the group model that assign annotators to groups.

**Parameters**  
* **model: *function or class of keras model***  
Group model based on [Keras](https://keras.io/).
* **optimizer: *string, {'sgd','rmsprop','adam','adadelta','adagrad'}, default=None***  
String name of optimizer used on the back-propagation SGD, based on https://keras.io/api/optimizers/, If *None* it uses the optimizer and epochs of the base model.
* **epochs: *int, default=None***  
Number of epochs (iteration over the entire set) to train the model based on https://keras.io/api/models/model_training_apis/, If *None* it uses the optimizer and epochs of the base model.

```python
set_priors(priors)
```
**Parameters**  
* **priors: *as state on init***  

```python
init_E(y_ann_var, A_idx_var, method="")
```
Initialization of the E-step, based on the following approximation:
<img src="https://render.githubusercontent.com/render/math?math=q_{i\ell}(g=m, z=k) = p(g=m, z=k | x_i, y_{i}^{(\ell)}, a_{i}^{(\ell)}) = p(g=m|a_i^{(\ell)}) p(z=k|x_i) ">. 
The groups *g* initialization is based on a K-means clustering and the ground truth *z* is based on *method*.

**Parameters**  
* **y_ann_var: *array-like of shape (n_samples,) of arrays of shape (n_annotations(i), n_classes)***  
Annotations of the data, should be on a categorical representation of variable length, from only anotators that annotate the data.
* **A_idx_var: *array-like of shape (n_samples,) of arrays of shape (n_annotations(i),)***  
Identifier of the annotator of each annotations in *y_ann_var*.
* **method: *string, {'softmv','hardmv',''}, default=''***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|x_i)">. Both posibilities are based on LabelAgg class. The empty string will use the method seted on init.


```python
E_step(X, y_ann_flatten, A_idx_flatten, predictions_Z=[], predictions_G=[])
```
Perform the inference on the E-step.

**Parameters**  
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **y_ann_flatten: *array-like of shape (n_annotations, n_classes)***  
Annotations of the data, the flatten format of the variable length categorical representation from *y_ann_var*.
* **A_idx_flatten: *array-like of shape (n_annotations, )***  
Identifier of the annotators on the data, the flatten format from *A_idx_var*.
* **predictions_Z: *array-like of shape (n_samples, n_classes)***  
Probability predictions of the ground truth on training set. If *X* is given, not necessary to give this parameter (*default=[]*).  
* **predictions_G: *array-like of shape (n_annotators, n_groups)***  
Probability predictions of the groups on the annotators. If *A_idx_flatten* is given, not necessary to give this parameter (*default=[]*).

```python
M_step(X, y_ann_flatten, A_idx_flatten)
```
Perform the inference on the M-step.

**Parameters**  
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **y_ann_flatten: *array-like of shape (n_annotations, n_classes)***  
Annotations of the data, the flatten format of the variable length categorical representation from *y_ann_var*.
* **A_idx_flatten: *array-like of shape (n_annotations, )***   
Identifier of the annotators on the data, the flatten format from *A_idx_var*.

```python
compute_logL()
```
Calculate the log-likelihood of the data.

```python
train(X_train, y_ann_var, A_idx_var, max_iter=50,relative=True,tolerance=3e-2)
```
Perform all the inference based on EM algorithm.

**Parameters** 
* **X_train: *array-like of shape (n_samples, ...)***  
Input patterns of the training data. 
* **y_ann_var: *array-like of shape (n_samples,) of arrays of shape (n_annotations(i), n_classes)***  
Annotations of the data, should be on a categorical representation of variable length, from only anotators that annotate the data.
* **A_idx_var: *array-like of shape (n_samples,) of arrays of shape (n_annotations(i),)***  
Identifier of the annotator of each annotations in *y_ann_var*.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.
* **relative: *boolean, default=True***  
If the difference for tolerance is relative to the previous value.

```python
multiples_run(Runs, X, y_ann_var, A_idx_var, max_iter=50, tolerance=3e-2)
```
Perform multiple runs of the EM algorithm and save the best execution based on log-likelihood.

**Parameters** 
* **Runs: *int***  
The number of times the EM will be run to obtain different results.
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of the training data. 
* **y_ann_var: *array-like of shape (n_samples,) of arrays of shape (n_annotations(i), n_classes)***  
Annotations of the data, should be on a categorical representation of variable length, from only anotators that annotate the data.
* **A_idx_var: *array-like of shape (n_samples,) of arrays of shape (n_annotations(i),)***  
Identifier of the annotator of each annotations in *y_ann_var*.
* **max_iter: *int, default=50***  
The maximum number of iterations to iterate between E and M.
* **tolerance: *float, default=3e-2***  
The maximus difference on the parameters and loss between  the iterations to train.

**Returns**  
* **found_logL: *list of length=Runs***  
A list with the history of log-likelihood for each iteration in the different runs
* **best_run: *int***  
The index of the best run between all executed, a number between *0* and *Runs-1*.

```python
fit(X,Y, runs = 1, max_iter=50, tolerance=3e-2)
```
same operation than *multiples_run* function.

```python
get_predictions_z(X)
```
**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 

**Returns**  
* **prob_Z_hat: *array-like of shape (n_samples, n_classes)***  
The probability predictions of the ground truth over some set <img src="https://render.githubusercontent.com/render/math?math=f(x) = p(z|x)">

```python
get_predictions_g(A)
```
**Parameters** 
* **A: *array-like of shape (n_annotators_pred, 1)***  
The identifier of *n_annotators_pred* annotators to assign groups.

**Returns**  
* **prob_G_hat: *array-like of shape (n_annotators_pred, n_groups)***  
The probability predictions over the groups of some annotators <img src="https://render.githubusercontent.com/render/math?math=h(a) = p(g|a)">

```python
get_global_confusionM(prob_Gt)
```
**Parameters** 
* **prob_Gt: *array-like of shape (n_annotators, n_groups)***  
Probabilities of the annotators over the groups.

**Returns**  
* **prob_Y_Z: *array-like of shape (n_classes, n_classes)***  
The global confusion matrix <img src="https://render.githubusercontent.com/render/math?math=\hat{\beta}_{k,j} =\sum_m \hat{\beta}_{k,j}^{(m)} \cdot p(g=m)">

```python
get_ann_confusionM(A)
```
**Parameters** 
* **A: *array-like of shape (n_annotators_pred, 1)***  
The identifier of *n_annotators_pred* annotators.

**Returns**  
* **prob_Y_Zt: *array-like of shape (n_annotators_pred, n_classes, n_classes)***  
The individual confusion matrices of *n_annotators_pred* annotators: <img src="https://render.githubusercontent.com/render/math?math=\{\beta}_{k,j}^{(t)} = \sum_m \hat{\beta}_{k,j}^{(m)}\cdot p(g=m|t)">

```python
get_predictions_groups(X, data=[])
```

**Parameters** 
* **X: *array-like of shape (n_samples, ...)***  
Input patterns of some data. 
* **data: *array-like of shape (n_samples, n_classes)***  
If the probability predictions of the ground truth over some set are delivered

**Returns**  
* **prob_Y_xg: *array-like of shape (n_samples, n_groups, n_classes)***  
The probability estimation of labels over some set for each group <img src="https://render.githubusercontent.com/render/math?math=p(y|x, g) = \sum_z p(y|z,g ) p(z|x)">.


---
### Model Inference based on EM - Label Noise
```python
class codeE.methods.ModelInf_EM_G(init_Z="softmv", n_init_Z=0, priors=0, DTYPE_OP='float32')
```
[UP](#crowdsourcing-methods)

global label noise

global learning from corews

learning from global annotations

**Parameters**  
* **init_Z: *string, {'softmv','hardmv'}, default='softmv'***  
The method used to initialize the ground truth probabilities on the EM step: <img src="https://render.githubusercontent.com/render/math?math=p(z=k|x_i)">. Both posibilities are based on LabelAgg class. 
* **n_init_Z: *int, default=0***  
The number of epochs that the predictive model is going to be pre-trained.
* **priors: *different options***  
The priors to be set on the confusion matrices of annotators, could be in different formats:
	* *string, {'laplace','none'}*  
	The 'laplace' stand for Laplace smoothing, a prior with the value of 1.
	* *int*  
	A number of annotations to be set prior for all the groups over all the data.
	* *array-like of shape (n_groups,)*  
	A vector of the number of annotations to be set prior for every possible group on the data.
	* *array-like of shape (n_groups, n_classes)*  
	A matrix of the number of annotations to be set priors for every group and every ground truth label on the data.
	* *array-like of shape (n_groups, n_classes, n_classes)*  
	A cube with the number of annotations to be set priors for every group, every ground truth label and every observed label on the data. 
	> Comments on the priors: The laplace smooth prior helps to stabilize traning and speeds up convergence. The disadvantage trade-off correspond to a slightly worse estimation of the ground truth.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html


##### References
[1] nada

##### Examples
```python
X_train = ...
... #read some data 
from codeE.representation import set_representation
r_obs = set_representation(y_obs, "global")
```
> Define predictive model (based on keras)
```python
F_model = Sequential()
... #add layers
```
> Set the model and train the method
```python
from codeE.methods import ModelInf_EM_G as G_Noise
GNoise_model = G_Noise() 
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
GNoise_model.set_model(F_model, **args)
GNoise_model.fit(X_train, r_obs)
```
> Train with different settings 
```python
from codeE.methods import ModelInf_EM_G as G_Noise
GNoise_model = G_Noise(init_Z='softmv', n_init_Z=3, priors=0) 
args = {'epochs':1, 'batch_size':BATCH_SIZE, 'optimizer':OPT}
GNoise_model.set_model(F_model, **args)
GNoise_model.fit(X_train, r_obs, runs=20)
```
> Get the base model to predict the ground truth on some data
```python
G_fx = GNoise_model.get_basemodel()
G_fx.predict(new_X)
```