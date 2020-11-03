# PyLearningCrowds
<img src="https://marketingland.com/wp-content/ml-loads/2016/01/DataPeople_1200.png" width="50%" />

Learning from crowds methods implemented in Python. The available methods:
* **Majority Voting**: soft, hard, weighted
* **Dawid and Skene**: ground truth (GT) inference based on confusion matrices (CM) of annotators.
* **Raykar et al**: predictive model over GT inference based on CM of annotators
* **Mixture Models**: inference of model and groups on annotations of the data or the annotators
* **Global Behavior**: based on label noise solutions, a global confusion matrix to infer a predictive model.
	* **Without predictive model**: As *Dawid and Skene*, infers only the GT based on a global confusion matrix.
* **Rodrigues et al (2013)**: predictive model over GT inference based on annotators reliability.


For examples of how to use the methods see the notebooks **Tutorials** on:
* [LabelMe](./Tutorial%20-%20LabelMe.ipynb): a real image dataset
* [Sentiment](./Tutorial%20-%20Sentiment.ipynb): a real text dataset
* [Synthetic](./Tutorial%20-%20Synthetic.ipynb): a synthetic dataset
* [Scalability Comparison](./Scalability%20Comparison.ipynb): over the  synthetic dataset


---
### Documentation
* [Comparison](./docs/comparison.md)
* [Notation](./docs/notation.md)
* CODE:
	* [Evaluation](./docs/evaluation.md)
	* [Generate](./docs/generate_data.md)
	* [Methods](./docs/methods.md)
	* [Representation](./docs/representation.md)
	* [Utils](./docs/utils.md)


---
#### Example
> Read some dataset annotations
```python
import numpy as np
y_obs = np.loadtxt("./data/LabelMe/answers.txt",dtype='int16') #not annotation symbol ==-1
T_weights = np.sum(y_obs != -1,axis=0) #number of annotations per annotator
print("Remove %d annotators that do not annotate on this set "%(np.sum(T_weights==0)))
y_obs = y_obs[:,T_weights!=0]
print("Shape (n_samples,n_annotators)= ",y_obs.shape)
```
For further details on representation see the [documentation](./docs/representation.md)
> You can estimate the ground truth with some aggregation technique: Majority Voting (MV)
```python
from codeE.representation import set_representation
r_obs = set_representation(y_obs,"global")
print("Global representation shape (n_samples, n_classes)= ",r_obs.shape)
from codeE.methods import LabelAgg
label_A = LabelAgg(scenario="global")
mv_soft = label_A.predict(r_obs, 'softMV')
mv_hard = label_A.predict(r_obs, 'hardMV')
```
> Read the dataset input patterns
```python
X_train = ... 
```
> Define a predictive model over the ground truth
```python
model_z = ... 
```
>> You can infer a predictive model with the ground truth
```python
from codeE.representation import set_representation
y_obs_categorical = set_representation(y_obs,'onehot') 
print("Individual representation shape (N,T,K)= ",y_obs_categorical.shape)
from codeE.methods import ModelInf_EM as Raykar
R_model = Raykar()
R_model.set_model(model_z)
R_model.fit(X_train, y_obs_categorical, runs=20)
raykar_fx = R_model.get_basemodel()
raykar_fx.predict(new_X)
```
>> You can infer the predictive model and groups of behaviors
```python
from codeE.methods import ModelInf_EM_CMM as CMM
CMM_model = CMM(M=3) 
CMM_model.set_model(model_z)
CMM_model.fit(X_train, r_obs, runs =20)
cmm_fx = CMM_model.get_basemodel()
cmm_fx.predict(new_X)
```

For the other available methods see the [methods documentation](./docs/methods.md)
---

#### News
* Base predictive model support Logistic Regression on sklearn 
> Only with **one run** in the configuration of the methods
>> Example 
```python
from sklearn.linear_model import LogisticRegression as LR 
model_sklearn_A = LR(C= 1, multi_class="multinomial")
from codeE.methods import ModelInf_EM as Raykar
R_model = Raykar(init_Z="softmv")
args = {'epochs':1, 'optimizer': "newton-cg", 'lib_model': "sklearn"}
R_model.set_model(model_sklearn_A, **args)
R_model.fit(Xstd_train, y_obs_categorical, runs=1)
```

#### Extensions
* Rodrigues et al. 2018
* Label noise without EM
* Guan et al. 2018 (models with label aggregation)
* Kajino et al. 2012 (models with model aggregation)
* Fast estimation, based on hard or discrete, on other methods besides DS