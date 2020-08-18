## State of the art methods in crowdsourcing
```python
from codeE.methods import ...
```
Methods proposed in the crowdsourcing scenario, also called learning from crowds.

---
### Simple aggregation techniques
```python
class codeE.methods.LabelAggregation(scenario="global")
```

Muy breve descripcion de estas tecncias en ingles..

**Parameters**  
* **scenario: *string, {'global','individual'}, default='global'***  
The scenario in which the annotations will be aggregated. Subject to the representation format, for further details see the Representation documentation.

**Attributes**
* scenario..

##### Examples
```python
... #read some data
from codeE.representation import set_representation
y_obs_categorical = set_representation(y_obs,'onehot') 
r_obs = set_representation(y_obs,"global")
```
> Infer over global scenario
```python
from codeE.methods import LabelAggregation
label_A = LabelAggregation(scenario="global")
mv_soft = label_A.infer(  r_obs, 'softMV')
mv_hard = label_A.predict(r_obs, 'hardMV')
```
> Infer over individual scenario
```python
from codeE.baseline import LabelAggregation
label_A = LabelAggregation(scenario="individual")
mv_soft = label_A.infer(  y_obs_categorical, 'softMV')
mv_hard = label_A.predict(y_obs_categorical, 'hardMV')
```
> Weighted
```python
from codeE.baseline import LabelAggregation
label_A = LabelAggregation(scenario="individual")
T_weights = np.sum(y_obs_categorical.sum(axis=-1) == 0, axis=0) #number of annotations given per annotator
Wmv_soft = label_A.infer(y_obs_categorical, 'softMV', weights=T_weights)
Wmv_soft
```

##### Class Methods u otro nombre
|Function|Description
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
The method used to infer the ground truth of the data based on the most used aggregation techniques: Majority Voting (MV).
> formulas de cada uno

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
### Raykar et al. 2010 - title? o nombre. mmm
```python
class codeE.methods.nombre
```