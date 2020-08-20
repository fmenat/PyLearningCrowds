## Representation functions
```python
from codeE.representation import ...
```
Function to change representation in the crowdsourcing scenario

##### Examples
```python
... #read some data 
y_obs = ... #classes of shape of (n_samples, n_annotators) not annotation symbol =-1
```
> Change representation to global
```python
from codeE.representation import set_representation
r_obs = set_representation(y_obs,"global")
print("Global representation shape (N,K)= ",r_obs.shape)
```
> Change representation to individual - dense
```python
from codeE.representation import set_representation
y_obs_categorical = set_representation(y_obs, 'onehot') 
print("Individual representation shape (N,T,K)= ",y_obs_categorical.shape)
```
> Change representation to individual - sparse
```python
from codeE.representation import set_representation
y_cat_var, A_idx_var = set_representation(y_obs, 'onehotvar') 
print("Individual sparse representation, variable shape (N,)= ",y_cat_var.shape)
print("one-hot vectors of K-dimensions, K=",y_cat_var[0].shape[1])
```

---
### Transform representation 
```python
codeE.representation.set_representation(y_obs, needed="onehot")
```
Change representation on crowdsourcing scenario.

**Parameters**  
* **y_obs: *array-like of shape (n_samples, n_annotators)***  
The multiple annotations observed by the data by every annotators, the class gives by each annotators, if annotator does not annotates there must be a "-1" in the representation. 
* **needed: *{'onehot','global','onehotvar','variable'}, default="onehot"***  
The representation to be seted

> 'onehot': categorical representation of the annotations by every annotator

> 'global': number of annotations gives per every label between all the annotators  

> 'onehotvar' or 'variable': categorical representation of variable length, from only anotators that annotate the data.

**Returns**  
* **new_annotations: *array-like of shape***  
The seted representation.

> 'onehot': *(n_samples, n_annotators, n_classes)*  

> 'global': *(n_samples, n_classes)*  

> 'onehotvar' or 'variable': *(n_samples,) of arrays of shape (n_annotations(i), n_classes)*
> * **identity_annotations: *same shape as new_annotations***  
 The identifier of each annotator is returned as well, $(\mathcal{L}_i, \mathcal{A}_i)$


---
### Transform function auxiliar
```python
codeE.representation.list_to_global(list_ann, K)
```

Change representation from list of variable annotations to global array representation.

**Parameters**  
* **list_ann: *list-like of len n_samples***  
The multiple annotations observed by the data in a list of variable number of annotations.
* **K: *int, n_classes of the data***  
The number of the classes of the data

**Returns**  
* **r_obs: *array-like of shape (n_samples, n_classes)***  
The global representation.