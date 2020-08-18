## Representation functions
```python
from codeE.representation import ...
```
Function to change representation in the crowdsourcing scenario

---
### Transform representation 
> ```python
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

> 'onehotvar' or 'variable': *(n_samples,) of array-like of shape (n_annotations(i),n_classes)*
* Besides an array of same dimensions with the identifier of each annotator


---
### Transform function auxiliar
> ```python
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


SET Option1: A_train, A = get_A_il(X, T=T) 
SET Option2: A_train, A = get_A_il(X, A=A) #if you already  had a representation for annotators
