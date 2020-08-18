## Utils functions
```python
from codeE.utils import ...
```
Function to use in the crowdsourcing scenario

-----
### Get Individual matrices
> ```python
codeE.utils.generate_Individual_conf(Z_data, annotations, DTYPE_OP='float32')
```
To generate the individual confusion matrix of multiple annotators
$$ \beta_{k,j}^{(t)} = p(y=j | z=k, a=t) $$


**Parameters**  
* **Z_data: *array-like of shape (n_samples, n_classes)***  
The ground truth of the data in a one-hot vector format.
* **annotations: *array-like of shape (n_samples, n_annotators, n_classes)***  
The multiple annotations observed by the data in the individual categorical (onehot) format, if an annotator does not annotate it has a one-hot of zeros.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html

**Returns**  
* **I_conf: *array-like of shape (n_annotators, n_classes, n_classes)***  
The $ \beta_{k,j}^{(t)}$

##### Examples
> ```python
import numpy as np
import keras
N = 100 #data
K = 8 #classes
T = 10 # annotators
Z = np.random.randint(K, size=(N,))
Y = np.random.randint(K, size=(N,T))
Y_ohv = keras.utils.to_categorical(Y)
from codeE.utils import generate_Individual_conf
generate_Individual_conf(Z, Y_ohv)
```

---
### Get Global matrix
> ```python
codeE.utils.generate_Global_conf(Z_data, annotations, DTYPE_OP='float32')
```

To generate the global confusion matrix of the multiple annotations
$$ \beta_{k,j} = p(y=j | z=k) $$

**Parameters**  
* **Z_data: *array-like of shape (n_samples, n_classes) or (n_samples,)***  
The ground truth of the data, could be in both format, classes or one-hot vector.
* **annotations: *array-like of shape (n_samples, n_annotators, n_classes) or (n_samples, n_classes)***  
The annotations observed by the data, could be in both format: individual (categorical one-hot) or global. For further details see documentation of representation.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html

**Returns**  
* **G_conf: *array-like of shape (n_classes, n_classes)***  
The $ \beta_{k,j}$


##### Examples
> ```python
import numpy as np
N = 100 #data
K = 8 #classes
Z = np.random.randint(K, size=(N,))
R = np.random.randint(3, size=(N,K))
from codeE.utils import generate_Global_conf
generate_Global_conf(Z, R)
```
to visualize you can use
```python
from codeE.utils import plot_confusion_matrix
plot_confusion_matrix(generate_Global_conf(Z, R), title= "Global Matrix")
```

---
```python
codeE.utils.get_confusionM(*args)
```
Perform same operation that *generate_Global_conf*
