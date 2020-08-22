## Utils functions
```python
from codeE.utils import ...
```
Function to use in the crowdsourcing scenario

-----
### Get Individual matrices
```python
codeE.utils.generate_Individual_conf(Z_data, annotations, DTYPE_OP='float32')
```
To generate the individual confusion matrix of multiple annotators 
<img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(t)} = p(y=j | z=k, a=t)">

**Parameters**  
* **Z_data: *array-like of shape (n_samples, n_classes)***  
The ground truth of the data in a one-hot vector format.
* **annotations: *array-like of shape (n_samples, n_annotators, n_classes)***  
The multiple annotations observed by the data in the individual categorical (onehot) format, if an annotator does not annotate it has a one-hot of zeros.
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html

**Returns**  
* **I_conf: *array-like of shape (n_annotators, n_classes, n_classes)***  
The <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(t)}">

##### Examples
```python
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
```python
codeE.utils.generate_Global_conf(Z_data, annotations, DTYPE_OP='float32')
```

To generate the global confusion matrix of the multiple annotations <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j} = p(y=j | z=k)">

**Parameters**  
* **Z_data: *array-like of shape (n_samples, n_classes) or (n_samples,)***  
The ground truth of the data, could be in both format, classes or one-hot vector.
* **annotations: *array-like of shape (n_samples, n_annotators, n_classes) or (n_samples, n_classes)***  
The annotations observed by the data, could be in both format: individual (categorical one-hot) or global. For further details see [representation documentation](representation.md).
* **DTYPE_OP: *string, default='float32'***  
dtype of numpy array, restricted to https://numpy.org/devdocs/user/basics.types.html

**Returns**  
* **G_conf: *array-like of shape (n_classes, n_classes)***  
The <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}">


##### Examples
```python
import numpy as np
N = 100 #data
K = 8 #classes
Z = np.random.randint(K, size=(N,))
R = np.random.randint(3, size=(N,K))
from codeE.utils import generate_Global_conf
generate_Global_conf(Z, R)
```
> To visualize you can use
```python
from codeE.utils import plot_confusion_matrix
plot_confusion_matrix(generate_Global_conf(Z, R), title= "Global Matrix")
```

---
```python
codeE.utils.get_confusionM(*args)
```
Perform same operation that *generate_Global_conf*

-----
### Pre-train neural network
```python
codeE.utils.pre_init_F(model, X_inp, Z_targ, n_init, batch_size=32)
```
Train the neural net model and reset the optimizer, as a pre-train step.

**Parameters**  
* **model: *function or class of keras model***  
Predictive model based on [Keras](https://keras.io/).
* **X_inp: *array-like of shape (n_samples, ...)***  
Input patterns of the data.
* **Z_targ: *array-like of shape (n_samples, n_classes)***  
The estimation of the ground truth to pre-train the model.
* **n_init_Z: *int, default=0***  
The number of epochs that the predictive model is going to be pre-trained.
* **batch_size: *int, default=32***  
Number of samples per gradient update, based on https://keras.io/api/models/model_training_apis/

##### Examples
```python
... #read some data 
X_data = ...
Z_hat = ...
```
> Define predictive model (based on keras)
```python
model_B = Sequential()
... #add layers
```
> Use it
```python
from codeE.utils import pre_init_F
pre_init_F(model_B, X_data, Z_hat, n_init=3)
```

# PENDIENTE


---
### Cluster Annotations
```python
codeE.utils.clusterize_annotators(y_o,M,no_label=-1,bulk=True,cluster_type='mv_close',data=[],model=None,DTYPE_OP='float32',BATCH_SIZE=64,option="hard",l=0.005)
```

To clusterize on crowdsourcing, as initial step of groups behavior *p(g)*.


---
### Visual comparison of confusion matrix
```python
codeE.utils.compare_conf_ma(pred_conf_mat, true_conf_mat=[], text=False):
```

To compare a predicted confusion matrix against the true values

---
### Visual comparison of set confusion matrices
```python
codeE.utils.compare_set_conf_ma(set_conf_ma, true_set_conf_ma = [], text=True, n_samp=0):
```

To compare a set of predicted confusion matrix against the set of true values

