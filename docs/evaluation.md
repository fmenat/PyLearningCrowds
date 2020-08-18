## Evaluation functions
```python
from codeE.evaluation import ...
```
Function to use in evaluation on crowdsourcing scenario

---
### Accuracy by model
```python
codeE.evaluation.accuracy_model(model, X_data, Z_data)
```

Evaluate a predictive model over some set X and ground truth Z based on **Accuracy**.
$$ formula $$

**Parameters**  
* **model: *class of model with predictive function 'predict'***  
The predictive model of the ground truth
* **X_data: *array-like of shape (n_samples, ...)***  
The input patterns, the points is for the different shapes of data representation, images, text, audio, etc.
* **Z_data: *array-like of shape or (n_samples,)***  
The ground truth of the data in class format.

**Returns**  
* **acc: *float***  
The accuracy on the set.

##### Examples 
```python
import numpy as np
N = 100
K = 8
X = np.random.rand(N,5)
Z = np.random.randint(K, size=(N,))
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X,Z)
from codeE.evaluation import accuracy_model
accuracy_model(model, X, Z, mode="weighted")
```

---
### F1-score by model
```python
codeE.evaluation.f1score_model(model, X_data, Z_data, mode='macro')
```

Evaluate a predictive model over some set X and ground truth Z based on **F1-score**.
$$ formula $$

**Parameters**  
* **model: *class of model with predictive function 'predict'***  
The predictive model of the ground truth
* **X_data: *array-like of shape (n_samples, ...)***  
The input patterns, the points is for the different shapes of data representation, images, text, audio, etc.
* **Z_data: *array-like of shape or (n_samples,)***  
The ground truth of the data in class format.
* **mode: *string, {'micro','macro','weighted'}, default='macro'***  
The average done over the f1 score, based on scikit-learn, further details in https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

**Returns**  
* **acc: *float***  
The accuracy on the set.

##### Examples
```python
import numpy as np
N = 100
K = 8
X = np.random.rand(N,5)
Z = np.random.randint(K, size=(N,))
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X,Z)
from codeE.evaluation import f1score_model
f1score_model(model, X, Z, mode="macro")
```

---
### Error on confusion matrix estimation by JS
```python
codeE.evaluation.D_JS(conf_true, conf_pred, raw=False)
```
Evaluate one confusion matrix estimation based on **Jensen-Shannon divergence** between the rows.
$$ formula $$

**Parameters**  
* **conf_true: *array-like of shape (n_classes, n_classes)***  
Real confusion matrix, $\beta_{k,j} = p(y=j | z=k)$
* **conf_pred: *array-like of shape (n_classes, n_classes)***  
Estimated confusion matrix, $\hat{\beta}_{k,j} = \hat{p}(y=j | z=k)$
* **raw: *bolean, default=False***  
If the error is returned per row (*True*) or global as a mean between the rows (*False*)

**Returns**  
* **d_js: *float***  
The d_js on the estimation.


---
### Error on confusion matrix estimation by NormF
```python
codeE.evaluation.D_NormF(conf_true, conf_pred)
```

Evaluate one confusion matrix estimation based on **normalized Frobenius** between the rows.
$$ formula $$

**Parameters**  
* **conf_true: *array-like of shape (n_classes, n_classes)***  
Real confusion matrix, $\beta_{k,j} = p(y=j | z=k)$
* **conf_pred: *array-like of shape (n_classes, n_classes)***  
Estimated confusion matrix, $\hat{\beta}_{k,j} = \hat{p}(y=j | z=k)$

**Returns**  
* **normF: *float***  
The normF on the estimation.

##### Examples of confusion matrix estimation
```python
import numpy as np
N = 100 #data
K = 8 #classes
Z = np.random.randint(K, size=(N,))
R = np.random.randint(3, size=(N,K))
from codeE.utils import generate_Global_conf
B = generate_Global_conf(Z, R)
B_hat = B + 1e-7
```
> Evaluation:
```python
from codeE.evaluation import D_KL, D_JS, D_NormF
print("D_KL = ",D_KL(B, B_hat))
print("D_JS = ",D_JS(B, B_hat))
print("D_NormF = ",D_NormF(B, B_hat))
```

---
### Error on set of confusion matrices estimation
```python
codeE.evaluation.Individual_D(confs_true, confs_pred, D)
```

Evaluate a set of confusion matrix estimation based on *D*.
$$ formula $$

**Parameters**  
* **confs_true: *array-like of shape (n_annotators, n_classes, n_classes)***  
Real set of confusion matrix, for example individual matrices $\beta_{k,j}^{(t)} = p(y=j | z=k, a=t)$
* **confs_pred: *array-like of shape (n_annotators, n_classes, n_classes)***  
Estimated set of confusion matrix, for example individual matrices $\hat{\beta}_{k,j}^{(t)} = \hat{p}(y=j | z=k, a=t)$
* **D: function, {D_KL,D_JS, D_NormF}**  
Function to measure error between two array-like confusion matrices.

**Returns**  
* **res: *float***  
The error result on the estimation.

##### Examples of confusion matrix estimation
```python
import numpy as np
N = 100 #data
K = 8 #classes
T = 10 # annotators
Z = np.random.randint(K, size=(N,))
Y = np.random.randint(K, size=(N,T))
Y_ohv = keras.utils.to_categorical(Y)
from codeE.utils import generate_Individual_conf
B_ind = generate_Individual_conf(Z, Y_ohv)
B_ind_hat = B_ind +1e-7
```
> Evaluation
```python
from codeE.evaluation import Individual_D, D_JS, D_NormF
print("Individual D_JS = ",Individual_D(B_ind, B_ind_hat, D=D_JS))
print("Individual D_NormF = ",Individual_D(B_ind, B_ind_hat, D=D_NormF))
```

para describir anotadoras...