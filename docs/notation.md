## Problem Notation


### Supervised Scenario
* Consider an input pattern <img src="https://render.githubusercontent.com/render/math?math=x \in \mathbb{X}"> observed with probability distribution <img src="https://render.githubusercontent.com/render/math?math=p(x)"> and a ground-truth label <img src="https://render.githubusercontent.com/render/math?math=z \in \mathbb{Z}"> observed with conditional probability distribution <img src="https://render.githubusercontent.com/render/math?math=p(z|x)">.
* Given a finite sample <img src="https://render.githubusercontent.com/render/math?math=S=\{\left(x_{i},z_{i}\right)\}_{i=1}^N">, where <img src="https://render.githubusercontent.com/render/math?math=\left(x_{i},z_{i}\right) \sim p(x,z)=p(z|x)p(x) \, \ \forall i \in [N]">. 
* Objective: estimate a predictive model <img src="https://render.githubusercontent.com/render/math?math=f(x)"> that maps <img src="https://render.githubusercontent.com/render/math?math=x \rightarrow z"> or learn statistics of <img src="https://render.githubusercontent.com/render/math?math=p(z|x)">.

<img src="https://miro.medium.com/max/1204/0*qf-O7Jm1mmZrXYqA" width="50%" />
    


### Crowdsourcing scenario
* Same objective that supervised scenario, but the ground-truth labels <img src="https://render.githubusercontent.com/render/math?math=z_{i}"> corresponding to the input patterns <img src="https://render.githubusercontent.com/render/math?math=x_{i}"> are not directly observed. 
* Consider labels <img src="https://render.githubusercontent.com/render/math?math=y \in \mathbb{Z}"> that do not follow the ground-truth distribution <img src="https://render.githubusercontent.com/render/math?math=p(z|x)">. Instead, they are generated from an unknown process <img src="https://render.githubusercontent.com/render/math?math=p(y^{(\ell)}|x,z)"> that represents the <img src="https://render.githubusercontent.com/render/math?math=\ell "> annotator **ability** to detect the ground truth.

> #### Individual
* Consider multiple noise labels <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_i = \{y_i^{(1)},\ldots, y_i^{(T_i)}\}"> given by <img src="https://render.githubusercontent.com/render/math?math=T_i"> annotators.
* These annotations come from a subset <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}_i"> of the set of all the annotators <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A} "> participating in the labelling process. (<img src="https://render.githubusercontent.com/render/math?math=T = |\mathcal{A}|"> )
* The annotator identity could be define as a input variable: <img src="https://render.githubusercontent.com/render/math?math=a_{i}^{(\ell)} \in \mathcal{A}">, with <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A} = \{ 1, \ldots, T\}"> 
    * Then <img src="https://render.githubusercontent.com/render/math?math=p(y^{(\ell)}|x,z)=p(y|x,z, a=\ell)">
* Given a sample <img src="https://render.githubusercontent.com/render/math?math=\{(x_i, \mathcal{L}_i )\}_{i=1}^N$ or $\{(x_i, (\mathcal{L}_i, \mathcal{A}_i) )\}_{i=1}^N">

> #### Global
* Consider that we do not known or do not care which annotators provided the labels: we know <img src="https://render.githubusercontent.com/render/math?math=|\mathcal{A}_i|"> but not <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}_i">
* Consider the number of times that all the annotators gives each possible labels: <img src="https://render.githubusercontent.com/render/math?math=r_{ij} \in \{0,1,\ldots,T_i\}">
* Given a sample <img src="https://render.githubusercontent.com/render/math?math=\{ (x_i,r_i) \}_{i=1}^N">.

<img src="https://minutes.co/wp-content/uploads/2019/08/crowdsourcing.jpg" width="50%" />



#### Focus
In this implementation, we study the pattern recognition case, that is, we let <img src="https://render.githubusercontent.com/render/math?math=\mathbb{Z}"> be a small set of *K* categories or classes <img src="https://render.githubusercontent.com/render/math?math=\{c_1,c_2,\ldots,c_K\}">.

---

One also can define two scenarios based on the annotation density and assumptions:
* **Dense**: 
    * All the annotators labels each data: <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}_i = \mathcal{A}">
    * The implementation is simpler since fixed size matrices are assumed.
* **Sparse**: 
    * The number of labels collected by data point and annotator varies: <img src="https://render.githubusercontent.com/render/math?math=|\mathcal{A}_i| \neq |\mathcal{A}_j| < |\mathcal{A}| = T">
    * An appropiate implementation lead to computational efficiency.


---
#### Confusion Matrices

* Individual confusion matrix (for an annotator *t*):  
>> <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j}^{(t)} = p(y=j | z=k, a=t)">
* Global confusion matrix (for all the annotations): 
>> <img src="https://render.githubusercontent.com/render/math?math=\beta_{k,j} = p(y=j | z=k)">