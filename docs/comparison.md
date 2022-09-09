## Learning from Crowds Solutions
<img src="../imgs/Learning_from_crowds_final-2.jpg" width="50%" />

As this is an unsupervised problem of learning the *ground truth* based on the labeling process it can be solved with different approaches. Each method will try to model the annotation behavior in different ways and in different settings, providing different solutions for what is necessary.

---
### Differences in Methodology

Some notation comments:
> To more details in the problem notation see the [documentation](notation.md).
* *z* correspond to the *ground truth* of the data.
* *e* correspondn to the *reliability* of the annotators.
* *T* correspond the number of annotators `n_annotators`
* *K* correspond to the number of classes `n_classes`
* *M* correspond to the number of groups in some models: `n_groups`
* *W* correspond to the number of parameters of some predictive model
* *Wm* correspond to the number of parameters of the group model of *Model Inference EM - Groups* (gating network of MoE)

|Method name |Inferred variable |Predictive model |Setting |Annotator model | Other model | Learnable parameters |
|------------|------------------|-----------------|--------|----------------|---------------------|--------------------------|
|[Label Aggregation](./methods.md#simple-aggregation-techniques)|-|:x:|Global|-|-|0|
|[Label Inference EM](./methods.md#label-inference-based-on-em---confusion-matrix)|*z*|:x:|Individual dense| Probabilistic confusion matrix|Class marginals| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(TK^2 %2B K)"> |
|[Label Inference EM - Global](./methods.md#label-inference-based-on-em---label-noise)|*z*|:x:|Global| -| *global* Probabilistic confusion matrix| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(K^2 %2B K)"> |
|[Model Inference EM](./methods.md#model-inference-based-on-em---confusion-matrix)|*z*|:heavy_check_mark:|Individual dense| Probabilistic confusion matrix| - | <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(TK^2 %2B W)"> |
|[Model Inference EM - Groups](./methods.md#model-and-annotators-group-inference-based-on-em---confusion-matrix)|*z*|:heavy_check_mark:|Individual sparse| - | Probabilistic confusion matrix *per group*, gating network over groups| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(MK^2 %2B W %2B W_m)"> |
|[Model Inference EM - Groups Global](./methods.md#model-and-annotations-group-inference-based-on-em---confusion-matrix)|*z*|:heavy_check_mark:|Global| - | Probabilistic confusion matrix *per group*, group marginals| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(MK^2 %2B W %2B M)"> |
|[Model Inference EM - Global](./methods.md#model-inference-based-on-em---label-noise)|*z*|:heavy_check_mark:|Global| - | *global* Probabilistic confusion matrix| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(K^2 %2B W)"> |
|[Model Inference - Reliability EM](./methods.md#model-inference-based-on-em---reliability)|*e*|:heavy_check_mark:|Individual dense| Probabilistic reliability number|-| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(T %2B W)"> |
|[Model Inference BP](./methods.md#model-inference-based-on-bp---confusion-matrix)|-|:heavy_check_mark:|Individual dense (masked)| Confusion matrix weights|-| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(TK^2 %2B W)"> |
|[Model Inference BP - Global](./methods.md#model-inference-based-on-bp---label-noise)|-|:heavy_check_mark:|Global|-|*global* confusion matrix weights| <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(K^2 %2B W)"> |



#### Comments
* The inference of the methods with an explicit model per annotator depends on the participation of the annotators on the labelling process.
	* Large number of annotations
* An explicit model per annotator could take inference advantage when the individual behavior is quite different from each other.
	* While more complex model will overfit to the desired behavior modeling.
* The methods with predictive model could take inference advantage when the input patterns are more complex.
* The methods without two-step inference (based on backpropagation) could take advantage of a more stable learning.

---
### Usability

|Method name |Two-step inference|Predictive model |Setting |Computational scalability |Use case |
|---|---|------------------|---------------|--------------------------|---------|
|[Label Aggregation](./methods.md#simple-aggregation-techniques)|:x:|:x:|Global|All cases|High density per data|
|[Label Inference EM](./methods.md#label-inference-based-on-em---confusion-matrix)|:heavy_check_mark:|:x:|Individual dense|Not scalable with `n_annotators`|High density per annotator|
|[Label Inference EM - Global](./methods.md#label-inference-based-on-em---label-noise)|:heavy_check_mark:|:x:|Global|Very large `n_annotators`|High density|
|[Model Inference EM](./methods.md#model-inference-based-on-em---confusion-matrix)|:heavy_check_mark:|:heavy_check_mark:|Individual dense|Not scalable with `n_annotators`|High density per annotator|
|[Model Inference EM - Groups](./methods.md#model-and-annotators-group-inference-based-on-em---confusion-matrix)|:heavy_check_mark:|:heavy_check_mark:|Individual sparse|Very large `n_annotators`|High density per annotator|
|[Model Inference EM - Groups Global](./methods.md#model-and-annotations-group-inference-based-on-em---confusion-matrix)|:heavy_check_mark:|:heavy_check_mark:|Global|Very large `n_annotators`|High density per data|
|[Model Inference EM - Global](./methods.md#model-inference-based-on-em---label-noise)|:heavy_check_mark:|:heavy_check_mark:|Global|Very large `n_annotators`|High density|
|[Model Inference - Reliability EM](./methods.md#model-inference-based-on-em---reliability)|:heavy_check_mark:|:heavy_check_mark:|Individual dense|Large `n_annotators`|High density per annotator|
|[Model Inference BP](./methods.md#model-inference-based-on-bp---confusion-matrix)|:x:|:heavy_check_mark:|Individual dense (masked)| Not scalable with `n_annotators`|High density per annotator|
|[Model Inference BP - Global](./methods.md#model-inference-based-on-bp---label-noise)|:x:|:heavy_check_mark:|Global|Very large `n_annotators`|High density per data|

**Use case** indicates that, the closer the method is to that setting, a better inference is performed. The **density** refers to the number of annotations per annotator/data/globally.

#### Comments
* The methods without a predictive model are independent of the choice of the learning model, only learns from labels.
	* On a second phase these methods could learn *f(x)* over the inferred *ground truth*.
* The methods with a predictive model depend on the chosen learning model.
	* Being able to take advantage of when the input patterns are more complex.
* The global methods could be set on the individual setting by changing the representation from individual to global (*not vice versa*).
* The methods without two-step inference are independent of the inference algorithm, where the learning is based in a single optimization framework.


#### Experimental details on the computational scalability can be found on [Scalability Comparison](../Scalability%20Comparison.ipynb)
