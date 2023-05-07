# Fine Grained Image Classification : A study using CNN architectures and Transformers
## Datasets
We have used the following four fine grained image datasets:
1. CUB-200
2. FGVC-Aircraft
3. Stanford Dogs
4. FoodX-251

 ## Models
We have fine tuned the following pre-trained models on our dataset.
1. ConvNext - Tiny, Base & Large variants
2. Vision Transformer - Base & Large variants
3. Data efficient image Transformer
4. Swin Transformer - Base & Large variants
 
The configuration parameters for the models are given in ./utils/configs.py.

 ## Knowledge Distillation
We adopt the approach of knowledge distillation to improve the performance of student model by transferring the knowledge from teacher model. The architecture of knowledge distillation is shown as follows.
![Knowledge-Distillation_3](https://user-images.githubusercontent.com/113207800/236699344-a24d8353-7240-4ebe-a9fb-d62f006b2b60.png)


## Requirements 
We have tested this code on Ubuntu 20.04 LTS with Python 3.8. The other requirements are <br>
 - PyTorch
 - torchvision
 
## Training and Evaluation
<!--The repository provides code for both fine-tuning and knowledge distillation:-->
To train and evaluate the code in a single run, execute 
`python models/train_dataset.py` <br> 
For example, to work with FGVC-Aircraft classification execute `python models/train_aircraft.py`


<!--### Fine-Tuning
The fine-tuning code allows for the training of the convolutional neural networks (ConvNets), Vision Transformers (ViT), and Swin Transformers on the specified dataset.

### Kfold Cross Validation
The Kfold.py file provides code for running k-fold cross validation on the dataset to evaluate model performance. The code allows for specifying the number of epochs and the optimizer used for training


### Knowledge Distillation
The knowledge distillation code allows for the transfer of knowledge from a pre-trained teacher model to a smaller student model, with the goal of improving the performance of the student model on Stanford Dogs and FoodX-251.

### Data Augmentation
The models are trained with data augmentation techniques to improve their generalization performance. The following transformations are applied to the images during training:

1. Random horizontal flip
2. Random rotation
3. Random autocontrast
4. Resize to 256 x 256
5. Center crop to 224 x 224
6. To tensor
7. Normalize with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225)-->


### Acknowledgement
This code repo is forked and modified from <!--the official [ConvNext repositry](https://github.com/facebookresearch/ConvNeXt) and -->[Training data-efficient image transformers & distillation through attention](https://github.com/facebookresearch/deit)

## References
1. Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
2. Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J (2021). Training data-efficient image transformers & distillation through attention. In International conference on machine learning (pp. 10347--10357).
