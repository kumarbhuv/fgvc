# Fine Grained Image Classification using knowledge distillation
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


## Training and Evaluation
<!--The repository provides code for both fine-tuning and knowledge distillation:
For training, execute 
'python models/train_dataset.py'
For evaluation, execute
'python models/Kfold.py

### Fine-Tuning
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
7. Normalize with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225)


## Requirements
 - Python 3
 - PyTorch
 - torchvision


## Citation
@techreport{maji13fine-grained,
   title         = {Fine-Grained Visual Classification of Aircraft},
   author        = {S. Maji and J. Kannala and E. Rahtu
                    and M. Blaschko and A. Vedaldi},
   year          = {2013},
   archivePrefix = {arXiv},
   eprint        = {1306.5151},
   primaryClass  = "cs-cv",
}

@techreport{WahCUB_200_2011,
	Title = ,
	Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
	Year = {2011}
	Institution = {California Institute of Technology},
	Number = {CNS-TR-2011-001}
}
