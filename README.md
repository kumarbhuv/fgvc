# Fine Grained Image Classification
Fine grained vision classification (Convnext, Vit and Swin) on Cub-200, fgvc-aircraft, stanford-dogs and foodx-251

## Datasets
This study involves the four fine grained image datasets
1. CUB-200: Caltech-UCSD Birds-200-2011 dataset
2. FGVC-Aircraft: FGVCAircraft dataset
3. Stanford Dogs: Stanford Dogs dataset
4. FoodX-251: FoodX-251 dataset


 ## Models
The repository provides pre-trained models for convolutional neural networks (ConvNets), Vision Transformers (ViT), and Swin Transformers trained on the specified datasets. The gvc/models/model.py file provides the implementation of the models, and the gvc/utils/configs.py file contains the configuration parameters for each model.

### ConvNets
The following ConvNets are provided:

convnext_tiny: a tiny ConvNet architecture for CUB-200, with 768 output features in the classifier.
convnext_base: a larger ConvNet architecture for CUB-200, with 1024 output features in the classifier.
convnext_large: a very large ConvNet architecture for CUB-200, with 1536 output features in the classifier.

### Vision Transformers (ViT)
The following ViT models are provided:

vit_b_16: a ViT base architecture for CUB-200, with 768 output features in the head.
vit_l_16: a ViT large architecture for CUB-200, with 1024 output features in the head.

### Swin Transformers
The following Swin Transformers are provided:

swin_t: a Swin Transformer architecture for CUB-200, with 768 or 1024 output features in the head.


## Training and Evaluation
The repository provides code for both fine-tuning and knowledge distillation:

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


##Citation
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
