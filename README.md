# Fine Grained Image Classification : A study using CNN architectures and Transformers
Fine grained Image Classification (FGIC) is one of the challenging tasks in Computer Vision. Many recent methodologies including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have tried to solve this problem. In this study, we introduce a teacher-student strategy specific to ConvNext model and show the effectiveness of using CNNs and transformers combined together
to produce state of the art results on challenging FGIC datasets. We show that by using ConvNext-base as a student model and swin-base transformer as teacher model in knowledge distillation settings, we achieve the highest accuracy of 86.9% on FGVC-Aircraft dataset and the least accuracy of 81.37% on a more challenging dataset named FoodX-251.

 ## Datasets
The instructions to download and install FGVC datasets is given below:-

<b> CUB Dataset </b>

Caltech-UCSD Birds (CUB) dataset is provided by Caltech Vision Lab. The dataset can be downloaded from this [link](https://www.vision.caltech.edu/datasets/cub_200_2011/).
The CUB-200 dataset consists of 11,788 images of 200 bird species, with each species having between 30 to 60 images.
The dataset folder should have the following structure:

```
CUB_dataset_root_folder/
    └─ images
    └─ image_class_labels.txt
    └─ train_test_split.txt
    └─ ....
```
<b> FGVC-Aircraft Dataset </b>

The FGVC-Aircraft-2013 dataset contains 10,000 images of airplanes, with each category having between 80 to 100 images. The Dataset is provided by Oxford Univeristy under the [link](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/). 

The dataset folder should have the following structure:

```
aircraft_dataset_root_folder/
    └─ data
        └─ images
            ├─ 0034309.jpg
            ├─ 0034958.jpg
            ├─ ...
        ├─ families.txt
        ├─ images_box.txt
        ├─ ...
    ├─ evaluation.m
    ├─ example_evaluation.m
    ├─ ...

```

<b> Stanford Dogs Dataset </b>

Stanford Dogs Dataset is provided by Stanford Univeristy under the [link](http://vision.stanford.edu/aditya86/ImageNetDogs/). 
The dataset consists of 20,580 images of 120 different dog breeds.
The dataset folder should have the following structure:

```
dog_dataset_root_folder/
    └─ Images
        ├─ n02092339-Weimaraner
            ├─ n02092339_107.jpg
            ├─ ....
        ├─ n02101388-Brittany_spaniel
            ├─ ....
        ├─ ....
    └─ splits
        ├─ file_list.mat
        ├─ test_list.mat
        ├─ train_list.mat

```

<b> FoodX Dataset </b>

FoodX-251 contains 251 food categories with a total of 158,000 images, making it one of the largest fgvc datasets present out there. the dataset can be downloaded from [here](https://github.com/karansikka1/iFood_2019). 
The dataset folder should have the following structure:

```
FoodX_dataset_root_folder/
    └─ annot
        ├─ class_list.txt
        ├─ train_info.csv
        ├─ val_info.csv
    └─ train_set
        ├─ train_039992.jpg
        ├─ ....
    └─ val_set
        ├─ val_005206.jpg
        ├─ ....
```



 ## Models
We have fine tuned the following pre-trained models on our dataset.
1. ConvNext - Tiny, Base & Large variants
2. Vision Transformer - Base & Large variants
3. Swin Transformer - Base variant
 
The configuration parameters for the models are given in ./utils/configs.py.

## Requirements 
We have tested this code on Windows 10 with Python 3.9. The other requirements are <br>
 - PyTorch
 - torchvision
 
## Training and Evaluation

The repository provides code for both fine-tuning and knowledge distillation:<br>
To train and evaluate individual models, execute 
`python models/train_(dataset).py` <br> 
For example, to work with FGVC-Aircraft classification execute `python models/train_aircraft.py`

### Kfold Cross Validation
The Kfold.py file provides code for running 5-fold cross validation on the dataset to evaluate model performance. The code allows for specifying the number of epochs and the optimizer used for training.
Validation test is done in every single epoch with 1/5th of the entire dataset i.e. test sampler.

### Data Augmentation
The models are trained with data augmentation techniques to improve their generalization performance. The following transformations are applied to the images during training:

1. Random horizontal flip
2. Random rotation
3. Random autocontrast
4. Resize to 256 x 256
5. Center crop to 224 x 224
6. To tensor
7. Normalize with mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225)

# Knowledge Distillation for Fine Grained Image Classification 
We forked https://github.com/MUKhattak/DeiT_ConvNeXt_KnowledgeDistillation.git repo and changes has been made to implement knowledge distillation with swin-base transformer as teacher model and convnext-base as student model.

## Training and Evaluation 

To finetune ConvNext model on CUB dataset, run the following command 

  ```bash
 $ python main.py --model convnext_base --drop-path 0.8 --input-size 384 --batch-size 16 --lr 5e-5 --warmup-epochs 0 --epochs 60 --weight-decay 1e-8 --cutmix 0 --mixup 0 --data-set CUB --data-path /path/to/dataset/root/folder --output_dir ./output/path --finetune /path/to/pretrained/swin/weights.pth/
```

To further finetune ConvNext model (already finetuned on CUB dataset) using Knowledge Distillation from Swin Transformer Base teacher model, run the following commad:

  ```bash
 $ python main.py --model convnext_base_distilled --distillation-type hard --teacher-model swin_transformer_base --drop-path 0.8 --input-size 384 --batch-size 16 --lr 5e-5 --warmup-epochs 0 --epochs 60 --weight-decay 1e-8 --cutmix 0 --mixup 0 --data-set CUB --data-path /path/to/dataset/root/folder --output_dir /path/to/save/output/files --finetune /path/to/pretrained/swin/weights.pth/ 
```
