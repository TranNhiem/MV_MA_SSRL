# Multi-View, multi-data augmentation (MVMA_SSRL)

<span style="color: red"><strong> </strong></span> This is offical implemenation of MVMA framework</a>.
<div align="center">
  <img width="50%" alt="MVMA Framework Illustration" src="images/MV_MA.png">
</div>

## Features 

+ Multi-view data augmentation: generate Local and Global views of the same object by Random Cropping.
+ Multi-data augmentation: apply different augmentation techniques to different parts of an image comprising (Random & Searched Policies).
+ Configurable pipeline: easily define your data augmentation pipeline by specifying the desired transformations and their parameters.
+ Batch processing: augment multiple images in parallel to speed up the data generation process.
+ Compatibility: integrate with popular deep learning libraries such as PyTorch and PyTorch Lightning 

# Table of Contents

  - [Installation](#installation)
  - [Configure Self-Supervised Pretraining](#Setup-self-supervised-pretraining)
    - [Dataset](#Natural-Image-Dataset)
    - [Hyperamters Setting](#Important-Hyperparameter-Setting)
    - [Choosing # augmentation Strategies](#Number-Augmentation-Strategies)
    - [Single or Multi GPUs](#Single-Multi-GPUS)
  - [Pretrained model](#model-weights)
  - [Downstream Tasks](#running-tests)
     - [Image Classification Tasks](#Natural-Image-Classification)
     - [Other Vision Tasks](#Object-Detection-Segmentation)
  - [Contributing](#contributing)
  
 ## Installation

```
pip or conda installs these dependents in your local machine
```
### Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm


## Self-supervised Pretraining

###  Preparing  Dataset: 

**NOTE:** Currently, This repository support self-supervised pretraining on the ImageNet dataset. 
+ 1. Download ImageNet-1K dataset (https://www.image-net.org/download.php). Then unzip folder follow imageNet folder structure. 


###  in pretraining Flags: 
`
Naviaging to the 

bash_files/pretrain/imagenet/HARL.sh
`

**1 Changing the dataset directory according to your path**
    `
    --train_dir ILSVRC2012/train \
    --val_dir ILSVRC2012/val \
    --mask_dir train_binary_mask_by_USS \
    `
**2 Other Hyperparameters setting** 
  
  - Use a large init learning rate {0.3, 0.4} for `short training epochs`. This would archieve better performance, which could be hidden by the initialization if the learning rate is too small.Use a small init learning rate for Longer training epochs should use value around 0.2.

    `
    --max_epochs 100 \
    --batch_size 512 \
    --lr 0.5 \
    `
**3 Distributed training in 1 Note**

`
Controlling number of GPUs in your machine by change the --gpus flag
    --gpus 0,1,2,3,4,5,6,7\
    --accelerator gpu \
    --strategy ddp \


## HARL Pre-trained models  

We opensourced total 8 pretrained models here, corresponding to those in Table 1 of the <a href="">HARL</a> paper:

|   Depth | Width   |    Param (M)  | Pretrained epochs| Linear eval  |
|--------:|--------:|--------:|-------------:|--------------:|
| [ResNet50 (1x)]() | 1X | 24 | 100 |    ## |     
| [ResNet50 (1x)]() | 1X  |  24 | 200 |    ## |  
| [ResNet50 (1x)]() | 1X  | 24 | 300 |    ## |  
| [ViT Small]() | 1X  |  22.2 | 100 |   ## |  
| [ViT Small]() | 1X  | 22.2 | 200 |  ## |  
| [ViT Small]() | 1X  |  22.2 | 300 |    ## |  


These checkpoints are stored in Google Drive Storage:

## Finetuning the linear head (linear eval)

To fine-tune a linear head (with a single GPU), try the following command:

For fine-tuning a linear head on ImageNet using GPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:
`
Stay tune! The instructions will update soon
`

## Semi-supervised learning and fine-tuning the whole network

You can access 1% and 10% ImageNet subsets used for semi-supervised learning via [tensorflow datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012_subset): simply set `dataset=imagenet2012_subset/1pct` and `dataset=imagenet2012_subset/10pct` in the command line for fine-tuning on these subsets.

You can also find image IDs of these subsets in `imagenet_subsets/`.

To fine-tune the whole network on ImageNet (1% of labels), refer to the following command:

`
Stay tune! The instructions will update soon
`

## Other resources
update soon


