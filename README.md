# SC-IBSR: Style-mixed Contrastive Learning for Image-based 3D Shape Retrieval

Code for **Cross-modal Contrastive Learning with a Style-mixed Bridge for Single Image 3D Shape Retrieval**.  **TOMM 2024.**

![Overview](/images/method.png)

## Introduction

Image-based 3D shape retrieval (IBSR) is a cross-modal matching task which searches similar shapes from a 3D repository using a natural image. Continuous attention has been paid to this topic, such as joint embedding, adversarial learning, and contrastive learning. Modality gap and diversity of instance similarities are two obstacles for accurate and fine-grained cross-modal matching. To overcome the two obstacles, we propose a style-mixed contrastive learning method (SC-IBSR). On one hand, we propose a style transition module to mix the styles of images and rendered shape views to form an intermediate style and inject it into image contents. The obtained style-mixed image features serve as a bridge for later contrastive learning in order to alleviate the modality gap. On the other hand, the proposed strategy of fine-grained consistency constraint aims at cross-domain contrast and considers the different importance of negative (positive) samples. Extensive experiments demonstrate the superiority of the style-mixed cross-modal contrastive learning on both the instance-level retrieval benchmark (i.e., Pix3D, Stanford Cars, and Comp Cars that annotate shapes to images), and the unsupervised category-level retrieval benchmark (i.e., MI3DOR-1 and MI3DOR-2 with unlabeled 3D shapes). Moreover, experiments are conducted on Office-31 dataset to validate the generalization capability of our method.

## About this repository

This repository provides **data**, **pre-trained models** and **code**.

## Installation
```zsh
# create anoconda environment
## please make sure that python version >= 3.7 (required by jittor)
conda create -n ibsr_jittor python=3.7
conda activate ibsr_jittor

# jittor installation
python3.7 -m pip install jittor==1.3.2
python3.7 -m jittor_utils.install_cuda
## testing jittor
### if errors appear, you can follow the instructions of jittor to fix them.
python3.7 -m jittor.test.test_example
# testing for cudnn
python3.7 -m jittor.test.test_cudnn_op

# other pickages
pip install pyyaml
pip install scikit-learn
pip install matplotlib
pip install scikit-image
pip install argparse
pip install easydl
pip install scipy
```



## How to use
## For instance-level retrieval
```zsh
# download pre-trained models, data and official ResNet pre-trained models from this links:
https://1drv.ms/u/s!Ams-YJGtFnP7mTQOACYHco1s2gXE?e=c87UnV

# put the unzip folder pre_trained, pretrained_resnet, data under SC-IBSR/Instance-level
cd SC-IBSR/Instance-level

# all codes are test under a single Nvidia RTX3090, Ubuntu 18.04
# training
python RetrievalNet_sag_con.py --config ./configs/pix3d.yaml

# testing
python RetrievalNet_test.py --config ./configs/pix3d.yaml --mode simple
# for full test
python RetrievalNet_test.py --config ./configs/pix3d.yaml --mode full
# for shapenet test
python RetrievalNet_test.py --config ./configs/pix3d.yaml --mode shapenet

# pay attention to:
# model_std_bin128 and model_std_ptc10k_npy are not uploaded.
# For model_std_ptc10k_npy, we randomly sample 10k points from the mesh by python igl package.
# For model_std_bin128, please refer to https://www.patrickmin.com/viewvox/ for more information.
```

## For category-level retrieval
```zsh
# download MI3DOR datasets from this links:
https://github.com/tianbao-li/MI3DOR

# put the unzip folder data under SC-IBSR/Category-level
cd SC-IBSR/Category-level

# all codes are test under a single Nvidia RTX3090, Ubuntu 18.04
# training
python train_image_sag_CO2_con.py
# remember to modify the path of the dataset

