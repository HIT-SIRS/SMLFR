# <center> Generative ConvNet Foundation Model with Sparse and Low-Frequency Filtered Masked Modeling for Remote Sensing Image Interpretation <center>

Introduction
---
This is the official repository for the paper "Generative ConvNet Foundation Model with Sparse and Low-Frequency Filtered Masked Modeling for Remote Sensing Image Interpretation".


**Abstract**: Foundation models offer a highly versatile and precise solution for intelligent interpretation of remote sensing images, thus greatly facilitating various remote sensing applications. Nevertheless, current foundational models for remote sensing predominantly employ vision transformers based on generative methods, with no corresponding exploration of ConvNets with masked image modeling (MIM). In this paper, we make the first attempt to propose a generative ConvNet foundation model tailored for remote sensing scenarios, which comprises two key components: Firstly, a large dataset named GeoSense, containing approximately nine million diverse remote sensing images, is constructed to enhance the robustness and generalization of the foundation model during the pre-training phase. Secondly, a sparse and low-frequency filtered masked modeling (SLFFM) self-supervised learning framework is designed for representation learning of ConvNet foundation model. Specifically, we introduce sub-manifold sparse convolutions to enable the ConvNet to process variable-length sequences for MIM self-supervised pre-training. Additionally, a low-frequency filtered reconstruction target is designed to guide the model's attention towards essential ground object features in remote sensing images, while mitigating unnecessary detail interference. To evaluate the general performance of our proposed foundation model, comprehensive experiments have been carried out on five datasets across three downstream tasks (i.e., object detection, semantic segmentation, and change detection.). Experimental results demonstrate  that our method consistently achieves state-of-the-art performance across all benchmark datasets and downstream tasks.

![flowchart](https://github.com/HIT-SIRS/SLFFM/assets/114158053/614d3211-da3e-44cf-9ed4-43f6b6e694b9)

## Pre-trained and Fine-tuned Models

### Pre-training

#### GeoSense

| Pretrain | Backbone      | Input Size | Paramters | Pretrained Model |
|----------|---------------|---------|-----------|------------------|
| SLFFM    | ConvNeXt-Base | 224x224 | 89M       | [Weights](<https://pan.baidu.com/s/1T5gmsvLh7mmGbzHI6r8gWw?pwd=ewjg>)|
|  SLFFM   | ConvNeXt-Large| 224x224 | 198M      | [Weights](<https://pan.baidu.com/s/1fTesshqGB3UCjYDP7jwWKQ?pwd=5yxn>)|

### Object Detection

#### Dota V1.0

| Method         | Pre-train | Backbone       | Lr Schd | mAP   | Config                                                                                                                | Model                                   |
|----------------|-----------|----------------|---------|-------|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| Oriented R-CNN | SLFFM     | ConvNeXt-Base  | 1x      | 79.15 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/ObjectDetection/configs/convnext/convnext_base_rcnn_dota.py>)   | [Weights](<https://pan.baidu.com/s/16llItvVn6iOXino1FNiDhQ?pwd=shcp>)  |
| Oriented R-CNN | SLFFM     | ConvNeXt-Large | 1x      | 79.33 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/ObjectDetection/configs/convnext/convnext_large_rcnn_dota.py>)  | [Weights](<https://pan.baidu.com/s/1X8LkrECyAfCXOaGFsm-dBw?pwd=oq36>)  |

#### DIOR-R

| Method         | Pre-train | Backbone       | Lr Schd | mAP   | Config                                                                                                                  | Model                                                                                                               |
|----------------|-----------|----------------|---------|-------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Oriented R-CNN | SLFFM     | ConvNeXt-Base  | 1x      | 71.50 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/ObjectDetection/configs/convnext/convnext_base_rcnn_dior.py>)     | [Weights](<https://pan.baidu.com/s/1ySSJZ596n8TYzcRubSqYPw?pwd=aczh>)                                                                              |
| Oriented R-CNN | SLFFM     | ConvNeXt-Large | 1x      | 72.33 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/ObjectDetection/configs/convnext/convnext_large_rcnn_dior.py>)    | [Weights](<https://pan.baidu.com/s/1rTCTfy3KGFJor4Hyd4zDRQ?pwd=hrpc>) |

### Semantic Segmentation

#### Potsdam

| Method   | Pre-train | Backbone       | Lr Schd | OA    | Config                                                                                                        | Model                                   |
|----------|-----------|----------------|---------|-------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| UperNet  | SLFFM     | ConvNeXt-Base  | 160k    | 91.72 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/SemanticSegmentation/configs/convnext_b_potsdam.py>)    | [Weights](<https://pan.baidu.com/s/1G7t-zao0crIuaOJ8AsRqaQ?pwd=28gt>)  |
| UperNet  | SLFFM     | ConvNeXt-Large | 160k    | 91.82 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/SemanticSegmentation/configs/convnext_l_potsdam.py>)    | [Weights](<https://pan.baidu.com/s/1fIAXmBhk2kuviXFyIXRJtQ?pwd=ascd>)  |

#### LoveDA

| Method   | Pre-train | Backbone       | Lr Schd | mIoU  | Config                                                                                                    | Model                                   |
|----------|-----------|----------------|---------|-------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------|
| UperNet  | SLFFM     | ConvNeXt-Base  | 160k    | 52.59 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/SemanticSegmentation/configs/convnext_b_loveda.py>) | [Weights](<https://pan.baidu.com/s/1w_A_3HQhVCd0-p5A5t_ezA?pwd=pf18>)  |
| UperNet  | SLFFM     | ConvNeXt-Large | 160k    | 53.03 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/SemanticSegmentation/configs/convnext_l_loveda.py>) | [Weights](<https://pan.baidu.com/s/1YbyscREPbH4ZbjRV4d4cuA?pwd=2ybd>)  |

### Change Detection

#### LEVIR-CD

| Method | Pre-train | Backbone       | Lr Schd | F1    | Config                                                                                                         | Model                                   |
|--------|-----------|----------------|---------|-------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| BIT    | SLFFM     | ConvNeXt-Base  | 20k     | 93.66 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/ChangeDetection/configs/convnext/convnext_base_bit.py>)  | [Weights](<https://pan.baidu.com/s/1WkpEJQe9o1nZ6hYqvG_s8g?pwd=9kql>)  |
| BIT    | SLFFM     | ConvNeXt-Large | 20k     | 93.89 | [Config](<https://github.com/HIT-SIRS/SLFFM/blob/main/ChangeDetection/configs/convnext/convnext_large_bit.py>) | [Weights](<https://pan.baidu.com/s/1ho2QDc49EbPYY177iKk60w?pwd=icko>)  |

## Usage

Environment:

- python 3.8.13
- pytorch 1.12.1+cu113
- torchvision 0.13.1+cu113
- timm 0.6.12
- mmdet 2.28.2
- mmsegmentation 0.30.0
- opencd 0.0.3

