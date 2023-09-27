# <center> Generative ConvNet Foundation Model with Sparse and Low-Frequency Filtered Masked Modeling for Remote Sensing Image Interpretation <center>

Introduction
---
This is the official repository for the paper "Generative ConvNet Foundation Model with Sparse and Low-Frequency Filtered Masked Modeling for Remote Sensing Image Interpretation".


**Abstract**: Foundation models offer a highly versatile and precise solution for intelligent interpretation of remote sensing images, thus greatly facilitating various remote sensing applications. Nevertheless, current foundational models for remote sensing predominantly employ vision transformers based on generative methods, with no corresponding exploration of ConvNets with masked image modeling (MIM). In this paper, we make the first attempt to propose a generative ConvNet foundation model tailored for remote sensing scenarios, which comprises two key components: Firstly, a large dataset named GeoSense, containing approximately nine million diverse remote sensing images, is constructed to enhance the robustness and generalization of the foundation model during the pre-training phase. Secondly, a sparse and low-frequency filtered masked modeling (SLFFM) self-supervised learning framework is designed for representation learning of ConvNet foundation model. Specifically, we introduce sub-manifold sparse convolutions to enable the ConvNet to process variable-length sequences for MIM self-supervised pre-training. Additionally, a low-frequency filtered reconstruction target is designed to guide the model's attention towards essential ground object features in remote sensing images, while mitigating unnecessary detail interference. To evaluate the general performance of our proposed foundation model, comprehensive experiments have been carried out on five datasets across three downstream tasks (i.e., object detection, semantic segmentation, and change detection.). Experimental results demonstrate  that our method consistently achieves state-of-the-art performance across all benchmark datasets and downstream tasks.



