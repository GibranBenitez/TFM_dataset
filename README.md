# TFM Dataset 

This repository is an official implementation of the paper:

[_**TFM a Dataset for Detection and Recognition of Masked Faces in the Wild**_](https://dl.acm.org/doi/10.1145/3551626.3564957)<br>
[Gibran Benitez-Garcia](https://gibranbenitez.github.io), Miguel Jimenez-Martinez, Jesus Olivares-Mercado and Hiroki Takahashi<br>
**Presented at** _**[ACM Multimedia Asia 2022](https://www.mmasia2022.org)**_

## Updates
- `[2022-12-15]:` Download links of zipped images have been disabled due to copyright issues. Links to download the original images from Twitter will be shared soon. 

## Coming soon
- [X] Share a small batch of public images where anyone can test our pretrained models.
- [ ] Instructions to download the complete TFM dataset from Twitter
## Introduction

In this paper we introduced a **TFM dataset** with sufficient size and variety to train and evaluate deep learning models
for face mask detection and recognition. This dataset contains **107, 598** images with **135, 958** maskedfaces in the wild mined from
Twitter within two years since the beginning of the COVID-19 pandemic. Thus, they include diverse scenes with real-world variations
in background and illumination.
<br>TFM is annotated into unmasked and four categories of facepieces (**surgical, cloth, respirator and valved**) based on WHO and CDC recommendations. We evaluate four one-stage object detection methods: **SSD**, **RetinaNet**, **FCOS**, and **YOLOv5**. The experimental results show that YOLOv5 can achieve about **90%**
of **mAP@0.5**, demonstrating that the TFM dataset can be used to train robust models and may help the community step forward in
detecting and recognizing masked faces in the wild.

## Dataset details
After the annotation process, we remove duplicates and images with explicit content, thus,  we have:

Unmasked |  Surgical	| Cloth	| Respirator| Valved 
:------: | :-------: | :---: | :-------: | :-------:
 46,917 | 25,498 | 38,432 | 20,971 | 4,140 
 
We divide the entire data set into training (with pictures that only include up to two faces per image), validation and testing sets (with pictures with up to six faces). The distribution of faces shown in the table below:                
 
 Set |Unmasked |  Surgical	| Cloth	| Respirator| Valved | Total 
:--: | :-----: | :-------: | :---: | :-------: | :----: | :-----------: 
 Train | 42,687 | 16,979 | 22, 058 | 10, 611 | 2,939 | 95,274
 Valid | 1,869 | 1,915 | 7,041 | 3,514 | 399 | 14,738
 Test | 2,361 | 6,604 | 9,333 | 6,846 | 802  | 25,946
 
 ## Download TFM dataset from:
 
 #### Instructions to download the complete TFM dataset directly from Twitter will be available SOON.
  
 ## Benchmark evaluation
 In general, all methods present the expected trend when evaluating faces of specific dimensions. Medium (mAP<sub>????</sub>) and large (mAP<sub>????</sub>) faces achieved higher performance than the general mAP, while small (mAP<sub>????</sub>) shows a significant accuracy drop.
 <br>You can download the pre-trained models from the table below:
 
Method | Valid mAP |  mAP	| mAP<sub>S</sub>	| mAP<sub>M</sub>| mAP<sub>L</sub> 
:-----:| :-------: | :--: | :-------------: | :------------: | :-------------: 
[SSD](https://drive.google.com/file/d/1Cd2YxcaCrxZWUcyWM2GAjblvu5P7X886/view?usp=share_link) | 76.6 |  74.9	| 55.8	| 79.8 | 85.1 |
[FCOS](https://drive.google.com/file/d/1MnMsGaSVfs6WEfAix_mOe6KYHpPOWJqM/view?usp=share_link) | 85.7 |  86.4	| 81.2	| 88.3 | 87.6 |
[RetinaNet](https://drive.google.com/file/d/1DJtsLel7qrkKzpKmf8O4Td2HN9dBSMZw/view?usp=share_link) | 87.1 |  86.9	| 71.3	| 91.7 | 92.8 |  
[YOLOv5](https://drive.google.com/file/d/1uAZioqd4Pvurl7eEiDawidve8FU0dFXA/view?usp=share_link) | **89.6** |  **89.6**	| **82.2**	| **92.9** | **93.0** |  

## Requeriments

- Python 3.8+
- PyTorch 1.0+
- TorchVision
- OpenCV
- Scikit-learn

## Usage example
We have prepared an example of detection and recognition of masked faces with our pretrained models.

### Preparation
- Clone this repository

```
git clone https://github.com/GibranBenitez/TFM_dataset.git
```
```
cd TFM_dataset && mkdir checkpoints
```
- Store all pretrained models in `./TFM_dataset/checkpoints`
### Mini-batch download for testing
We have prepared a minibatch with images downloaded from internet.
- Dowload the [MM_minibatch](https://drive.google.com/file/d/1bZrOjHg2vOQPzliB2qftARLDfNYNUPlB/view?usp=share_link)
- Unzip and move the MM_minibatch to `./TFM_dataset`
### Run
 Choose the model and source directory.
- **Yolov5:**
```
python main.py --detect --model yolov5 --source MM_minibatch
```
- **SSD:**
```
python main.py --detect --model ssd --source MM_minibatch
```
- **FCOS:**
```
python main.py --detect --model fcos --source MM_minibatch
```
- **RetinaNet:**
```
python main.py --detect --model retina --source MM_minibatch
```
- To add your own images you must create the directory in the root and choose it in the source:
```
python main.py --detect --model "choose model" --source "your directory name"
```
## Usage YOLOv5 Testing and Validation 
### Valid Set:
- Unzip the "MM_valid.zip"
 ```
python main.py --data2yolo --set valid --path_annots "add the path to the Annotations directory" --path_images "add path to the JPEGImages directory"
```   
 - Start evaluation:
 ```
python main.py --eval --set valid --path_annots "add the path to the Annotations directory"
```

### Test Set:

 - Unzip the "MM_test.zip"
 ```
python main.py --data2yolo --set test --path_annots "add the path to the Annotations directory" --path_images "add path to the JPEGImages directory"
```   
 - Start evaluation:
 ```
python main.py --eval --set test --path_annots "add the path to the Annotations directory"
```
## Citation
If you find useful the TFM dataset for your research, please cite the paper:

```bibtex
@inproceedings{10.1145/3551626.3564957,
author = {Benitez-Garcia, Gibran and Takahashi, Hiroki and Jimenez-Martinez, Miguel and Olivares-Mercado, Jesus},
title = {TFM a Dataset for Detection and Recognition of Masked Faces in the Wild},
year = {2022},
isbn = {9781450394789},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3551626.3564957},
doi = {10.1145/3551626.3564957},
booktitle = {Proceedings of the 4th ACM International Conference on Multimedia in Asia},
articleno = {1},
numpages = {7},
keywords = {datasets, face mask recognition, Twitter image mining, face mask detection},
location = {Tokyo, Japan},
series = {MMAsia '22}
}
```
## Acknowledgement

This project is inspired by many previous works, including:

- Yolov5, Glenn Jocher et al. [[code](https://github.com/ultralytics/yolov5)].
- [Real-Time
Face Mask Detection Method Based on YOLOv3](https://doi.org/10.3390/electronics10070837), Xinbei Jiang et al, Electronics 2021.
- [SSD: Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325), Wei Liu et al, ECCV 2016 [[code](https://github.com/weiliu89/caffe/tree/ssd)].
- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355). Zhi Tian et al, CVPR 2019 [[code](https://github.com/tianzhi0549/FCOS)].
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), Tsung-Yi Lin et al, CVPR.
- [Detecting Masked Faces in the
Wild with LLE-CNNs](https://ieeexplore.ieee.org/document/8099536), Shiming Ge, et al, CVPR 2017. 

