# TFM_dataset 

[__TFM a Dataset for Detection and Recognition of Masked Faces in the Wild__](https://dl.acm.org/doi/10.1145/3551626.3564957)<br>
[Gibran Benitez-Garcia](https://gibranbenitez.github.io), Miguel Jimenez-Martinez, Jesus Olivares-Mercado and Hiroki Takahashi<br>
___Accepted at [ACM Multimedia Asia 2022](https://www.mmasia2022.org)___

In this paper we introduced a **TFM dataset** with sufficient size and variety to train and evaluate deep learning models
for face mask detection and recognition. This dataset contains **107, 598** images with **135, 958** maskedfaces in the wild mined from
Twitter within two years since the beginning of the COVID-19 pandemic. Thus, they include diverse scenes with real-world variations
in background and illumination.
<br>TFM is annotated into unmasked and four categories of facepieces (**surgical, cloth, respirator and valved**) based on WHO and CDC recommendations. We evaluate four one-stage object detection methods: **SSD**, **RetinaNet**, **FCOS**, and **YOLOv5**. The experimental results show that YOLOv5 can achieve about **90%**
of **mAP@0.5**, demonstrating that the TFM dataset can be used to train robust models and may help the community step forward in
detecting and recognizing masked faces in the wild.

### Dataset details
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
 
 ### Download TFM dataset from:
  - [MM_train1.zip (5.4 GB)](https://drive.google.com/file/d/1XFv41kuujoG3uKfeZ3CsIM7ZCjJ0QXO0/view?usp=sharing)
  - [MM_train2.zip (5.03GB)](https://drive.google.com/file/d/1XZeNA9M1b-6bbIzd3XXmsQ9VF9ziGtMT/view?usp=sharing)
  - [MM_valid.zip (1GB)](https://drive.google.com/file/d/1X7tMmI_zXT89UxUdsn6xhIvmSo7j2edU/view?usp=sharing)
  - [MM_test.zip (1.26 GB)](https://drive.google.com/file/d/1XBB760jFFbhlYFILXXER-JgGaBDKcSMQ/view?usp=share_link)
 ### Benchmark evaluation
 In general, all methods present the expected trend when evaluating faces of specific dimensions. Medium (mAP<sub>𝑀</sub>) and large (mAP<sub>𝐿</sub>) faces achieved higher performance than the general mAP, while small (mAP<sub>𝑆</sub>) shows a significant accuracy drop.
 <br>You can download the pre-trained models from the table below:
 
Method | Valid mAP |  mAP	| mAP<sub>S</sub>	| mAP<sub>M</sub>| mAP<sub>L</sub> 
:-----:| :-------: | :--: | :-------------: | :------------: | :-------------: 
[SSD](https://drive.google.com/file/d/1Cd2YxcaCrxZWUcyWM2GAjblvu5P7X886/view?usp=share_link) | 76.6 |  74.9	| 55.8	| 79.8 | 85.1 |
[FCOS](https://drive.google.com/file/d/1MnMsGaSVfs6WEfAix_mOe6KYHpPOWJqM/view?usp=share_link) | 85.7 |  86.4	| 81.2	| 88.3 | 87.6 |
[RetinaNet](https://drive.google.com/file/d/1DJtsLel7qrkKzpKmf8O4Td2HN9dBSMZw/view?usp=share_link) | 87.1 |  86.9	| 71.3	| 91.7 | 92.8 |  
[YOLOv5](https://drive.google.com/file/d/1uAZioqd4Pvurl7eEiDawidve8FU0dFXA/view?usp=share_link) | **89.6** |  **89.6**	| **82.2**	| **92.9** | **93.0** |  

### Requeriments

- Python 3.8+
- PyTorch 1.0+
- TorchVision
- OpenCV

## Usage example
We have prepared an example of the detection and recognition of the maskedfaces with our pre-entrenated models.

### Preparation
- Clone this repository

```
git clone https://github.com/GibranBenitez/TFM_dataset.git
```
```
cd TFM_dataset && mkdir checkpoints
```
- Store all pretrained models in `./TFM_dataset/checkpoints`
- Choose the model and source directory.
- **Yolov5:**
```
python main.py --model yolov5 --source eg
```
- **SSD:**
```
python main.py --model ssd --source eg
```
- **FCOS:**
```
python main.py --model fcos --source eg
```
- **RetinaNet:**
```
python main.py --model retina --source eg
```
- To add your own images you must create the directory in the root and choose it in the source:
```
python main.py --model "choose model" --source "your directory name"
```