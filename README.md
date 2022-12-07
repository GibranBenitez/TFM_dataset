# TFM_dataset 

__TFM a Dataset for Detection and Recognition of Masked Faces in the Wild__<br>
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
-------- | ---------- | ----- | ----------| -------
 46,917 | 25,498 | 38,432 | 20,971 | 4,140 
 
We divide the entire data set into training (with pictures that only include up to two faces per image), validation and testing sets (with pictures with up to six faces). The distribution of faces shown in the table below:                
 
 Set |Unmasked |  Surgical	| Cloth	| Respirator| Valved | Total 
---- | ------- | ---------- | ----- | ----------| ------ | ----------- 
 Train | 42,687 | 16,979 | 22, 058 | 10, 611 | 2,939 | 95,274
 Valid | 1,869 | 1,915 | 7,041 | 3,514 | 399 | 14,738
 Test | 2,361 | 6,604 | 9,333 | 6,846 | 802  | 25,946
 
 
