import torch
from glob import glob
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

import cv2


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
size_font = 1
grosor = 2


def boundingBoxColor(label):
    if label == 'surgical':  # Color Cian
        return (255, 255, 0)
    elif label == 'valved':  # Color Magenta
        return (255, 0, 255)
    elif label == 'cloth':  # Color Naranja
        return (0, 165, 255)
    elif label == 'respirator':  # Color azul
        return (0, 0, 255)
    elif label == 'other':  # Color Naranja
        return (0, 165, 255)
    elif label == 'unmasked':  # Color rojo
        return (255, 0, 0)
    return (0, 0, 0)

def setBoxesToImage(path, dir ,labels ,lis={}):

    image = cv2.imread(path)

    split_img=path.split('\\')

    colors = []
    count = 0
    for box, label in zip(lis['boxes'], lis['labels']):
        color = boundingBoxColor(labels[label])
        colors.append(color)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(image, str(labels[label]),
                    (box[0], box[3] + 18), font, size_font, color, grosor)
        count += 1
    #print(dir + '/' +split_img[1])
    cv2.imwrite(dir + '/' +split_img[1] , image)
    return colors



def detect(path_model, path_imgs, threshold, to_dir, modelo = 'fcos'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path_model, map_location=device)

    model = checkpoint['model']

    model.eval()

    paths = glob(path_imgs + '/*')

    if modelo == 'retina':
        labelst = { 1: 'cloth', 2: 'unmasked', 3: 'respirator', 4: 'surgical', 5: 'valved'}
    else:
        labelst = { 0: 'cloth', 1: 'unmasked', 2: 'respirator', 3: 'surgical', 4: 'valved'}



    for img in paths:

        image = Image.open(img)
        image = transforms.ToTensor()(image).unsqueeze_(0)
        output = model(image)
        boxes = output[0]['boxes'].to(torch.int32)
        labels = output[0]['labels']
        scores = output[0]['scores']
        
        predicciones = {'boxes': [], 'labels': []}

        for i, box in enumerate(boxes):
            bbox = []

            if float(scores[i]) < threshold :
                break

            for c in box:
                bbox.append(int(c))
            predicciones['boxes'].append(bbox)
        
        for i, lbl in enumerate(labels):
            if float(scores[i]) < threshold :
                break
            predicciones['labels'].append(int(lbl))

        setBoxesToImage(img, to_dir, labelst,predicciones, )


if __name__ == "__main__":
    detect('C:/Users/MIke/Desktop/TFM_dataset/checkpoints/Checkpoint_FT_MM_Dataset_epoca_30_fcos.pth.rar', 'C:/Users/MIke/Desktop/TFM_dataset/imgpruebas', '', 0.5)
