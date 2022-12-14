import xml.etree.ElementTree as ET
import glob
import json
import os
import shutil
from pathlib import Path


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def data2yolov5(set: str, path_annot: str, path_images):

    if not os.path.exists('./datasets/tfm'):
        os.makedirs('./datasets', exist_ok=True)

        if not os.path.exists('./datasets/tfm/images'):
            os.makedirs('./datasets', exist_ok=True)

        if not os.path.exists('./datasets/tfm/labels'):
            os.makedirs('./datasets', exist_ok=True)

        dirs = ['./datasets/tfm/images', './datasets/tfm/labels']

        for dir in dirs:
            for subset in ['train', 'test', 'valid']:
                os.makedirs(os.path.join(dir, subset), exist_ok=True)

    moveImages(set, path_images)
    creaTxt(set, path_annot)


def moveImages(set: str, path_images: str):

    listImages = glob.glob(os.path.join(path_images, '*'))

    for image in listImages:

        src = r'{}'.format(image)
        dest = r'./datasets/tfm/images/' + set + '/' + str(Path(image).name)

        shutil.copy2(src, dest)


def creaTxt(set: str, path_annot: str):

    folders_db = [set]
    for folder in folders_db:
        classes = ['cloth', 'none', 'respirator', 'surgical', 'valve']
        input_dir = path_annot
        output_dir = "./datasets/tfm/labels/{}/".format(
            folder)

        # create the labels folder (output directory)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # identify all the xml files in the annotations folder (input directory)
        files = glob.glob(os.path.join(input_dir, '*.xml'))
        print("Generating txt annots for {} files in {} folder".format(
            len(files), folder))
        # loop through each
        for fil in files:
            basename = os.path.basename(fil)
            filename = os.path.splitext(basename)[0]
            result = []
            # parse the content of the xml file
            tree = ET.parse(fil)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)

            for obj in root.findall('object'):
                label = obj.find("label").text
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                # convert data to string
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{label} {bbox_string}")

            if len(result) > 0:
                with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(result))
            else:
                print("{} with NO annots".format(basename))

    # generate the classes file as reference
    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))
