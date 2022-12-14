import os
from glob import glob
from xml.etree import ElementTree as ET
import json


def checkdir(filePath):

    if os.path.exists(filePath):
        numb = 1
        while True:
            newName = "{0}".format(filePath+'(' + str(numb) + ')')
            if os.path.exists(newName):
                numb += 1
            else:
                return newName

    return filePath


def xmls2jsonGT(path_xmls, set='VALID'):
    # create the labels folder (output directory)
    # if not os.path.exists('./eval'):
    #     os.makedirs('./eval')

    # if not os.path.exists('./eval/runs'):
    #     os.makedirs('./eval/runs')

    # path = './eval/runs/yolov5'
    # path = checkdir(path)
    # os.makedirs(path, exist_ok=True)

    # identify all the xml files in the annotations folder (input directory)
    files = glob(os.path.join(path_xmls, '*.xml'))

    result = []

    print("Generating JSON annots GT for {} files".format(len(files)))
    # loop through each
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()

        boxes = []
        labels = []
        sizes = []
        for obj in root.findall('object'):
            label = obj.find("label").text
            pil_bbox = [int(x.text) for x in obj.find("bndbox")][:4]
            size = int(obj.find("size").text)

            sizes.append(size)
            labels.append(int(label))
            boxes.append(pil_bbox)

        result.append({'id': filename, 'boxes': boxes,
                      'labels': labels, 'sizes': sizes})

    # with open(os.path.join(path, "{}_GT_objects.json".format(set)), 'w') as file:
    #     json.dump(result, file)
    return result


# xmls2jsonGT('/Users/agustincastillo/Downloads/lotes/valid/Annotations')
