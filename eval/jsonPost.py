import json
import os


def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def formatJson(path_json_yolo, path_save, set='valid'):

    with open(path_json_yolo, 'r') as file:
        data = json.load(file)

    result = []

    conj = {}

    boxes = []
    labels = []
    scores = []

    for i, obj in enumerate(data):

        ids = str(obj["image_id"])

        label = int(obj["category_id"])
        score = float(obj["score"])
        box_fill = [int(box) for box in obj["bbox"]]

        box_fill = [box_fill[0],
                    box_fill[3],
                    box_fill[0] + box_fill[1],
                    box_fill[3] + box_fill[2]]

        if ids not in conj.keys() and score > 0.25:
            conj[ids] = {'id': ids, "boxes": [box_fill],
                         "labels": [label], "scores": [score]}
        elif score > 0.25:
            conj[ids]["boxes"].append(box_fill)
            conj[ids]["labels"].append(label)
            conj[ids]["scores"].append(score)

    result = list(conj.values())

    with open(os.path.join(path_save, 'YOLO_{}_Epoch_30.json'.format(set)), 'w') as file:
        json.dump(result, file)


formatJson('/Users/agustincastillo/Downloads/best_predictions.json',
           'eval/runs/yolov5')
