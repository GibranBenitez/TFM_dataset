import json
import os


def formatJson(dict, set='valid'):

    # with open(path_json_yolo, 'r') as file:
    #     data = json.load(file)
    data = dict
    result = []

    conj = {}

    for i, obj in enumerate(data):

        ids = str(obj["image_id"])

        label = int(obj["category_id"])
        score = float(obj["score"])
        box_fill = [int(box) for box in obj["bbox"]]
        box_fill = [
            box_fill[0],
            box_fill[1],
            box_fill[0] + box_fill[2],
            box_fill[1] + box_fill[3]
        ]

        if ids not in conj.keys() and score > 0.25:
            conj[ids] = {'id': ids, "boxes": [box_fill],
                         "labels": [label], "scores": [score]}
        elif score > 0.25:
            conj[ids]["boxes"].append(box_fill)
            conj[ids]["labels"].append(label)
            conj[ids]["scores"].append(score)

    result = list(conj.values())

    # with open(os.path.join(path_save, 'YOLO_{}_Epoch_30.json'.format(set)), 'w') as file:
    #     json.dump(result, file)

    return result


# formatJson('/Users/agustincastillo/Documents/Repositorios/TFM_dataset/yolov5/runs/val/exp6/yolo_predictions.json',
#            'eval/runs/yolov5')
