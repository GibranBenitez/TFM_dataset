from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import json
import pdb

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['cloth', 'none', 'respirator', 'surgical', 'valve']
filename = 'YOLO_val_cm.json'
iou_th = 0.5
# gt_json = 'TEST_GT_objects_S.json'
gt_json = 'VALID_GT_objects.json'
# pred_json = 'RETINA_ep30_preds_Gral.json'
pred_json = 'YOLO_valid_Epoch_30.json'


def main_all():
    # Read json files
    with open(gt_json, "r") as file:
        gt_ = json.load(file)
    with open(pred_json, "r") as file:
        pd_all = json.load(file)

    print(gt_json)
    print(pred_json)

    gt_ids = [d['id'] for d in gt_]
    pd_ids = [d['id'] for d in pd_all]
    seto = set(pd_ids)
    not_pds = [i for i, x in enumerate(gt_ids) if x not in seto]
    not_pds.sort(reverse=True)
    poped = [gt_.pop(x) for x in not_pds]  # images excluded from predictions
    print('## {} images not predicted (not found in GT) u.u'.format(len(poped)))

    pd_all.sort(key=lambda x: x['id'])
    gt_.sort(key=lambda x: x['id'])
    pd_ids = [d['id'] for d in pd_all]

    gt_all = [{k: torch.tensor(v).to(device)
               for k, v in t.items() if type(v) != str} for t in gt_]
    pred_all = [{k: torch.tensor(v).to(
        device) for k, v in t.items() if type(v) != str} for t in pd_all]

    # calculate gral mAP & APs
    print('\ncalculating results...')
    sam_mets = get_batch_statistics(pred_all, gt_all, iou_threshold=iou_th)
    labels_ = [ann['labels'].to('cpu') for ann in gt_all]
    true_positives, pred_scores, pred_labels = [
        torch.cat(x, 0) for x in list(zip(*sam_mets))]
    labels = torch.cat(labels_, 0)
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)
    mAP = torch.mean(AP)

    # calculate AP per image
    print('almost done...')
    img_error_idx = []
    ap_error = []
    full_preds = []
    full_gts = []
    ap_full = []
    true_positives_, pred_scores_, pred_labels_ = [
        x for x in list(zip(*sam_mets))]
    for i_, (tp, ps, pl, lb) in enumerate(zip(true_positives_, pred_scores_, pred_labels_, labels_)):
        # pdb.set_trace()
        _, _, AP_, _, ap_class_ = ap_per_class(tp, ps, pl, lb)

        if len(pl) > 0:
            for j_ in range(len(lb)):
                ious = bbox_iou(gt_all[i_]['boxes'][j_].repeat(
                    len(pl), 1), pred_all[i_]['boxes']) >= iou_th
                full_preds += pl[ious].tolist()
                full_gts += lb[j_].repeat(1, len(pl[ious])).tolist()[0]

        try:
            ap_full.append(torch.mean(AP_.float()).item())
            if torch.mean(AP_.float()) != 1:
                img_error_idx.append(i_)
                ap_error.append(torch.mean(AP_.float()).item())
        except:
            print(AP_, i_)

    datos = [mAP.item(), AP.numpy().tolist(), img_error_idx, ap_error, [pd_ids[x]
                                                                        for x in img_error_idx], full_gts, full_preds, ap_full, pd_ids]
    # pdb.set_trace()
    cm = confusion_matrix(full_gts, full_preds)

    print(f'\nmAP : {mAP}\n')
    for cls_, ap_ in zip(classes, AP.numpy().tolist()):
        print(f'{cls_} : {ap_}')
    print(f'\nerrors_len : {len(img_error_idx)}/{len(labels_)}')
    print('\nconfusion matrix:')
    print(cm)
    print(classes)

    # 3. Write json file
    try:
        with open(filename, "w") as file:
            json.dump(datos, file)
    except:
        print('ERROR @ saving json file')
        pdb.set_trace()
    print('\n... {} saved'.format(filename))


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output = outputs[sample_i]  # predict
        true_positives = torch.zeros(output['boxes'].shape[0])

        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(output['boxes'], output['labels'])):
                # If targets are found break
                if len(detected_boxes) == len(target_labels):  # annotations -> target_labels
                    break
                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                iou, box_index = bbox_iou(
                    pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append(
            [true_positives, output['scores'], output['labels']])

    return batch_metrics


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes
    unique_classes = torch.unique(target_cls)   # 2가 거의 예측안됨
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i], -1)
            tpc = torch.cumsum(tp[i], -1)
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])
            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])
            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))
    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = torch.tensor(np.array(p)), torch.tensor(
        np.array(r)), torch.tensor(np.array(ap))
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


if __name__ == "__main__":
    main_all()
