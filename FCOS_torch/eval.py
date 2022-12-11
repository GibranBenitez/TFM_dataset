from create_dataset import PascalVOCDataset, Dataset_MM
from FCOS_utils import get_batch_statistics,ap_per_class, collate_fn
from transformation import get_transform
#from others.utils import collate_fn
from pprint import PrettyPrinter
from tqdm import tqdm
import pdb
import os
import torch
import json

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = '/home/fp/Escritorio/Datasets/MM_Dataset/test/JSONFiles' #JSON del Mask dataset
batch_size = 8
workers = 4
iou_th = 0.5
conf_th = 0.25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'checkpoints\Checkpoint_FT_MM_Dataset_epoca_25_retina.pth.rar'
filename = 'prueba_retina.json'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint,map_location=device)
model = checkpoint['model']
model = model.to(device)

model.eval() # Switch to eval mode

# Load test data
# test_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=keep_difficult)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

#dataset = VOCDataset('/home/fp/Escritorio/LuisBringas/FCOS/JSONfiles', 'TEST', get_transform(True))

dataset = Dataset_MM(data_folder, 'TEST', get_transform(False))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)


def meanAP(data_loader_test, model):
    
    labels = []
    img_id = []
    preds_adj_all = []
    annot_all = []

    for im, annot in tqdm(data_loader_test, position = 0, leave = True):
        im = list(img.to(device) for img in im)
        annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']
            img_id += [os.path.basename(dataset.imgs[t['image_id'][0]]).split('.')[0]]

        with torch.no_grad():
            preds_adj = make_prediction(model, im, conf_th)
            preds_adj = [{k: v.to(device) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)
        # pdb.set_trace()
    
    sample_metrics = []
    labels_ = []
    img_error_idx = []
    ap_error = []
    pred_boxes_ = []
    boxes_ = []

    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=iou_th) 
        labels_ += [ann['labels'].to('cpu') for ann in annot_all[batch_i]]
        pred_boxes_ += [x['boxes'].to('cpu') for x in preds_adj_all[batch_i]]
        boxes_ += [x['boxes'].to('cpu') for x in annot_all[batch_i]]
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
    true_positives_, pred_scores_, pred_labels_ = [x for x in list(zip(*sample_metrics))]
    for i_, (tp, ps, pl, lb) in enumerate(zip(true_positives_, pred_scores_, pred_labels_, labels_)):
        # pdb.set_trace()
        _, _, AP_, _, ap_class = ap_per_class(tp, ps, pl, lb)
        try:
            if torch.mean(AP_.float()) != 1:
                img_error_idx.append(i_)
                ap_error.append(torch.mean(AP_.float()).item())
        except:
            print(AP_, i_)
    # pdb.set_trace()
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    mAP = torch.mean(AP)
    # 2. Update json object
    datos = [mAP.item(), AP.numpy().tolist(), img_error_idx, ap_error]
    # 3. Write json file
    try:
        with open(filename, "w") as file:
            json.dump(datos, file)
    except:
        print('ERROR @ saving json file')
        pdb.set_trace()
    generate_JSON(img_id, pred_scores_, pred_labels_, pred_boxes_)
    print(f'mAP : {mAP}')
    print(f'AP : {AP}')
    print(f'errors_len : {len(img_error_idx)}/{len(labels_)}')

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def generate_JSON(img_ids, pred_scores, pred_labels, pred_boxes):
    all_preds = []
    for ids, ps, pl, bx in zip(img_ids, pred_scores, pred_labels, pred_boxes):
        temp_d = dict()
        temp_d['id'] = ids
        temp_d['scores'] = ps.tolist()
        temp_d['labels'] = pl.tolist()
        temp_d['boxes'] = bx.tolist()
        # temp_d['gt_labels'] = gt.tolist()
        # temp_d['gt_boxes'] = gtb.tolist()
        all_preds.append(temp_d)
    try:
        with open(filename, "w") as file:
            json.dump(all_preds, file)
    except:
        print('ERROR @ saving json file')
        pdb.set_trace()


def runEvaluate():
    meanAP(data_loader, model)

if __name__ == '__main__':
    runEvaluate()