import argparse
from yolov5.detect import run
from yolov5.val import mainEvalSingle
import os
from os import path as osPath
from SSD.detect import runssd
from eval.xml2yolo_TFM import data2yolov5
from eval.xml2json import xmls2jsonGT
from eval.jsonPost import formatJson
from eval.eval_fromJSON import main_all
from FCOS_RETINA.detect import detect

checkpoints = {'yolov5': 'checkpoints\yolo.pt', 'retina': 'checkpoints\weight_retina.pth.rar',
               'fcos': 'checkpoints\weight_fcos.pth.rar', 'ssd': 'checkpoints\ssd.pth'}


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments for detect images
    parser.add_argument('--detect', action='store_true', default=False)

    parser.add_argument('--model', type=str, default='yolov5')
    parser.add_argument('--source', type=str, default='./data')

    # Arguments for prepair data a yolov5
    parser.add_argument('--data2yolo', action='store_true', default=False)

    parser.add_argument('--set', type=str,
                        help='val, test, train', default=None)
    parser.add_argument('--path_annots', type=str,  default=None,
                        help='path of annotation set')
    parser.add_argument('--path_images', type=str,  default=None,
                        help='path of annotation set')

    # Arguments for eval sets in YoloV5
    parser.add_argument('--eval', action='store_true', default=False)
    # 'set' argument
    # 'path_annots'

    parser.add_argument('--val')

    args = parser.parse_args()

    if args.eval:
        dictYolo = mainEvalSingle(
            './yolov5/data/tfm.yaml', checkpoints['yolov5'], 640, args.set)
        dictYolo = formatJson(dictYolo, args.set)
        gt_set = xmls2jsonGT(args.path_annots, args.set)
        main_all(gt_set, dictYolo)

    if args.data2yolo:
        if not os.path.exists('./datasets'):
            os.makedirs('./datasets', exist_ok=True)

        if args.set is not None and args.path_annots is not None and args.path_images is not None:
            if os.path.exists(args.path_annots) and os.path.exists(args.path_images):
                data2yolov5(args.set, args.path_annots, args.path_images)
                print('Data in format YoloV5 cretated in datasets/tfm')
                exit()

    if not os.path.exists('./runs'):
        os.makedirs('./runs', exist_ok=True)

    source = args.source

    if not os.path.exists(source):
        print("Error Dir not found")
        exit()

    elif args.detect:

        if args.model == 'yolov5':
            path = "./runs/yolov5"
            path = checkdir(path)
            #print (path)
            run(weights=checkpoints['yolov5'],
                source=args.source, project=path, name='./')

        if args.model == 'ssd':
            path = "./runs/ssd"
            path = checkdir(path)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            # print(path)
            runssd(source, path)
            print("\nResults saved to: ", path)

        if args.model == 'retina':
            path = "./runs/retina"
            path = checkdir(path)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            detect(checkpoints['retina'], source, 0.5, path, 'retina')
            print("\nResults saved to: ", path)

        if args.model == 'fcos':
            path = "./runs/fcos"
            path = checkdir(path)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            detect(checkpoints['fcos'], source, 0.5, path)
            print("\nResults saved to: ", path)
