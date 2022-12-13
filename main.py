import argparse
from yolov5.detect import run
import os
from os import path as osPath
from SSD.detect import runssd
from FCOS_RETINA import detect
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
    parser.add_argument('--model', type=str, default='yolov5', required=True)
    parser.add_argument('--source', type=str, default='./data', required=True)
    args = parser.parse_args()

    if not os.path.exists('./runs'):
        os.makedirs('./runs', exist_ok=True)

    source = args.source

    if not os.path.exists(source):
        print("Error Dir not found")
        exit()

    else:

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
