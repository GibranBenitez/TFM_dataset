import argparse 
from yolov5.detect import run
import os
from os import path as osPath
from SSD_artesanal.detect import runssd

checkpoints = {'yolov5':'checkpoints\yolo.pt','retina':'checkpoints\Checkpoint_FT_MM_Dataset_epoca_25_retina.pth.rar',
            'fcos':'checkpoints\Checkpoint_FT_MM_Dataset_epoca_30_fcos.pth.rar','ssd':'checkpoints\SSD_MMask_E_28.pth'}
 

def checkdir(filePath):
    """ Pasar como parametros: directorio de archivo y nombre de archivo completo """

    if os.path.exists(filePath):
        numb = 1
        while True:
            # Separa el nombre del archivo de su extensión colocando el numero en el medio (al final del nombre)
            newName = "{0}".format(filePath+'(' + str(numb) +')')

            # Si existe un archivo con ese nombre incrementa el numero
            if os.path.exists(newName):
                numb += 1
            else:
                # Devuelve el nombre modificado si el archivo existe
                return newName
    # Devuelve el nombre original si el archivo no existe
    return filePath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5',required=True)
    parser.add_argument('--source', type=str, default='./data/*.jpg',required=True)
    parser.add_argument('--to', type=str, default='./runs/',required=True)
    args = parser.parse_args()

    if not os.path.exists('./runs'):
        os.makedirs('./runs', exist_ok=True)

    if args.model == 'yolov5':
        path = "./runs/yolov5"
        path = checkdir(path)
        print (path)  
        run(weights = checkpoints['yolov5'],source='imgpruebas',project=path,name='./')

    if args.model == 'ssd':
        path = "./runs/ssd"
        path = checkdir(path)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        print(path)
        runssd('imgpruebas', path)

    if args.model == 'retina':
        path = "./runs/retina"
        path = checkdir(path)

    if args.model == 'fcos':
        path = "./runs/fcos"
        path = checkdir(path)
        

