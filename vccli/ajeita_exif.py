# -*- coding: utf-8 -*-
import argparse
import os
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
import exifread

CONTADOR_ARQUIVOS = 0

def force_directory(direc):
    if not os.path.exists(direc):
        path,_ = os.path.split(direc)
        if force_directory(path):
            try:
                os.mkdir(direc)
            except:
                return False
    else:
        if not os.path.isdir(direc):
            return False
    
    return True

def vl(tags,nome):
    try:
        if nome in tags.keys():
            return tags[nome].values
        else:
            return ""
    except:
        return ""

def get_tags(arquivo):
    if os.path.isdir(arquivo):
        return None
    f = open(arquivo,"rb")
    try:
        tags = exifread.process_file(f)
    except:
        tags = None
    f.close()
    return tags

def rotate_bound(imge, angle):
    while angle < 0: angle += 360.0
    while angle >= 360.0: angle -= 360.0
    if abs(angle) < 2:
        return imge

    if abs(angle - 90.0) < 2: angle = 90
    if abs(angle - 180.0) < 2: angle = 180
    if abs(angle - 270.0) < 2: angle = 270

    # grab the dimensions of the image 
    h, w = imge.shape[:2]
    dp = 1 if len(imge.shape) < 3 else imge.shape[2]
    # determine the center
    (cX, cY) = (w // 2, h // 2)

    if angle in [90,180,270]:
        M = np.array([[0.0, -1.0, (w+h) / 2.0], [1.0, 0.0, (h-w) / 2.0]] if angle == 90 else \
            [[-1.0, 0.0, w], [0.0, -1.0, h]] if angle == 180 else \
            [[0.0, 1.0, (w-h) / 2.0], [-1.0, 0.0, (h+w) / 2.0]])
    else:
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # calculate border color
    grim = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY) if dp == 3 else imge
    mediana = np.mean(grim)
    borda = (mediana,mediana,mediana) if dp == 3 else mediana

    # rotate image and return
    imgO = cv2.warpAffine(imge, M, (nW, nH), borderValue=borda)
    return imgO

def intdef(s,default = 0):
    try:
        return int(s)
    except:
        return default

def ajeitaArquivo(nomearq):
    global CONTADOR_ARQUIVOS
    novonome = ""
    nomearq = os.path.abspath(nomearq)
    nme,ext = os.path.splitext(nomearq)
    if ext.lower() == ".bmp": return
    if ext.lower() not in [".jpg",".jpeg",".png",".webp"]: return
    print("\033[KVerificando {}".format(nomearq) ) #, end='\r')
    tags = get_tags(nomearq)
    img = cv2.imread(nomearq)
    novaext = ".bmp"
    novonome = f"{nme}{novaext}"
    if tags is None: 
        print("Sem orientação")
        return
    orientacao = intdef(vl(tags,'Image Orientation'))
    print(f"Orientação = {orientacao}")
    transformacao = [0,0,0,180,180,90,90,-90,-90]
    angulo = transformacao[orientacao] if orientacao in range(9) else 0
    if angulo == 0: return
    CONTADOR_ARQUIVOS += 1
    #img = rotate_bound(img,angulo)
    #img = rotate_bound(img,-angulo)
    cv2.imwrite(novonome,img)
    print(f"Novo arquivo {novonome}")
    return

def ajeitaRecursivo(pasta):
    pasta = os.path.split(os.path.join(pasta,'teste'))[0]
    print("Pasta", pasta)
    for root, _, files in os.walk(pasta):
        if pasta != root: continue
        print(files)
        for fi in files:
            ajeitaArquivo(os.path.join(root,fi))
    return

# --------------------- Inicio do Programa
def main():
    global CONTADOR_ARQUIVOS
    # Defino argumentos da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("images_or_paths", nargs='+', type=str, help="Images or paths with images to be processed")
    args = parser.parse_args()

    arquivos = args.images_or_paths

    horaInicialProcesso = datetime.now()
    
    CONTADOR_ARQUIVOS = 0

    for nomearq in arquivos:
        if os.path.isdir(nomearq):
            ajeitaRecursivo(nomearq)
        else:
            ajeitaArquivo(nomearq)
    
    print("\033[KFim de execução")

    horaFinalProcesso = datetime.now()

    tempo_processamento = horaFinalProcesso - horaInicialProcesso
    print("Tempo de processamento = {}".format(tempo_processamento))
    print("Número de imagens válidas = {}".format(CONTADOR_ARQUIVOS))

if __name__ == "__main__":
    main()
