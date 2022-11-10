# -*- coding: utf-8 -*-

import os
import sys
import cv2
import math
import numpy as np
import pytesseract as tess
from scipy.ndimage import interpolation
import argparse


ESPACO_MINIMO_IMG_INTER = 7


def delete_file(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False

def delete_files(path):
    for r,d,f in os.walk(path):
        for arq in f:
            delete_file(os.path.join(r,arq))

def getOsPathName(arquivo, newExt = None, newPath = None, addToName = None, addToExt = None):
    nomePath,nomeArq = os.path.split(arquivo)
    novoNome,extParte = os.path.splitext(nomeArq)

    if newExt == None:
        newExt = extParte

    if addToExt != None:
        newExt = newExt + addToExt
    
    if newPath == None:
        newPath = nomePath

    if addToName != None:
        novoNome = novoNome + addToName

    if newExt:
        novoNome = novoNome + newExt

    if newPath:
        novoNome = os.path.join(newPath, novoNome)

    return novoNome

def forceDirectoy(direc, deletingFiles = True):
    try:
        os.makedirs(direc)
    except:
        pass
    if deletingFiles:
        try:
            delete_files(direc)
        except:
            pass
    
def diagonal(linha,y1=None,x2=None,y2=None):
    x1 = linha
    if y1 == None and x2 == None and y2 == None:
        while not isint(x1[0]):
            x1 = x1[0]
        y1 = x1[1]
        x2 = x1[2]
        y2 = x1[3]
        x1 = x1[0]
    c1 = abs(x1-x2)
    c2 = abs(y1-y2)
    return (c1*c1 + c2*c2) ** 0.5
    
def getImgProps(img):
    alt = img.shape[0]
    larg = img.shape[1]
    try:
        prof = img.shape[2]
    except:
        prof = 1
    return alt,larg,prof

def copy_image(image):
    return image.copy()

def makeOTSU(img):
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if getImgProps(img)[2] > 1 else img
    thr,imgg = cv2.threshold(imgg,127,255,cv2.THRESH_OTSU)
    return thr,imgg

def resize_image(img,boundX,boundY,useMax = False):
    alt,larg,prof = getImgProps(img)
    if useMax:
        escala = max((boundY + 0.0) / larg, (boundX + 0.0) / alt)
    else:
        escala = min((boundY + 0.0) / larg, (boundX + 0.0) / alt)

    if escala == 1:
        return copy_image(img)

    if escala < 1:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    newImg = cv2.resize(img, None, fx=escala, fy=escala, interpolation = inter)
    return newImg

def adaptiveThresholdGauss(img,kind):
    blockSize = kind + 9 + (kind & 1)
    pc = (kind >> 1) + 2
    if getImgProps(img)[2] > 1:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blockSize,pc)

def writeTextOnImage(txt, img=None, posicao=(ESPACO_MINIMO_IMG_INTER,ESPACO_MINIMO_IMG_INTER), fs=None):
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1 if fs is None else fs
    thick = 2
    tam = cv2.getTextSize(txt,fontFace,fontScale,thick)
    ml,ma = tam[0]
    if img is not None:
        cv2.putText(img,txt,(posicao[0],posicao[1]+ma),fontFace,fontScale,(0,0,0),thick)
    next_pos = (posicao[0],posicao[1] + ma + ESPACO_MINIMO_IMG_INTER)
    return ml,ma,next_pos

def createPlainImage(altura, largura, corDeFundo=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((altura,largura), corDeFundo[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

def select_foreground_contours(contornos,alt,larg):
    novo_contornos = []
    for cnt in contornos:
        (x,y,w,h) = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or w == 0 or h == 0 or x+w == larg or y+h == alt:
            continue
        novo_contornos += [cnt]
    return novo_contornos

titulo_anterior = ""
seqTitulo = 0

def showImageWithContour(titulo,img,contour = None):
    global titulo_anterior
    global seqTitulo

    return

    if contour is not None:
        img2 = copy_image(img)
        if getImgProps(img2)[2] == 1: img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img2,[contour],0,(0,0,255),cv2.FILLED)
    else:
        img2 = img

    if titulo != titulo_anterior:
        titulo_anterior = titulo
        seqTitulo = 1
    else:
        seqTitulo += 1

    if seqTitulo > 25:
        return
        
    alt,larg,_ = getImgProps(img2)
    if alt > 480 or larg > 480:
        img2 = resize_image(img2,480,480)
    wname = "{} [{}]".format(titulo,seqTitulo)
    cv2.imshow(wname,img2)
    cv2.waitKey()
    if seqTitulo > 1:
        cv2.destroyWindow(wname)

def addPixel(a_ini,value,min_a,max_a):
    r = a_ini + value
    if r < min_a:
        r = min_a
    elif r > max_a:
        r = max_a
    return r

def find_possible_letters(img):
    def sort_dic(x):
        return {k: v for k, v in sorted(x.items(), reverse=True, key=lambda item: item[1])}
    # -----

    imgo = img
    try:
        cv2.destroyAllWindows()
    except:
        pass
    showImageWithContour("Img Original",imgo)
    # -----
    alt,larg,prof = getImgProps(imgo)

    imgg = cv2.cvtColor(imgo,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(imgo)
    _,threshold = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # -----
    showImageWithContour("Img Inv+OTSU",threshold)
    # -----
    
    #tipo_retr = cv2.RETR_EXTERNAL
    tipo_retr = cv2.RETR_TREE

    contours,_ = cv2.findContours(threshold, tipo_retr, cv2.CHAIN_APPROX_SIMPLE)
    #contours = select_foreground_contours(contours,alt,larg)
    lista_h = []
    lista_q = []
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3: continue
        (x1,y1,w,h) = cv2.boundingRect(cnt)
        h_hull = h
        if w < h: w,h = h,w
        if float(h) / float(w) > 0.67:
            lista_q += [h_hull]
        lista_h += [h_hull]

    d_qud = sort_dic( {x:lista_q.count(x) for x in set(lista_q)} )
    d_min = sort_dic( {x:lista_h.count(x) for x in set(lista_h)} )
    
    max_d = min_d = 0
    if d_qud == {}:
        d_qud = d_min

    try:
        for i in range(2):
            for item in d_qud.items():
                n = item[0]
                if max_d == 0:
                    max_d = n + 6
                    min_d = n - 6
                else:
                    max_d = max(n+6,max_d)
                    min_d = min(n-6,min_d)
                break
            d_min = {}
            for item in d_qud.items():
                if item[0] not in range(min_d,max_d+1):
                    d_min[item[0]] = item[1]
            d_qud = d_min
    except:
        pass
    print("Altura de letra Min:{} Max:{}".format(min_d, max_d))
    
    blank = createPlainImage(alt,larg)
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3: continue
        (x1,y1,w,h) = cv2.boundingRect(cnt)
        x2,y2 = x1+w, y1+h
        xm = int(x1 + w / 2)
        ym = int(y1 + h / 2)
        h_hull = h
        if w < h: w,h = h,w
        if h_hull in range(min_d,max_d+1):
            #cv2.drawContours(blank,[cnt],0,(0,0,0),cv2.FILLED)
            cv2.circle(blank,(xm,ym),2,(0,0,255),thickness=1)
            cv2.rectangle(blank, (x1,y1), (x2,y2), (255,192,128),1)

    # -----
    showImageWithContour("IMAGEM DE RETORNO",blank)
    # -----
    return blank, min_d, max_d

def criaFundoPadrao(altura,largura):
    c = np.full((altura,largura), 96, dtype="uint8")
    for a in range(400):
        for b in range(400):
            if (a+b) & 1: c[a,b] = 192
    fundo = cv2.merge([c,c,c])
    return fundo

def makeShowImage(img=None,titulo=None):
    TAMANHO = 500
    fundo = criaFundoPadrao(TAMANHO,TAMANHO)
    if titulo is not None:
        _,at,_ = writeTextOnImage(titulo,fundo,fs=0.5)
        at += 4
    else:
        at = 0
    if img is not None:
        img = resize_image(img,TAMANHO-at,TAMANHO-2)
        na,nl,p = getImgProps(img)
        if p == 1: img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        px = (TAMANHO - nl) // 2
        py = (TAMANHO - at - 1 - na) // 2 + at + 1
        # aqui tem crop tambem
        fundo[py:py+na,px:px+nl] = img

    return fundo

if __name__ == "__main__":
    path_prog = sys.path[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+', type=str, help="Images to be processed")
    args = parser.parse_args()
    filenames = args.images
    print(filenames)
    for filename in filenames:
        print("==================================================================================")
        print("Processando imagem \"{}\"".format(filename))
        print("==================================================================================")
        img = cv2.imread(filename)
        if img is None:
            continue

        a,l,_ = getImgProps(img)
        if a > 1600 or l > 1600:
            img = resize_image(img,1600,1600)
        a,l,p = getImgProps(img)
        print("Imagem: {}\nLargura x Altura: {} x {}\n".format(filename,l,a))
        imgg = adaptiveThresholdGauss(img, 12)
        _, imgg = makeOTSU(imgg)
        print("Letters in AGT")
        img_a,_,_ = find_possible_letters(imgg)
        print("\nLetters in Original")
        img_l,_,_ = find_possible_letters(img)

        imgg = makeShowImage(imgg, "AGT")
        img_a = makeShowImage(img_a, "Letters in AGT")
        img2 = makeShowImage(img, "Original Image")
        img_l = makeShowImage(img_l, "Letters in Original")

        imgf = np.vstack([
            np.hstack([imgg,img_a]),
            np.hstack([img2,img_l]),
            ])
        cv2.imshow('Images', imgf)
        cv2.waitKey(0)
