#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys

import cv2
import imutils
import numpy as np
import pytesseract as tess
from scipy.ndimage import interpolation
from random import randint
#import albumentations.augmentations.transforms as TT


MINIMUM_TEXT_SPACE = 7
FACE_CASCADE = None
EYE_CASCADE = None


def transformation(img,transname):
    # if transname == "solarize":
    #     t = TT.Solarize()
    #     return t.apply(img,72)
    # elif transname == "fancy_pca":
    #     t = TT.FancyPCA()
    #     return t.apply(img, 0.2)
    # elif transname == "grid_dropout":
    #     t = TT.GridDropout() 
    #     return t.apply(img)
    # else:
    return img
        
        
def posterize(img):
    return img

def equalize(img):
    return img

def delete_file(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False

def delete_files(path):
    for r,_,f in os.walk(path):
        for arq in f:
            delete_file(os.path.join(r,arq))

def force_directory(direc, deletingFiles = True):
    if not os.path.exists(direc):
        path,_ = os.path.split(direc)
        if force_directory(path,False):
            try:
                os.mkdir(direc)
            except:
                return False
        
    if deletingFiles:
        try:
            delete_files(direc)
        except:
            pass
    
    return True

def manage_path_name(arquivo, newExt = None, newPath = None, addToPath = None, 
                     addToName = None, addToExt = None):
    nomePath,nomeArq = os.path.split(arquivo)
    novoNome,extParte = os.path.splitext(nomeArq)

    if newExt == None:
        newExt = extParte

    if addToExt != None:
        newExt = newExt + addToExt
    
    if newPath == None:
        newPath = nomePath

    if addToPath != None:
        newPath = os.path.join(newPath,addToPath)

    if addToName != None:
        novoNome = novoNome + addToName

    if newExt:
        novoNome = novoNome + newExt

    if newPath:
        novoNome = os.path.join(newPath, novoNome)

    return novoNome

def is_int(a):
    try:
        if isinstance(a,str):
            return False
        _ = int(a)
        return True
    except:
        return False
    
def treat_ocr_string(s):
    r = s.replace('\n','\\n').replace('\r',' ').replace('\t',' ')
    while '  ' in r:
        r = r.replace('  ',' ')
    r = ''.join([ch 
                 if ch.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ\\/,.-0123456789 " 
                 else '' 
                 for ch in r])
    return r

def euclidean_distance(linha,y1=None,x2=None,y2=None):
    x1 = linha
    if y1 == None and x2 == None and y2 == None:
        while not is_int(x1[0]):
            x1 = x1[0]
        y1 = x1[1]
        x2 = x1[2]
        y2 = x1[3]
        x1 = x1[0]
    c1 = abs(x1-x2)
    c2 = abs(y1-y2)
    return (c1*c1 + c2*c2) ** 0.5

def get_image_shape(img):
    alt = img.shape[0]
    larg = img.shape[1]
    try:
        prof = img.shape[2]
    except:
        prof = 1
    return alt,larg,prof

def copy_image(image):
    return image.copy()

def make_OTSU(img):
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if get_image_shape(img)[2] > 1 else img
    thr,imgg = cv2.threshold(imgg,127,255,cv2.THRESH_OTSU)
    return thr,imgg

def resize_image(img,bound_height,bound_width,use_max = False):
    alt,larg,prof = get_image_shape(img)
    if use_max:
        escala = max(float(bound_width) / larg, float(bound_height) / alt)
    else:
        escala = min(float(bound_width) / larg, float(bound_height) / alt)

    if escala == 1:
        return copy_image(img)

    if escala < 1:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    newImg = cv2.resize(img, None, fx=escala, fy=escala, interpolation = inter)
    return newImg

def adaptive_gaussian_threshold(img,kind):
    blockSize = kind + 9 + (kind & 1)
    pc = (kind >> 1) + 2
    if get_image_shape(img)[2] > 1:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blockSize,pc)

def single_image_contrast(img):
    _,_,prof = get_image_shape(img)
    lab = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if prof == 1 else img
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
    ll, aa, bb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(ll)
    limg = cv2.merge((cl,aa,bb))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if prof == 1:
        final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    return final

def write_text_on_image(txt, img=None, position=(MINIMUM_TEXT_SPACE,MINIMUM_TEXT_SPACE), fs=None):
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1 if fs is None else fs
    thick = 2
    tam = cv2.getTextSize(txt,fontFace,fontScale,thick)
    ml,ma = tam[0]
    if img is not None:
        cv2.putText(img,txt,(position[0],position[1]+ma),fontFace,fontScale,(0,0,0),thick)
    next_pos = (position[0],position[1] + ma + MINIMUM_TEXT_SPACE)
    return ml,ma,next_pos

def find_contours(img, modo, metodo, cnts=None, hier=None, ofst=None):
    tupla = cv2.findContours(img,modo,metodo,contours=cnts,hierarchy=hier,offset=ofst)
    if len(tupla) == 2:
        return tupla[0]
    else:
        return tupla[1]

def remove_bounding_boxes(img):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
   
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
   
    # Specify size on horizontal axis
    rows = vertical.shape[0]
    cols = horizontal.shape[1]
    horizontal_size = cols // 10
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
   
    contours = find_contours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    center_x = rows // 2
    center_y = cols // 2

    min_y = 0
    max_y = rows
   
    for _, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        #minimum width
        if w >= cols * 0.1:
            if y < center_x:
                #top
                if min_y < y + h:
                    min_y = y + h
            else:
                #bottom
                if max_y > y:
                    max_y = y
   
    # Specify size on vertical axis
    verticalsize = rows // 5
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
   
    contours = find_contours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    min_x = 0
    max_x = cols
    max_h = 0
   
    for _, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        #minimum height
        if h >= center_x:
            #print(x, y, w, h)
            if x < center_y:
                #left
                if max_h < h: # and min_x < x + w
                    min_x = x + w
                    max_h = h
            else:
                #right
                if max_x > x:
                    max_x = x
   
    crop_img = img[min_y:max_y, min_x:max_x]
   
    return crop_img

def correct_skew(image, delta=1, limit=6, original_image = None, border_value=None):
    def determine_score(arr, angle):
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if get_image_shape(image)[2] > 1 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    original_rotated = None
    if limit > 6:
        rotated = rotate_bound(image, best_angle, border_value)
        if original_image is not None:
            original_rotated = rotate_bound(original_image, best_angle, border_value)
    else:
        h,w,_ = get_image_shape(image)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        if original_image is not None:
            original_rotated = cv2.warpAffine(original_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if original_rotated is None:
        return best_angle, rotated
    else:
        return best_angle, rotated, original_rotated

def correct_skew_tess(image, simple=True):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if get_image_shape(image)[2] > 1 else copy_image(image)
    #gray = resize_image(gray,560,560,True) # resize apenas para não perder muito tempo no processamento do tesseract
    maior_len = 10
    melhor_angulo = None
    if simple:
        s = treat_ocr_string(tess.image_to_string(gray))
        maior_len = len(s)
        melhor_angulo = 0
        #print("Len(0) =", maior_len)
    else:
        limit = 45
        delta = 9
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            rotated = rotate_bound(gray,angle)
            s = treat_ocr_string(tess.image_to_string(rotated))
            if len(s) > maior_len:
                maior_len = len(s)
                melhor_angulo = angle
    
    angle = melhor_angulo + 90
    rotated = rotate_bound(gray,angle)
    s = treat_ocr_string(tess.image_to_string(rotated))
    #print("Len(90) =", len(s))
    if len(s) > maior_len:
            maior_len = len(s)
            melhor_angulo = angle

    angle = angle + 180
    rotated = rotate_bound(gray,angle)
    s = treat_ocr_string(tess.image_to_string(rotated))
    #print("Len(270) =", len(s))
    if len(s) > maior_len:
            maior_len = len(s)
            melhor_angulo = angle

    return melhor_angulo, rotate_bound(image,melhor_angulo)

def find_skeleton3(img):
    # https://stackoverflow.com/a/42846932/7690982
    skeleton = np.zeros(img.shape,np.uint8)
    eroded = np.zeros(img.shape,np.uint8)
    temp = np.zeros(img.shape,np.uint8)

    _, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

    iters = 0
    while(True):
        cv2.erode(thresh, kernel, eroded)
        cv2.dilate(eroded, kernel, temp)
        cv2.subtract(thresh, temp, temp)
        cv2.bitwise_or(skeleton, temp, skeleton)
        thresh, eroded = eroded, thresh # Swap instead of copy

        iters += 1
        if cv2.countNonZero(thresh) == 0:
            return (skeleton,iters,kernel)

def get_normalization_angle(img, skeletonize = True, plot_line=False):
    '''
    #https://stackoverflow.com/a/47483966/7690982
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    ## slice no verde
    imask = mask>0
    verde = np.zeros_like(img, np.uint8)
    verde[imask] = img[imask]

    #Threshold
    (canal_h, canal_s, canal_v) = cv2.split(verde)
    retval, threshold = cv2.threshold(canal_v, 130, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    '''
    
    grayAN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if get_image_shape(img)[2] > 1 else img
    
    #Transformada de Hough
    minLineLength = int(max(10, min(grayAN.shape[0],grayAN.shape[1]) / 20))
    maxLineGap = int(max(3,minLineLength / 10))

    if skeletonize:
        #Skeletonize
        skel, iters, kern = find_skeleton3(grayAN)
        skel = cv2.dilate(skel,kern,iterations = 4)
        lines = cv2.HoughLinesP(skel,1,np.pi/180,255,minLineLength,maxLineGap)
    else:
        _, threshold = cv2.threshold(grayAN, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(threshold,1,np.pi/180,255,minLineLength,maxLineGap)
    try:
        maxLine = max(lines,key=euclidean_distance)
        x1,y1,x2,y2 = maxLine[0]
        atg = - (math.atan2(y1-y2, x1-x2) * 180.0 / np.pi)
        if 180 - abs(atg) <= 45:
            atg += 180 if atg < 0 else -180
        if plot_line:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)
    except:
        atg = 0.0
    return atg

def create_plain_image(height, width, bckcolor=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((height,width), bckcolor[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

def select_foreground_contours(cnts,height,width):
    novo_contornos = []
    for cnt in cnts:
        (x,y,w,h) = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or w == 0 or h == 0 or x+w == width or y+h == height:
            continue
        novo_contornos += [cnt]
    return novo_contornos

previous_title = ""
seq_title = 0

def show_image_with_contour(title,img,contour = None):
    global previous_title
    global seq_title

    return

    if contour is not None:
        img2 = copy_image(img)
        if get_image_shape(img2)[2] == 1: img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img2,[contour],0,(0,0,255),cv2.FILLED)
    else:
        img2 = img

    if title != previous_title:
        previous_title = title
        seq_title = 1
    else:
        seq_title += 1

    if seq_title > 25:
        return
        
    alt,larg,_ = get_image_shape(img2)
    if alt > 480 or larg > 480:
        img2 = resize_image(img2,480,480)
    wname = "{} [{}]".format(title,seq_title)
    cv2.imshow(wname,img2)
    cv2.waitKey()
    if seq_title > 1:
        cv2.destroyWindow(wname)

def add_pixel(a_ini,value,min_a,max_a):
    r = a_ini + value
    if r < min_a:
        r = min_a
    elif r > max_a:
        r = max_a
    return r

def create_simple_background(height,width):
    c = np.full((height,width), 96, dtype="uint8")
    for a in range(height):
        for b in range(width):
            if (a+b) & 1: c[a,b] = 192
    fundo = cv2.merge([c,c,c])
    return fundo

def make_display_image(img=None,title=None):
    ALTURA = 600
    LARGURA = 600
    fundo = create_simple_background(ALTURA,LARGURA)
    if title is not None:
        _,at,_ = write_text_on_image(title,None,fs=0.5)
        at += 4
    else:
        at = 0
    if img is not None:
        img = resize_image(img,ALTURA-at,LARGURA-2)
        na,nl,p = get_image_shape(img)
        if p == 1: img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        px = (LARGURA - nl) // 2
        py = (ALTURA - at - 1 - na) // 2 + at + 1
        # aqui tem crop tambem
        fundo[py:py+na,px:px+nl] = img

    if title is not None:
        write_text_on_image(title,fundo,fs=0.5)

    return fundo

def get_contour_box(cnt):
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    return np.array(box,dtype="int")

def get_contour_min_rect(cnt):
    box = get_contour_box(cnt)
    l = [0,0,0,0]
    min_l = max_l = 0
    for i in range(4):
        j = (i + 1) & 3
        l[i] = euclidean_distance(box[i][0], box[i][1], box[j][0], box[j][1])
        if i == 0:
            continue
        if l[i] < l[min_l]: min_l = i
        if l[i] > l[max_l]: max_l = i

    box_width = l[max_l]
    box_height = l[min_l]
    if box_height / float(box_width) > 0.67:
        angulo = None
    else:
        i = max_l
        j = (i + 1) & 3
        x1,y1,x2,y2 = box[i][0], box[i][1], box[j][0], box[j][1]
        atg = - (math.atan2(y1-y2, x1-x2) * 180.0 / np.pi)
        if 180 - abs(atg) <= 45:
            atg += 180 if atg < 0 else -180
        angulo = atg

    box_width = int(round(max(l)))
    box_height = int(round(min(l)))
    return box.astype("int"), box_height, box_width, angulo

def find_min_max_letter_height_from_contours(cnts):
    def sort_dic(x):
        return {k: v for k, v in sorted(x.items(), reverse=True, key=lambda item: item[1])}
    # -----

    lista_h = []
    lista_q = []
    for cnt in cnts:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3: continue
        (x1,y1,w,h) = cv2.boundingRect(cnt)
        h_hull = h
        if w < h: w,h = h,w
        aspecto = float(h) / float(w)
        if aspecto > 0.67:
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
    if min_d < 0: min_d = 0
    return min_d, max_d

def rotate_bound(imge, angle, border_value=None):
    while angle < 0: angle += 360.0
    while angle >= 360.0: angle -= 360.0
    if abs(angle) < 1:
        return imge

    if abs(angle - 90.0) < 1: angle = 90
    if abs(angle - 180.0) < 1: angle = 180
    if abs(angle - 270.0) < 1: angle = 270

    # grab the dimensions of the image 
    h, w, dp = get_image_shape(imge)
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

    if border_value is None:
        # calculate border color
        grim = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY) if dp > 1 else imge
        mediana = np.mean(grim)
        borda = (mediana,mediana,mediana) if dp == 3 else mediana
    else:
        borda = border_value

    # rotate image and return
    imgO = cv2.warpAffine(imge, M, (nW, nH), borderValue=borda)
    return imgO

def find_best_angle(img,using_faces=True):
    alt,larg,prof = get_image_shape(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if prof > 1 else img
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    ites = 5
    edged = cv2.dilate(edged, None, iterations=ites)
    edged = cv2.erode(edged, None, iterations=ites)

    cnts = find_contours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_d,max_d = find_min_max_letter_height_from_contours(cnts)
    range_altura = range(min_d,max_d)

    angulos = []
    faixas = {}
    faixa_mais_frequente = None
    blank = create_plain_image(alt, larg)
    blank = blank - np.array((blank - cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)) * 0.5, dtype="uint8")
    for cnt in cnts:
        (_,_,w,h) = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        box,rh,rw,ang = get_contour_min_rect(cnt)
        if rh == 0 or rw == 0:
            continue

        aspecto = float(rh) / float(rw)
        ok_cnt = aspecto > 0.02 and \
            (h in range_altura \
            or w in range_altura \
            or rh in range_altura \
            or rw in range_altura)
        
        if not ok_cnt:
            continue
        
        cv2.drawContours(blank, [box], 0, (128,0,255))

        if ang is not None:
            faixa = round(ang / 5.0) * 5
            if ang != 0:
                if faixa in faixas: 
                    faixas[faixa] += rw+rh
                else:
                    faixas[faixa] = rw+rh
                    if faixa_mais_frequente is None:
                        faixa_mais_frequente = faixa

                if faixas[faixa] > faixas[faixa_mais_frequente]:
                    faixa_mais_frequente = faixa

                angulos += [ang]

    if faixa_mais_frequente is None:
        ang = 0
    else:
        angulos2 = []
        for ang in angulos:
            if round(ang/5.0)*5 == faixa_mais_frequente:
                angulos2 += [ang]
        ang = np.median(angulos2)
    if using_faces:
        imgr = rotate_bound(img, ang)
        ang2,_ = face_recognize_versao_lenta(imgr)
        ang = ang + ang2
    return ang, blank

def crop_contour(img,contour,make_crop=True,return_inv_mask=False):
    alt,larg,prof = get_image_shape(img)
    mask = create_plain_image(alt,larg,(0,0,0))
    cv2.drawContours(mask, [contour], 0, (255,255,255), cv2.FILLED)
    if prof == 1: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cnt_hull = cv2.convexHull(contour, False)
    area_hull = cv2.contourArea(cnt_hull)
    area_cnt = cv2.contourArea(contour)
    try:
        percentHull = float(area_cnt) / float(area_hull) * 100
    except:
        percentHull = 0

    if make_crop:
        (min_x,min_y,max_x,max_y) = cv2.boundingRect(contour)
        area_util = max_x * max_y
        max_x += min_x
        max_y += min_y
        if min_x > 0: min_x -= 1
        if min_y > 0: min_y -= 1
        if max_x < larg: max_x += 1
        if max_y < alt: max_y += 1
        img_crop = img[min_y:max_y, min_x:max_x]
        mask_crop = mask[min_y:max_y, min_x:max_x]
    else:
        img_crop = img
        mask_crop = mask
        min_x = 0
        max_x = larg
        min_y = 0
        max_y = alt
        area_util = larg*alt

    if return_inv_mask:
        img_ret = cv2.bitwise_not(mask_crop)
    else:
        img_ret = cv2.bitwise_and(img_crop, mask_crop) + cv2.bitwise_not(mask_crop)
    
    bbox = {'left': min_x, 'top': min_y, 'width': max_x-min_x, 'height': max_y-min_y}
    try:
        percentArea = 100.0 * (float(area_cnt)/area_util)
    except:
        percentArea = 0
    return img_ret, bbox, percentArea, percentHull

def find_external_borders(img):
    alt,larg,prof = get_image_shape(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    img2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    #gray = cv2.GaussianBlur(gray, (3,3), 0)
    #_,threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    threshold = adaptive_gaussian_threshold(img, 12)
    contour_mode = cv2.RETR_EXTERNAL
    while True:
        contours = find_contours(threshold, contour_mode, cv2.CHAIN_APPROX_TC89_L1)
        contours = select_foreground_contours(contours,alt,larg)

        # create hull array for convex hull points
        big_cnt = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                continue
            big_cnt.extend(cnt)
            cv2.drawContours(img2,[cnt],0,(0,0,255),1)
        
        if len(big_cnt) == 0:
            if contour_mode == cv2.RETR_EXTERNAL:
                contour_mode = cv2.RETR_TREE
                continue
            else:
                break

        big_cnt = np.array(big_cnt)
        cnt_hull = cv2.convexHull(big_cnt, False)
        cv2.drawContours(img2,[cnt_hull],0,(255,0,0),2)
        break

    return img2

def discretize_gray(img, to_white=False, threshold=None, darkness_increase = 0.15, lightness_increase = 0.75):
    imgo = copy_image(img) if get_image_shape(img)[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        thr,_ = make_OTSU(imgo)
    else:
        thr = threshold

    def trans_ci(x):
        if x <= thr: 
            return x - int(x * darkness_increase)
        else:
            return 255 if to_white else x + int((255-x) * lightness_increase)

    vtrans = np.vectorize(trans_ci)
    imgt = np.array(vtrans(imgo), dtype='uint8')
    return imgt

def red_enhance(img):
    if get_image_shape(img)[2] == 1:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0, 64, 48), (15, 255, 255)) \
         | cv2.inRange(hsv, (165, 64, 48), (180, 255, 255))
    mask_green = cv2.inRange(hsv, (45, 64, 48), (75, 255,255))
    gmask = (mask_red | mask_green) == 0
    rmask = mask_red > 0
    vmask = mask_green > 0
    reds = np.zeros_like(img, np.uint8)
    blank = reds.copy()
    blank = cv2.bitwise_not(blank)
    greens = blank.copy()
    reds[rmask] = img[rmask]
    reds = np.array(reds * 0.9, dtype=np.uint8)
    greens[vmask] = img[vmask]
    greens = blank - greens
    greens = np.array(greens * 0.2, dtype=np.uint8)
    greens = blank - greens
    reds[vmask] = greens[vmask]
    reds[gmask] = img[gmask]
    return reds

# def red_enhance(img, threshold):
#     if getImgProps(img)[2] == 1:
#         return img
    
#     if threshold is None:
#         threshold, _ = makeOTSU(img)

#     b,g,r = cv2.split(img)
#     media = np.array(b/3 + g/3 + r/3, dtype='uint8')
#     media_bg = np.array(b/2 + g/2, dtype='uint8')
#     media_br = np.array(b/2 + r/2, dtype='uint8')
    
#     def _fator(n, valor):
#         if valor > 0:
#             return n + (255 - n) * valor
#         else:
#             return n * (1 + valor)
    
#     FATOR = 0.08
#     oi = [ [(_fator(media[x,y],-FATOR), _fator(media[x,y],-FATOR), _fator(media[x,y],-FATOR)) \
#         if r[x,y] - int(media_bg[x,y]) > 32 and media[x,y] <= threshold \
#         else (_fator(media[x,y], FATOR), _fator(media[x,y], FATOR), _fator(media[x,y], FATOR)) \
#         if int(media_br[x,y]) - g[x,y] > 32 and media[x,y] > threshold
#         else (media[x,y], media[x,y], media[x,y]) \
#         for y in range(len(img[x]))] for x in range(len(img))]

#     oi = np.array(oi, dtype='uint8')
#     return oi

def apply_erosion_dilation(img, make_erosion=True, iters=1):
    alt,larg,prof = get_image_shape(img)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    # Taking a matrix of size 5 as the kernel
    #kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    if make_erosion:
        img_apply = cv2.erode(imgg, kernel, iterations=iters)
    else:
        img_apply = cv2.dilate(imgg, kernel, iterations=iters)
    _, img_back_inv = cv2.threshold(img_apply, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if prof > 1:
        img_back_inv = cv2.cvtColor(img_back_inv, cv2.COLOR_GRAY2BGR)
    img_back = cv2.bitwise_not(img_back_inv)
    imgg = cv2.bitwise_and(np.array(img * 0.92, np.uint8), img_back_inv) + cv2.bitwise_and(img, img_back)
    return imgg

CORNER_HARRIS_METHOD = "harris"
CORNER_EIGEN_METHOD = "eigen"
CORNER_GFTT_METHOD = "gftt"

def find_corners(img, method=CORNER_HARRIS_METHOD):
    alt,larg,prof = get_image_shape(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    img2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    blank = create_plain_image(alt,larg)
    img2 = np.array(blank*3/4 + img2/4, dtype="uint8")
    #_,threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

    if method == CORNER_GFTT_METHOD:
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.05, 10)
        corners = np.int0(corners) 
        
        # we iterate through each corner,  
        # making a circle at each point that we think is a corner. 
        for i in corners: 
            x, y = i.ravel()
            cv2.circle(img2, (x, y), 8, (255,0,255), 3)
            
    elif method == CORNER_EIGEN_METHOD:
        eigen = cv2.cornerEigenValsAndVecs(gray, 15, 5)
        eigen = eigen.reshape(alt, larg, 3, 2)  # [[e1, e2], v1, v2]
        flow = eigen[:,:,2]
        vis = img2
        vis[:] = (192 + np.uint32(vis)) / 2
        d = 12
        points =  np.dstack( np.mgrid[d/2:larg:d, d/2:alt:d] ).reshape(-1, 2)

        for x, y in np.int32(points):
            vx, vy = np.int32(flow[y, x]*d)
            cv2.line(vis, (x-vx, y-vy), (x+vx, y+vy), (255, 0, 255), 1, cv2.LINE_AA)

        img2 = vis.copy()
        
    elif method == CORNER_HARRIS_METHOD:  
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
    
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
    
        # Threshold for an optimal value, it may vary depending on the image.
        img2[dst>0.01*dst.max()]=[0,0,0]

    return img2

def find_letters(img):
    img2 = resize_image(img,800,800)
    alt,larg,prof = get_image_shape(img2)
    imgg = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img2)
    if prof == 1:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    _,imgg = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(imgg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    contours = find_contours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_d,max_d = find_min_max_letter_height_from_contours(contours)

    blank = create_plain_image(alt,larg,(0,0,0))
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3: continue
        cnt_hull = cv2.convexHull(cnt, False)
        (x1,y1,w,h) = cv2.boundingRect(cnt_hull)
        if w < h: w,h = h,w
        if h in range(min_d,max_d+1) and float(h) / float(w) > 0.12:
            cv2.drawContours(blank,[cnt],0,(255,255,255),cv2.FILLED)
    blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
    blank = cv2.bitwise_not( cv2.bitwise_and(imgg,blank) )
    blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
    return blank

def crop_image(image_np, min_x, min_y, max_x, max_y):
    alt,larg = image_np.shape[:2]
    min_x = int(max(min_x, 0))
    min_y = int(max(min_y, 0))
    max_x = int(min(max_x, larg))
    max_y = int(min(max_y, alt))
    crop_img_np = image_np[min_y:max_y, min_x:max_x]
    return crop_img_np

def find_possible_lines(img,return_crops = False):
    alt,larg,prof = get_image_shape(img)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    _,threshold = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = find_contours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rs = sorted([cv2.boundingRect(cnt) + (n,) for n,cnt in enumerate(contours) if cv2.contourArea(cnt) >= 20])
    min_d,max_d = find_min_max_letter_height_from_contours(contours)
    dist_letra = (max_d + min_d)*4 // 2
    linhas = []
    cnt_linhas = []
    for (x1,y1,w,h,n) in rs:
        #if w < h: w,h = h,w
        if h in range(min_d,max_d+1): # and float(h) / float(w) > 0.12:
            x2 = x1+w
            y2 = y1+h
            encontrou = False
            for i,(lx1,ly1,lx2,ly2) in enumerate(linhas):
                lym = (ly2-ly1) // 2
                lh = ly2-ly1
                difh = abs(h-lh) / float(lh)
                disx = x1-lx2
                if (difh <= 0.2) and (disx >= 0) and (disx <= dist_letra) and (y1 in range(ly1,ly1+lym+1) or y2 in range(ly2-lym,ly2+1)):
                    lx2 = x2
                    ly1 = min(ly1,y1)
                    ly2 = max(ly2,y2)
                    linhas[i] = (lx1,ly1,lx2,ly2)
                    cnt_linhas[i] = np.array(list(cnt_linhas[i]) + list(contours[n]))
                    encontrou = True
                    break
            if encontrou: continue
            linhas += [(x1,y1,x2,y2)]
            cnt_linhas += [contours[n]]

    img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if prof == 1 else img
    cor_media = (
        int(round(np.mean(img2[:,:,0]))),
        int(round(np.mean(img2[:,:,1]))),
        int(round(np.mean(img2[:,:,2])))
    )

    if return_crops:
        blank = create_plain_image(alt,larg,(0,0,0))
        faixas = []
        espaco_y = max_d // 8
        for cnt in cnt_linhas:
            (_,y1,_,h) = cv2.boundingRect(cnt)
            if h <= espaco_y: continue
            y2 = y1+h
            faixas += [(y1,y2)]
        faixas = sorted(faixas)
        for i in range(len(faixas)):
            if i == 0: continue
            x1,x2 = faixas[i-1]
            y1,y2 = faixas[i]
            if y1 <= x2:
                faixas[i-1] = None
                faixas[i] = (x1,max(x2,y2))
        crop_list = []
        text_list = ""
        for faixa in faixas:
            if faixa is None: continue
            (y1,y2) = faixa
            y1,y2 = max(y1-espaco_y,0), min(y2+espaco_y,alt)
            imgm = crop_image(img2,0,y1,larg,y2)
            crop_list += [imgm]
            text_list = text_list + tess.image_to_string(imgm) + '\n'
        return crop_list, text_list
    else:
        blank = create_plain_image(alt,larg,(0,0,0))
        imgm = create_plain_image(alt,larg,cor_media)
        espaco_y = max_d // 8
        for n,cnt in enumerate(cnt_linhas):
            # ---- Opção 1: Contour Box
            box = get_contour_box(cnt)
            cv2.drawContours(blank,[box],0,(255,255,255),cv2.FILLED)

            # ---- Opção 2: Extended Bounding Rect
            #soma_x = dist_letra
            #soma_y = max_d // 8

            #(x1,y1,w,h) = cv2.boundingRect(cnt)
            #x2,y2 = x1+w, y1+h
            #x1,y1 = max(0, x1-soma_x) , max(0, y1-soma_y)
            #x2,y2 = min(x2+soma_x, larg) , min(y2+soma_y, alt)
            #cv2.rectangle(blank,(x1,y1),(x2,y2),(255,255,255),cv2.FILLED)

            # ---- Opção 3: All-Width Bounding Rect
            #(_,y1,_,h) = cv2.boundingRect(cnt)
            #if h <= espaco_y: continue
            #cv2.rectangle(blank,(0,y1),(larg,y1+h),(255,255,255),cv2.FILLED)

        img2 = cv2.bitwise_and(img2, blank)
        imgm = cv2.bitwise_and(imgm,cv2.bitwise_not(blank))
        img2 = img2 + imgm
        return img2

def find_good_contours(img, force_depth = None):
    alt,larg,prof = get_image_shape(img)
    if force_depth is None or not (force_depth in [1, 3]):
        forceDepth = prof
    img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if prof == 1 else copy_image(img)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    #gray = cv2.GaussianBlur(imgg, (3,3), 0)
    _,threshold = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    blank = create_plain_image(alt,larg)
    imgg = copy_image(img2)

    # Abaixo tem o comando findContours, usando a opção cv2.RETR_TREE, que traz mais contornos (um dentro do outro normalmente)
    # ... com hierarquia, tipo árvore.  
    contours = find_contours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1) 
    
    # # e esse outro abaixo usa a opção cv2.RETR_EXTERNAL, que traz menos contornos, somente os mais externos
    # contours = find_contours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = select_foreground_contours(contours,alt,larg)
    #areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    #areas = areas[areas > 10.0] # ignoro áreas muito pequenas

    #area_media = np.mean(areas)
    #dez_pc = area_media * 2
    #area_mediana = np.median(areas)
    #range_media = range(int(round(area_media-dez_pc)), int(round(area_media+dez_pc)))
    seqFile = 0
    angulos = []
    faixas = {}
    faixa_mais_frequente = None
    for cnt in contours:
        thic = 1
        #thic = cv2.FILLED if area < 600 else 1 if area < 1800 else 2
        #approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
        approx = cnt
        # Checking the no. of sides of the selected region.
        #num_sides = len(approx)
        #cor = (192*bool(num_sides & 4), 192*bool(num_sides & 2), 192*bool(num_sides & 1))
        cor = (0,0,0)
        img_crop,bbox,percentArea,percentHull = crop_contour(img2,cnt,return_inv_mask=True)
        if percentHull > 15 and percentHull * 0.26 < percentArea:
            cv2.drawContours(blank, [approx], 0, cor, thic)
        else:
            cv2.drawContours(imgg, [approx], 0, (255,255,255), cv2.FILLED)
        if True: # percentArea <= 33:
            #intPercent = int(percentArea)
            #intPercHull = int(percentHull)
            #parea = "{:7.4f}".format(percentArea).replace('.','_')
            #pareaHull = "{:7.4f}".format(percentHull).replace('.','_')
            ang = get_normalization_angle(img_crop)
            faixa = round(ang / 5.0) * 5
            if ang != 0 and percentArea <= 33:
                if faixa in faixas: 
                    faixas[faixa] += 1
                    if faixas[faixa] > faixas[faixa_mais_frequente]:
                        faixa_mais_frequente = faixa
                else:
                    faixas[faixa] = 1
                    if faixa_mais_frequente is None or (faixa == 0 and faixas[faixa_mais_frequente] == 1):
                        faixa_mais_frequente = faixa
                angulos += [ang]
                #img_crop = rotate_bound(img_crop, ang, (255,255,255))

            seqFile += 1
            #cv2.imwrite(os.path.join(OUT_DIR,"hull_{:03d}_{}-area_{:03d}-{:05d}.png".format(intPercHull,pareaHull,intPercent,seqFile)),img_crop)

    #print("Áreas")
    #print(" - Min:",np.min(areas))
    #print(" - Max:",np.max(areas))
    #print(" - Media:",np.mean(areas))
    #print(" - Mediana:",np.median(areas))
    #print("Foram criados {} arquivos com subáreas da imagem".format(seqFile))
    if faixa_mais_frequente is None:
        ang = 0
    else:
        #print("Faixa mais frequente de ângulo de correção", faixa_mais_frequente)
        angulos2 = []
        for ang in angulos:
            if round(ang/5.0)*5 == faixa_mais_frequente:
                angulos2 += [ang]
        ang = np.median(angulos2)
    #_,imgg = make_OTSU(imgg)
    profg = get_image_shape(imgg)[2]
    if profg != forceDepth:
        if profg == 1:
            imgg = cv2.cvtColor(imgg,cv2.COLOR_GRAY2BGR)
        else:
            imgg = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)
    return ang, imgg # blank, imgg

def create_all_colors(nbits):
    try:
        n = int(nbits)
        if n < 1: n = 1
        if n > 8: n = 8
    except:
        n = 8
    
    npixels = 1 << int((n*3+1) / 2)
    img = create_plain_image(npixels, npixels)
    sequencial = 0
    for x in range(npixels):
        for y in range(npixels):
            b = g = r = 0
            for i in range(n):
                sl = i*3
                sr = i*2
                su = int(7*(i+1)/n) - i
                br = (sequencial & (1 << sl)) >> sr
                bg = (sequencial & (1 << (sl+1))) >> (sr+1)
                bb = (sequencial & (1 << (sl+2))) >> (sr+2)
                if su > 0:
                    br <<= su
                    bg <<= su
                    bb <<= su
                r += br
                g += bg
                b += bb
            img[x,y] = [b,g,r]
            sequencial += 1

    return img

def encontre_letras(img, expectedLines = 1, image_original = None):
    def addPixel(a_ini,value,min_a,max_a):
        r = a_ini + value
        if r < min_a:
            r = min_a
        elif r > max_a:
            r = max_a
        return r
    def sort_dic(x):
        return {k: v for k, v in sorted(x.items(), reverse=True, key=lambda item: item[1])}
    # -----

    imgo = img if image_original is None else image_original
    try:
        cv2.destroyAllWindows()
    except:
        pass
    show_image_with_contour("Img Original",imgo)
    # -----
    alt,larg,prof = get_image_shape(imgo)
    pedacinho = int(round(alt * 0.1))

    imgo = copy_image(imgo[pedacinho:,:])
    alt,larg,prof = get_image_shape(imgo)
    # -----
    show_image_with_contour("Img 90%",imgo)
    # -----

    imgg = cv2.cvtColor(imgo,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(imgo)
    _,threshold = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    # -----
    show_image_with_contour("Img Inv+OTSU",threshold)
    # -----

    contours = find_contours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = select_foreground_contours(contours,alt,larg)
    lista_h = []
    lista_q = []
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3: continue

        #cnt_hull = cv2.convexHull(cnt, False)
        #(x1,y1,w,h) = cv2.boundingRect(cnt_hull)
        (x1,y1,w,h) = cv2.boundingRect(cnt)

        h_hull = h
        if w < h: w,h = h,w
        if float(h) / float(w) > 0.67:
            lista_q += [h_hull]
            # -----
            show_image_with_contour("Provavel letra",imgg,cnt)
            # -----
        
        lista_h += [h_hull]

    d_qud = sort_dic( {x:lista_q.count(x) for x in set(lista_q)} )
    d_min = sort_dic( {x:lista_h.count(x) for x in set(lista_h)} )
    
    if expectedLines > 0:
        provaveis_linhas = []
        alt_linha = float(alt) / (expectedLines + 1)
        y_ini = alt_linha / 2.0
        for i in range(expectedLines):
            y_fim = y_ini + alt_linha
            provaveis_linhas += [(int(round(y_ini)), int(round(y_fim)))]
            y_ini = y_fim
        
        lista_cnt = []
        if d_qud == {}: d_qud = d_min
        for cnt in contours:
            if cv2.contourArea(cnt) <= 0:
                continue
            #cnt_hull = cv2.convexHull(cnt, False)
            #(x1,y1,w,h) = cv2.boundingRect(cnt_hull)
            (x1,y1,w,h) = cv2.boundingRect(cnt)
            #if h not in d_qud:
            #    continue
            x2,y2 = x1+w,y1+h
            if float(h) / alt_linha > 1.1:
                pl = expectedLines + 10
                # -----
                show_image_with_contour("NÃO ENTROU NA LINHA",imgg,cnt)
                # -----

            else:
                pl = -1
                ym = int(round((y1+y2) / 2.0))
                for i in range(expectedLines):
                    (y_ini,y_fim) = provaveis_linhas[i]
                    if ym in range(y_ini,y_fim):
                        pl = i
                        # -----
                        show_image_with_contour("Entrou na linha {}".format(pl+1),imgg,cnt)
                        # -----
                        break
            lista_cnt += [[pl,x1,y1,x2,y2,cnt]]
        
        reinicia_pl = []
        x_ini = x_fim = -1
        for i in range(expectedLines):
            reinicia_pl += [True]
        repete = False
        alt_linha = 0
        for _ in range(2):
            for (pl,x1,y1,x2,y2,cnt) in lista_cnt:
                if pl < 0 or pl >= expectedLines:
                    continue
                if x_ini == -1:
                    x_ini = x1
                    x_fim = x2
                else:
                    x_ini = min(x_ini,x1)
                    x_fim = max(x_fim,x2)
                if reinicia_pl[pl]:
                    reinicia_pl[pl] = False
                else:
                    (y_ini,y_fim) = provaveis_linhas[pl]
                    if (y1 <= y_fim and y2 >= y_ini):
                        y1 = min(y1, y_ini)
                        y2 = max(y2, y_fim)
                    else:
                        if y2-y1 < y_fim - y_ini:
                            y1 = y_ini
                            y2 = y_fim
                        else:
                            repete = True
                provaveis_linhas[pl] = (y1,y2)
                alt_linha = max(alt_linha, y2-y1)
            if not repete: break
        if x_ini >= 2: x_ini -= 2
        if x_fim <= larg-2: x_fim += 2
        alt_linha = int(round(alt_linha))+4
        corMedia = int(np.mean(imgg))
        corMedia = addPixel(corMedia,2,0,255)
        imgs_crop = create_plain_image(alt_linha,larg * expectedLines,(corMedia,corMedia,corMedia))
        img_crop = cv2.cvtColor(imgo,cv2.COLOR_GRAY2BGR) if prof == 1 else copy_image(imgo)
        x1 = x_ini = 0
        x2 = x_fim = larg
        #print("\n\nProvaveis Linhas=", provaveis_linhas)
        #print("Reinicia_PL=",reinicia_pl)
        #print("Altura da Linha =", alt_linha,"Altura da imagem =",alt)
        for i in range(expectedLines):
            if not reinicia_pl[i]:
                (y_ini,y_fim) = provaveis_linhas[i]
                y_ini = addPixel(y_ini,-2,0,alt)
                y_fim = addPixel(y_fim, 2,0,alt)
                y1 = int((alt_linha - y_fim + y_ini) / 2)
                y2 = y1 + y_fim - y_ini
                #print("X1=",x1," X2=",x2," X_ini=",x_ini," X_fim=",x_fim,sep="")
                #print("Y1=",y1," Y2=",y2," Y_ini=",y_ini," Y_fim=",y_fim,sep="")
                imgs_crop[y1:y2,x1:x2] = img_crop[y_ini:y_fim,x_ini:x_fim]
            x1 += larg
            x2 += larg
        _,imgs_crop = make_OTSU(imgs_crop)
        blank = imgs_crop
    else:
        max_d = min_d = 0
        try:
            for i in range(2):
                #print(d_qud.items())
                for item in d_qud.items():
                    #print(item)
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
        #print(min_d, max_d)
        
        blank = create_plain_image(alt,larg,(0,0,0))
        for cnt in contours:
            area_cnt = cv2.contourArea(cnt)
            if area_cnt < 3: continue
            #cnt_hull = cv2.convexHull(cnt, False)
            #(x1,y1,w,h) = cv2.boundingRect(cnt_hull)
            (x1,y1,w,h) = cv2.boundingRect(cnt)
            h_hull = h
            if w < h: w,h = h,w
            if h_hull in range(min_d,max_d+1) and float(h) / float(w) > 0.12:
                cv2.drawContours(blank,[cnt],0,(255,255,255),cv2.FILLED)
        blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        blank = cv2.bitwise_not( cv2.bitwise_and(imgg,blank) )
        blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        
    # -----
    show_image_with_contour("IMAGEM DE RETORNO",blank)
    # -----
    return blank

def face_recognize_versao_lenta(img):
    global FACE_CASCADE
    global EYE_CASCADE
    
    encontrou = False
    angles = [0,90,-90,180]
    melhor_ang = 0
    melhor_tam = -1
    melhor_area = 0
    str_f = ""
    str_e = ""
    list_a = []
    for _ in range(2):
        for ang in angles:
            imgr = rotate_bound(img, ang, border_value = (255,255,255))
            gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY) if get_image_shape(imgr)[2] > 1 else imgr
            imgo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    eyes = EYE_CASCADE.detectMultiScale(roi_gray)
                tam = len(faces) + len(eyes)
                area = w*h
                str_f += str(len(faces))
                str_e += str(len(eyes))
                list_a += [area]
                if (tam > melhor_tam) or (tam == melhor_tam and area > melhor_area):
                    melhor_ang = ang
                    melhor_tam = tam
                    melhor_area = area
                encontrou = True
            else:
                str_f += "0"
                str_e += "0"
                list_a += [0]
        if encontrou: break
        angles = [45,-45,135,-135]
        
    #print("Faces:",str_f)
    #print("Olhos:",str_e)
    #print("Areas:",list_a)
    ang = melhor_ang
    imgr = rotate_bound(img, ang, border_value = (255,255,255))
    gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY) if get_image_shape(imgr)[2] > 1 else imgr
    imgo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.circle(imgo,(x+w//2,y+h//2),min(w,h)//2,(255,0,0), 2)
        #cv2.rectangle(imgo, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = imgo[y:y + h, x:x + w]

        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(roi_color,(ex+ew//2,ey+eh//2),min(ew,eh)//2,(0,255,0),2)
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return ang, imgo

def face_recognize(img):
    global FACE_CASCADE
    global EYE_CASCADE
    
    encontrou = False
    angles = [0,180,90,-90]
    for _ in range(2):
        for ang in angles:
            imgr = rotate_bound(img, ang, border_value = (255,255,255))
            gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY) if get_image_shape(imgr)[2] > 1 else imgr
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                encontrou = True
                break
        if encontrou: break
        angles = [45,-45,135,-135,0]
        
    imgo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in faces:
        menor_t = min(w,h)
        qt = menor_t // 5
        cv2.circle(imgo,(x+w//2,y+h//2),menor_t//2+qt,(64,64,72),-1)
        #cv2.rectangle(imgo, (x-qt, y-qt), (x + w + qt, y + h + qt), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = imgo[y:y + h, x:x + w]

        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(roi_color,(ex+ew//2,ey+eh//2),min(ew,eh)//2,(84,84,84),-1)
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return ang, imgo

def compare_rate(s,t,related_to_s = True):
    positions = {}
    
    def get_p_t(i,j):
        try:
            return positions[t[i]][j]
        except:
            return 0
    
    if related_to_s and len(s) == 0: return 0.0
    if not related_to_s and len(t) == 0: return 0.0
    positions = {c : [i for i in range(len(s)) if s[i] == c] for c in set([a for a in t])}
    maiores = [ [] for i in t]
    mais_maior = 0
    mais_r = ''
    for i in range(len(t))[::-1]:
        for j in range(len(s)):
            m = get_p_t(i,j)
            if m == 0: break
            maior = 0
            r = ''
            for k in range(i+1,len(t)):
                for l in range(len(s)):
                    n = get_p_t(k,l)
                    if n == 0: break
                    if n > m and len(maiores[k][l]) > maior:
                        r = maiores[k][l]
                        maior = len(r)
            r = ''.join(t[i],r)
            maiores[i] += [r]
            if len(r) > mais_maior:
                mais_maior = len(r)
                mais_r = r
    print(mais_r)
    return float(mais_maior) / float(len(s) if related_to_s else len(t))

# ======================================================================
# ======================================================================

# ----------------------------------------------------- Inicio do Módulo

# ======================================================================
# ======================================================================

if __name__ == "__main__":
    path_prog = sys.path[0]
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(path_prog,'haarcascade_frontalface_default.xml'))
    EYE_CASCADE = cv2.CascadeClassifier(os.path.join(path_prog,'haarcascade_eye_tree_eyeglasses.xml'))

    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+', type=str, help="Images to be processed")
    args = parser.parse_args()
    n = 0
    for filename in args.images:
        print("==================================================================================")
        print("Processando imagem \"{}\"".format(filename))
        print("==================================================================================")
        img = cv2.imread(filename)
        if img is None:
            continue
    
        a,l,_ = get_image_shape(img)
        if a > 1600 or l > 1600:
            img = resize_image(img,1600,1600)
        a,l,p = get_image_shape(img)
        print("Imagem: {}\nLargura x Altura: {} x {}".format(filename,l,a))
        imgg = adaptive_gaussian_threshold(img, 12)
        _, imgg = make_OTSU(imgg)
        ang,_ = find_best_angle(img)
        print("Angulo calculado = {}".format(ang))
        img_b = rotate_bound(img, ang, border_value=(255,255,255))
        ang2, img_r = face_recognize(img_b)
        ang += ang2
        img_x = rotate_bound(img,ang)
        img_x = find_possible_lines(img_x)
        img_sol = transformation(img,"grid_dropout") # "solarize" ou "fancy_pca" ou "grid_dropout"
        img_pst = posterize(img)
        img_equ = equalize(img)
        #img_anon = anonymize(img)
        
        _,texto = find_possible_lines(img_x,True)
        print(texto)
        
        xch_ang = ''
        if ang != 0:
            xch_ang = " - Angle {}".format(ang)

        img_o = make_display_image(img, "Original Image")
        img_g = make_display_image(img_r, "Rotation + Face Recogn")
        img_l = make_display_image(img_x, "Lines"+xch_ang)
        img_s = make_display_image(img_sol, "grid_dropout")
        img_p = make_display_image(img_pst, "Posterize")
        img_e = make_display_image(img_equ, "Equalize")
        imgf = np.vstack([ 
            np.hstack([img_o,img_g,img_l]),
            np.hstack([img_s,img_p,img_e]),
            ])
        novo_nome = manage_path_name(filename, newExt = ".png", addToPath = "rodado")

        cv2.imwrite(novo_nome, img_b)    
        cv2.imshow('Images', imgf)
        cv2.waitKey(0)
