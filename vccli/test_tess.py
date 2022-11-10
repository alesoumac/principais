import os
import cv2
import math
import numpy as np
import pytesseract as tess
from scipy.ndimage import interpolation
import argparse

OUT_DIR = ""

def getImgProps(img):
    alt = img.shape[0]
    larg = img.shape[1]
    try:
        prof = img.shape[2]
    except:
        prof = 1
    return alt,larg,prof

def copy_image(image):
    return cv2.copyMakeBorder(image,0,0,0,0,cv2.BORDER_REPLICATE)

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

def getOsPathName(arquivo, newExt = None, newPath = None, addToPath = None, addToName = None, addToExt = None):
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

def createPlainImage(altura, largura, corDeFundo=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((altura,largura), corDeFundo[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

def resize_image(img,boundAlt,boundLarg,useMax = False):
    alt,larg,prof = getImgProps(img)
    if useMax:
        escala = max((boundLarg + 0.0) / larg, (boundAlt + 0.0) / alt)
    else:
        escala = min((boundLarg + 0.0) / larg, (boundAlt + 0.0) / alt)

    if escala == 1:
        return copy_image(img)

    if escala < 1:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    newImg = cv2.resize(img, None, fx=escala, fy=escala, interpolation = inter)
    return newImg

def rotate_bound(imge, angle, borderValue=None):
    while angle < 0: angle += 360.0
    while angle >= 360.0: angle -= 360.0
    if abs(angle) < 2:
        return imge

    if abs(angle - 90.0) < 2: angle = 90
    if abs(angle - 180.0) < 2: angle = 180
    if abs(angle - 270.0) < 2: angle = 270

    # grab the dimensions of the image 
    h, w, dp = getImgProps(imge)
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

    if borderValue is None:
        # calculate border color
        grim = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY) if dp > 1 else imge
        mediana = np.mean(grim)
        borda = (mediana,mediana,mediana) if dp == 3 else mediana
    else:
        borda = borderValue

    # rotate image and return
    imgO = cv2.warpAffine(imge, M, (nW, nH), borderValue=borda)
    return imgO

def correct_skew(image, delta=5, limit=45, borderValue=None):
    def determine_score(arr, angle):
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if getImgProps(image)[2] > 1 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(0, 2*limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    rotated = rotate_bound(image, best_angle, borderValue)

    return best_angle, rotated

'''
def correct_skew_2(image):
    alt,larg,prof = getImgProps(image)
    menor,maior = min(alt,larg),max(alt,larg)

    if menor < 8:
        return 0,copy_image(image)

    if maior <= 64:
        return correct_skew(image)
    
    divisor = 32 if maior <= 800 else 64
    divisor = min(divisor,menor)
    ndiv_alt = int(round(float(alt) / divisor))
    ndiv_larg = int(round(float(larg) / divisor))
    y2 = 0
    angulos = {}
    angulo_frequente = None
    for iy in range(ndiv_alt):
        y1 = y2
        y2 = int(round((float(iy+1)/ndiv_alt)*alt))
        x2 = 0
        for ix in range(ndiv_larg):
            x1 = x2
            x2 = int(round((float(ix+1)/ndiv_larg)*larg))
            angulo, _ = correct_skew(image[y1:y2,x1:x2])
            if angulo not in angulos:
                angulos[angulo] = 1
            if angulo_frequente is None:
                angulo_frequente = angulo
            else:
                if angulos[angulo_frequente] < angulos[angulo]:
                    angulo_frequente = angulo
    
    if angulo_frequente is None:
        angulo_frequente = 0
    return angulo_frequente, rotate_bound(image,angulo_frequente)
'''
def trata_tess_str(s):
    r = s.replace('\n','\\n').replace('\r',' ').replace('\t',' ')
    while '  ' in r:
        r = r.replace('  ',' ')
    r = ''.join([ch if ch.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ\\/.-0123456789 " else '' for ch in r])
    return r

def correct_skew_tess(image,simples=True):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if getImgProps(image)[2] > 1 else copy_image(image)
    #gray = resize_image(gray,560,560,True) # resize apenas para não perder muito tempo no processamento do tesseract
    maior_len = 10
    melhor_angulo = None
    if simples:
        s = trata_tess_str(tess.image_to_string(gray))
        maior_len = len(s)
        melhor_angulo = 0
        #print("Len(0) =", maior_len)
    else:
        limit = 45
        delta = 9
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            rotated = rotate_bound(gray,angle)
            s = trata_tess_str(tess.image_to_string(rotated))
            if len(s) > maior_len:
                maior_len = len(s)
                melhor_angulo = angle
    
    angle = melhor_angulo + 90
    rotated = rotate_bound(gray,angle)
    s = trata_tess_str(tess.image_to_string(rotated))
    #print("Len(90) =", len(s))
    if len(s) > maior_len:
            maior_len = len(s)
            melhor_angulo = angle

    angle = angle + 180
    rotated = rotate_bound(gray,angle)
    s = trata_tess_str(tess.image_to_string(rotated))
    #print("Len(270) =", len(s))
    if len(s) > maior_len:
            maior_len = len(s)
            melhor_angulo = angle

    return melhor_angulo, rotate_bound(image,melhor_angulo)

def adaptiveThresholdGauss(img,kind):
    blockSize = kind + 9 + (kind & 1)
    pc = (kind >> 1) + 2
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if getImgProps(img)[2] > 1 else img
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blockSize,pc)

def isint(a):
    try:
        if isinstance(a,str):
            return False
        _ = int(a)
        return True
    except:
        return False

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

def calcula_angulo_normalizacao(img, skeletonize = True, plot_line=False):
    #img = single_resize_image(img,1000,1000)
    '''
    #HSV para extrair a cor verde
    #https://stackoverflow.com/a/47483966/7690982
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    ## slice no verde
    imask = mask>0
    verde = np.zeros_like(img, np.uint8)
    verde[imask] = img[imask]
    cv2.imwrite('C:\\Users\\Desktop\\teste\\2.jpg', verde)

    #Threshold
    (canal_h, canal_s, canal_v) = cv2.split(verde)
    retval, threshold = cv2.threshold(canal_v, 130, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite('C:\\Users\\Desktop\\teste\\3.jpg', canal_v)
    '''
    
    grayAN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if getImgProps(img)[2] > 1 else img
    
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
        #imgOnes = np.ones((3000,4000,3), np.uint8)
        #imgOnes[imgOnes==1]=255
        #print("Número de linhas encontrado: {:d}".format(len(lines)))
        maxLine = max(lines,key=diagonal)
        x1,y1,x2,y2 = maxLine[0]
        atg = - (math.atan2(y1-y2, x1-x2) * 180.0 / np.pi)
        if 180 - abs(atg) <= 45:
            atg += 180 if atg < 0 else -180
        if plot_line:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)
    except:
        atg = 0.0
    return atg

def img_crop_contour(img,contorno,make_crop=True,return_inv_mask=False):
    alt,larg,prof = getImgProps(img)
    mask = createPlainImage(alt,larg,(0,0,0))
    cv2.drawContours(mask, [contorno], 0, (255,255,255), cv2.FILLED)
    if prof == 1: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cnt_hull = cv2.convexHull(contorno, False)
    area_hull = cv2.contourArea(cnt_hull)
    area_cnt = cv2.contourArea(contorno)
    try:
        percentHull = float(area_cnt) / float(area_hull) * 100
    except:
        percentHull = 0

    if make_crop:
        (min_x,min_y,max_x,max_y) = cv2.boundingRect(contorno)
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

def select_foreground_contours(contornos,alt,larg):
    novo_contornos = []
    for cnt in contornos:
        (x,y,w,h) = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or w == 0 or h == 0 or x+w == larg or y+h == alt:
            continue
        novo_contornos += [cnt]
    return novo_contornos

def encontre_borda_externa(img):
    alt,larg,prof = getImgProps(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    img2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    #gray = cv2.GaussianBlur(gray, (3,3), 0)
    #_,threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    threshold = adaptiveThresholdGauss(img, 12)
    contour_mode = cv2.RETR_EXTERNAL
    while True:
        contours,_ = cv2.findContours(threshold, contour_mode, cv2.CHAIN_APPROX_TC89_L1)
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

def apply_erosion(img):
    alt,larg,prof = getImgProps(img)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    # Taking a matrix of size 5 as the kernel
    #kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(imgg, kernel, iterations=1)
    #img_dilation = cv2.dilate(imgg, kernel, iterations=1)
    #img_back = cv2.erode(img_dilation, kernel, iterations=1)
    _, img_back_inv = cv2.threshold(img_erosion, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if prof > 1:
        img_back_inv = cv2.cvtColor(img_back_inv, cv2.COLOR_GRAY2BGR)
    img_back = cv2.bitwise_not(img_back_inv)
    imgg = cv2.bitwise_and(np.array(img * 0.92, np.uint8), img_back_inv) + cv2.bitwise_and(img, img_back)
    return imgg

def encontre_letras(img):
    def sort_dic(x):
        return {k: v for k, v in sorted(x.items(), reverse=True, key=lambda item: item[1])}

    img2 = resize_image(img,800,800)
    alt,larg,prof = getImgProps(img2)
    imgg = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img2)
    if prof == 1:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    _,imgg = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(imgg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lista_min = []
    lista_qud = []
    for cnt in contours:
        area_cnt = cv2.contourArea(cnt)
        if area_cnt < 3: continue

        cnt_hull = cv2.convexHull(cnt, False)
        (x1,y1,w,h) = cv2.boundingRect(cnt_hull)
        if w < h: w,h = h,w
        if float(h) / float(w) > 0.67:
            lista_qud += [h]
        lista_min += [h]

    d_qud = sort_dic( {x:lista_qud.count(x) for x in set(lista_qud)} )
    d_min = sort_dic( {x:lista_min.count(x) for x in set(lista_min)} )

    if d_qud == {}: d_qud = d_min
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
    blank = createPlainImage(alt,larg,(0,0,0))
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

def encontre_interesses(img, forceDepth = None):
    alt,larg,prof = getImgProps(img)
    if forceDepth is None or not (forceDepth in [1, 3]):
        forceDepth = prof
    img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if prof == 1 else copy_image(img)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    #gray = cv2.GaussianBlur(imgg, (3,3), 0)
    _,threshold = cv2.threshold(imgg, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    blank = createPlainImage(alt,larg)
    imgg = copy_image(img2)

    # Abaixo tem o comando findContours, usando a opção cv2.RETR_TREE, que traz mais contornos (um dentro do outro normalmente)
    # ... com hierarquia, tipo árvore.  
    contours,_ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1) 
    
    # # e esse outro abaixo usa a opção cv2.RETR_EXTERNAL, que traz menos contornos, somente os mais externos
    # contours,_ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
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
        img_crop,bbox,percentArea,percentHull = img_crop_contour(img2,cnt,return_inv_mask=True)
        if percentHull > 15 and percentHull * 0.26 < percentArea:
            cv2.drawContours(blank, [approx], 0, cor, thic)
        else:
            cv2.drawContours(imgg, [approx], 0, (255,255,255), cv2.FILLED)
        if True: # percentArea <= 33:
            intPercent = int(percentArea)
            intPercHull = int(percentHull)
            parea = "{:7.4f}".format(percentArea).replace('.','_')
            pareaHull = "{:7.4f}".format(percentHull).replace('.','_')
            ang = calcula_angulo_normalizacao(img_crop)
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
            cv2.imwrite(os.path.join(OUT_DIR,"hull_{:03d}_{}-area_{:03d}-{:05d}.png".format(intPercHull,pareaHull,intPercent,seqFile)),img_crop)

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
    _,imgg = cv2.threshold(imgg, 110, 255, cv2.THRESH_BINARY_INV)
    profg = getImgProps(imgg)[2]
    if profg != forceDepth:
        if profg == 1:
            imgg = cv2.cvtColor(imgg,cv2.COLOR_GRAY2BGR)
        else:
            imgg = cv2.cvtColor(imgg,cv2.COLOR_BGR2GRAY)
    return ang, imgg # blank, imgg

def encontre_cantos(img):
    alt,larg,prof = getImgProps(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img)
    img2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    blank = createPlainImage(alt,larg)
    img2 = np.array(blank*3/4 + img2/4, dtype="uint8")
    #_,threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

    '''
    # detect corners with the goodFeaturesToTrack function. 
    corners = cv2.goodFeaturesToTrack(gray, 20, 0.01, 10)
    print(corners)
    corners = np.int0(corners) 
    
    # we iterate through each corner,  
    # making a circle at each point that we think is a corner. 
    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(blank, (x, y), 8, (0,0,0), -1) 

    return blank
    '''
    
    '''
    eigen = cv2.cornerEigenValsAndVecs(gray, 15, 3)
    eigen = eigen.reshape(alt, larg, 3, 2)  # [[e1, e2], v1, v2]
    flow = eigen[:,:,2]
    vis = blank
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points =  np.dstack( np.mgrid[d/2:larg:d, d/2:alt:d] ).reshape(-1, 2)
    for x, y in np.int32(points):
        vx, vy = np.int32(flow[y, x]*d)
        cv2.line(vis, (x-vx, y-vy), (x+vx, y+vy), (0, 0, 0), 1, cv2.LINE_AA)

    return vis
    '''

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    blank[dst>0.01*dst.max()]=[0,0,0]
    return blank

def writeTextOnImage(txt, img=None, posicao=(1,1), fs=None):
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1 if fs is None else fs
    thick = 1
    tam = cv2.getTextSize(txt,fontFace,fontScale,thick)
    ml,ma = tam[0]
    if img is not None:
        cv2.putText(img,txt,(posicao[0],posicao[1]+ma),fontFace,fontScale,(0,0,0),thick)
    next_pos = (posicao[0],posicao[1] + ma)
    return ml,ma,next_pos

def criaFundoPadrao(altura,largura):
    c = np.full((altura,largura), 96, dtype="uint8")
    for a in range(400):
        for b in range(400):
            if (a+b) & 1: c[a,b] = 192
    fundo = cv2.merge([c,c,c])
    return fundo

def makeShowImage(img=None,titulo=None):
    TAMANHO = 400
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

def find_best_angle(img, usingOCR = False):
    alt,larg,prof = getImgProps(img)
    if alt > 640 and larg > 640:
        img2 = resize_image(img,640,640,True)
    else:
        img2 = copy_image(img)
    imgg = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) if prof > 1 else copy_image(img2)
    #imgg = adaptiveThresholdGauss(imgg, 8)
    imgg = apply_erosion(imgg)
    ang,_ = encontre_interesses(imgg,prof)
    img2 = rotate_bound(img2,ang,(255,255,255))
    if usingOCR:
        ang2, img2 = correct_skew_tess(img2)
        ang = ang+ang2
    return ang, img2

def red_enhance(img, threshold = None):
    if getImgProps(img)[2] == 1:
        return img
    
    if threshold is None:
        img_ci = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (threshold, _) = cv2.threshold(img_ci, 127, 255, cv2.THRESH_OTSU)

    b,g,r = cv2.split(img)
    media = np.array(b/3 + g/3 + r/3, dtype='uint8')
    media_bg = np.array(b/2 + g/2, dtype='uint8')
    media_br = np.array(b/2 + r/2, dtype='uint8')
    
    def _fator(n, valor):
        if valor > 0:
            return n + (255 - n) * valor
        else:
            return n * (1 + valor)
    
    FATOR = 0.08
    oi = [ [(_fator(media[x,y],-FATOR), _fator(media[x,y],-FATOR), _fator(media[x,y],-FATOR)) \
        if r[x,y] - int(media_bg[x,y]) > 32 and media[x,y] <= threshold \
        else (_fator(media[x,y], FATOR), _fator(media[x,y], FATOR), _fator(media[x,y], FATOR)) \
        if int(media_br[x,y]) - g[x,y] > 32 and media[x,y] > threshold
        else (media[x,y], media[x,y], media[x,y]) \
        for y in range(len(img[x]))] for x in range(len(img))]

    oi = np.array(oi, dtype='uint8')
    return oi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+', type=str, help="Images to be processed")
    args = parser.parse_args()
    filenames = args.images
    for filename in filenames:
        print("==================================================================================")
        print("Processando imagem \"{}\"".format(filename))
        print("==================================================================================")
        img = cv2.imread(filename)
        if img is None:
            continue
        '''
        alt,larg,_ = getImgProps(img)
        if alt > 640 and larg > 640:
            img = resize_image(img,640,640,True)
        img2 = red_enhance(img)
        img2 = apply_erosion(img2)
        ang, img3 = find_best_angle(img2,True)
        print("Ângulo de Correção:", ang)
        #img2 = encontre_letras(img2)

        fundo = makeShowImage()
        writeTextOnImage("Ângulo de Correção: {:5.2f}".format(ang),fundo)
        img1 = makeShowImage(img, "Input")
        img2 = makeShowImage(img2, "Erosion")
        img3 = makeShowImage(img3, "Best Angle")

        imgf = np.vstack([
            np.hstack([img1,fundo]),
            np.hstack([img2,img3]),
            ])
        cv2.imshow('Images', imgf)
        cv2.waitKey(0)

        continue
        '''
        OUT_DIR = getOsPathName(filename,addToPath='parts', addToExt='.dir')
        try:
            os.makedirs(OUT_DIR)
        except:
            pass

        try:
            delete_files(OUT_DIR)
        except:
            pass

        # read the image and get the dimensions

        alt,larg,_ = getImgProps(img)
        if alt > 640 and larg > 640:
            img = resize_image(img,640,640,True)
        
        imgg = adaptiveThresholdGauss(img, 8)
        _,imgo = cv2.threshold(imgg,127,255, cv2.THRESH_OTSU) 
        img_3 = encontre_cantos(imgg)
        ang, img_2 = encontre_interesses(imgg)
        img_5 = encontre_borda_externa(img)
        
        angFinal, img_4 = find_best_angle(img)
    

        #imgo = rotate_bound(imgo,ang)
        #ang2, imgo = correct_skew_tess(imgo)
        #img_4 = rotate_bound(img_4,ang2)
        
        #print("Angulo 1",ang)
        #print("Angulo de correção:", angFinal)
        #s = trata_tess_str(tess.image_to_string(imgo))
        #print(s)
        #print("")
        
        fundo = makeShowImage()
        imgg = makeShowImage(imgg, "Adaptive Gauss Threshold")
        img_3 = makeShowImage(img_3, "Corner Harris")
        img_2 = makeShowImage(img_2, "Find Contours")
        img_5 = makeShowImage(img_5, "Convex Hull")
        img_4 = makeShowImage(img_4, "Imagem Final")

        imgf = np.vstack([
            np.hstack([imgg,img_3,fundo]),
            np.hstack([img_2,img_5,img_4]),
            ])
        cv2.imshow('Images', imgf)
        cv2.waitKey(0) 

'''
#angulo = calcula_angulo_normalizacao(imgg)
angulo,_ = correct_skew_tess(img)
print("Angulo de rotação:",angulo)
imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgg = cv2.cvtColor(imgg,cv2.COLOR_GRAY2BGR)
imgg = rotate_bound(imgg,angulo)
'''




'''
d = tess.image_to_data(img, output_type=tess.Output.DICT)
print(d)
n_boxes = len(d['level'])
for i in range(n_boxes):
    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    pontoMeio = (x + int(w/2), y + int(h/2))
    cv2.circle(imgg,pontoMeio,2,(0,128,255),cv2.FILLED)
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', imgg)
cv2.waitKey(0)
'''

'''
h, w, _ = imgg.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = tess.image_to_boxes(imgg) # also include any config options you use
print(boxes)

# draw the bounding boxes on the image
for b in boxes.splitlines():
    b = b.split(' ')
    (x1,y1,x2,y2) = ( int(b[1]), h - int(b[2]), int(b[3]), h - int(b[4]) )
    if x1 > x2: x1,x2 = x2,x1
    if y1 > y2: y1,y2 = y2,y1
    alt =  y2 - y1
    larg = x2 - x1
    if alt == 0 or larg == 0:
        continue
    if float(min(alt,larg)) / float(max(alt,larg)) <= 0.1:
        continue
    xm = x1 + int(larg/2)
    ym = y1 + int(alt/2)
    pontoMeio = (xm, ym)
    cv2.circle(imgg,pontoMeio,2,(128,96,255),cv2.FILLED)
    cv2.line(imgg,(xm,y1),(xm,y2),(144,255,205),1)
    #img = cv2.rectangle(imgg, (x1,y1), (x2,y2), (0, 255, 0), 2)
    
# show annotated image and wait for keypress
cv2.imshow(filename, imgg)
cv2.waitKey(0)
'''