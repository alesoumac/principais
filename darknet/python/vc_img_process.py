# Reestruturação da aplicacao
# -*- coding: utf-8 -*-
import base64
import io
import os

import cv2
import imutils
import numpy as np
from PIL import Image
from scipy.ndimage import interpolation
from skimage.filters import threshold_local

DNN_MODEL = None
FACE_CASCADE = None
EYE_CASCADE = None

# -----------
def get_depth(img_np):
    try:
        return img_np.shape[2]
    except:
        return 1

# -----------
def makeOTSU(img_np_rgb):
    try:
        imgg = cv2.cvtColor(img_np_rgb,cv2.COLOR_RGB2GRAY)
    except:
        return None,None
        
    thr,imgg = cv2.threshold(imgg,127,255,cv2.THRESH_OTSU)
    return thr,imgg

# -----------
def adjustAngle(angulo):
    while angulo > 180: angulo -= 360
    while angulo <= -180: angulo += 360
    if abs(angulo) < 2: angulo = 0
    if abs(angulo - 90) < 2: angulo = 90
    if abs(angulo + 90) < 2: angulo = -90
    if 180 - abs(angulo) < 2: angulo = 180
    return angulo

# -----------
def create_plain_image(height, width, bckcolor=(255,255,255)):
    if (type(bckcolor) is int) or (type(bckcolor) is float):
        backg = int(bckcolor)
        backg = [backg,backg,backg]
    else:
        backg = bckcolor

    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((height,width), backg[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

# -----------
def crop_image(image_np, min_x, min_y, max_x, max_y):
    alt,larg = image_np.shape[:2]
    min_x = int(max(min_x, 0))
    min_y = int(max(min_y, 0))
    max_x = int(min(max_x, larg))
    max_y = int(min(max_y, alt))
    crop_img_np = image_np[min_y:max_y, min_x:max_x]
    return crop_img_np

# -----------
def ResizeImage(img,boundHeight,boundWidth,useMax = False):
    alt,larg = img.shape[:2]
    if alt == 0 or larg == 0: return img.copy()

    if useMax:
        escala = max((boundWidth + 0.0) / larg, (boundHeight + 0.0) / alt)
    else:
        escala = min((boundWidth + 0.0) / larg, (boundHeight + 0.0) / alt)

    if escala == 1:
        return img.copy()

    if escala < 1:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    newImg = cv2.resize(img, None, fx=escala, fy=escala, interpolation = inter)
    return newImg

# -----------
def rotateImage(imge, angle):
    angle = adjustAngle(angle)
    if angle == 0:
        return imge

    # grab the dimensions of the image 
    h, w = imge.shape[:2] 
    dp = get_depth(imge)
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

# -----------
def changeImageCV2PIL(img_np_bgr):
    img = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# -----------
def save_image_to_bytes(PIL_Image, formato):
    with io.BytesIO() as outfile:
        PIL_Image.save(outfile, format=formato)
        return outfile.getvalue()

# -----------
def save_image_to_base64(PIL_Image, formato='JPEG'):
    img_bytes = save_image_to_bytes(PIL_Image, formato)
    return base64.b64encode(img_bytes).decode("utf-8")

# -----------
def get_image_bgr_from_bytes(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_np_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np_bgr

# -----------
def get_image_bgr_from_base64(img_data_base64):
    img_np_bgr = Image.open(io.BytesIO(img_data_base64))
    try:
        #uts.printLog(f"Informação EXIF { img_np_rgb._getexif().get(274, 0) }", self.logger)
        deg = {3:180, 6:270, 8:90}.get(img_np_bgr._getexif().get(274, 0),0)
    except:
        deg = 0
        
    img_np_bgr = np.array(img_np_bgr.convert("RGB"))
    if deg != 0:
        img_np_bgr = rotateImage(img_np_bgr,deg)
    img_np_bgr = cv2.cvtColor(img_np_bgr,cv2.COLOR_RGB2BGR)
    return img_np_bgr

# -----------
def find_contours(img, modo, metodo, cnts=None, hier=None, ofst=None):
    tupla = cv2.findContours(img,modo,metodo,contours=cnts,hierarchy=hier,offset=ofst)
    if len(tupla) == 2:
        return tupla[0]
    else:
        return tupla[1]

# -----------
def borderRemoval1(image_np_rgb):
    profundidade = get_depth(image_np_rgb)
    rows,cols = image_np_rgb.shape[:2]
    gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY) if profundidade > 1 else image_np_rgb

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
   
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
   
    # Specify size on horizontal axis
    horizontal_size = cols // 10
    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
   
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
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
   
    contours = find_contours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    min_x = 0
    max_x = cols
    max_h = 0
   
    for _, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        #minimum height
        if h >= center_x:
            #uts.printLog(f"{x}, {y}, {w}, {h}")
            if x < center_y:
                #left
                if max_h < h: # and min_x < x + w
                    min_x = x + w
                    max_h = h
            else:
                #right
                if max_x > x:
                    max_x = x

    rect = min_x, min_y, max_x, max_y
    return rect

# -----------
def borderRemoval2(image, min_area = 1800, min_height = 25, min_width = 22):
    # apply image contrast enhance
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    enhanced = clahe.apply(gray)

    T = threshold_local(enhanced, 29, offset=35, method="gaussian", mode="mirror")
    thresh = (enhanced > T).astype("uint8") * 255
    
    # invert image
    thresh = cv2.bitwise_not(thresh)

    # find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    biggest_area = 0
    biggest_cnt = []
    rectangle = []
    # loop trhough
    for c in cnts:
        rect = cv2.minAreaRect(c)
        (_, _),(rh, rw),_ = rect
        if (rh > 0):
            ratio = float(rw)/rh
            area = rw*rh
            if (area > biggest_area and area > min_area and rh > min_height and rw > min_width and (ratio > 1 or ratio < 0.5)):
                # add to the rois list
                biggest_area = area
                biggest_cnt = c
    
    if len(biggest_cnt) > 0:
        rectangle = cv2.boundingRect(biggest_cnt)
        x, y, w, h = rectangle
        x += 3
        w -= 6
        y += 3
        h -= 6
        rectangle = x, y, x+w, y+h
    
    return rectangle

# -----------
def borderRemoval3(img_np_rgb):
    # apply image contrast enhance
    altura,largura = img_np_rgb.shape[:2]
    profundidade = get_depth(img_np_rgb)

    gray = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY) if profundidade > 1 else img_np_rgb
    _,threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    countours = find_contours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank = create_plain_image(altura,largura,255)
    faixas = find_possible_lines(gray)

    for cnt in countours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        okcnt = False
        if w > 0.8*largura and h > 0.5*altura:
            blank = create_plain_image(altura,largura,255)
            cv2.drawContours(blank,[cnt],0,(0,0,0),cv2.FILLED)
            for (y1f,y2f,_) in faixas:
                if y2f <= y or y2f > y+h: continue
                proporcao = (y2f-y) / h
                if proporcao > 0.5: continue
                proporcao = int(proporcao * 1000) / 10.0
                crop_faixa = crop_image(blank,x+w//10,y1f,x+w-w//10,y2f)
                crop_faixa = cv2.bitwise_not(crop_faixa)
                #print(f"Faixa ({y1f} - {y2f}): está a {proporcao}% da moldura com média = {np.mean(crop_faixa)}")
                if np.mean(crop_faixa) > 10:
                  cv2.rectangle(blank,(x+w//10,y1f),(x+w-w//10,y2f),(0,0,0),cv2.FILLED)
            break
        if w > 0.8*largura or h > 0.75*altura:
            okcnt = True
        else:
            okcnt = (w > largura // 10 or h > altura // 2) and (x+w < largura // 8 or x > largura*0.875 or y+h < altura // 8 or y > altura*0.875)
        if not okcnt:
            continue
        cv2.drawContours(blank,[cnt],0,(0,0,0),cv2.FILLED)
    return borderRemoval1(blank)

# -----------
def deskew(image_np_rgb, delta=1, limit=6):
    def determine_score(arr, angle):
        data = interpolation.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    best_angle = 0
    rotated_np_rgb = image_np_rgb
    try:
        gray = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            _, score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image_np_rgb.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated_np_rgb = cv2.warpAffine(image_np_rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    except:
        best_angle = 0
        rotated_np_rgb = image_np_rgb

    return best_angle, rotated_np_rgb

# -----------
def align_face_image(img_np_bgr):
    bgr = img_np_bgr
    (x1,y1,x2,y2,n,_,_,_) = reconhece_face_dnn(bgr,False)
    if n == 0:
        return bgr
    (h,w) = get_image_shape(bgr)[:2]
    mw = int(abs(x2-x1) / 2)
    mh = int(abs(y2 - y1) / 2)
    x1 = max(0, x1 - mw)
    y1 = max(0, y1 - mh)
    x2 = min(w, x2 + mw)
    y2 = min(h, y2 + mh)
    faceForHaar = imutils.resize(bgr[y1:y2, x1:x2], width=256)
    
    angulo = face_align_haar(faceForHaar)

    return adjustAngle(angulo)

# -----------
def get_contour_box(cnt):
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    return np.array(box,dtype="int")

# -----------
def is_int(a):
    try:
        if isinstance(a,str):
            return False
        _ = int(a)
        return True
    except:
        return False

# -----------
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

# -----------
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
        atg = - (np.arctan2(y1-y2, x1-x2) * 180.0 / np.pi)
        if 180 - abs(atg) <= 45:
            atg += 180 if atg < 0 else -180
        angulo = atg

    box_width = int(round(max(l)))
    box_height = int(round(min(l)))
    return box.astype("int"), box_height, box_width, angulo

# -----------
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

# -----------
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
        imgr = rotateImage(img, ang)
        ang2 = face_align_haar(imgr)
        ang = ang + ang2
    return adjustAngle(ang)

# -----------
def reconhece_face_dnn(img,use_padding = True):
    global DNN_MODEL
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    DNN_MODEL.setInput(blob)
    res = DNN_MODEL.forward()
    facebox = (0,0,0,0)
    areabox = 0
    for i in range(res.shape[2]):
        confianca = res[0,0,i,2]
        if confianca > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1,y1,x2,y2) = box.astype('int')
            area = abs(x2-x1) * abs(y2-y1)
            if area > areabox:
                facebox = (x1,y1,x2,y2)
                areabox = area
    if areabox > 0:
        (x1,y1,x2,y2) = facebox
        w = x2-x1
        h = y2-y1
        if use_padding:
            xpadd = w * 0.4
            y1padd = h * 0.25
            y2padd = h * 0.3
        else:
            xpadd = y1padd = y2padd = 0
        return x1-xpadd, y1-y1padd, x2+xpadd, y2+y2padd, 1, 1, 1, 1
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0

# -----------
def shrinkToWidth(img, desired_width = 800, try_resize=False):
    if not try_resize:
        return 1.0, img

    h, w = img.shape[:2]
    scale = desired_width / w
    
    if scale > 1.0:
        return 1.0, img

    width = int(w * scale)
    height = int(h * scale)

    # dsize
    dsize = (width, height)
    return scale, cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

# -----------
def calculateAngleBetweenPoints(A,B):
    return np.degrees( np.arctan2(float(B[1] - A[1]), float(B[0] - A[0])) )

# -----------
def get_image_shape(img):
    alt = img.shape[0]
    larg = img.shape[1]
    try:
        prof = img.shape[2]
    except:
        prof = 1
    return alt,larg,prof

# -----------
def face_align_haar(img, ajusta_olhos = False):
    global FACE_CASCADE
    global EYE_CASCADE
    
    encontrou = False
    angles = [0,90,-90,180]
    melhor_ang = 0
    melhor_area = 0
    for _ in range(2):
        for ang in angles:
            imgr = rotateImage(img, ang)
            gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY) if get_image_shape(imgr)[2] > 1 else imgr
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
            if len(faces) <= 0: continue
            
            for (x, y, w, h) in faces:
                area = w*h
                if area > melhor_area:
                    melhor_ang = ang
                    melhor_area = area
                    if ajusta_olhos:
                        F = (int(x + w/2), int(y + h/2))
                        roi_gray = gray[y:y + h, x:x + w]
                        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
                        if len(eyes) == 2:
                            (ex, ey, ew, eh) = eyes[0]
                            A = (int(ex + ew/2), int(ey + eh/2))
                            (ex, ey, ew, eh) = eyes[1]
                            B = (int(ex + ew/2), int(ey + eh/2))
                            angAB = calculateAngleBetweenPoints(A,B)
                            angAF = calculateAngleBetweenPoints(A,F)
                            difFB = adjustAngle(angAF - angAB)
                            somaAng = 180 if difFB > 0 else 0
                            melhor_ang += (180 - angAB) + somaAng

                encontrou = True
        if encontrou: break
        angles = [45,-45,135,-135]

    return melhor_ang

# -----------
def find_possible_lines(img):
    '''
    Essa função retorna uma lista com as faixas do eixo Y, 
    onde estão localizadas as possíveis linhas com texto da imagem.
    A busca é feita usando função de contorno do OpenCV, e fazendo 
    alguns cálculos de proporção de altura da possível linha. 
    Esse método pode eventualmente não encontrar nenhuma linha,
    e nesse caso, retornará uma lista vazia de faixas.
    Cada faixa possui 3 elementos:
      - Y inicial (posição superior da linha)
      - Y final (posição inferior da linha)
      - Y do meio (posição do meio da linha)
    A lista já vem em ordem crescente do 2º elemento da faixa (Y final)
    Ex.: 
      Retorno = [(10,20,15), (22,32,27), ...]
    '''
    _,_,prof = get_image_shape(img)
    imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if prof > 1 else img.copy()
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

    faixas = []
    espaco_y = max_d // 8
    for cnt in cnt_linhas:
        (_,y1,_,h) = cv2.boundingRect(cnt)
        if h <= espaco_y: continue
        y2 = y1+h
        faixas += [[y2,y1]]
    for i in range(len(faixas)):
        for j in range(len(faixas)):
            if i == j: continue
            y2,y1 = faixas[i]
            y4,y3 = faixas[j]
            if y3 >= y2 or y1 >= y4: continue
            y1 = min(y1,y3)
            y2 = max(y2,y4)
            faixas[i] = faixas[j] = [y2,y1]
    faixas = sorted(faixas)
    faixas = [(y1,y2,(y1+y2) // 2) for n,(y2,y1) in enumerate(faixas) if n == 0 or y2 > faixas[n-1][0]]
    return faixas

# def align_cnh_by_perspective(img,centros):
#     pontos_ref = {
#         "width": 800,
#         "height": 1074,
#         "foto_CNH": (273, 330),
#         "nome_CNH": (432, 173),
#         "identidade_CNH": (550, 219),
#         "cpf_CNH": (487, 268),
#         "nascimento_CNH": (638, 268),
#         "filiacao_CNH": (550, 352),
#         "registro_CNH": (270, 488),
#         "validade_CNH": (458, 488),
#         "pri_habilitacao_CNH": (619, 488),
#         "local_emissao_CNH": (356, 836),
#         "data_emissao_CNH": (628, 836)
#     }

#     height, width = pontos_ref["height"], pontos_ref["width"]
#     p1 = []
#     p2 = []
#     for obj in centros:
#         if obj not in pontos_ref: continue
#         p1 += [ np.array(centros[obj]) ]
#         p2 += [ np.array(pontos_ref[obj]) ]
#     p1 = np.array(p1)
#     p2 = np.array(p2)

#     homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
#     transf_img = cv2.warpPerspective(img, homography, (width, height))
#     return transf_img

# -----------
def inicializa_image_models(path_prog):
    global DNN_MODEL
    global FACE_CASCADE
    global EYE_CASCADE

    # Model and config file downloaded from https://github.com/vardanagarwal/Proctoring-AI/tree/master/models 
    # Arquivos de configuração do modelo OpenCV DNN para reconhecimento facial
    models_dir = os.getenv('VCDOC_API_FACES_CAMINHO')
    if models_dir is None:
        models_dir = os.path.join(path_prog, "models")

    modelfile_dnn = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    cfgfile_dnn = os.path.join(models_dir,"deploy.prototxt")

    DNN_MODEL = cv2.dnn.readNetFromCaffe(cfgfile_dnn, modelfile_dnn)
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))
    EYE_CASCADE = cv2.CascadeClassifier(os.path.join(models_dir, 'haarcascade_eye_tree_eyeglasses.xml'))
