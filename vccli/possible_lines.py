import argparse
import cv2
import numpy as np
import imutils
import os
import sys

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
def find_contours(img, modo, metodo, cnts=None, hier=None, ofst=None):
    tupla = cv2.findContours(img,modo,metodo,contours=cnts,hierarchy=hier,offset=ofst)
    if len(tupla) == 2:
        return tupla[0]
    else:
        return tupla[1]

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
def get_contour_box(cnt):
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    return np.array(box,dtype="int")

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
def rotate_image(imge, angle):
    angle = ajusta_angulo(angle)
    if angle == 0:
        return imge

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

    # calculate border color
    grim = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY) if dp == 3 else imge
    mediana = np.mean(grim)
    borda = (mediana,mediana,mediana) if dp == 3 else mediana

    # rotate image and return
    imgO = cv2.warpAffine(imge, M, (nW, nH), borderValue=borda)
    return imgO

# -----------
def ajusta_angulo(angulo):
    while angulo > 180: angulo -= 360
    while angulo <= -180: angulo += 360
    if abs(angulo) < 2: angulo = 0
    if abs(angulo - 90) < 2: angulo = 90
    if abs(angulo + 90) < 2: angulo = -90
    if 180 - abs(angulo) < 2: angulo = 180
    return angulo

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
        imgr = rotate_image(img, ang)
        ang2 = 0 # face_align_haar(imgr)
        ang = ang + ang2
    return ajusta_angulo(ang)

def find_possible_lines(img):
    alt,larg,prof = get_image_shape(img)
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
        faixas += [(y2,y1)]
    faixas = sorted(faixas)
    faixas = [(y1,y2,(y1+y2) // 2) for (y2,y1) in faixas]
    # for i in range(len(faixas)):
    #     if i == 0: continue
    #     x1,x2 = faixas[i-1]
    #     y1,y2 = faixas[i]
    #     if y1 <= x2:
    #         faixas[i-1] = None
    #         faixas[i] = (x1,max(x2,y2))
    return faixas

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
def resize_image(img,boundX,boundY,useMax = False):
    alt,larg = img.shape[:2]
    if useMax:
        escala = max((boundY + 0.0) / larg, (boundX + 0.0) / alt)
    else:
        escala = min((boundY + 0.0) / larg, (boundX + 0.0) / alt)

    if escala == 1:
        return img.copy()

    if escala < 1:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    newImg = cv2.resize(img, None, fx=escala, fy=escala, interpolation = inter)
    return newImg

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
        ang = find_best_angle(img)
        img_x = rotate_image(img, ang)
        a,l,_ = get_image_shape(img_x)
        metade = a // 2
        print(f"Imagem: {filename}\nLargura x Altura: {l} x {a}\nMeio: {metade}")
        print("Angulo calculado = {}".format(ang))
        faixas = find_possible_lines(img_x)
        yma = None
        ymb = None
        for (y1,y2,_) in faixas:
            if y2 > metade:
                ymb = y1-2 if ymb is None or ymb > y1-2 else ymb
            else:
                yma = y2+2 if yma is None or yma < y2+2 else yma
        
        if yma is None and ymb is None:
            yma = ymb = metade
        elif yma is None:
            yma = ymb
        elif ymb is None:
            ymb = yma
        elif ymb > yma:
            dist_a = abs(yma - metade)
            dist_b = abs(ymb - metade)
            if dist_a < dist_b:
                ymb = yma
            else:
                yma = ymb
        print(f"Meio = { (0,yma) } e { (ymb,a) }")
        imga = crop_image(img_x,0,0,l,yma)
        imgb = crop_image(img_x,0,ymb,l,a)
        cv2.imshow('Image A', imga)
        cv2.imshow('Image B', imgb)
        cv2.waitKey(0)