import os
import sys
import cv2
import numpy as np
import imutils

FACE_CASCADE = None
EYE_CASCADE = None

def get_image_shape(img):
    alt = img.shape[0]
    larg = img.shape[1]
    try:
        prof = img.shape[2]
    except:
        prof = 1
    return alt,larg,prof

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
    
def create_plain_image(height, width, bckcolor=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((height,width), bckcolor[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

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

def find_contours(img, modo, metodo, cnts=None, hier=None, ofst=None):
    tupla = cv2.findContours(img,modo,metodo,contours=cnts,hierarchy=hier,offset=ofst)
    if len(tupla) == 2:
        return tupla[0]
    else:
        return tupla[1]

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

def get_contour_box(cnt):
    box = cv2.minAreaRect(cnt)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    return np.array(box,dtype="int")

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

    img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if prof == 1 else img.copy()
    cor_media = (
        int(round(np.mean(img2[:,:,0]))),
        int(round(np.mean(img2[:,:,1]))),
        int(round(np.mean(img2[:,:,2])))
    )

    blank = create_plain_image(alt,larg,(0,0,0))
    imgm = create_plain_image(alt,larg,cor_media)
    espaco_y = max_d // 8
    for n,cnt in enumerate(cnt_linhas):
        # ---- Opção 1: Contour Box
        box = get_contour_box(cnt)
        cv2.drawContours(blank,[box],0,(255,255,255),cv2.FILLED)

    img2 = cv2.bitwise_and(img2, cv2.bitwise_not(blank))
    imgm = cv2.bitwise_and(imgm, blank)
    img2 = img2 + imgm
    return img2

def face_recognize_with_angle(img):
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
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if get_image_shape(img)[2] > 1 else img.copy()
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
    imgo = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if get_image_shape(img)[2] == 1 else img.copy()
    for (x, y, w, h) in faces:
        menor_t = min(w,h)
        qt = menor_t // 5
        cv2.circle(imgo,(x+w//2,y+h//2),menor_t//2 + qt,(64,64,72),-1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = imgo[y:y + h, x:x + w]

        eyes = EYE_CASCADE.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(roi_color,(ex+ew//2,ey+eh//2),min(ew,eh)//2,(84,84,84),-1)

    return imgo

def anonymize(img):
    img1 = face_recognize(img)
    img1 = find_possible_lines(img1)
    return img1

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
        ang2,_ = face_recognize_with_angle(imgr)
        ang = ang + ang2
    return ang, blank

def inicializa_anonym(path_classifier):
    global FACE_CASCADE
    global EYE_CASCADE
    
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(path_classifier,'haarcascade_frontalface_default.xml'))
    EYE_CASCADE = cv2.CascadeClassifier(os.path.join(path_classifier,'haarcascade_eye_tree_eyeglasses.xml'))
