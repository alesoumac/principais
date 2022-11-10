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
    try:
        imgg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if getImgProps(img)[2] > 1 else img
    except:
        print("Deu galho")
        print(img.shape)
        
    thr,imgg = cv2.threshold(imgg,127,255,cv2.THRESH_OTSU)
    return thr,imgg

def resize_image(img,boundX,boundY,useMax = False):
    alt,larg,_ = getImgProps(img)
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

    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blockSize,pc)

def find_contours(img, modo, metodo, cnts=None, hier=None, ofst=None):
    tupla = cv2.findContours(img,modo,metodo,contours=cnts,hierarchy=hier,offset=ofst)
    if len(tupla) == 2:
        return tupla[0]
    else:
        return tupla[1]

def createPlainImage(altura, largura, corDeFundo=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((altura,largura), corDeFundo[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

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

def correct_skew(image, delta=1, limit=6, image_original = None):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    _,thresh = makeOTSU(image)

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        _, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if image_original is None:
        return best_angle, rotated
    else:
        (h, w) = image_original.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated_original = cv2.warpAffine(image_original, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return best_angle, rotated, rotated_original

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
