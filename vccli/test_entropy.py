"""
=======
Entropy
=======

In information theory, information entropy is the log-base-2 of the number of
possible outcomes for a message.

For an image, local entropy is related to the complexity contained in a given
neighborhood, typically defined by a structuring element. The entropy filter can
detect subtle variations in the local gray level distribution.

In the first example, the image is composed of two surfaces with two slightly
different distributions. The image has a uniform random distribution in the
range [-14, +14] in the middle of the image and a uniform random distribution in
the range [-15, 15] at the image borders, both centered at a gray value of 128.
To detect the central square, we compute the local entropy measure using a
circular structuring element of a radius big enough to capture the local gray
level distribution. The second example shows how to detect texture in the camera
image using a smaller structuring element.

"""
import os
import cv2
import numpy as np
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

def isint(a):
    try:
        if isinstance(a,str):
            return False
        _ = int(a)
        return True
    except:
        return False

def createPlainImage(altura, largura, corDeFundo=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((altura,largura), corDeFundo[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

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
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if getImgProps(img)[2] > 1 else img
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blockSize,pc)

def select_foreground_contours(contornos,alt,larg):
    novo_contornos = []
    for cnt in contornos:
        (x,y,w,h) = cv2.boundingRect(cnt)
        if x == 0 or y == 0 or x+w == larg or y+h == alt:
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
            contour_mode = cv2.RETR_TREE
            continue

        big_cnt = np.array(big_cnt)
        cnt_hull = cv2.convexHull(big_cnt, False)
        cv2.drawContours(img2,[cnt_hull],0,(255,0,0),2)
        break

    return img2

def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent

def get_entropy_image(img, useGray = True, threshold = None, nivel=5):
    def getEntropyColor(n):
        c = [0.0,0.0,0.0]
        if n < 64:
            cA = np.array([255,224,204], dtype=float)
            cB = np.array([255,128,0],dtype=float)
            na = 0
            nb = 64
        elif n < 128:
            cA = np.array([255,128,0],dtype=float)
            cB = np.array([128,255,32],dtype=float)
            na = 64
            nb = 128
        elif n < 192:
            cA = np.array([128,255,32],dtype=float)
            cB = np.array([0,192,192],dtype=float)
            na = 128
            nb = 192
        else:
            cA = np.array([0,192,192],dtype=float)
            cB = np.array([0,0,255],dtype=float)
            na = 192
            nb = 255
        cor = ((cB - cA) * n - cB * na + cA * nb) / (nb - na)
        cor = np.array(cor,dtype="uint8")
        return cor

    gray = adaptiveThresholdGauss(img, 12)
    alt,larg,_ = getImgProps(gray)
    N = nivel if isint(nivel) and nivel > 0 else 1
    max_nent = None
    min_nent = None
    E = copy_image(gray)
    E = np.array(E, dtype=float)
    Ecolor = createPlainImage(alt,larg,(0,0,0))
    for row in range(alt):
        for col in range(larg):
            Lx=np.max([0,col-N])
            Ux=np.min([larg,col+N])
            Ly=np.max([0,row-N])
            Uy=np.min([alt,row+N])
            region=gray[Ly:Uy,Lx:Ux].flatten()
            nent = entropy(region)
            if max_nent is None:
                max_nent = min_nent = nent
            else:
                if nent > max_nent: max_nent = nent
                if nent < min_nent: min_nent = nent
            E[row,col]=nent
    for row in range(alt):
        for col in range(larg):
            nent = round(255.0 * (E[row,col] - min_nent) / (max_nent - min_nent))
            if threshold is not None:
                if nent < threshold:
                    nent = 0.0
            if useGray:
                E[row,col] = nent
            else:
                Ecolor[row,col] = getEntropyColor(nent)
    
    if useGray:
        E = np.array(E,dtype="uint8")
        E = cv2.bitwise_not(E)
        return E
    else:
        return Ecolor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs='+', type=str, help="Images to be processed")
    args = parser.parse_args()

    filenames = args.images
    for filename in filenames:

        #OUT_DIR = getOsPathName(filename,addToPath='parts', addToExt='.dir')
        #try:
        #    os.makedirs(OUT_DIR)
        #except:
        #    pass

        #try:
        #    delete_files(OUT_DIR)
        #except:
        #    pass

        img = cv2.imread(filename)
        h,w,_ = getImgProps(img)
        if h > 560 or w > 560:
            img = resize_image(img,560,560,False)
        
        E = get_entropy_image(img,useGray = False, nivel=1, threshold = 224)
        BE = encontre_borda_externa(E)
        cv2.imshow('Image Entropy', E)
        cv2.imshow('Image Convex Hull', BE)
        cv2.waitKey(0)

