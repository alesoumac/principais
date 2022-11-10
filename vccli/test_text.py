import cv2
import numpy as np

def writeTextOnImage(txt, img=None, posicao=(2,2)):
    tam = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
    ml,ma = tam[0]
    if img is not None:
        cv2.putText(img,txt,(posicao[0],posicao[1]+ma),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
    next_pos = (posicao[0],posicao[1] + ma + 3)
    return ml,ma,next_pos

def createPlainImage(altura, largura, corDeFundo=(255,255,255)):
    newC = [None, None, None]
    for i in range(3):
        newC[i] = np.full((altura,largura), corDeFundo[i], dtype="uint8")
    plain_img = cv2.merge(newC)
    return plain_img

img = createPlainImage(200,200,(243,145,196))
writeTextOnImage('Tudo Azul',img)
cv2.imwrite('test_text.jpg',img)
