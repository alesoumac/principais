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

class ImageVC():
    def __init__(self):
        self._imagem = None
        self._extensao = '.png'
    
    @property
    def dimensions(self):
        '''Retorna o par (altura, largura) da imagem
        '''
        return self._imagem.shape[:2]

    @property
    def height(self):
        '''Retorna a altura da imagem
        '''
        return self._imagem.shape[0]

    @property
    def width(self):
        '''Retorna a largura da imagem
        '''
        return self._imagem.shape[1]

    @property
    def depth(self):
        '''Retorna a profundidade da imagem
        '''
        return self._imagem.shape[2] if len(self._imagem.shape) > 2 else 1

    def load_base64(self, str_b64):
        '''Carrega uma imagem a partir de uma string codificada em Base64
        '''
        self._imagem = Image.open(io.BytesIO(base64.b64decode(str_b64)))
        try:
            deg = {3:180, 6:270, 8:90}.get(self._imagem._getexif().get(274, 0),0)
        except:
            deg = 0
            
        self._imagem = np.array(self._imagem.convert("RGB"))
        if deg != 0:
            self._imagem = self.rotate_image(deg)

    def adjust_angle(self,angulo,limiar_de_proximidade = 2):
        '''Retorna o ângulo ajustado. O ângulo será ajustado de maneira que fique entre -180 e 180 graus.
        Além disso, se o angulo estiver próximo dos valores 0, 90, -90, 180 ou -180, então ele é aproximado para esses valores.
        ''' 
        while angulo > 180: angulo -= 360
        while angulo <= -180: angulo += 360
        if abs(angulo) < limiar_de_proximidade: angulo = 0
        if abs(angulo - 90) < limiar_de_proximidade: angulo = 90
        if abs(angulo + 90) < limiar_de_proximidade: angulo = -90
        if 180 - abs(angulo) < limiar_de_proximidade: angulo = 180
        return angulo

    def rotate_image(self,degrees):
        '''Retorna a imagem rotacionada. O parãmetro degrees indica o grau de rotação (ex. 0, 90 ou -35.7)
        '''
        angle = self.adjust_angle(degrees)
        if angle == 0: return self._imagem.copy()
        h, w = self.dimensions
        dp = self.depth
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
        grim = cv2.cvtColor(self._imagem, cv2.COLOR_BGR2GRAY) if dp == 3 else self._imagem
        mediana = np.mean(grim)
        grim = None
        borda = (mediana,mediana,mediana) if dp == 3 else mediana
        # rotate image and return
        return cv2.warpAffine(self._imagem, M, (nW, nH), borderValue=borda)

    def crop_image(self, min_x, min_y, max_x, max_y):
        '''Retorna a imagem recortada. Os parâmetros min_x, min_y, max_x e max_y definem a área de recorte.
        '''
        alt,larg = self.dimensions
        min_x = int(max(min_x, 0))
        min_y = int(max(min_y, 0))
        max_x = int(min(max_x, larg))
        max_y = int(min(max_y, alt))
        return self._imagem[min_y:max_y, min_x:max_x]

    def resize_image(self,boundX,boundY,useMax = False):
        '''Retorna a imagem redimensionada. Os parâmetros boundX e boundY definem, respectivamente,
        a largura e a altura de um retângulo
        '''
        alt,larg = self.dimensions
        escala = max(float(boundY) / larg, float(boundX) / alt) if useMax \
            else min(float(boundY) / larg, float(boundX) / alt)
        if escala == 1: return self._imagem.copy()
        interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
        return cv2.resize(self._imagem, None, fx=escala, fy=escala, interpolation = interpolacao)

    def apply_threshold(self, threshold = None):
        imgg = cv2.cvtColor(self._imagem,cv2.COLOR_RGB2GRAY) if self.depth > 1 else self._imagem.copy()
        thr,imgg = cv2.threshold(imgg,127,255,cv2.THRESH_OTSU)
        return thr,imgg

