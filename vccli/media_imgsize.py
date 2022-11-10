#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import statistics

import cv2

ALTURAS = []
LARGURAS = []
FILESIZE = []

def verifyArquivo(arquivo):
    global ALTURAS
    global LARGURAS
    global FILESIZE

    try:
        fs = os.path.getsize(arquivo)
        img = cv2.imread(arquivo)
        h, w = img.shape[:2]
        ALTURAS += [h]
        LARGURAS += [w]
        FILESIZE += [fs]
    except:
        pass

def verifyRecursivo(pasta, listaExt, recursivo = False):
    for root, _, files in os.walk(pasta):
        if not recursivo and pasta != root:
            continue
        filessrt = sorted(files)
        for fi in filessrt:
            _,ex = os.path.splitext(fi)
            if ex[1:].lower() in listaExt:
                verifyArquivo(os.path.join(root,fi))
    return

# --------------------- Inicio do Programa
def main():
    global ALTURAS
    global LARGURAS
    global FILESIZE

    # Defino argumentos da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("images_or_paths", nargs='+', type=str, 
        help="Images or paths with images to be processed")
    args = parser.parse_args()

    arquivos = args.images_or_paths
    ALTURAS = []
    LARGURAS = []
    for nomearq in arquivos:
        if os.path.isdir(nomearq):
            verifyRecursivo(nomearq, ['bmp','jpg','jpeg','png','tif','tiff','webp'])
        else:
            verifyArquivo(nomearq)

    num_imgs = len(ALTURAS)
    print("Número de imagens processadas = {}".format(num_imgs))
    print("Média de Larg. x Alt. = ({}, {})".format(
        statistics.mean(LARGURAS), statistics.mean(ALTURAS)))
    print("Mediana de Larg. x Alt. = ({}, {})".format(
        statistics.median(LARGURAS), statistics.median(ALTURAS)))
    print("Média de Tamanho de Arquivo = {} bytes [stdev: {}]".format(
        statistics.mean(FILESIZE), statistics.stdev(FILESIZE)))

if __name__ == "__main__":
    main()
