# -*- coding: utf-8 -*-
import argparse
import os
import time
from datetime import datetime, timedelta
import cv2

CONTADOR_ARQUIVOS_VALIDOS = 0

def force_directory(direc):
    if not os.path.exists(direc):
        path,_ = os.path.split(direc)
        if force_directory(path):
            try:
                os.mkdir(direc)
            except:
                return False
    else:
        if not os.path.isdir(direc):
            return False
    
    return True

def ajeitaArquivo(nomearq):
    global CONTADOR_ARQUIVOS_VALIDOS
    
    try:
        nomearq = os.path.abspath(nomearq)
    except:
        pass

    print("\033[KVerificando {}".format(nomearq), end='\r')

    parte_nome,parte_ext = os.path.splitext(nomearq)
    if parte_ext == "":
        parte_ext = ".jpg"
        os.rename(nomearq, nomearq + parte_ext)
        nomearq = nomearq + parte_ext
    
    arquivo_valido = True
    pasta_destino = ""
    
    img = None
    try:
        img = cv2.imread(nomearq)
        alt,larg = img.shape[:2]
    except:
        arquivo_valido = False
        pasta_destino = '_excluir/0'

    if arquivo_valido:
        if alt < 300 and larg < 300:
            arquivo_valido = False
            pasta_destino = '_excluir/1'
        elif larg < 450:
            arquivo_valido = False
            pasta_destino = '_excluir/2'
    
    if not arquivo_valido:
        parte_pasta,parte_nome = os.path.split(nomearq)
        nova_pasta = os.path.join(parte_pasta,pasta_destino)
        force_directory(nova_pasta)
        os.rename(nomearq,os.path.join(nova_pasta,parte_nome))
    else:
        parte_pasta,parte_nome = os.path.split(nomearq)
        parte_nome,parte_ext = os.path.splitext(parte_nome)
        if parte_ext in [".jpg",".jpeg",".webp",".bmp"]:
            parte_ext = ".png"
        ultima_pasta = os.path.split(parte_pasta)[1]
        if parte_nome.startswith(ultima_pasta):
            novo_nome = os.path.join(parte_pasta, f"{parte_nome}{parte_ext}")
        else:
            novo_nome = os.path.join(parte_pasta, f"{ultima_pasta}_{parte_nome}{parte_ext}")
        os.remove(nomearq)
        cv2.imwrite(novo_nome,img)
        CONTADOR_ARQUIVOS_VALIDOS += 1

    return

def ajeitaRecursivo(pasta):
    for root, _, files in os.walk(pasta):
        if '_excluir/' in root:
            continue

        for fi in files:
            ajeitaArquivo(os.path.join(root,fi))

    return

# --------------------- Inicio do Programa
def main():
    global CONTADOR_ARQUIVOS_VALIDOS    
    # Defino argumentos da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("images_or_paths", nargs='+', type=str, help="Images or paths with images to be processed")
    args = parser.parse_args()

    arquivos = args.images_or_paths

    horaInicialProcesso = datetime.now()
    
    CONTADOR_ARQUIVOS_VALIDOS = 0

    for nomearq in arquivos:
        if os.path.isdir(nomearq):
            ajeitaRecursivo(nomearq)
        else:
            ajeitaArquivo(nomearq)
    
    print("\033[KFim de execução")

    horaFinalProcesso = datetime.now()

    tempo_processamento = horaFinalProcesso - horaInicialProcesso
    print("Tempo de processamento = {}".format(tempo_processamento))
    print("Número de imagens válidas = {}".format(CONTADOR_ARQUIVOS_VALIDOS))

if __name__ == "__main__":
    main()
