import os
import sys

PATH_PROG = sys.path[0]
DIR_TEMP = os.path.join(PATH_PROG,'.temp__')
DIR_TRAIN = os.path.join(PATH_PROG,'train_imgs')

def define_dir_temp(pasta):
    global DIR_TEMP
    global DIR_TRAIN
    DIR_TEMP = os.path.join(PATH_PROG,'.temp__',pasta)
    DIR_TRAIN = os.path.join(PATH_PROG,'train_imgs')
    force_directory(DIR_TEMP)
    force_directory(DIR_TRAIN,deletingFiles=False)
    force_directory(os.path.join(DIR_TEMP,'results'))
    return DIR_TEMP

def delete_file(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False

def delete_files(path):
    for r,_,f in os.walk(path):
        for arq in f:
            delete_file(os.path.join(r,arq))

def getOsPathName(arquivo, newExt = None, newPath = None, addToName = None, addToExt = None):
    nomePath,nomeArq = os.path.split(arquivo)
    novoNome,extParte = os.path.splitext(nomeArq)

    if newExt == None:
        newExt = extParte

    if addToExt != None:
        newExt = newExt + addToExt
    
    if newPath == None:
        newPath = nomePath

    if addToName != None:
        novoNome = novoNome + addToName

    if newExt:
        novoNome = novoNome + newExt

    if newPath:
        novoNome = os.path.join(newPath, novoNome)

    return novoNome

def force_directory(direc, deletingFiles = True):
    if not os.path.exists(direc):
        path,_ = os.path.split(direc)
        if force_directory(path,False):
            try:
                os.mkdir(direc)
            except:
                return False
        
    if deletingFiles:
        try:
            delete_files(direc)
        except:
            pass
    
    return True
