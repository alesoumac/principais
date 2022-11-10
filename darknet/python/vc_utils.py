# Reestruturação da aplicação, divisão em classes, separação do darknet_server.py 30% performance

import configparser
import os
import re
import shutil
from datetime import datetime

import vc_constants
import vc_strings

# -------------------------------------------
# ----------------------------- função de log

# A variável USE_LOGGER define o uso do logger ou do print para mostrar mensagens de Log.
#   Se quiser usar o Print como sistema de log, colocar False na variável USE_LOGGER.
#   Se quiser usar o Logger como sistema de log, colocar True na variável USE_LOGGER.
USE_LOGGER = True
ULTIMO_LOGGER_USADO = None

def dictValue(dic, chave, padrao = None, as_type = None):
    valor = dic[chave] if chave in dic else padrao
    if as_type is None: return valor
    if as_type == 'bool': 
        valor = vc_strings.str_to_int_def(valor,None)
        if valor is None: return padrao
        return valor != 0
    if as_type == 'str': return str(valor)
    if as_type == 'int': return vc_strings.str_to_int_def(valor,padrao)
    return valor
    
def printLog(msg, logger = None):
    global USE_LOGGER
    global ULTIMO_LOGGER_USADO

    if USE_LOGGER:
        if logger is not None:
            local_logger = logger
            ULTIMO_LOGGER_USADO = logger
        else:
            local_logger = ULTIMO_LOGGER_USADO

        if local_logger is not None:
            local_logger.info(msg)

    print(msg)

def printErr(msg, logger = None):
    global USE_LOGGER
    global ULTIMO_LOGGER_USADO

    if USE_LOGGER:
        if logger is not None:
            local_logger = logger
            ULTIMO_LOGGER_USADO = logger
        else:
            local_logger = ULTIMO_LOGGER_USADO

        if local_logger is not None:
            local_logger.error(msg)

    print(msg)

# -------------------------------------------
# ------------- algumas funções básicas úteis
def deleteFile(filename):
    try:
        os.remove(filename)
        return True
    except:
        return False

def deleteFiles(path):
    for r,_,f in os.walk(path):
        for arq in f:
            deleteFile(os.path.join(r,arq))

def copyFile(arquivo,pasta):
    nomedest = os.path.join( pasta, os.path.split(arquivo)[1] ) if os.path.isdir(pasta) else pasta
    shutil.copy2(arquivo,nomedest)
    return True

def forceDirectory(direc, deletingFiles = False):
    if not os.path.exists(direc):
        path,_ = os.path.split(direc)
        if forceDirectory(path,False):
            try:
                os.mkdir(direc)
            except:
                return False
        
    if deletingFiles:
        try:
            deleteFiles(direc)
        except:
            pass
    
    return True

def makeTemporaryFile(path_base,original_filename=None):
    horaAtual = datetime.now()
    tempData = datetime.strftime(horaAtual,'%Y%m%d')
    tempHora = datetime.strftime(horaAtual,'%H%M%S') + f'_{horaAtual.microsecond:06d}'
    tempdir = os.path.join(path_base,tempData,tempHora)
    forceDirectory(tempdir)
    if original_filename is None:
        baseFilename = tempHora
        extensao = ''
    else:
        baseFilename = os.path.split(original_filename)[1]
        baseFilename,extensao = os.path.splitext(original_filename)
    baseFilename = os.path.join(tempdir,baseFilename)
    return baseFilename,extensao

def basicCompareRate(s,t,principal = None):
    if principal is not None and principal == 1 and len(s) == 0: return 1.0, 0
    if principal is not None and principal == 2 and len(t) == 0: return 1.0, 0
    packt = ''.join(set(t))
    positions = {c : [i for i in range(len(s)) if s[i] == c] for c in packt}
    #
    maiores = [ [] for i in t]
    mais_maior = 0
    #
    for i in range(len(t))[::-1]:
        c = t[i]
        for m in positions[c]:
            maior = 0
            r = ''
            for k in range(i+1,len(t)):
                ck = t[k]
                for l,n in enumerate(positions[ck]):
                    if n > m and len(maiores[k][l]) > maior:
                        r = maiores[k][l]
                        maior = len(r)
            r = f"{c}{r}"
            maiores[i] += [r]
            if len(r) > mais_maior:
                mais_maior = len(r)
    #print(positions)
    #print(maiores)
    numerador = denominador = 0
    if principal is None or principal == 1:
        numerador += mais_maior
        denominador += len(s)
    
    if principal is None or principal == 2:
        numerador += mais_maior
        denominador += len(t)
        
    erro = len(t) + len(s) - 2*mais_maior

    return float(numerador) / float(denominador), erro

# -------------------------------------------
def responseError(st_code, msg_err):
    resp_media = {vc_constants.FIELD_STATUS: vc_constants.STATUS_FATAL,
        vc_constants.FIELD_MSG: msg_err}
    return st_code, resp_media

# -------------------------------------------
def setSuccessResponseJSON(resp,json):
    resp.status_code = 200
    resp.media = json

# -------------------------------------------
def preprocessingName(pp):
    if pp == 1: return "BorderRemoval[1]"
    if pp == 2: return "BorderRemoval[2]"
    if pp == 3: return "BorderRemoval[1+2]"
    return "None"

def getConfig(config, section, key, as_int=False):
    return config.getint(section,key) if as_int else config.get(section,key)

def getConfigParameters(configfilepath):
    """
    getting parameter from config
    """

    config = configparser.ConfigParser()
    config.read(configfilepath)

    # for YOLO
    namesfilepath = os.getenv('VCDOC_API_YOLO_CAMINHO_ARQUIVO_NOMES_CLASSES')  # obj.names
    cfgfilepath = os.getenv('VCDOC_API_YOLO_CAMINHO_ARQUIVO_CONFIG_REDE')      # yolo-obj.cfg
    weightfilepath = os.getenv('VCDOC_API_YOLO_CAMINHO_ARQUIVO_PESOS')         # yolo-obj_25000.weights
    jwksaddress = os.environ['VCDOC_JWKSADDR']

    try:
        # for Server
        host = getConfig(config, 'Server', 'host')
        port = getConfig(config, 'Server', 'port', True)
        logfilepath = getConfig(config, 'Server', 'logfilepath')
        if namesfilepath is None: namesfilepath = getConfig(config, 'YOLO', 'namesfilepath')
        if cfgfilepath is None: cfgfilepath = getConfig(config, 'YOLO', 'cfgfilepath')
        if weightfilepath is None: weightfilepath = getConfig(config, 'YOLO', 'weightfilepath')
    except configparser.Error as config_parse_err:
        raise config_parse_err
    
    return namesfilepath, cfgfilepath, weightfilepath, host, port, logfilepath, jwksaddress

def pathExists(targetpath):
    """
    checking path
    """
    check_flg = None
    if not os.path.exists(targetpath):
        printLog(f'{targetpath} não existe')
        check_flg = False
    else:
        check_flg = True
    return check_flg

def isValidHostName(hostname):
    """
    validate host name
    refference:
    https://stackoverflow.com/questions/2532053/validate-a-hostname-string
    """
    valid_flg = True
    if len(hostname) > 255:
        valid_flg = False
    if hostname[-1] == ".":
        # strip exactly one dot from the right, if present
        hostname = hostname[:-1]
        allowed = re.compile(r"(?!-)[A-Z\d-]{1,63}(?<!-)$", re.IGNORECASE)
        valid_flg = all(allowed.match(x) for x in hostname.split("."))
    return valid_flg

def checkParameters(namesfilepath, cfgfilepath, weightfilepath, host, port, logger):
    """
    checking parameters
    """
    printLog("Checando parâmetros", logger)
    validation_flg = True
    #for targetpath in [darknetlibfilepath, datafilepath,
    #                   cfgfilepath, weightfilepath]:
    #    validation_flg &= pathExists(targetpath)

    for targetpath in [namesfilepath, cfgfilepath, weightfilepath]:
        validation_flg &= pathExists(targetpath)
    
    if not isValidHostName(host):
        printErr(f'Nome de host inválido: {host}', logger)
        validation_flg = False

    if port < 0 or port > 65535:
        printErr(f'Número da porta deve estar entre 0 and 65535. Porta: {port}', logger)
        validation_flg = False
    elif 0 < port < 1024:
        printLog(f'[Aviso] Porta bem conhecida sendo usada: {port}\n  --> É recomendado usar outra porta.')

    return validation_flg
