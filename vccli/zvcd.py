# -*- coding: utf-8 -*-
import argparse
import base64
import io
import sys
import json
import os
import time
from datetime import datetime, timedelta
import platform

import cv2
import difflib
import requests
from PIL import Image
import numpy as np
import jwt

import modulo_env as ME
import modulo_ocr as MO
import modulo_anonym as MA

CURRENT_OS = platform.system().lower()

# Valores para a variável AMBIENTE:
#   P - Produção
#   D - Desenvolvimento
#   H - Homologação
AMBIENTE = "D"

TAG_FILE_COM_PREFIXO_PERCENTUAL = False

USE_LOCAL_SERVER = False
PERFORM_SKEW_CORRECTION = False
SHOW_RESULT_JSON = False
SAVE_RESULT_IMAGE = False
SAVE_TAG_MARKER_FILE = False
ANONYMIZE_IMAGES = False
SAVE_PERCENTUAL_ACCURACY = False
ASK_ROTATE = False
ASK_DIVIDE = False
CROP_HEIGHT = -1
LAP_TIME = None
OCR_TEST_TYPE = 3
SHOW_EXTRACTED_VALUES = False

CONTADOR_ARQUIVOS = 0
OUTFILE = None

FFPP = '3'
OFPP = '3'
USE_RESIZE = False
LARG_MAXIMA = 800

IGNORE_AUTENTIKUS = False
USE_EASYOCR = True

HEADER_DIC = {}
HEADER_EXPIRATION = None

CPF_OK = 0
CPF_NOK = 0

CAMPOS_ACURACIA = {"cnh": "CNH", "cnh_frente": "CNH Frente", 
    "nome_cnh":"Nome CNH (%)", "identidade_cnh": "Identidade CNH (%)", "cpf_cnh": "CPF CNH (%)", "nascimento_cnh": "Data Nasc. CNH (%)",
    "filiacao_cnh": "Filiação CNH (%)", "categoria_cnh": "Categoria CNH (%)", "registro_cnh": "Nº Registro CNH (%)", 
    "validade_cnh": "Data Validade CNH (%)", "pri_habilitacao_cnh": "Data 1ª Habil. CNH (%)", "observacao_cnh": "Observações CNH (%)",
    "local_emissao_cnh": "Local Emissão CNH (%)", "data_emissao_cnh": "Data Emissão CNH (%)", "foto_cnh": "Foto CNH", 
    "rg_frente": "RG Frente", "rg_verso": "RG Verso", 
    "nome_rg": "Nome RG (%)", "registro_geral_rg": "Nº Reg. Geral RG (%)", "data_expedicao_rg": "Data Exped. RG (%)",
    "filiacao_rg": "Filiação RG (%)", "naturalidade_rg": "Naturalidade RG (%)", "nascimento_rg": "Data Nasc. RG (%)",
    "cpf_rg": "CPF RG (%)", "doc_origem_rg": "Doc. Origem RG (%)", "cabecalho_rg": "Cabecalho RG (%)",
    "foto_rg": "Foto RG", "assinatura_rg": "Assinatura RG", "digital_rg": "Digital RG", 
    "rg2_frente": "RG2 Frente", "rg2_verso": "RG2 Verso", 
    "nome_rg2": "Nome RG2 (%)", "registro_geral_rg2": "Nº Reg. Geral RG2 (%)",
    "filiacao_rg2": "Filiação RG2 (%)", "naturalidade_rg2": "Naturalidade RG2 (%)", "nascimento_rg2": "Data Nasc. RG2 (%)",
    "cpf_rg2": "CPF RG2 (%)", "cabecalho_rg2": "Cabecalho RG2 (%)",
    "foto_rg2": "Foto RG2", 
    "predict_time_secs": "Tempo Predição (s)", "detect_time_secs": "Tempo Detecção (s)",
    "ocr_time_secs": "Tempo OCR (s)", "detect_ocr_total": "Tempo Total (s)", "latency_total": "Tempo Total + Latência (s)",
    "timers": "Timers"
    }

def tty_color(cor = None):
    '''
    Cores para imprimir no terminal

    TTY_RED   = "\033[1;31m"
    TTY_BLUE  = "\033[1;34m"
    TTY_CYAN  = "\033[1;36m"
    TTY_GREEN = "\033[0;32m"
    TTY_RESET = "\033[0;0m"
    TTY_BOLD    = "\033[;1m"
    TTY_REVERSE = "\033[;7m"

    vermelho = '\033[31m'
    verde = '\033[32m'
    azul = '\033[34m'

    ciano = '\033[36m'
    magenta = '\033[35m'
    amarelo = '\033[33m'
    preto = '\033[30m'

    branco = '\033[37m'

    restaura cor original = '\033[0;0m'
    negrito = '\033[1m'
    reverso = '\033[2m'

    fundo preto = '\033[40m'
    fundo vermelho = '\033[41m'
    fundo verde = '\033[42m'
    fundo amarelo = '\033[43m'
    fundo azul = '\033[44m'
    fundo magenta = '\033[45m'
    fundo ciano = '\033[46m'
    fundo branco = '\033[47m'
    '''
    if CURRENT_OS != "linux":
        return ""

    if cor is None:
        return "\033[0;0m" # Reset
    c = cor.lower()
    if c == "reset":   return "\033[0;0m"
    if c == "red":     return "\033[1;31m"
    if c == "blue":    return "\033[1;34m"
    if c == "cyan":    return "\033[1;36m"
    if c == "dark green":   return "\033[0;32m"
    if c == "green":  return "\033[1;32m"
    if c == "bold":    return "\033[;1m"
    if c == "reverse": return "\033[;7m"
    return ""

def trata_string(s,caracteres_permitidos = ""):
    newS = MO.translate_extchar(s).upper()
    if caracteres_permitidos:
        newS = ''.join([c for c in newS if c in caracteres_permitidos])
    while '  ' in newS:
        newS = newS.replace('  ',' ')
    return newS.strip()

def trata_campo(campo,lista):
    campo = campo.lower()

    if campo.startswith('foto') or campo.startswith('cnh') or campo.startswith('rg') \
    or campo.startswith('assinatura') or campo.startswith('digital'):
        return lista
    
    filtro = ""
    if campo.startswith('cpf_'):
        filtro = "0123456789"
    # if campo.startswith('identidade') or campo.startswith('registro_geral'):
    #     filtro = ""
    # elif campo.startswith('cpf') or campo.startswith('registro'):
    #     filtro = "0123456789"
    # elif campo.startswith('nome') or campo.startswith('filiacao'):
    #     filtro = "ABCDEFGHIJKLMNOPQRSTUVWXYZ =\n"
    # elif campo.startswith('nascimento') or campo.startswith('validade') \
    # or campo.startswith('pri_habilitacao') or campo.startswith('data_'):
    #     filtro = "0123456789/-."
    # elif campo.startswith('local_emissao') or campo.startswith('naturalidade'):
    #     filtro = "ABCDEFGHIJKLMNOPQRSTUVWXYZ ,"
    # elif campo.startswith('doc_origem'):
    #     filtro = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-"

    return [trata_string(s,filtro) for s in lista]

def compare_and_rate(s_orig, t_orig):
    '''
    A função compare_and_rate compara duas strings 's' e 't' e retorna um valor entre 0 e 1,
    que indica a taxa de similaridade entre essas strings. A string s é a string base,
    ou seja, a string correta, e a string t é a string candidata.
    Valor de retorno 0 significa que não houve nenhuma similaridade
    e o valor 1 significa similaridade total (ou seja s é igual a t).
    '''
    s = s_orig.replace('\n',' ')
    t = t_orig.replace('\n',' ')
    if s == t:
        rate = 1
    else:
        #rate,difs = MO.cmp_rate(s,t,1)
        seqs = difflib.SequenceMatcher(None, s, t)
        nch = sum([m.size for m in seqs.get_matching_blocks()])
        rate = float(nch) / len(s) if s else 1
        difs = len(s)+len(t) - 2 * nch

        if rate == 1:
            if s.replace(' ','') == t.replace(' ',''):
                rate = 0.9998
            elif difs <= 2:
                rate = 0.9997
            elif s.replace(' ','') in t.replace(' ',''):
                rate = 0.9996
            else:
                rate = 0.9995
    rate_str = "PERFECT MATCH       " if rate == 1 \
          else "ALMOST PERFECT MATCH" if rate >= 0.9995 \
          else "MATCH               " if rate >= 0.75 \
          else "GoodPartialMatch    " if rate >= 0.5 \
          else "BadPartialMatch     " if rate >= 0.125 \
          else "Fail                "

    return rate,rate_str

def rotate_bound(imge, angle, borderValue=None):
    while angle < 0: angle += 360.0
    while angle >= 360.0: angle -= 360.0
    if abs(angle) < 2:
        return imge

    if abs(angle - 90.0) < 2: angle = 90
    if abs(angle - 180.0) < 2: angle = 180
    if abs(angle - 270.0) < 2: angle = 270

    # grab the dimensions of the image 
    h, w = imge.shape[:2]
    dp = 1 if len(imge.shape) <= 2 else imge.shape[2]

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

def cria_header_autentikus():
    global HEADER_DIC
    global HEADER_EXPIRATION
    global AMBIENTE
    global USE_LOCAL_SERVER

    if IGNORE_AUTENTIKUS or USE_LOCAL_SERVER:
        HEADER_DIC = {}
        HEADER_EXPIRATION = 3600
        return HEADER_DIC

    data_agora = datetime.now()
    #print(f"Peguei essa hora: {data_agora}")
    if (HEADER_EXPIRATION is not None) and ((HEADER_EXPIRATION - data_agora).total_seconds() > 3):
        nseconds = int((HEADER_EXPIRATION - data_agora).total_seconds())
        #print(f"Segundos para a expiração: {nseconds}")
        return HEADER_DIC

    if HEADER_EXPIRATION is not None:
        nseconds = int((HEADER_EXPIRATION - data_agora).total_seconds()) + 1
        if nseconds > 0:
            #print(f"Tive que fazer um sleep de {nseconds} segundos.")
            time.sleep(nseconds)

    if AMBIENTE in "HD":
        # Chave de serviço do ValAutentikus
        chave_servico = b"uhjflqa60eec7ofhe843o9pcm0:m1sgpug50s9gl2ktg5ccgh5hnr"
        url_servico = "https://valautentikus.estaleiro.serpro.gov.br/autentikus-authn/api/v1/token"
    else:
        # Chave de serviço do Autentikus (Produção)
        chave_servico = b"3c3eem214ti7fl512o71cpek11:9vf5h9h4hpi8dafstnkjliqb7d"
        url_servico = "https://autentikus.estaleiro.serpro.gov.br/autentikus-authn/api/v1/token"

    autorizacao_basic = f"{jwt.utils.base64url_encode(chave_servico).decode()}="
    # A variável autorizacao_basic tem que ter o valor "dWhqZmxxYTYwZWVjN29maGU4NDNvOXBjbTA6bTFzZ3B1ZzUwczlnbDJrdGc1Y2NnaDVobnI="
    #print(f"Autorizacao Basic = {autorizacao_basic}")

    while True:
        header_autk = {
            'content-type': 'application/x-www-form-urlencoded', 
            'Authorization': f'Basic {autorizacao_basic}'
            }
        data_autk = 'grant_type=client_credentials&scope=escopo_vcdoc'
        #print("Iniciando o request do token")
        resp_token_antes = requests.post(url_servico, data=data_autk, headers=header_autk)
        if resp_token_antes.status_code != 200:
            print(f"Erro de conexão Autentikus: {resp_token_antes.content.decode()}")
            exit(-500)
            #HEADER_DIC = {}
            #HEADER_EXPIRATION = 3600
            #return HEADER_DIC

        resp_token = resp_token_antes.json()
        #print(f"RespToken = {resp_token}")
        if 'error' in resp_token:
            raise Exception("Erro na obtenção do token de autenticação")

        exp_time = resp_token['expires_in']
        if exp_time > 2:
            break
        time.sleep(exp_time)

    bearer = resp_token['access_token']

    HEADER_EXPIRATION = datetime.now() + timedelta(seconds=exp_time)
    #bearer = "eyJhbGciOiJSUzUxMiJ9.eyJzdWIiOiIzMzY4MzExMTAwMDEwNyIsImF1ZCI6Ijk4IiwicGVybWlzc2lvbnMiOlsicG9zdCB2YWxpZGFjYW9fZGVfZG9jdW1lbnRvcyJdLCJzY29wZSI6ImVzY29wb192Y2RvYyIsInJvbGVzIjpbImNsaWVudGUiXSwibmFtZSI6IjExNjc0LXZjZG9jIiwiaXNzIjoidmFsLmF1dGVudGlrdXMuZXN0YWxlaXJvLnNlcnBybyIsImNsaWVudCI6IjIyIiwiY2xpZW50X2lwIjoiMTAuNDMuNC4xMTUiLCJleHAiOjE1OTUyNjUzMTIsInBhcmFtcyI6e30sImlhdCI6MTU5NTI2MTcxMn0.FFzdXxZOI7GJXaotq-DyjVrXJgbgfz3wNrYn3oq1du4u-LQE7V-_l1QqltqJ87tFxObdsjPggI7JTLyggXv86iD7hgcb8ZHlApsI2XUPfeahTXr89aoh8MC5FdcxZyWFeax1HdzbIuw-nwMsSJaXDM5wJNpkb-2w1baw-4yK762PQ7-QTq69ld4nUxqalrtuurdlQoiW5T52VFDc4p4GuNzxuuEHlW52rVi1PCRVpCdacneFpR6fLZqR2Esj1cG5AY9OHK9Apv0Q_-ehByvORBOkgvPNv-WMQ567_MYHwyqAz5tKbmJQEGNEOUCgSgs4fWohdRMWgoQZCUPee2707A"

    HEADER_DIC = {'Authorization': 'Bearer ' + bearer, 'typ': 'JWT', 'alg': 'RS512'}
    return HEADER_DIC

def write_ocr_and_adjusted(ocr,adjusted,expected):
    if expected == "": return
    if ocr == adjusted: return

    filename = os.path.join(ME.PATH_PROG, "adjusted_ocr.csv")
    if os.path.exists(filename):
        f = open(filename,"a")
    else:
        f = open(filename,"w")
        f.write('ocr,adjusted,expected,rate_oe,err_oe,rate_ae,err_ae\n')
    ocr2 = ocr.upper()
    rate_oe, err_oe = MO.cmp_rate(expected,ocr2,1)
    rate_ae, err_ae = MO.cmp_rate(expected,adjusted,1)
    ocr2 = ocr2.replace('"','´')
    adj2 = adjusted.replace('"','´')
    exp2 = expected.replace('"','´')
    f.write(f'"{ocr2}","{adj2}","{exp2}",{rate_oe},{err_oe},{rate_ae},{err_ae}\n')
    f.close()
    return
    
def get_json_imagem(img_filename, use_local_server, dir_work):
    global FFPP
    global OFPP
    global LARG_MAXIMA
    global USE_RESIZE
    global CPF_OK
    global CPF_NOK
    global AMBIENTE
    global CAMPOS_ACURACIA
    global LAP_TIME
    global OCR_TEST_TYPE
    global SHOW_EXTRACTED_VALUES

    if LAP_TIME is None:
        LAP_TIME = datetime.now()

    detected_objs = None
    himg,wimg = (0,0)
    if img_filename is None or img_filename == "":
        return 0, 0, detected_objs

    with open(img_filename, "rb") as inputfile:
        data = inputfile.read()

    try:
        filename_valores = ME.getOsPathName(img_filename,newExt='.json')
        valfile = open(filename_valores, "r")
        expected_values = json.loads(valfile.read())
        expected_values_novo = {}
        for obj_name in expected_values:
            expected_values_novo[obj_name.lower()] = expected_values[obj_name]
        expected_values = expected_values_novo
        for obj_name in ['tipo_doc']:
            if obj_name in expected_values:
                del expected_values[obj_name]
        valfile.close()
    except:
        expected_values = {}

    num_matches = 0

    num_ocr_nao_vazio = 0
    num_campos_esperados = len(expected_values)
    num_campos_detectados = 0
    print(f'Numero de Campos Esperados = {num_campos_esperados}')

    # Obter o Bearer Token
    header_dic = cria_header_autentikus()

    #header_req = header_dic.copy()
    header_dic["requester"]   = os.path.split(sys.argv[0])[-1]
    header_dic["vcdoc_param_avaliar_rotacao"] = '1' if ASK_ROTATE else '0' 
    header_dic["vcdoc_param_ocr_solicitado"]  = '2' if USE_EASYOCR else '1'
    header_dic["vcdoc_param_crop_height"] = '48' if CROP_HEIGHT < 24 else str(CROP_HEIGHT)
    header_dic["vcdoc_param_largura_de_redimensionamento_imagem"] = LARG_MAXIMA if USE_RESIZE else 0
    header_dic["vcdoc_param_preprocessamento_multiline"] = FFPP
    header_dic["vcdoc_param_preprocessamento_singleline"] = OFPP

    #header_dic = {}
    post_data = {"image": base64.b64encode(data).decode("utf-8") }

    try:
        if use_local_server:
            # --> print("Using local server...")
            resj = requests.post("http://localhost:8081/v1/detect-ocr", json=post_data, headers=header_dic)
            #print("ResJ\n===================================================")
            #print(resj.content)
            res = resj.json()
            #res = requests.post("http://localhost:8081/v1/detect-ocr", json=post_data, verify=False).json()

            # DÚVIDA: Estou passando a JWT_STR no post_data e também passando o cabeçalho header_dic no request
            # feito pro darknet_server
        else:
            if AMBIENTE == "D":
                # Estaleiro Dev
                resj = requests.post("https://des-vcdocgpu.app.spesp0003.estaleiro.serpro.gov.br/v1/detect-ocr", json=post_data, headers=header_dic, verify=False)
            elif AMBIENTE == "H":
                # Estaleiro Homolog
                resj = requests.post("https://hom-vcdocgpu.app.spesp0003.estaleiro.serpro.gov.br/v1/detect-ocr", json=post_data, headers=header_dic, verify=False)
            else:
                # Estaleiro Prod
                resj = requests.post("https://vcdocgpu.app.spesp0003.estaleiro.serpro.gov.br/v1/detect-ocr", json=post_data, headers=header_dic)

            #print(resj.status_code)
            #print("ResJ\n===================================================")
            #print(resj.content.decode())
            #faux = open('vvcd_resp.html','w+')
            #s = "" + str(resj.content)
            #faux.write(s)
            #faux.close()
            try:
                res = resj.json()
            except:
                print(resj.content.decode())
                res = {'status':'exception', 'msg': 'Resposta inválida. Retorno = {}'.format(resj.status_code) }
  
    except Exception as e:
        res = {'status':'exception', 'msg': str(e)}

    if "pred_img" in res:
        pred_data = base64.b64decode(res["pred_img"])
        img = Image.open(io.BytesIO(pred_data))
        del res["pred_img"]
        nome_temp = os.path.join(dir_work,"bounding_boxes.png")
        img.save(nome_temp)

    if "work_img" in res:
        work_data = base64.b64decode(res["work_img"])
        img = Image.open(io.BytesIO(work_data))
        del res["work_img"]
        nome_work = os.path.join(dir_work,"work_image.png") 
        img.save(nome_work)
    else:
        nome_work = img_filename

    angulo = 0
    if "angle" in res:
        angulo = res["angle"]
        if angulo != 0:
            print("A imagem parece ficar melhor se for rotacionada em {}°".format(angulo))

    if "face_img" in res:
        face_data = base64.b64decode(res["face_img"])
        img = Image.open(io.BytesIO(face_data))
        del res["face_img"]
        nome_face = os.path.join(dir_work, '_face_' + os.path.split(nome_work)[1])
        img.save(nome_face)

    dic_acuracia = {
        "arquivo" : img_filename
    }
    for campo in expected_values:
        dic_acuracia[campo.lower()] = 0.0

    if "resultlist" not in res:
        print("Problema de conexão com a API.\nJson retornado:")
        print(res)
        detected_objs = None
    else:
        nomesObjetos = [element['obj_name'].lower() for element in res['resultlist']] # crio um array só com os obj_names detectados
        if "cpf_rg" in expected_values and "cpf_rg" not in nomesObjetos:
            res['resultlist'] += [{'obj_name':'cpf_rg', 'score':1.0, 'ocr_text':'', 'adjusted_ocr':''}]

        num_objs = len(res['resultlist'])

        print('Num. de objetos detectados: {}'.format(num_objs))
        detected_objs = []
        img = cv2.imread(nome_work)
        himg,wimg = img.shape[:2]
        num_matches = 0
        if num_objs > 0:
            prim_nres = 0
            for nres in range(num_objs):
                if res['resultlist'][nres]['obj_name'].lower() in ['cnh','rg_frente','rg_verso']:
                    prim_nres = nres
                    break

        for nres in range(num_objs):
            id_nres = nres
            if prim_nres > 0:
                id_nres = prim_nres if nres == 0 else 0 if nres == prim_nres else nres
            yolo_result = res['resultlist'][id_nres]
            detec = yolo_result['obj_name'].lower()
            scoreObj = yolo_result['score'] * 100
            detected_objs.extend([detec])
            candidatos = []
            rt = ""
            rc = ""
            rt0 = yolo_result['ocr_text'] if 'ocr_text' in yolo_result else ""
            rt = yolo_result['adjusted_ocr'] if 'adjusted_ocr' in yolo_result else rt0
            rt1 = rt
            if 'ocr_text' in yolo_result:
                rt = yolo_result['adjusted_ocr'] if 'adjusted_ocr' in yolo_result else yolo_result['ocr_text']
                if rt != "" and detec in expected_values:
                    num_ocr_nao_vazio += 1
                #if detec.lower() in ['filiacao_cnh','nome_cnh','identidade_cnh','local_emissao_cnh']:
                #    rc = MO.verifica_spell(rt)
                #    candidatos = MO.get_spell_candidates(rt) if rc != rt else [] 
            match_str = "n/a             "
            cor_fonte = ""

            if detec in expected_values:
                num_campos_detectados += 1
                #print(f"Detectando campo '{detec}'")
                
                esperado = expected_values[detec]
                if detec.startswith('filiacao'):
                    if '=' in esperado:
                        rt = rt.replace('\n','=')
                        rt0 = rt0.replace('\n','=')
                        rt1 = rt1.replace('\n','=')

                rt,rt0,esperado = trata_campo(detec,(rt,rt0,esperado))

                rate_match, match_str = compare_and_rate(esperado, rt)
                ocr_rate,ocr_match = compare_and_rate(esperado, rt0)
                if detec.startswith('nome_') or detec.startswith('filiacao_'):
                    write_ocr_and_adjusted(rt0,rt1,expected_values[detec])

                if OCR_TEST_TYPE == 1:
                    rate_match = ocr_rate
                    match_str = ocr_match
                    rt = rt0

                if (OCR_TEST_TYPE == 3) and (ocr_rate > rate_match):
                    rate_match = ocr_rate
                    match_str = ocr_match
                    rt = rt0
                    if cor_fonte == "": cor_fonte = tty_color("blue")

                if detec.startswith("cpf_"):
                    if rate_match == 1:
                        CPF_OK += 1
                        cor_fonte = tty_color("green")
                    else:
                        CPF_NOK += 1
                        cor_fonte = tty_color("red")
                
                dic_acuracia[detec] = rate_match
                num_matches += rate_match

            if SHOW_EXTRACTED_VALUES:
                print("{:3d}: {:32s} - Score={:6.2f}% =>".format(nres,detec,scoreObj), end=" ")
                print("{}{} - {}{}".format(cor_fonte,match_str,rt,tty_color("reset") if cor_fonte != "" else ""))
                print(f"      OCR      = '{rt0}'\n      Ajustado = '{rt1}'")
            else:
                print("{:3d}: {:32s} =>".format(nres,detec), end=" ")
                print("{}{} - {}{}".format(cor_fonte,match_str,rt,tty_color("reset") if cor_fonte != "" else ""))

            #if candidatos != []:
            #    print("     - Ajustado:   \"{}\"".format(rc))
            #    print("     - Candidatos: {}".format(str(candidatos)))
        
        detected_objs = list(set(detected_objs))
        detected_objs.sort()
        if num_campos_detectados <= 2 and num_ocr_nao_vazio <= 1: # significa que não conseguiu detectar direito o documento
            num_campos_detectados = 0

    if 'timers' in res: print(f'Timers: {res["timers"]}')

    metricas = {'num_matches': num_matches, 
        'num_ocr_nao_vazio': num_ocr_nao_vazio, 
        'num_campos_detectados': num_campos_detectados, 
        'num_campos_esperados': num_campos_esperados}

    print(f'Tempo do Processo de Detecção/OCR: {res["detect_ocr_time"] if "detect_ocr_time" in res else "N/A"}')
    print(" ")
    if "timers" in res:
        dic_acuracia["timers"] = res["timers"]

    if "predict_time_secs" in res:
        dic_acuracia["predict_time_secs"] = res["predict_time_secs"]

    if "detect_time_secs" in res and "detect_ocr_time" in res:
        dic_acuracia["detect_time_secs"] = res["detect_time_secs"]
        dic_acuracia["detect_ocr_total"] = res["detect_ocr_time"]
        dic_acuracia["ocr_time_secs"] = dic_acuracia["detect_ocr_total"] - dic_acuracia["detect_time_secs"]

    novo_lap = datetime.now()
    dic_acuracia["latency_total"] = (novo_lap - LAP_TIME).total_seconds()
    LAP_TIME = novo_lap

    if SAVE_PERCENTUAL_ACCURACY:
        acc_line = f'\"{os.path.split( dic_acuracia["arquivo"] )[1] }\"'
        for campo in CAMPOS_ACURACIA:
            acc_line += ',{}'.format( dic_acuracia[campo] if campo in dic_acuracia else '' )
        printCSVAcuracia(acc_line)

    if SHOW_RESULT_JSON:
        print("RESULT JSON:", res, "\n")

    if SAVE_RESULT_IMAGE and 'resultlist' in res:
        nome_result_img = os.path.splitext(os.path.split(nome_work)[1])
        nome_result_json = os.path.join(dir_work, nome_result_img[0] + '.json')
        nome_result_frente = os.path.join(dir_work, nome_result_img[0] + ".__frente__" + nome_result_img[1]) 
        nome_result_verso = os.path.join(dir_work, nome_result_img[0] + ".__verso__" + nome_result_img[1]) 
        nome_result_img = os.path.join(dir_work, nome_result_img[0] + ".__marked__" + nome_result_img[1])

        print(f"Salvando resultado nos arquivos:\n{nome_result_img}\n{nome_result_json}")
        file_json = open(nome_result_json,"w")
        file_json.write("{")
        separador = '\n'
        y1, y2, y3, y4 = None,None,None,None
        pad = 0
        for elemento in res['resultlist']:
            detec = elemento['obj_name']
            dlower = detec.lower()
            if ASK_DIVIDE:
                bbox = elemento['bounding_box']
                (xmin,ymin,widt,heig) = (bbox['x_min'],bbox['y_min'],bbox['width'],bbox['height'])
                if dlower in ["registro_cnh", "validade_cnh", "pri_habilitacao_cnh"]:
                    pad = int(max(pad,heig / 8))
                    y1 = ymin if y1 is None else min(ymin,y1)
                    y2 = ymin+heig if y2 is None else max(y2,ymin+heig)
                elif dlower in ["local_emissao_cnh","data_emissao_cnh"]:
                    y3 = ymin if y1 is None else min(ymin,y3)
                    y4 = ymin+heig if y4 is None else max(y4,ymin+heig)
            if dlower in ["cnh","cnh_frente","rg_frente","rg_verso","foto_rg","foto_cnh"]:
                continue
            rt = elemento['adjusted_ocr'] if 'adjusted_ocr' in elemento else elemento['ocr_text'] if 'ocr_text' in elemento else ""
            file_json.write(f"{separador}  \"{detec}\" : \"{rt}\"")
            separador = ',\n'
        file_json.write("\n}\n")
        file_json.close()

        if ASK_DIVIDE and y1 is not None and y2 is not None and y3 is not None and y4 is not None and himg > 0:
            imagem_recog = cv2.imread(nome_work)
            if y1 > y4:
                ydiv = y1 - pad
                nome_result_frente, nome_result_verso = nome_result_verso, nome_result_frente
            elif y3 > y2:
                ydiv = y2 + pad
            else:
                ydiv = himg
            
            if ydiv < himg:
                print(f"\nGerando arquivos de frente/verso da CNH\n{nome_result_frente}\n{nome_result_verso}")
                cv2.imwrite(nome_result_frente,imagem_recog[0:ydiv, :])
                cv2.imwrite(nome_result_verso,imagem_recog[ydiv:himg, :])

        imagem_recog = cv2.imread(nome_work, cv2.IMREAD_GRAYSCALE)
        imagem_recog = cv2.cvtColor(imagem_recog, cv2.COLOR_GRAY2BGR)
        for elemento in res['resultlist']:
            bbox = elemento['bounding_box']
            (xmin,ymin,widt,heig) = (bbox['x_min'],bbox['y_min'],bbox['width'],bbox['height'])
            cv2.rectangle(imagem_recog, (xmin, ymin), (xmin + widt, ymin + heig), (255, 0, 0), 2)
        cv2.imwrite(nome_result_img,imagem_recog)

    if SAVE_TAG_MARKER_FILE and 'resultlist' in res:
        imagem_recog = cv2.imread(nome_work)
        if angulo != 0:
            imagem_recog = rotate_bound(imagem_recog,angulo)

        if ANONYMIZE_IMAGES:
            imagem_recog = MA.anonymize(imagem_recog)

        himg,wimg = imagem_recog.shape[:2]
        area_foto_cnh = 0
        area_cnh = 0
        for elemento in res['resultlist']:
            bbox = elemento['bounding_box']
            (xmin,ymin,widt,heig) = (bbox['x_min'],bbox['y_min'],bbox['width'],bbox['height'])
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmin+widt > wimg: widt = wimg-xmin
            if ymin+heig > himg: heig = himg-ymin

            nome_objeto = elemento["obj_name"].lower()
            if nome_objeto == "foto_cnh":
                area_foto_cnh = max(area_foto_cnh,widt * heig)
            elif nome_objeto in ["cnh","cnh_frente"]:
                area_cnh = max(area_cnh,widt * heig)

        if area_foto_cnh != 0 and area_cnh == 0:
            area_foto_cnh = 0

        if area_foto_cnh <= 0 and area_cnh <= 0:
            prefixo = "00"
        elif area_foto_cnh >= area_cnh:
            prefixo = "99"
        else:
            prefixo = "{:02d}".format(int(100 * float(area_foto_cnh) / float(area_cnh)))

        prefixo = prefixo + '_' if TAG_FILE_COM_PREFIXO_PERCENTUAL else ''

        nome_result_img = os.path.join(ME.DIR_TRAIN,prefixo + os.path.split(nome_work)[1])
        nome_result_tag = os.path.join(ME.DIR_TRAIN,prefixo + os.path.split(nome_work)[1])
        nome_result_tag = os.path.splitext(nome_result_tag)[0] + '.txt'

        nome_result_ground_truth = os.path.join(ME.DIR_TRAIN,prefixo + os.path.split(nome_work)[1])
        nome_result_ground_truth = os.path.splitext(nome_result_tag)[0] + '.json'

        print(f"\nGerando arquivo de tag em \'{nome_result_tag}\'")
        print(f"\nGerando arquivo de ground truth em \'{nome_result_ground_truth}\'")

        nRG = nCNH = 0
        for elemento in res['resultlist']:
            nome = elemento['obj_name']
            if   "_rg" in nome:
                nRG += 1
            elif nome.startswith("rg"):
                nRG += 1
            elif "_cnh" in nome:
                nCNH += 1
            elif nome.startswith("cnh"):
                nCNH += 1
        tipoDoc = 'cnh' if nCNH > nRG else 'rg'

        jsonStr = '{' + f'\n  "tipo_doc" : "{tipoDoc}",'
        for elemento in res['resultlist']:
            nome = elemento['obj_name']
            valor = elemento['adjusted_ocr']
            jsonStr += f'\n  "{nome}" : "{valor.upper()}",'
        jsonStr = jsonStr[:-1] + '\n}'

        gt = open(nome_result_ground_truth,'w')
        gt.write(jsonStr)
        gt.close()

        ftag = open(nome_result_tag,"w")
        for elemento in res['resultlist']:
            idx_class = elemento["class_index"]
            bbox = elemento['bounding_box']
            (xmin,ymin,widt,heig) = (bbox['x_min'],bbox['y_min'],bbox['width'],bbox['height'])
            if xmin < 0: xmin = 0
            if ymin < 0: ymin = 0
            if xmin+widt > wimg: widt = wimg-xmin
            if ymin+heig > himg: heig = himg-ymin
            line_class = f"{idx_class} {(xmin+widt/2.0)/float(wimg)} {(ymin+heig/2.0)/float(himg)} {widt/float(wimg)} {heig/float(himg)}"
            ftag.write(f"{line_class}\n")

            nome_objeto = elemento["obj_name"].lower()
            if ANONYMIZE_IMAGES:
                if nome_objeto.startswith("foto_"):
                    cv2.rectangle(imagem_recog,(int(xmin+widt//10), int(ymin+heig//2)), (int(xmin+widt-widt//10),int(ymin+heig-widt//10)), (124,124,124), cv2.FILLED)
                    cv2.circle( imagem_recog, (int(xmin+widt//2), int(ymin+heig//3)), min(heig//4,widt//3), (124,124,124), cv2.FILLED)
                elif nome_objeto not in ["cnh","cnh_frente","rg_frente","rg_verso","rg2_frente","rg2_verso"]:
                    fator = 0.7 if nome_objeto.startswith("filiacao_") else 0.4 
                    fator = (1 - fator) / 2.0
                    xanon = int(xmin+heig*fator)
                    yanon = int(ymin+heig*fator)
                    wanon = int(widt - heig * fator * 2.0)
                    hanon = int(heig - heig * fator * 2.0)
                    cv2.rectangle(imagem_recog, (xanon, yanon), (xanon+wanon, yanon+hanon), (132,132,132), cv2.FILLED)
                    if nome_objeto == "local_emissao_cnh":
                        yanon = yanon - 4 * (hanon + 3)
                        for idx in range(8):
                            if idx not in (3,4,5):
                                cv2.rectangle(imagem_recog, (xanon, yanon), (xanon+wanon, yanon+hanon), (144,144,144), cv2.FILLED)
                            yanon += hanon + 2

        ftag.close()
        cv2.imwrite(nome_result_img,imagem_recog)

    return metricas, detected_objs

def str_now(formato = None):
    now = datetime.now()
    if formato is None:
        return now.isoformat(sep=' ', timespec='seconds')
    else:
        return now.strftime(formato)

def verifyArquivo(nomearq, dir_work):
    global CONTADOR_ARQUIVOS
    
    CONTADOR_ARQUIVOS += 1
    print("Verificando {}".format(nomearq))
    matches = 0
    max_matches = 0
    horaInicio = str_now()
    try:
        nomearq = os.path.abspath(nomearq)
    except:
        pass

    metricas, detectado = get_json_imagem(nomearq, USE_LOCAL_SERVER, dir_work)
    
    num_matches = metricas['num_matches']
    num_ocr_nao_vazio = metricas['num_ocr_nao_vazio']
    num_campos_detectados = metricas['num_campos_detectados']
    num_campos_esperados = metricas['num_campos_esperados']

    if num_campos_detectados > 0:
        percentual_ocr = "{:5.1f}".format(float(num_ocr_nao_vazio) * 100 / float(num_campos_detectados)).strip().replace('.',',')
        percentual_acerto = "{:5.1f}".format(float(num_matches) * 100 / float(num_campos_detectados)).strip().replace('.',',')
    else:
        percentual_ocr = percentual_acerto = "N/A"

    classeRG = 0
    classeCNH = 0
    docRGF = 0
    docRGV = 0
    docCNH = 0

    if detectado is not None:
        for dc in detectado:
            if dc in ["cnh","cnh_frente"]:
                docCNH += 1
            if dc in ["rg_frente","rg2_frente"]:
                docRGF += 1
            if dc in ["rg_verso","rg2_verso"]:
                docRGV += 1
            if dc in [ \
                "nome_cnh","identidade_cnh","cpf_cnh", \
                "nascimento_cnh","filiacao_cnh","registro_cnh", \
                "validade_cnh","pri_habilitacao_cnh", \
                "local_emissao_cnh","data_emissao_cnh", \
                "categoria_cnh","observacao_cnh"]:
                classeCNH += 1
            elif dc in [ \
                "nome_rg","assinatura_rg","digital_rg", \
                "registro_geral_rg","data_expedicao_rg","filiacao_rg", \
                "naturalidade_rg","nascimento_rg","doc_origem_rg", \
                "cpf_rg","cabecalho_rg", \
                "nome_rg2","filiacao_rg2","nascimento_rg2","naturalidade_rg2", \
                "cpf_rg2","registro_geral_rg2","cabecalho_rg2"]:
                classeRG += 1

        detectedRG = "S" if docCNH == 0 and classeRG > 0 and docRGF <= 1 and docRGV <= 1 and docRGF+docRGV > 0 else "N"
        detectedCNH = "S" if docRGF+docRGV == 0 and classeCNH > 1 and docCNH == 1 else "N"
        directoryRG = "S" if "/rg" in nomearq.lower() else "N"
        directoryCNH = "S" if "/cnh" in nomearq.lower() else "N"

        doc_detectado = "Outros" if detectedRG == detectedCNH else "RG" if detectedRG == "S" else "CNH"
        doc_esperado = "Outros" if directoryRG == directoryCNH else "RG" if directoryRG == "S" else "CNH"
        ver_ok = "OK" if doc_esperado == doc_detectado else "" 

        printCSV(f"{horaInicio},{doc_esperado},{doc_detectado},{ver_ok},{detectedRG},{directoryRG},{detectedCNH},{directoryCNH},\"{nomearq}\"")
        printOUT("{:22s},{:>8s},{:>8s}".format( \
            '"' + os.path.split(nomearq)[1] + '"', \
            "\"" + percentual_acerto + "\"", \
            "\"" + percentual_ocr + "\"" \
            ))
    
    if max_matches == 0:
        print("                      Percentual de acerto dessa imagem: N/A\n")
    else:
        print("                      Percentual de acerto dessa imagem: {:5.2f}%\n".format(float(matches)/max_matches*100.0))
    return metricas

def verifyRecursivo(pasta, listaExt, dir_work, recursivo = False):
    sum_matches = sum_max_matches = matches = max_matches = 0
    for root, _, files in os.walk(pasta):
        if not recursivo and pasta != root:
            continue
        filessrt = sorted(files)
        for fi in filessrt:
            _,ex = os.path.splitext(fi)
            if ex[1:].lower() in listaExt and '.temp__' not in root:
                metricas = verifyArquivo(os.path.join(root,fi), dir_work)
                matches, max_matches = metricas['num_matches'], metricas['num_campos_detectados']
                sum_matches += matches
                sum_max_matches += max_matches
    return sum_matches, sum_max_matches

arquivoCSV = None
arquivoAcuracia = None

def printCSV(msg):
    global arquivoCSV
    arquivoCSV.write(msg + "\n")

def printCSVAcuracia(linha):
    global arquivoAcuracia
    arquivoAcuracia.write(linha + '\n')

def printOUT(s):
    global OUTFILE
    print(s)
    OUTFILE.write(s + "\n")

# --------------------- Inicio do Programa
def main():
    global USE_LOCAL_SERVER
    global SHOW_RESULT_JSON
    global SAVE_RESULT_IMAGE
    global SAVE_TAG_MARKER_FILE
    global SAVE_PERCENTUAL_ACCURACY
    global ASK_ROTATE
    global ASK_DIVIDE
    global CROP_HEIGHT
    global USE_EASYOCR
    global FFPP
    global OFPP
    global USE_RESIZE
    global LARG_MAXIMA
    global CONTADOR_ARQUIVOS
    global arquivoCSV
    global OUTFILE
    global arquivoAcuracia
    global CAMPOS_ACURACIA
    global CPF_OK
    global CPF_NOK
    global OCR_TEST_TYPE
    global SHOW_EXTRACTED_VALUES
    
    # Defino argumentos da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("images_or_paths", nargs='+', type=str, help="Images or paths with images to be processed")
    parser.add_argument("-l","--local", action='store_true', help="Use local server for Neural Net")
    parser.add_argument("-j","--json", action='store_true', help="Show result JSON")
    parser.add_argument("-s","--save", action='store_true', help="Save recognized image result")
    parser.add_argument("-t","--tags", action='store_true', help="Save text file with tag labels for training")
    parser.add_argument("-a","--accuracy", action='store_true', help="Save CSV file with percents of accuracy for OCR process")
    parser.add_argument("-r","--ask_rotate", action='store_true', help="Require server to make a rotation alignment if necessary")
    parser.add_argument("-e","--easyocr", action='store_true', help="Try to use EasyOCR instead Tesseract as OCR Engine")
    parser.add_argument("-d","--divide", action='store_true', help="Save CNH divided")
    parser.add_argument("-x","--show_extracted_values", action='store_true', help="Show extracted values (Raw OCR and Adjusted OCR)")
    parser.add_argument("-c","--crop", type=int, default=48, help="Crop height used on server-side")
    parser.add_argument("-o","--ocr_test", type=int, default=3, help="OCR Test Type. 1=Raw OCR || 2=Adjusted OCR || 3=Both")
    parser.add_argument("-pm","--ffpp", type=str, default='3', help="PreProc-Multiline Type")
    parser.add_argument("-ps","--ofpp", type=str, default='1', help="PreProc-SingleLine Type")
    parser.add_argument("-nr","--no_resize", action='store_true',  help="Try resize image in OCR")
    parser.add_argument("-w","--width", type=int, default=None, help="Max width for resize in OCR")
    args = parser.parse_args()

    # Deasabilito mensagens de erro de certificado
    requests.packages.urllib3.disable_warnings() 

    arquivos = args.images_or_paths
    USE_LOCAL_SERVER = args.local
    SHOW_RESULT_JSON = args.json
    SHOW_EXTRACTED_VALUES = args.show_extracted_values
    SAVE_RESULT_IMAGE = args.save
    SAVE_TAG_MARKER_FILE = args.tags
    SAVE_PERCENTUAL_ACCURACY = args.accuracy
    ASK_ROTATE = args.ask_rotate
    USE_EASYOCR = args.easyocr
    CROP_HEIGHT = args.crop
    OCR_TEST_TYPE = args.ocr_test if args.ocr_test in [1,2,3] else 3
    if SAVE_TAG_MARKER_FILE: 
        ASK_ROTATE = False

    ASK_DIVIDE = args.divide
    if ASK_DIVIDE:
        ASK_ROTATE = False
        SAVE_RESULT_IMAGE = True

    LARG_MAXIMA = args.width
    USE_RESIZE = not args.no_resize
    OFPP = args.ofpp
    FFPP = args.ffpp

    ME.define_dir_temp('vvcd')
    MA.inicializa_anonym(ME.PATH_PROG)

    if LARG_MAXIMA is None:
        LARG_MAXIMA = 800 if USE_RESIZE else 0

    horaInicialProcesso = datetime.now()
    sufixoCSV = horaInicialProcesso.strftime("_%Y%m%d_%H%M%S")
    nomeArquivoCSV = os.path.join(ME.PATH_PROG, 'vvcd_{}.csv'.format(sufixoCSV))
    print("Gerando arquivo CSV:", nomeArquivoCSV)
    arquivoCSV = open(nomeArquivoCSV, 'w')
    printCSV("Timestamp,Doc_Esperado,Doc_Detectado,Verificado,Detectou_RG,Diretorio_RG,Detectou_CNH,Diretorio_CNH,Arquivo")
    
    if SAVE_PERCENTUAL_ACCURACY:
        nomeArquivoCSVAcuracia = os.path.join(ME.PATH_PROG, f"acuracia_vvcd_{sufixoCSV}.csv")
        arquivoAcuracia = open(nomeArquivoCSVAcuracia, 'w')

        printCSVAcuracia(
            '"Arquivo"' + ''.join([f",\"{CAMPOS_ACURACIA[campo_acuracia]}\"" for campo_acuracia in CAMPOS_ACURACIA])
        )

    CONTADOR_ARQUIVOS = 0

    OUTFILE = open(os.path.join(ME.PATH_PROG,'zvcd.csv'),'a+')
    
    printOUT("=======================================================================================")
    printOUT('CONFIG')
    printOUT('  - FFPP: {} / OFPP: {}'.format(FFPP,OFPP))
    printOUT('  - Resizing: {}'.format(str(USE_RESIZE)))
    if USE_RESIZE: printOUT('     - Max Width = {}'.format(str(LARG_MAXIMA)))
    printOUT('Arquivo               , %Acerto,    %OCR')
    sum_matches = sum_max_matches = matches = max_matches = 0
    for nomearq in arquivos:
        if os.path.isdir(nomearq):
            matches, max_matches = verifyRecursivo(nomearq, ['bmp','jpg','jpeg','png','tif','tiff','webp'], ME.DIR_TEMP)
        else:
            metricas = verifyArquivo(nomearq, ME.DIR_TEMP)
            matches, max_matches = metricas['num_matches'], metricas['num_campos_detectados']

        sum_matches += matches
        sum_max_matches += max_matches

    printOUT("")
    if sum_max_matches > 0:
        printOUT("Percentual de acerto = {:5.2f}%".format(float(sum_matches)/sum_max_matches*100.0))
    
    horaFinalProcesso = datetime.now()

    arquivoCSV.close()
    if SAVE_PERCENTUAL_ACCURACY:
        arquivoAcuracia.close()

    tempo_processamento = horaFinalProcesso - horaInicialProcesso
    printOUT("Tempo de processamento = {}".format(tempo_processamento))
    printOUT("Número de imagens processadas = {}".format(CONTADOR_ARQUIVOS))
    printOUT("Tempo médio de processamento por imagem = {}".format(tempo_processamento / CONTADOR_ARQUIVOS))
    printOUT("=======================================================================================")
    NCPFS = CPF_OK + CPF_NOK
    if NCPFS > 0:
        printOUT("CPFs testados: {} | CPFs com Erro: {} | Taxa de Acerto de CPF: {:5.2f}%".format(
            NCPFS, CPF_NOK, float(CPF_OK)*100/(NCPFS)))
    else:
        printOUT("Não foram testados CPFs nessa imagem ou grupo de imagens")

    OUTFILE.close()
    
if __name__ == "__main__":
    main()
