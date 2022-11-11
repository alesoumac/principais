#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
darknet yolo server
"""

import base64
import configparser
import io
import os
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import cnh_process as cnhp
import constants
from darknet import Darknet

SERVIDOR = None

# -------------------------------------------
# ------------- algumas funções básicas úteis
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

def killProcessDarknet(nproc):
    _, s = subprocess.getstatusoutput('ps -f -p {}'.format(nproc))
    if "darknet_server" in s:
        print("Process #{} terminated".format(nproc))
        os.kill(nproc,9)

def force_directory(direc, deletingFiles = False):
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
# -------------------------------------------


class DarknetServer:
    """
    Darknet Yolo server
    """
    
    def __init__(self, upload_dir, yolo):
        """
        init server class
        """
        self.upload_dir = upload_dir
        self.output_dir = os.path.join(os.path.split(upload_dir)[0],'out_images')
        self.yolo = yolo
        self.initialTime = None
        self.finishTime = None

    def get_image_rgb_from_base64(self, img_data_base64):
        img_np_rgb = Image.open(io.BytesIO(img_data_base64))
        try:
            deg = {3:180, 6:270, 8:90}.get(img_np_rgb._getexif().get(274, 0),0)
        except:
            deg = 0
            
        img_np_rgb = np.array(img_np_rgb.convert("RGB"))
        if deg != 0:
            img_np_rgb = cnhp.rotate_image(img_np_rgb,deg)

        return img_np_rgb

    def get_yolo_results(self, img_data, thresh=0.9):
        """
        get result
        """
        img_np_rgb = None
        try:
            img_np_rgb = self.get_image_rgb_from_base64(img_data)
            yolo_results = self.yolo.detect(img_np_rgb, thresh)
        except Exception as err:
            print(str(err))

        return img_np_rgb, yolo_results

    def create_filename_png(self, base_path, base_name, process_instant):
        dir_name = os.path.join(base_path, process_instant.strftime("%Y%m%d"))
        force_directory(dir_name, False)
        file_name = "{}_{}.png".format(base_name,process_instant.strftime("%Y%m%d_%H%M%S"))
        return os.path.join(dir_name,file_name)

    def save_to_file(self, img_filename, img_bytes):
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_pil.save(img_filename)
        return img_pil.size

    def detect_json(self, req_data, class_filter = ()):
        img_np_rgb = None
        nomearq_request = ""
        nomearq_response = ""
        resp_status_code = None
        resp_media = {}

        if constants.FIELD_IMAGE in req_data:
            img_data = base64.b64decode(req_data[constants.FIELD_IMAGE])
            nomearq_request = self.create_filename_png(self.upload_dir, 'vcdoc_imgreq', datetime.now())
            img_larg, img_alt = self.save_to_file(nomearq_request, img_data)

            #quality checker
            quality = self.quality_checker(img_data)
            if quality < constants.THRESH_QUALITY_CHECKER:
                resp_status_code = sc.HTTP_412
                resp_media = {constants.FIELD_STATUS: constants.STATUS_FATAL,
                    constants.FIELD_MSG: "Quality checker fail"}
            else:
                #darknet object detection
                img_np_rgb, result = self.yolo_detect(img_data, constants.THRESH_DETECTION)

                if result[constants.FIELD_STATUS] == constants.STATUS_FATAL:
                    resp_status_code = sc.HTTP_500
                else:
                    #filter results
                    result = self.filter_result(result, class_filter, img_larg, img_alt)

                #return result
                resp_media = result

            del req_data[constants.FIELD_IMAGE]

        else:
            resp_status_code = sc.HTTP_400
            resp_media = {constants.FIELD_STATUS: constants.STATUS_FATAL,
                            constants.FIELD_MSG: "request should have the key of image"}

        return resp_status_code, resp_media, img_np_rgb, nomearq_request, nomearq_response


    def detect_ocr(self, img_ba64):
        """
        Detection using yolo. '/v1/detect-ocr'
        """
        self.hora_inicio = datetime.now()
        
        req_data = {"image": img_ba64, "get_img_flg": True}

        #quality checker for OCR
        img_base64 = base64.b64decode(req_data[constants.FIELD_IMAGE])
        quality = self.quality_checker_ocr(img_base64)
        if quality < constants.THRESH_QUALITY_CHECKER:
            resp_status_code = 412
            resp_media = {constants.FIELD_STATUS: constants.STATUS_FATAL,
                constants.FIELD_MSG: "Quality checker fail"}
            return resp_status_code, resp_media

        resp_status_code, resp_media, img_np_rgb, nomearq_request, nomearq_response = self.detect_json(req_data)

        if resp_status_code is None:
            resp_status_code = 200
            resp_media = self.classify_and_apply_ocr(img_np_rgb, resp_media)

        self.hora_fim = datetime.now()
        return resp_status_code, resp_media

    def quality_checker(self, img_data):
        quality = 1.0
        return quality

    def quality_checker_ocr(self, img_data_base64):
        quality = 1.0

        image = self.get_image_rgb_from_base64(img_data_base64)
        _, width = image.shape[:2]

        if width < constants.DETECT_OCR_MIN_WIDTH:
            quality = 0.0

        return quality
    
    def classify_and_apply_ocr(self, image_np_rgb, json):
        non_text_fields = ("cnh", "cnh_frente", "foto_cnh", "rg_frente", "rg_verso", "foto_rg", "assinatura_rg", "digital_rg")
        
        if cnhp.is_class(json, "cnh") or cnhp.is_class(json, "cnh_frente"):
            scale, cnh_resized = cnhp.resize_cnh(image_np_rgb, desired_width = cnhp.MAX_WIDTH, try_resize = cnhp.TRY_RESIZE)

            resultlist = json[constants.FIELD_RESULT_LIST]
            alturas = []
            altura_filiacao = 0
            for n in range(len(resultlist)):
                ocr_text = ""
                spell_ocr = ""
                element = resultlist[n]
                obj_name = element[constants.FIELD_OBJ_NAME]
                if obj_name.lower() not in non_text_fields:
                    bbox = element[constants.FIELD_BOUNDING_BOX]
                    x_min, y_min, w, h = \
                        int(bbox[constants.FIELD_X_MIN] * scale), \
                        int(bbox[constants.FIELD_Y_MIN] * scale), \
                        int(bbox[constants.FIELD_WIDTH] * scale), \
                        int(bbox[constants.FIELD_HEIGHT] * scale)
                    x_max, y_max = x_min + w, y_min + h
                    crop_np_rgb = cnhp.crop_image(cnh_resized, x_min, y_min, x_max, y_max)
                    h, w = crop_np_rgb.shape[:2]
                    if w > 0 and h > 0:
                        ocr_text,altura_ocr = cnhp.detectar_campo_cnh(obj_name, crop_np_rgb)
                        spell_ocr = cnhp.spell_verify(ocr_text)
                        if obj_name in ["cnh","cnh_frente"] \
                        or obj_name.lower().startswith("rg") \
                        or obj_name.lower().endswith("rg"):
                            pass
                        elif obj_name == "filiacao_cnh":
                            altura_filiacao = altura_ocr
                        else:
                            if altura_ocr > 0: alturas += [altura_ocr]
                
                json[constants.FIELD_RESULT_LIST][n][constants.FIELD_OCR_TEXT] = ocr_text
                json[constants.FIELD_RESULT_LIST][n][constants.FIELD_ADJUSTED_OCR] = spell_ocr
            
            json["altura_media"] = np.mean(alturas) if alturas != [] else 0
            json["altura_filiacao"] = altura_filiacao

        return json

    def filter_result(self, result, filter, x_limit, y_limit):
        if constants.FIELD_PRED_IMG in result:
            del result[constants.FIELD_PRED_IMG]

        if len(filter) > 0:
            result_filtered = result.copy()
            result_filtered[constants.FIELD_RESULT_LIST] = []
    
            for element in result[constants.FIELD_RESULT_LIST]:
                if element[constants.FIELD_OBJ_NAME].lower() in filter:    
                    result_filtered[constants.FIELD_RESULT_LIST].append(element)
        else:
            result_filtered = result.copy()

        for element in result_filtered[constants.FIELD_RESULT_LIST]:
            bbox = element[constants.FIELD_BOUNDING_BOX]
            x_min = bbox[constants.FIELD_X_MIN]
            y_min = bbox[constants.FIELD_Y_MIN]
            w = bbox[constants.FIELD_WIDTH]
            h = bbox[constants.FIELD_HEIGHT]
            x_max, y_max = x_min + w, y_min + h
            x_min = int(max(x_min,0))
            y_min = int(max(y_min,0))
            x_max = int(min(x_max,x_limit))
            y_max = int(min(y_max,y_limit))
            bbox[constants.FIELD_X_MIN]  = x_min
            bbox[constants.FIELD_Y_MIN]  = y_min
            bbox[constants.FIELD_WIDTH]  = x_max - x_min
            bbox[constants.FIELD_HEIGHT] = y_max - y_min
        
        return result_filtered

    def yolo_detect(self, img_data, thresh):
        img_np_rgb = None
        try:
            img_np_rgb, yolo_results = self.get_yolo_results(img_data, thresh)
            res = {constants.FIELD_STATUS: constants.STATUS_SUCCESS,
                    constants.FIELD_RESULT_LIST: [yolo_result.get_detect_result() for yolo_result in yolo_results]}

            return img_np_rgb, res

        except Exception as err:
            print(str(err))
            return img_np_rgb, {constants.FIELD_STATUS: constants.STATUS_FATAL,
                    constants.FIELD_MSG: "An error occured in server"}


def get_params(configfilepath):
    """
    getting parameter from config
    """

    config = configparser.ConfigParser()
    config.read(configfilepath)

    try:

        # for YOLO
        darknetlibfilepath = config.get('YOLO', 'darknetlibfilepath')
        datafilepath = config.get('YOLO', 'datafilepath')
        cfgfilepath = config.get('YOLO', 'cfgfilepath')
        weightfilepath = config.get('YOLO', 'weightfilepath')

        # for Server
        host = config.get('Server', 'host')
        port = config.getint('Server', 'port')
        logfilepath = config.get('Server', 'logfilepath')
        # A pasta de upload não será mais obtida do arquivo de parãmetros ".ini"
        #uploaddir = config.get('Server', 'uploaddir')
        uploaddir = ""
        return darknetlibfilepath, datafilepath, cfgfilepath, \
            weightfilepath, logfilepath, uploaddir

    except configparser.Error as config_parse_err:
        raise config_parse_err


def check_path(targetpath):
    """
    checking path
    """
    check_flg = None
    if not os.path.exists(targetpath):
        print('%s does not exist' % targetpath)
        check_flg = False
    else:
        check_flg = True
    return check_flg


def check_params(darknetlibfilepath, datafilepath, cfgfilepath, weightfilepath):
    """
    checking parameters
    """

    validation_flg = True
    for targetpath in [darknetlibfilepath, datafilepath,
                       cfgfilepath, weightfilepath]:
        validation_flg = check_path(targetpath)

    return validation_flg

def inicializa_srv(path_prog, use_r2, use_resize, larg_maxima):
    global SERVIDOR
    cnhp.USE_RECT_2 = use_r2
    cnhp.TRY_RESIZE = use_resize
    cnhp.MAX_WIDTH = larg_maxima
    path_audit = os.path.join(path_prog,".temp__","audit")
    path_upload = os.path.join(path_audit,"images")
    force_directory(path_upload)
    cnhp.inicializa_variaveis_globais(path_prog)
    
    darknet_server_conffilepath = os.path.join(path_prog,"conf","darknet_server.ini")
    darknetlibfilepath, datafilepath, cfgfilepath, \
        weightfilepath, logfilepath, \
        _ = get_params(darknet_server_conffilepath)

    if check_params(darknetlibfilepath, datafilepath, cfgfilepath, weightfilepath):

        darknet = Darknet(libfilepath=darknetlibfilepath,
                          cfgfilepath=cfgfilepath.encode(),
                          weightsfilepath=weightfilepath.encode(),
                          datafilepath=datafilepath.encode())

        darknet.load_conf()

        SERVIDOR = DarknetServer(path_upload, darknet)
    else:
        print("Erro ao inicializar o serviço darknet")
        sys.exit(1)
