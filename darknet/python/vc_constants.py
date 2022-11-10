#!/usr/bin/env python3
# -*- coding: utf-8 -*-

THRESH_QUALITY_CHECKER = 0.8
THRESH_DETECTION = 0.9
FIELD_ENV = "env"
FIELD_IMAGE = "image"
FIELD_RESIZE            = "vcdoc_param_largura_de_redimensionamento_imagem" # Largura para resize da imagem inteira. 
                                                                            # Esperado valor inteiro. Valor padrão = 800

FIELD_ALIGN             = "vcdoc_param_avaliar_rotacao"                     # Indica se será feita rotação na imagem, para melhorar a 
                                                                            # obtenção dos campos do documento1 
                                                                            # Esperado valor 0 (não  faz rotação) ou 1 (tenta rotação).
FIELD_CROP_HEIGHT       = "vcdoc_param_crop_height"
FIELD_OCR_KIND          = "vcdoc_param_ocr_solicitado"
FIELD_MULTILINE_PREPROC = "vcdoc_param_preprocessamento_multiline"
FIELD_COMMON_PREPROC    = "vcdoc_param_preprocessamento_singleline"
FIELD_STATUS = "status"
FIELD_MSG = "msg"
FIELD_CLASS_INDEX = "class_index"
FIELD_OBJ_NAME = "obj_name"
FIELD_PRED_IMG = "pred_img"
FIELD_WORK_IMG = "work_img"
FIELD_FACE_IMG = "face_img"
FIELD_RESULT_LIST = "resultlist"
FIELD_BOUNDING_BOX = "bounding_box"
FIELD_X_MIN = "x_min"
FIELD_Y_MIN = "y_min"
FIELD_WIDTH = "width"
FIELD_HEIGHT = "height"
FIELD_OCR_TEXT = "ocr_text"
FIELD_ADJUSTED_OCR = "adjusted_ocr"
FIELD_ANGLE = "angle"
STATUS_SUCCESS = "success"
STATUS_FATAL = "fatal"
DETECT_OCR_MIN_WIDTH = 450
OCR_KIND_NONE = 0
OCR_KIND_TESSERACT = 1
OCR_KIND_EASYOCR = 2
OCR_KIND_DEFAULT = OCR_KIND_EASYOCR
CROP_HEIGHT_DEFAULT = 48
PDF_WIDTH_DEFAULT = 720

PREPROC_BORDER_REMOVAL_1 = 1
PREPROC_BORDER_REMOVAL_2 = 2
PREPROC_BORDER_REMOVAL_3 = 4

IGNORE_AUTENTIKUS = True
IGNORE_SPELL = False
IGNORE_DATA_SPELL = True