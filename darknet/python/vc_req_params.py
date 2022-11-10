#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vc_constants
import vc_ocr
import vc_utils

class RequestParams:

    def __init__(self, req_header):
        self.crop_height             = vc_utils.dictValue(req_header, vc_constants.FIELD_CROP_HEIGHT, vc_constants.CROP_HEIGHT_DEFAULT, "int")
        self.max_resize_width        = vc_utils.dictValue(req_header, vc_constants.FIELD_RESIZE, 800, 'int')
        self.use_resize              = self.max_resize_width > vc_constants.DETECT_OCR_MIN_WIDTH and self.max_resize_width <= 4 * vc_constants.DETECT_OCR_MIN_WIDTH
        self.multiline_field_preproc = vc_utils.dictValue(req_header, vc_constants.FIELD_MULTILINE_PREPROC, 3, 'int')
        self.common_field_preproc    = vc_utils.dictValue(req_header, vc_constants.FIELD_COMMON_PREPROC, 1, 'int')
        self.req_header_align        = vc_utils.dictValue(req_header, vc_constants.FIELD_ALIGN, 1, 'int')
        self.try_rotate_document     = self.req_header_align == 1
        self.ocr_kind                = vc_utils.dictValue(req_header, vc_constants.FIELD_OCR_KIND, vc_constants.OCR_KIND_DEFAULT, "int")
        new_ocr_kind, ocr_name       = vc_ocr.availableOcrEngine(self.ocr_kind)
        self.ocr_kind                = new_ocr_kind

        # Log request parameters
        if self.use_resize:
            vc_utils.printLog(f"Usando 'resizing' (max={self.max_resize_width})")
        else:
            vc_utils.printLog("Não usando 'resizing'")
        vc_utils.printLog("Tentativa de rotação: " + ('SIM' if self.try_rotate_document else 'NÃO'))
        vc_utils.printLog("Preprocessamento de Campos Comuns: {}".format(vc_utils.preprocessingName(self.common_field_preproc)))
        vc_utils.printLog("Preprocessamento de Campos Multilinha: {}".format(vc_utils.preprocessingName(self.multiline_field_preproc)))
        vc_utils.printLog(f"Motor OCR Padrão: {ocr_name}")
        vc_utils.printLog(f"Altura padrão de Crop: {self.crop_height}")
