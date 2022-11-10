#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VCDOC - Visual Cognitive Document
Using darknet yolo
"""

import base64
import logging
import os
import sys

import cv2
import numpy as np
import PIL
import responder
import spellchecker
from responder import status_codes as sc

import vc_audit
import vc_autentikus
import vc_constants
import vc_doc_image
import vc_excepts
import vc_img_process
import vc_metrics
import vc_ocr
import vc_utils
import vc_yolo
import vc_spell

class VCDOCServer:
    """
    VCDOC Server
    """
    
    def __init__(self, host, port, jwksaddress, path_upload, logger):
        """
        init server class
        """
        self.api = responder.API(cors=True, cors_params={'allow_origins': ['*'],'allow_methods': ['*'],'allow_headers': ['*'],}, allowed_hosts=["*"])
        self.host = host
        self.port = port
        self.define_uri()
        self.logger = logger
        self.multiline_field_preproc = 3
        self.common_field_preproc = 1
        self.use_resize = True
        self.max_resize_width = 800
        self.try_rotate_document = True
        self.ocr_kind = vc_constants.OCR_KIND_DEFAULT
        self.crop_height = vc_constants.CROP_HEIGHT_DEFAULT
        self.jwksaddress = jwksaddress
        self.path_upload = path_upload # usado para salvar imagens temporárias. No momento está sendo usado para imagens obtidas de PDF's. 
                                       # Algumas referências:
                                       # - Arquivo 'vcdoc_server.py' (este arquivo): Função 'detect_pdf' é o end-point da aplicação que recebe PDF's.
                                       # - Arquivo 'vc_pdf2img': Contém funções para tratamento de PDF's
        self.rotation_angle = 0
        vc_autentikus.initialize_public_key_autentikus(jwksaddress)

    def define_uri(self):
        """
        definition of uri
        """
        self.api.add_route('/v1/detect', self.detect) # faz a detecção (Yolo), não faz o OCR, e no JSON de retorno só vem as classes mais externas (CNH, CNH verso, RG frente, RG verso)
        self.api.add_route('/v1/detect-ocr', self.detect_ocr) # faz a detecção (Yolo), faz o OCR, e no JSON de retorno vem todas as classes, sejam externas ou dos campos internos
        self.api.add_route('/v1/detect-ocr-image', self.detect_ocr_image) # igual ao detect-ocr, porém a imagem na requisição não é base64, e sim a imagem binária normal. 
        self.api.add_route('/v1/detect-all', self.detect_all) # faz a detecção, não faz o OCR, e no JSON de retorno vem todas as classes, sejam externas ou dos campos internos
        self.api.add_route('/v1/metrics', self.gather_metrics) # usado pelo prometheus/grafana

    async def detect_ocr_image(self, req, resp):
        vc_utils.printLog("Chamada da API: 'detect-ocr-image'", self.logger)
        if not vc_autentikus.securityVerification(req,resp): 
            return
        try:
            ifile = await req.media(format='files')
            fileNameRequisicao = ifile['Image']['filename']
            image              = ifile['Image']['content'] # retorna array de bytes

            req_data = base64.b64encode(image).decode('utf-8')
            document = vc_doc_image.DocumentImage(req.headers, req_data)
            document.runWholeProcess()
            document.getResultJSON()['original_filename'] = fileNameRequisicao
            vc_utils.setSuccessResponseJSON(resp, document.getResultJSON())
            
            vc_metrics.includeRequestTimesMetrics(
                detect_yolo_time = vc_utils.dictValue(resp.media, 'detect_time_secs', 0),
                detect_ocr_time  = vc_utils.dictValue(resp.media, 'detect_ocr_time', 0))

        except Exception as exc:
            resp.status_code, resp.media = vc_excepts.convertExceptionToHtmlResponse(exc)

    async def extrairImagemDaRequisicao(self, req):
        req_header = req.headers
        # { for k in req_header}
        if req_header['content-type'] == "application/octet-stream":
            vc_utils.printLog('Requisição contém uma imagem em formato application/octet-stream')
            req_data = await req.content # bytes
        else:
            req_data = await req.media()
            req_data = vc_utils.dictValue(req_data,vc_constants.FIELD_IMAGE)
            if req_data is None:
                raise vc_excepts.RequestDoesntHaveImageError("Requisição não possui imagem")
            else:
                if type(req_data) is bytes:
                    req_data = req_data.decode() # str
                vc_utils.printLog('Requisição contém uma imagem no formato string base64')
        return req_header, req_data

    def gather_metrics(self, req, resp):
        resp.status_code = sc.HTTP_200
        resp.content = vc_metrics.snapshot()

    async def detect_ocr(self, req, resp):
        vc_utils.printLog("Chamada da API: 'detect-ocr'", self.logger)
        if not vc_autentikus.securityVerification(req,resp): 
            return
        try:
            req_header, req_data = await self.extrairImagemDaRequisicao(req)
            document = vc_doc_image.DocumentImage(req_header, req_data)
            document.runWholeProcess()
            vc_utils.setSuccessResponseJSON(resp, document.getResultJSON())
            
            vc_metrics.includeRequestTimesMetrics(
                detect_yolo_time = vc_utils.dictValue(resp.media, 'detect_time_secs', 0),
                detect_ocr_time  = vc_utils.dictValue(resp.media, 'detect_ocr_time', 0))

        except Exception as exc:
            resp.status_code, resp.media = vc_excepts.convertExceptionToHtmlResponse(exc)

    async def detect_all(self, req, resp):
        vc_utils.printLog("Chamada da API: 'detect-all'", self.logger)
        if not vc_autentikus.securityVerification(req,resp): 
            return
        try:
            req_header, req_data = await self.extrairImagemDaRequisicao(req)
            document = vc_doc_image.DocumentImage(req_header, req_data)
            document.runWholeProcess(run_ocr=False)
            vc_utils.setSuccessResponseJSON(resp, document.getResultJSON())
            
            vc_metrics.includeRequestTimesMetrics(
                detect_yolo_time = vc_utils.dictValue(resp.media, 'detect_time_secs', 0),
                detect_ocr_time  = vc_utils.dictValue(resp.media, 'detect_ocr_time', 0))

        except Exception as exc:
            resp.status_code, resp.media = vc_excepts.convertExceptionToHtmlResponse(exc)

    async def detect(self, req, resp):
        vc_utils.printLog("Chamada da API: 'detect'", self.logger)
        if not vc_autentikus.securityVerification(req,resp): 
            return
        try:
            req_header, req_data = await self.extrairImagemDaRequisicao(req)
            document = vc_doc_image.DocumentImage(req_header, req_data)
            document.runWholeProcess(calculate_additional_fields=False, run_ocr=False, filter_fields=True)
            vc_utils.setSuccessResponseJSON(resp, document.getResultJSON())

            vc_metrics.includeRequestTimesMetrics(
                detect_yolo_time = vc_utils.dictValue(resp.media, 'detect_time_secs', 0),
                detect_ocr_time  = vc_utils.dictValue(resp.media, 'detect_ocr_time', 0))

        except Exception as exc:
            resp.status_code, resp.media = vc_excepts.convertExceptionToHtmlResponse(exc)

    def run_server(self):
        """
        run server
        """
        try:
            self.api.run(address=self.host,
                        port=self.port,
                        logger=self.logger) 
        except Exception as e:
            if "unexpected keyword argument 'logger'" in str(e):
                self.api.run(address=self.host, port=self.port)
            else:
                raise

def include_path_prog(path_prog,some_path):
    return some_path if some_path.startswith(os.sep) or some_path.startswith('~') else os.path.join(path_prog,some_path)

def inicializa_variaveis_globais(path_prog):
    vc_utils.forceDirectory(os.path.join(path_prog,'log'))
    vc_img_process.inicializa_image_models(path_prog)
    vc_ocr.inicializa_easyocr(path_prog)
    vc_spell.inicializa_spellchecker(path_prog)

def main():
    """
    Main
    """
    vc_metrics.inicializa_metricas()
    # parte de código para manter a compatiblidade para rodar tanto em servidor como em computador local
    path_prog = sys.path[0]
    path_opt = path_prog
    #path_opt = "/opt/appfiles/hom_11674_vcdoc"
    #vc_utils.forceDirectory(path_opt)
    #if not os.path.exists(path_opt):
    #    path_opt = path_prog
    
    # inicializar auditoria
    _, path_upload = vc_audit.inicializa_audit(path_opt)
    
    inicializa_variaveis_globais(path_prog)
    
    darknet_server_conffilepath = os.path.join(path_prog,"conf","darknet_server.ini")
    namesfilepath, cfgfilepath, weightfilepath, host, port, logfilepath, jwksaddress = vc_utils.getConfigParameters(darknet_server_conffilepath)

    # Nas 4 linhas abaixo, estou obrigando que os paths dos arquivos names,
    # cfg, weight e log sejam diretórios relativos ao path do programa.
    namesfilepath = include_path_prog(path_prog, namesfilepath)
    cfgfilepath = include_path_prog(path_prog,cfgfilepath)
    weightfilepath = include_path_prog(path_prog,weightfilepath)
    logfilepath = include_path_prog(path_prog,logfilepath)

    logging.basicConfig(filename=logfilepath,
                        format="[%(asctime)s]\t[%(levelname)s]\t%(message)s",
                        level=logging.INFO)
    logger = logging.getLogger("darknet_server")

    vc_utils.printLog(f"VCDOC Server - Yolo DNN", logger)
    vc_utils.printLog(f"=======================", logger)
    vc_utils.printLog(f"Bibliotecas:", logger)
    vc_utils.printLog(f" - cv2: {cv2.__version__}", logger)
    vc_utils.printLog(f" - numpy: {np.__version__}", logger)
    vc_utils.printLog(f" - PIL: {PIL.__version__}", logger)
    vc_utils.printLog(f" - spellchecker: {spellchecker.__version__}", logger)
    vc_utils.printLog(f"=======================", logger)
    vc_utils.printLog(f"Configuração do Yolo: {cfgfilepath}", logger)
    vc_utils.printLog(f"Pesos da Rede do Yolo: {weightfilepath}", logger)
    vc_utils.printLog(f"Arquivo de Nomes: {namesfilepath}", logger)
    vc_utils.printLog(f"Arquivo de Log: {logfilepath}", logger)
    vc_utils.printLog(f"Host: {host}", logger)
    vc_utils.printLog(f"Port: {port}", logger)
    if vc_ocr.VCDOC_USE_EASYOCR: vc_utils.printLog("EasyOCR Engine disponível")
    if vc_ocr.VCDOC_USE_TESSERACT: vc_utils.printLog("Tesseract-OCR Engine disponível")
    
    #vc_utils.printLog("Verificação dos parâmetros", logger)
    if vc_utils.checkParameters(namesfilepath, cfgfilepath, weightfilepath, host, port, logger):
        #vc_utils.printLog("Parâmetros OK. Iniciando verificação do processo darknet", logger)
        
        vc_yolo.Load_DNN(namesfilepath,cfgfilepath,weightfilepath)

        server = VCDOCServer(host, port, jwksaddress, path_upload, logger)
        vc_utils.printLog("Iniciando API Server", logger)
        server.run_server()
    else:
        vc_utils.printErr('Erro de validação dos parâmetros', logger)
        sys.exit(1)

if __name__ == "__main__":
    main()
