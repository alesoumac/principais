#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64

import cv2

import vc_clocks
import vc_constants
import vc_excepts
import vc_fields
import vc_img_process
import vc_req_params
import vc_utils
import vc_yolo

EXTERNAL_DOC_FIELDS = ['cnh', 'cnh_frente', 'rg_frente', 'rg_verso']

POSSIBLE_DOC_CLASSES = ['cnh','rg']

ALIGNED_FIELDS_LIST = {}
ALIGNED_FIELDS = {
    'cnh': [
        ('cpf_cnh','nascimento_cnh'),
        ('registro_cnh','validade_cnh','pri_habilitacao_cnh'),
        ('local_emissao_cnh','data_emissao_cnh')
    ],
    'rg': []
}
# A variável acima ALIGNED_FIELDS contém as sequencias de campos que devem estar 
# alinhados em cada classe de documento, e será usada na função "calculateAngle",
# que calcula o ângulo de rotação, para ajustar a imagem do documento, fazendo com
# que ele fique o mais horizontal possível. Por enquanto, só calculamos ângulo de 
# rotação para documentos do tipo CNH.
#  
# Obs: Esse cálculo do ângulo é um código sensível a mudanças no layout da CNH,
# pois ele "confia" que os seguintes campos da CNH estejam visualmente na mesma linha:
#   1) "CPF" e "Data de Nascimento"
#   2) "Nº de Registro", "Data de Validade" e "Data da 1ª Habilitação"
#   3) "Local de Emissão" e "Data de Emissão"
# Caso tenhamos uma mudança no layout da CNH, será necessário adaptar esse código,
# colocando outras sequencias de campos para serem testados

class DocumentImage:

    def __init__(self, request_headers, img_b64_or_bytes):
        self.originalImage = None
        self.workImage = None
        self.rotationAngle = 0
        self.extractedFields = []
        self.resultJSON = None
        self.requestParams = vc_req_params.RequestParams(request_headers)
        self.scale = 1.0
        self.timers = vc_clocks.ClockTimer()
        self.filterFields = False
        self.loadImageFromRequest(img_b64_or_bytes)

    # ----------------------------------------------------------------
    def quality_checker(self):
        width = self.originalImage.shape[1] # shape[1] é a largura
        if width < vc_constants.DETECT_OCR_MIN_WIDTH:
            quality = 0.0
        else:
            quality = 1.0
        return quality

    # ----------------------------------------------------------------
    def loadImageFromRequest(self, req_b64_or_bytes):
        try:
            self.timers.startClock('load_image')
            if type(req_b64_or_bytes) is str: # string base64
                try:
                    base64Image = base64.b64decode(req_b64_or_bytes)
                except:
                    raise vc_excepts.InvalidBase64ImageError("Base64 da imagem inválido")
                self.originalImage = vc_img_process.get_image_bgr_from_base64(base64Image)
            else:
                self.originalImage = vc_img_process.get_image_bgr_from_bytes(req_b64_or_bytes)
            quality = self.quality_checker()
            if quality < vc_constants.THRESH_QUALITY_CHECKER:
                raise vc_excepts.ImageQualityCheckerError("Falha na verificação das pré-condições (Quality Checker)")
        finally:
            self.timers.endClock('load_image')

    # ----------------------------------------------------------------
    def testJsonWithResultList(self):
        json = self.resultJSON
        return json is not None \
               and vc_constants.FIELD_RESULT_LIST in json

    # ----------------------------------------------------------------
    def documentClassIs(self, doc_class):
        '''
        Verifica se o documento pertence à classe contida no parâmetro doc_class.
        Parâmetros:
            doc_class: string ou lista de string, contendo o(s) nome(s) da(s) classe(s)
                       a ser testada(s) 
        Retorno:
            Se o documento for de uma classe contida no parâmetro doc_class, retorna True.
            Do contrário, retorna False
        '''
        global POSSIBLE_DOC_CLASSES

        if type(doc_class) is list:
            # Se o parâmetro doc_class for uma lista, verifico se o documento é de uma das classes dentro dessa lista.
            # Por exemplo:
            #     instanciaDocumento.documentClassIs(['rg_frente','rg_verso'])
            #     retorna True, se instanciaDocumento for uma rg_frente ou uma rg_verso
            for a_class in doc_class:
                if self.documentClassIs(a_class):
                    return True
            return False

        # Se o parâmetro doc_class não é lista, então tem que ser uma string
        # contendo o nome da classe que se deseja testar (ex: "cnh", "rg_frente")
        if not self.testJsonWithResultList():
            return False

        class_lower = doc_class.lower()
        sufix = ""
        for main_class in POSSIBLE_DOC_CLASSES:
            if class_lower.startswith(main_class):
                sufix = main_class
                break

        returnValue = None
        for element in self.resultJSON[vc_constants.FIELD_RESULT_LIST]:
            element_name = element[vc_constants.FIELD_OBJ_NAME].lower()
            if element_name == class_lower:
                return True
            if sufix == "" or returnValue == False:
                continue
            returnValue = True
            if not element_name.startswith(sufix) and not element_name.endswith(sufix):
                returnValue = False

        if returnValue is None:
            returnValue = False

        return returnValue

    # ----------------------------------------------------------------
    def calculateAngle(self):
        global POSSIBLE_DOC_CLASSES
        global ALIGNED_FIELDS
        global ALIGNED_FIELDS_LIST

        classToAlign = None
        for docClass in POSSIBLE_DOC_CLASSES:
            if self.documentClassIs(docClass):
                classToAlign = docClass
                break

        possibleAngle = None
        if classToAlign is not None:
            self.timers.startClock('calculate_angle')

            # Primeiro, obtemos os centros dos campos que serão usados para testar
            # o alinhamento da imagem
            centers = {}
            for detec in self.resultJSON[vc_constants.FIELD_RESULT_LIST]:
                obj_name = detec['obj_name'].lower()
                bbox = detec['bounding_box']
                if obj_name in ALIGNED_FIELDS_LIST[classToAlign]:
                    x_min,y_min,width,height = list(bbox.values())
                    centers[obj_name] = (x_min + width // 2, y_min + height // 2)

            # Com estes centros, podemos testar o alinhamento da imagem usando pares desses campos
            possibleAngle = None
            for c1,c2 in ALIGNED_FIELDS[classToAlign]:
                if c1 in centers and c2 in centers:
                    possibleAngle = vc_img_process.calculateAngleBetweenPoints(centers[c1], centers[c2])
                    break

            self.timers.endClock('calculate_angle')

        if possibleAngle is not None:
            possibleAngle = vc_img_process.adjustAngle( - possibleAngle )
            return possibleAngle

        # If possibleAngle is None, then we'll reach this point of program, 
        # and it means that it wasn't possible to align the image by pair of fields
        # so we have to find the rotation angle by other ways
        self.timers.startClock('calculate_angle')
        possibleAngle = vc_img_process.find_best_angle(self.workImage)
        self.timers.endClock('calculate_angle')
        return possibleAngle

    # ----------------------------------------------------------------
    def runYolo(self):
        try:
            self.timers.startClock('yolo')
            # Prepara a imagem para processamento do yolo, guardando no atributo self.workImage.
            # Para isso, faz o redimensionamento para uma resolucao menor se a imagem for muito grande:
            if self.workImage is None:
                self.workImage = self.originalImage
                self.rotationAngle = 0
                needs_scale = self.documentClassIs(['cnh','cnh_frente','rg_frente'])
                if needs_scale:
                    self.timers.startClock('yolo.shrink')
                    self.scale, doc_resized = vc_img_process.shrinkToWidth(
                        self.workImage, 
                        desired_width = self.requestParams.max_resize_width, 
                        try_resize    = self.requestParams.use_resize)
                    self.workImage = doc_resized
                    self.timers.endClock('yolo.shrink')
                else:
                    self.scale = 1.0

            # Faz a chamada ao Yolo, eventualmente tentando duas vezes caso seja solicitada uma tentativa de rotação:
            self.timers.startClock('yolo.detect')
            self.resultJSON = vc_yolo.yoloDetectDNN(self.workImage, vc_constants.THRESH_DETECTION)
            self.timers.endClock('yolo.detect')
            self.resultJSON[vc_constants.FIELD_ANGLE] = self.rotationAngle
            if self.requestParams.try_rotate_document:
                possibleAngle = self.calculateAngle()
                if abs(possibleAngle) < 2:
                    return
                vc_utils.printLog(f"Tentativa com rotação da imagem - [{int(possibleAngle)}°]")
                self.rotationAngle = possibleAngle
                self.timers.startClock('yolo.rotation')
                self.workImage = vc_img_process.rotateImage(self.workImage, possibleAngle)
                self.timers.endClock('yolo.rotation')
                self.timers.startClock('yolo.detect')
                self.resultJSON = vc_yolo.yoloDetectDNN(self.workImage, vc_constants.THRESH_DETECTION)
                self.timers.endClock('yolo.detect')
                self.resultJSON[vc_constants.FIELD_ANGLE] = self.rotationAngle
        finally:
            self.timers.endClock('yolo')

    def runCalculateAdditionalFields(self):
        # self.timers.startClock('calc_additional_fields')
        # processMissingPhoto(self.getResultJSON(), self.workImage)
        # self.timers.endClock('calc_additional_fields')
        return

    # ----------------------------------------------------------------
    def getResultJSON(self):
        global EXTERNAL_DOC_FIELDS
        if self.resultJSON is None:
            raise vc_excepts.NeededToRunYoloDetectionError("O método de detecção (runYolo) não foi executado.")
        if self.filterFields:
            res = self.resultJSON[vc_constants.FIELD_RESULT_LIST]
            for n in range(len(res))[::-1]:
                element = res[n]
                obj_name = element[vc_constants.FIELD_OBJ_NAME].lower()
                if obj_name not in EXTERNAL_DOC_FIELDS:
                    del(res[n])
        return self.resultJSON

    # ----------------------------------------------------------------
    def runDocumentOCR(self):
        if not self.testJsonWithResultList():
            return
        
        try:
            self.timers.startClock('ocr')
            doc_original_rotated = vc_img_process.rotateImage(self.originalImage, self.rotationAngle)
            resultlist = self.resultJSON[vc_constants.FIELD_RESULT_LIST]

            doc_type = ''
            if self.documentClassIs(['cnh','cnh_frente']):
                doc_type = 'cnh'
            elif self.documentClassIs(['rg_frente','rg_verso']):
                doc_type = 'rg'

            for n in range(len(resultlist)):
                element = resultlist[n]
                obj_name = element[vc_constants.FIELD_OBJ_NAME].lower()
                
                newField = self.createNewField(obj_name, doc_type)
                preproc_kind = \
                    self.requestParams.multiline_field_preproc \
                    if newField.isMultilineField else          \
                    self.requestParams.common_field_preproc

                if newField.preprocessingKind is not None:
                    preproc_kind = None

                requestOcrKind = self.requestParams.ocr_kind if newField.ocrKind is None else None
                newField.initializeRequiredParameters(
                    source_image       = doc_original_rotated,
                    field_name         = obj_name,
                    bounding_box       = element[vc_constants.FIELD_BOUNDING_BOX], 
                    scale              = self.scale,
                    ocr_kind           = requestOcrKind,
                    preprocessing_kind = preproc_kind
                    )

                newField.extractText()

                element[vc_constants.FIELD_OCR_TEXT]     = newField.extractedValues[0]
                element[vc_constants.FIELD_ADJUSTED_OCR] = newField.extractedValues[1]
                #buf = b''
                #if newField.workImage is not None:
                #    _,buf = cv2.imencode('.jpg', newField.workImage)
                #element['box_image'] = base64.b64encode(buf).decode('utf8')

                vc_utils.printLog(f"  Field {n}: {obj_name} --> {newField.extractedValues[1]}")
        finally:
            self.timers.endClock('ocr')

    # ----------------------------------------------------------------
    def createNewField(self,obj_name,doc_type):
        if doc_type == 'cnh':
            if obj_name.startswith('nome'):            return vc_fields.CnhNameField()
            if obj_name.startswith('identidade'):      return vc_fields.CnhIdentityField()
            if obj_name.startswith('cpf'):             return vc_fields.CnhCpfField()
            if obj_name.startswith('nascimento'):      return vc_fields.CnhDateField()
            if obj_name.startswith('filiacao'):        return vc_fields.CnhFiliationField()
            if obj_name.startswith('registro'):        return vc_fields.CnhNumericField()
            if obj_name.startswith('validade'):        return vc_fields.CnhDateField()
            if obj_name.startswith('pri_habilitacao'): return vc_fields.CnhDateField()
            if obj_name.startswith('local_emissao'):   return vc_fields.CnhCityField()
            if obj_name.startswith('data_emissao'):    return vc_fields.CnhDateField()
            if obj_name.startswith('cnh'):             return vc_fields.CnhField()
            if obj_name.startswith('cnh_frente'):      return vc_fields.CnhField()
            if obj_name.startswith('observ'):          return vc_fields.CnhMultilineField()
            if obj_name.startswith('categoria'):       return vc_fields.CnhCategoryField()
        if doc_type == 'rg':
            if obj_name.startswith('nome'):            return vc_fields.RgNameField()
            if obj_name.startswith('assinatura'):      return vc_fields.RgField()
            if obj_name.startswith('digital'):         return vc_fields.RgField()
            if obj_name.startswith('registro_geral'):  return vc_fields.RgNumericField()
            if obj_name.startswith('data_expedicao'):  return vc_fields.RgDateField()
            if obj_name.startswith('filiacao'):        return vc_fields.RgMultilineField()
            if obj_name.startswith('naturalidade'):    return vc_fields.RgCityField()
            if obj_name.startswith('nascimento'):      return vc_fields.RgDateField()
            if obj_name.startswith('doc_origem'):      return vc_fields.RgMultilineField()
            if obj_name.startswith('cpf'):             return vc_fields.RgCpfField()
            if obj_name.startswith('rg_verso'):        return vc_fields.RgField()
            if obj_name.startswith('rg_frente'):       return vc_fields.RgField()
        return vc_fields.ExtractedField()

    # ----------------------------------------------------------------
    def adjustJSONDetails(self):
        '''
        Função que ajusta as bounding boxes dos elementos existentes no JSON resultante 
        do processo de extração de informações de um documento.
        Essa função altera as bounding boxes dos elementos, para que estejam dentro dos 
        limites da imagem, e adicionalmente inclui no JSON um campo com uma foto ("face_img"),
        caso exista uma foto nos elementos extraídos.
        '''
        result = self.getResultJSON()
        resultlist = vc_utils.dictValue(result,vc_constants.FIELD_RESULT_LIST)
        if resultlist is None:
            return
        # Ajusto as bounding boxes de cada elemento para ficarem dentro dos limites da imagem
        # e aproveito para verificar quantos elementos são do tipo "foto", gerando uma lista 
        # de bounding boxes dessas fotos.

        # A princípio, a lista de fotos está vazia
        fotos = []

        # Obtenho os limites da imagem 
        y_limit,x_limit = self.workImage.shape[:2]  # Altura e largura

        for element in resultlist:
            # Para cada elemento pego sua bounding box
            bbox = element[vc_constants.FIELD_BOUNDING_BOX]
            x_min = bbox[vc_constants.FIELD_X_MIN]
            y_min = bbox[vc_constants.FIELD_Y_MIN]
            wid   = bbox[vc_constants.FIELD_WIDTH]
            hei   = bbox[vc_constants.FIELD_HEIGHT]
            x_max, y_max = x_min + wid, y_min + hei

            # Ajusto os valores da bounding box para que fiquem nos limites da imagem
            x_min = int(max(x_min,0))
            y_min = int(max(y_min,0))
            x_max = int(min(x_max,x_limit))
            y_max = int(min(y_max,y_limit))

            # Atualizo a bounding box
            bbox[vc_constants.FIELD_X_MIN]  = x_min
            bbox[vc_constants.FIELD_Y_MIN]  = y_min
            bbox[vc_constants.FIELD_WIDTH]  = x_max - x_min
            bbox[vc_constants.FIELD_HEIGHT] = y_max - y_min

            # Se o elemento for uma foto, incluo na lista de fotos
            if element[vc_constants.FIELD_OBJ_NAME].lower().startswith('foto'):
                fotos += [(x_min,y_min,x_max,y_max)] 
        
        # Só vou adicionar o campo "face_img" ao JSON resultante, 
        # se houver uma (e apenas uma) foto nos elementos extraídos
        if len(fotos) == 1:
            x_min,y_min,x_max,y_max = fotos[0]
            face_img = self.workImage[y_min:y_max, x_min:x_max] # crop da foto contida na imagem
            if face_img.shape[1] < 256: face_img = vc_img_process.ResizeImage(face_img,256,256,True)
            _,buf = cv2.imencode('.jpg', face_img)
            result[vc_constants.FIELD_FACE_IMG] = base64.b64encode(buf).decode('utf8')

    # ----------------------------------------------------------------
    def runWholeProcess(self, calculate_additional_fields=True, run_ocr=True, filter_fields=False):
        try:
            self.timers.startClock('total')

            self.filterFields = filter_fields
            self.runYolo()
            
            if calculate_additional_fields:
                self.runCalculateAdditionalFields()
            
            if run_ocr:
                self.runDocumentOCR()

            self.adjustJSONDetails()
        finally:
            self.timers.endClock('total')
            self.resultJSON['timers'] = str(self.timers)

            if 'detect_time_secs' not in self.resultJSON:
                self.resultJSON['detect_time_secs'] = self.timers.totals['yolo']
            if 'detect_ocr_time' not in self.resultJSON:
                self.resultJSON['detect_ocr_time'] = self.timers.totals['total']


# def processMissingPhoto(result_json_cnh, img_face_rec):
#     return result_json_cnh

    # if vc_constants.FIELD_RESULT_LIST not in result_json_cnh:
    #     return result_json_cnh

    # tem_foto = False
    # n_cnh = 0

    # x_foto = 0
    # n_x = 0
    # y_foto = 0
    # n_y = 0
    # x2_foto = 0
    # n_x2 = 0
    # y2_foto = 0
    # n_y2 = 0
    # provavel_rg = False

    # for element in result_json_cnh[vc_constants.FIELD_RESULT_LIST]:
    #     bbox = element[vc_constants.FIELD_BOUNDING_BOX]
    #     nome_elemento = element[vc_constants.FIELD_OBJ_NAME].lower()
    #     if nome_elemento.startswith('foto_'):
    #         tem_foto = True

    #     if nome_elemento in ['cnh','cnh_frente']:
    #         n_cnh += 1
    #     if nome_elemento in ['nome_cnh', 'registro_cnh']:
    #         x_foto += bbox['x_min']
    #         n_x += 1
    #     if nome_elemento in ['nome_cnh', 'identidade_cnh']:
    #         y_foto += bbox['y_min'] + (bbox['height'] if nome_elemento == 'nome_cnh' else 0) 
    #         n_y += 1
    #     if nome_elemento in ['identidade_cnh', 'cpf_cnh', 'filiacao_cnh','validade_cnh']:
    #         x2_foto += bbox['x_min']
    #         n_x2 += 1
    #     if nome_elemento in ['registro_cnh', 'validade_cnh', 'pri_habilitacao_cnh']:
    #         y2_foto += bbox['y_min'] 
    #         n_y2 += 1
    #     if '_rg' in nome_elemento or 'rg_' in nome_elemento:
    #         provavel_rg = True

    # if not tem_foto and (n_x == 0 or n_x2 == 0 or n_y == 0 or n_y2 == 0):
    #     # # executar o reconhecimento facial via dlib
    #     # # para obter: x_foto, x2_foto, y_foto e y2_foto
    #     # x_foto, y_foto, x2_foto, y2_foto, n_x, n_y, n_x2, n_y2 = reconhece_face_dlib(img_face_rec)

    #     # executar o reconhecimento facial via DNN
    #     # para obter: x_foto, x2_foto, y_foto e y2_foto
    #     x_foto, y_foto, x2_foto, y2_foto, n_x, n_y, n_x2, n_y2 = vc_img_process.reconhece_face_dnn(img_face_rec)

    # if tem_foto \
    # or n_cnh >= 2 \
    # or n_x == 0 or n_x2 == 0 or n_y == 0 or n_y2 == 0:
    #     return result_json_cnh

    # x_foto = int(x_foto / n_x)
    # y_foto = int(y_foto / n_y)
    # x2_foto = int(x2_foto / n_x2)
    # y2_foto = int(y2_foto / n_y2)

    # if provavel_rg:
    #     result_json_cnh[vc_constants.FIELD_RESULT_LIST] += [         
    #         {
    #             'class_index': 13, 
    #             'obj_name': 'foto_rg', 
    #             'bounding_box': {'x_min': x_foto, 'y_min': y_foto, 'width': x2_foto-x_foto, 'height': y2_foto-y_foto}, 
    #             'score': 0.939876 
    #         }]
    # else:
    #     result_json_cnh[vc_constants.FIELD_RESULT_LIST] += [         
    #         {
    #             'class_index': 0, 
    #             'obj_name': 'foto_cnh', 
    #             'bounding_box': {'x_min': x_foto, 'y_min': y_foto, 'width': x2_foto-x_foto, 'height': y2_foto-y_foto}, 
    #             'score': 0.939876 
    #         }]

    # return result_json_cnh

# =======================================================
from itertools import combinations


def adjustAlignedFields():
    global ALIGNED_FIELDS
    global ALIGNED_FIELDS_LIST

    for doc in ALIGNED_FIELDS:
        newList = []
        newFields = set()
        listAligned = ALIGNED_FIELDS[doc]
        for n in range(len(listAligned)):
            fields = listAligned[n]
            sizeFields = len(fields)
            if sizeFields < 2: continue

            newFields = newFields.union(set(fields))

            if sizeFields == 2:
                newList += [list(fields)]
                continue

            pairs = sorted([[-abs(p[0]-p[1]),p[0],p[1]] for p in combinations(range(sizeFields),2)])
            for _,i,j in pairs:
                newList += [[fields[i],fields[j]]]

        ALIGNED_FIELDS[doc] = newList
        ALIGNED_FIELDS_LIST[doc] = newFields

adjustAlignedFields()
