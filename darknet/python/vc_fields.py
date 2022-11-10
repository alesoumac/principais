#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

import vc_constants
import vc_excepts
import vc_img_process
import vc_ocr
import vc_spell
import vc_strings

# =================================================
# Classe pai de todas as outras classes de campos
# de documentos
# =================================================

class ExtractedField:
    """
    ExtractedField class
    """

    def __init__(self):
        self.fieldName = ""
        self.isMultilineField = False
        self.isNonValueField = True
        self.isNumeric = False
        self.isCPF = False
        self.isFiliation = False
        self.fieldCroppedImage = None
        self.fieldDeskewedImage = None
        self.preprocessingKind = None
        self.ocrKind = None
        self.extractedValues = ['', '']
        self.cropHeight = vc_constants.CROP_HEIGHT_DEFAULT
        self.baseTimestamp = datetime.now()
        #self.workImage = None
        # Crop Height é a altura usada para fazer um resize do pedacinho da imagem desse campo
        # Para campos comuns, normalmente estamos usando o valor de cropHeight = 48
        # Para campos especiais (que necessitam de maior acurácia, tais como Nome e Identidade)
        # estamos usando cropHeight = 120
        # Para campos especiais multilinha (Filiação e Observações)
        # estamos usando cropHeight = 320
        # Explicando: se o campo estiver num pedaço de imagem com altura de 38 pixels, 
        # ele será redimensionado para que fique com a altura padrão definida pra ele (48 por exemplo)

    def initializeRequiredParameters(self,source_image,field_name,bounding_box,scale,ocr_kind,preprocessing_kind):
        self.setParameters(field_name=field_name,ocr_kind=ocr_kind,preprocessing_kind=preprocessing_kind)
        self.setCroppedImage(source_image,bounding_box,scale)

    def setParameters(self, field_name=None,
    crop_height=None, ocr_kind=None, preprocessing_kind=None,
    is_multiline_field=None, is_nonvalue_field=None,
    is_numeric=None, is_cpf=None, is_filiation=None):
        if field_name is not None:         self.fieldName = field_name.lower()
        if is_multiline_field is not None: self.isMultilineField = is_multiline_field
        if is_nonvalue_field is not None:  self.isNonValueField = is_nonvalue_field
        if is_numeric is not None:         self.isNumeric = is_numeric
        if is_cpf is not None:             self.isCPF = is_cpf
        if is_filiation is not None:       self.isFiliation = is_filiation
        if crop_height is not None:        self.cropHeight = crop_height
        if ocr_kind is not None:           self.ocrKind,_ = vc_ocr.availableOcrEngine(ocr_kind)

        # Para setar o valor do atributo preprocessingKind, temos as seguintes situações:
        # (Não confundir o parâmetro preprocessing_kind - underline - e o atributo preprocessingKind - camel case)
        # O parâmetro preprocessing_kind, em primeiro lugar tem que ser diferente de None,
        # para que se possa atribuir algum valor ao atributo preprocessingKind.
        # O atributo preprocessingKind ao final deverá conter uma lista de valores inteiros que representam os tipos
        # de preprocessamento, que serão usados no campo extraído do documento (CNH, RG, etc...)
        # Se o parâmetro preprocessing_kind for uma lista, tupla ou conjunto, o atributo preprocessingKind
        # receberá a lista com os elementos de preprocessing_kind.
        # Se o parâmetro preprocessing_kind for um inteiro (ou qualquer valor que possa ser transformado em inteiro),
        # o atributo preprocessingKind receberá uma lista com os valores de cada bit do valor de preprocessing_kind.
        # Por exemplo: 
        #   se preprocessing_kind = 3, então o atributo preprocessingKind receberá a lista [1,2]
        #   se preprocessing_kind = 5, então o atributo preprocessingKind receberá a lista [1,4]
        #   se preprocessing_kind = 2, então o atributo preprocessingKind receberá a lista [2]
        #   se preprocessing_kind = 7, então o atributo preprocessingKind receberá a lista [1,2,4]
        # Por enquanto, nós só temos dois tipos de preprocessamento de campo, cujos valores são 1 e 2.
        # E esses tipos de preprocessamento podem ser encontrados no módulo vc_constants.py
        #   PREPROC_BORDER_REMOVAL_1 = 1
        #   PREPROC_BORDER_REMOVAL_2 = 2
        # Se no futuro for necessário criar outros tipos de preprocessamento, basta manter o valor como sendo um exponencial de 2
        # ou seja, valores na sequencia 1, 2, 4, 8, 16, 32, ...

        if preprocessing_kind is not None:
            typePP = type(preprocessing_kind)
            preprocList = []
            try:
                if typePP is list or typePP is tuple or typePP is set:
                    preprocList = list(preprocessing_kind)
                else:
                    preproc = int(preprocessing_kind)
                    for nbits in range(2): # por enquanto, só testamos dois bits, mas, se no futuro houver mais tipos de processamento, é necessário rever esse range
                        testBit = 1 << nbits
                        if testBit & preproc != 0:
                            preprocList += [testBit]
            except:
                raise vc_excepts.PreprocessingKindTypeError("Tipo inválido no parâmetro preprocessing_kind")

            self.preprocessingKind = preprocList
    
    def setCroppedImage(self,sourceImage,boundingBox,scale):
        x_box      = int(boundingBox[vc_constants.FIELD_X_MIN] * scale)
        y_box      = int(boundingBox[vc_constants.FIELD_Y_MIN] * scale)
        width_box  = int(boundingBox[vc_constants.FIELD_WIDTH] * scale)
        height_box = int(boundingBox[vc_constants.FIELD_HEIGHT] * scale)
        self.fieldCroppedImage = vc_img_process.crop_image(sourceImage, x_box, y_box, x_box + width_box, y_box + height_box)

    def getCroppedImage(self):
        return self.fieldCroppedImage

    def getDeskewedImage(self):
        if self.fieldDeskewedImage is None:
            _, self.fieldDeskewedImage = vc_img_process.deskew(self.getCroppedImage())
        return self.fieldDeskewedImage

    def doPreprocessing(self):
        img = self.getDeskewedImage()
        altura, largura = img.shape[:2]
        retangulo_total = [0,0,largura,altura]
        min_x,min_y,max_x,max_y = retangulo_total
        retangulo1 = ()
        retangulo2 = ()

        if vc_constants.PREPROC_BORDER_REMOVAL_1 in self.preprocessingKind:
            retangulo1 = vc_img_process.borderRemoval1(img)

        if vc_constants.PREPROC_BORDER_REMOVAL_2 in self.preprocessingKind:
            retangulo2 = vc_img_process.borderRemoval2(img)

        if vc_constants.PREPROC_BORDER_REMOVAL_3 in self.preprocessingKind:
            retangulo1 = vc_img_process.borderRemoval3(img)

        if len(retangulo1) > 0 and len(retangulo2) > 0 and retangulo1 != retangulo_total:
            min_x = min(retangulo1[0], retangulo2[0])
            min_y = min(retangulo1[1], retangulo2[1])
            max_x = max(retangulo1[2], retangulo2[2])
            max_y = max(retangulo1[3], retangulo2[3])
        elif len(retangulo2) > 0:
            min_x,min_y,max_x,max_y = retangulo2
        elif len(retangulo1) > 0:
            min_x, min_y, max_x, max_y = retangulo1
        else:
            return img

        return vc_img_process.crop_image(img, min_x, min_y, max_x, max_y)

    def doOCR(self,img, img_alternativa=None):
        img_resized = vc_img_process.ResizeImage(img,self.cropHeight,1,True)
        alt,larg = img_resized.shape[:2]
        if alt == 0 or larg == 0: return ""

        #self.workImage = img_resized
        texto = vc_ocr.run_ocr(img_resized, self.ocrKind, self.isCPF, self.isFiliation)
        if texto == "" and img_alternativa is not None:
            texto = self.doOCR(img_alternativa, None)
        return texto

    def getFieldTitlesFilter(self):
        if self.fieldName.startswith('nome'): return ['NOME']
        if self.fieldName.startswith('filiacao'): return ['FILIACAO']
        if self.fieldName.startswith('identidade'): return ['DOC','IDENTIDADE','ORG', 'EMISSOR','UF']
        if self.fieldName.startswith('registro_geral'): return ['REGISTRO','GERAL']
        if self.fieldName.startswith('local_emissao'): return ['LOCAL']
        if self.fieldName.startswith('naturalidade'): return ['NATURALIDADE']
        if self.fieldName.startswith('observacao'): return ['OBSERVACOES']
        if self.fieldName.startswith('categoria'): return ['CAT','HAB']
        return []

    def doSpellChecker(self, texto):
        return texto

    def doRemoveCharacters(self, texto):
        return texto

    def doSpecialCorrection(self, texto):
        return texto

    def doPosprocessing(self,texto):
        s = self.doSpellChecker(texto)
        s = self.doRemoveCharacters(s)
        s = self.doSpecialCorrection(s)
        return s

    def extractText(self):
        if self.isNonValueField:
            self.extractedValues = ['', '']
            return

        processedImage = self.doPreprocessing()
        ocrText      = self.doOCR(processedImage, self.getDeskewedImage())
        adjustedText = self.doPosprocessing(ocrText)
        self.extractedValues = [ocrText, adjustedText]


# =================================================
# Classes para os campos da CNH
# =================================================

class CnhField(ExtractedField):
    pass

# ------------------------------ Text Field
class CnhTextField(CnhField):
    def __init__(self):
        super().__init__()
        self.setParameters(is_nonvalue_field=False)

# ------------------------------ Fine Text Field
class CnhFineTextField(CnhTextField):
    def __init__(self):
        super().__init__()
        self.setParameters(crop_height=120)

# ------------------------------ Name Field
class CnhNameField(CnhFineTextField):
    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,True,False,filtroTitulos)

    def doRemoveCharacters(self, texto):
        s = vc_strings.KeepChars(texto, "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÑÁÉÍÓÚÀÈÌÒÙÃÕÂÊÎÔÛÄËÏÖÜ' \n").strip()
        s = vc_strings.strip_spaces(s)
        return s

    def doSpecialCorrection(self, texto):
        return vc_spell.SpellAdjust(texto)

# ------------------------------ City Fields
class CnhCityField(CnhTextField):
    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,False,False,filtroTitulos)

    def doSpecialCorrection(self, texto):
        return vc_spell.AdjustCity(texto)

# ------------------------------ Identity Field
class CnhIdentityField(CnhFineTextField):
    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,True,True,filtroTitulos)

    def doSpecialCorrection(self, texto):
        return vc_spell.AdjustIdentity(texto)

# ------------------------------ Numeric Field
class CnhNumericField(CnhField):
    def __init__(self):
        super().__init__()
        self.setParameters(
            is_nonvalue_field=False,
            is_numeric=True)

    def doRemoveCharacters(self, texto):
        return vc_strings.KeepChars(texto, "0123456789 ")

    def doSpecialCorrection(self, texto):
        return texto.replace(' ','')

# ------------------------------ CPF Field
class CnhCpfField(CnhNumericField):
    def __init__(self):
        super().__init__()
        self.setParameters(
            is_cpf=True,
            ocr_kind=vc_constants.OCR_KIND_TESSERACT) 
        # Se houver Tesseract disponível, usamos esse OCR engine para os campos CPF,
        # pois durante testes tivemos melhor acurácia para esse campo usando o Tesseract.

    def doSpecialCorrection(self, texto):
        cpfs = vc_strings.FindCPF(texto)
        if cpfs != []:
            return cpfs[0]
        texto_sem_espaco = texto.replace(' ','')
        cpfs = vc_strings.FindCPF(texto_sem_espaco)
        if cpfs != []:
            return cpfs[0]
        return texto_sem_espaco

# ------------------------------ Date Field
class CnhDateField(CnhField):
    def __init__(self):
        super().__init__()
        self.setParameters(is_nonvalue_field=False)

    def doSpecialCorrection(self, texto):
        return vc_strings.RecognizeDate(texto)

# ------------------------------ Multiline Fields
class CnhMultilineField(CnhField):
    def __init__(self):
        super().__init__()
        self.setParameters(
            is_multiline_field=True,
            crop_height=320, # Esse valor foi obtido através de testes, pra ver qual resize dava uma acurácia melhor para o campo
            is_nonvalue_field=False)

    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,False,False,filtroTitulos)

# ------------------------------ Filiation Fields
class CnhFiliationField(CnhMultilineField):
    def __init__(self):
        super().__init__()
        self.setParameters(is_filiation=True)

    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,True,False,filtroTitulos)

    def doRemoveCharacters(self, texto):
        s = vc_strings.KeepChars(texto, "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÑÁÉÍÓÚÀÈÌÒÙÃÕÂÊÎÔÛÄËÏÖÜ' \n").strip()
        s = vc_strings.strip_spaces(s)
        return s

    def doSpecialCorrection(self, texto):
        return vc_spell.SpellAdjust(texto)

    def doPosprocessing(self, texto):
        if '\n' in texto:
            pai,mae = texto.split('\n')[:2]
            pai = self.doPosprocessing(pai)
            mae = self.doPosprocessing(mae)
            return f"{pai}\n{mae}"
        else:
            return super().doPosprocessing(texto)

class CnhCategoryField(CnhFineTextField):
    def doRemoveCharacters(self, texto):
        s = vc_strings.KeepChars(texto, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ").strip()
        s = vc_strings.strip_spaces(s)
        return s

    def doSpecialCorrection(self, texto):
        listaValoresAceitos = ["acc", "a", "a1", "b", "b1", "c", "c1", "d", "d1", "be", "ce", "c1e","de","d1e", "ab", "ad", "ae"]
        s = [parte for parte in texto.lower().split(' ') if parte in listaValoresAceitos]
        if s == []: return ""
        return s[-1].upper()

# =================================================
# Classes para os campos da Identidade (RG)
# =================================================

class RgField(ExtractedField):
    pass

# ------------------------------ Text Field
class RgTextField(RgField):
    def __init__(self):
        super().__init__()
        self.setParameters(is_nonvalue_field=False)

# ------------------------------ Fine Text Field
class RgFineTextField(RgTextField):
    def __init__(self):
        super().__init__()
        self.setParameters(crop_height=120)

# ------------------------------ Name Field
class RgNameField(RgFineTextField):
    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,True,False,filtroTitulos)

    def doRemoveCharacters(self, texto):
        s = vc_strings.KeepChars(texto, "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÑÁÉÍÓÚÀÈÌÒÙÃÕÂÊÎÔÛÄËÏÖÜ' \n").strip()
        s = vc_strings.strip_spaces(s)
        return s

    def doSpecialCorrection(self, texto):
        return vc_spell.SpellAdjust(texto)

# ------------------------------ City Fields
class RgCityField(RgTextField):
    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,False,False,filtroTitulos)

    def doSpecialCorrection(self, texto):
        return vc_spell.AdjustCity(texto)

# ------------------------------ Date Field
class RgDateField(RgField):
    def __init__(self):
        super().__init__()
        self.setParameters(is_nonvalue_field=False)

    def doSpecialCorrection(self, texto):
        return vc_strings.RecognizeDate(texto)

# ------------------------------ Numeric Field
class RgNumericField(RgField):
    def __init__(self):
        super().__init__()
        self.setParameters(
            is_nonvalue_field=False,
            is_numeric=True)

    def doRemoveCharacters(self, texto):
        return vc_strings.KeepChars(texto, "0123456789 ")

    def doSpecialCorrection(self, texto):
        return texto.replace(' ','')

# ------------------------------ CPF Field
class RgCpfField(RgNumericField):
    def __init__(self):
        super().__init__()
        self.setParameters(
            is_cpf=True,
            ocr_kind=vc_constants.OCR_KIND_TESSERACT) 
        # Se houver Tesseract disponível, usamos esse OCR engine para os campos CPF,
        # pois durante testes tivemos melhor acurácia para esse campo usando o Tesseract.

    def doSpecialCorrection(self, texto):
        cpfs = vc_strings.FindCPF(texto)
        if cpfs != []:
            return cpfs[0]
        texto_sem_espaco = texto.replace(' ','')
        cpfs = vc_strings.FindCPF(texto_sem_espaco)
        if cpfs != []:
            return cpfs[0]
        return texto_sem_espaco

# ------------------------------ Multiline Fields
class RgMultilineField(RgField):
    def __init__(self):
        super().__init__()
        self.setParameters(
            is_multiline_field=True,
            crop_height=320, # Esse valor foi obtido através de testes, pra ver qual resize dava uma acurácia melhor para o campo
            is_nonvalue_field=False)

    def doSpellChecker(self, texto):
        filtroTitulos = self.getFieldTitlesFilter()
        return vc_spell.SpellAdjust(texto,False,False,False,filtroTitulos)
