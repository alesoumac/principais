#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from itertools import combinations

import vc_fields
import vc_utils

DOC_MODELS_DIR = None
DOC_MODELS_DIR_ALTERNATIVE = None
POSSIBLE_DOC_CLASSES = []  # ['cnh', 'cnh_frente', 'rg_frente', 'rg_verso', ... ]

class DocumentModel:
    def __init__(self, kind, default_ocr):
        self.kind = kind.lower()
        self.defaultOCR = default_ocr
        modelJson = findDocModelFile(self.kind)
        self.fieldsModel = vc_utils.dictValue(modelJson,"fields")
        self.alignedFields = vc_utils.dictValue(modelJson,"aligned")
        self.alignedFieldsList = []
        self.alignedFields, self.alignedFieldsList = adjustAlignedFields(self.alignedFields)
        self.fields = []

    def model(self,fieldname):
        return vc_utils.dictValue(self.fieldsModel, fieldname)

    def atribute(self,fieldname,fieldatribute):
        return vc_utils.dictValue(self.model(fieldname),fieldatribute)

    def createFieldBox(self, field_name, source_image, bounding_box, scale):
        '''
        Cria uma instância de objeto ExtractedField, seguindo algumas regras, listadas abaixo:

        Se tiver "divideBox" no campo, atribuir true para isDivided
        Se fieldType=="cpf", atribuir true para tryTesseract

        Pré-Processamento:
        "borderRemoval" pode ser 1, 2 ou 3
        "enhanceHeight": se False, usa cropHeight=48  para campos singleline ou 120 para campos multiline
                        se True,  usa cropHeight=120 para campos singleline ou 320 para campos multiline
                        Esses valores de cropHeight foram obtidos através de testes para verificar qual o
                        valor que traz uma boa relação acurácia/desempenho

        Tipos de Pós-Processamento:
        "divideBox":
            faz a divisão e recursivamente entra na função de posProcessamento com cada pedaço do texto dividido
        fieldType == "name":
            spellAdjust(correcting=False, strip_short_noums=True, first_word_is_numeric=False, filter=['NOME'])
            keepChars("ABCDEFGHIJKLMNOPQRSTUVWXYZÇÑÁÉÍÓÚÀÈÌÒÙÃÕÂÊÎÔÛÄËÏÖÜ \n").strip()
            stripSpaces()
            spellAdjust()
        fieldType == "identity":
            spellAdjust(correcting=False, strip_short_noums=True, first_word_is_numeric=True, filter=['DOC', 'IDENTIDADE', 'ÓRG', 'ORG', 'EMISSOR', 'UF'])
            adjustIdentity()
        fieldType == "cpf":
            // keepChars("0123456789 ")
            findCPF()
        fieldType == "numeric":
            keepChars("0123456789")
        fieldType == "date":
            recognizeDate()
        fieldType == "city":
            spellAdjust(correcting=False, strip_short_noums=False, first_word_is_numeric=False, filter=['LOCAL'])
            adjustCity()
        fieldType == "free" or fieldType is None :
            spellAdjust(correcting=False, strip_short_noums=False, first_word_is_numeric=False, filter=['OBSERVACOES', 'OBSERVAÇÕES'])
        '''
        campo = vc_fields.ExtractedField()
        nome_campo = field_name.lower()
        campo.setRequiredParameters(nome_campo, source_image, bounding_box, scale)

        fieldType     = self.atribute(nome_campo,"fieldtype")
        borderRemoval = self.atribute(nome_campo,"borderremoval")
        enhanceHeight = self.atribute(nome_campo,"enhanceheight") == True
        isNonValue    = self.atribute(nome_campo,"nonvaluefield") == True
        isNumeric     = self.atribute(nome_campo,"numeric") == True
        isMultiline   = self.atribute(nome_campo,"multiline") == True
        isDivided     = self.atribute(nome_campo,"dividebox") == True
        tryTesseract  = True if fieldType == 'cpf' else False

        if borderRemoval is None: borderRemoval = 0
        
        cropHeight = 48
        if enhanceHeight:
            cropHeight = 320 if isMultiline else 120
        else:
            cropHeight = 120 if isMultiline else 48
        # "enhanceHeight": se False, usa cropHeight=48  para campos singleline ou 120 para campos multiline
        #                 se True,  usa cropHeight=120 para campos singleline ou 320 para campos multiline

        campo.setParameters(
            field_type=self.atribute(nome_campo, 'fieldtype'),
            crop_height=cropHeight,
            ocr_kind=self.defaultOCR,
            preprocessing_kind=borderRemoval,
            is_multiline_field=isMultiline,
            is_nonvalue_field=isNonValue,
            is_numeric=isNumeric,
            is_divided=isDivided,
            try_tesseract=tryTesseract
            )
        return campo

    def setFieldsParameters(self, source_image, scale, result_list):
        
        pass

# ------------------------------------------------------------------------
def inicializa_doc_models(path_prog):
    '''
    A função inicializa_doc_models inicializa as seguintes variáveis:
    - DOC_MODELS_DIR: Caminho de diretório definido pela variável de ambiente VCDOC_API_DOC_MODELS_CAMINHO
    - DOC_MODELS_DIR_ALTERNATIVE: Subdiretório 'doc_models', contido na pasta raiz da aplicação 'vcdoc_server.py'
    - POSSIBLE_DOC_CLASSES: Lista com todos os tipos de doc_models encontrados nas pastas acima. 
                            Essa lista é preenchida usando a subfunção 'getKinds', definida logo abaixo.
    '''
    #-------------------------
    def getKinds(diretorio): 
        '''
        Procuro todos os arquivos .json que estão no diretório passado como parâmetro
        e retorno um set com os nomes dos arquivos sem a extensão.
        Esses nomes definem quais são os tipos de documento que a aplicação reconhece
        Por exemplo, até o dia de hoje (25/07/2022), só temos 4 documentos definidos:
        - cnh.json
        - cnh_frente.json
        - rg_frente.json
        - rg_verso.json
        Sendo assim, a função getKinds retornará o set {'cnh', 'cnh_frente', 'rg_frente', 'rg_verso'}
        '''
        if diretorio is None: return set()
        for root,_,files in os.walk(diretorio):
            if root != diretorio: continue
            pares_nome_extensao = [os.path.splitext(nome_extensao) for nome_extensao in files]
            tipos_doc = []
            for nome,extensao in pares_nome_extensao:
                if extensao != '.json': continue # Somente nos interessa arquivos do tipo '.json'
                nomelower = nome.lower()         # E outro detalhe: não tem problema se houver letras maiúsculas no nome do arquivo,
                if nomelower != nome:            # mas ele será renomeado com letras minúsculas
                    os.rename(os.path.join(root,nome+extensao), os.path.join(root,nomelower+extensao))
                tipos_doc += [nomelower]
            return set(tipos_doc)

    #-------------------------
    global DOC_MODELS_DIR
    global DOC_MODELS_DIR_ALTERNATIVE
    global POSSIBLE_DOC_CLASSES
    try:
        DOC_MODELS_DIR = os.getenv('VCDOC_API_DOC_MODELS_CAMINHO') 
    except:
        DOC_MODELS_DIR = None
    DOC_MODELS_DIR_ALTERNATIVE = os.path.join(path_prog,'doc_models')
    kinds = getKinds(DOC_MODELS_DIR).union(
            getKinds(DOC_MODELS_DIR_ALTERNATIVE))
    POSSIBLE_DOC_CLASSES = list(kinds)

# ------------------------------------------------------------------------
def convertToLower(dicionario):
    '''
    Função que converte as strings contidas dentro de um dicionario para lowercase.
    Todas as strings, incluindo as chaves, valores, strings dentro de listas, tuplas,
    sets, subdicionários ou qualquer subestrutura dentro do dicionario principal,
    serão convertidas para lowercase.
    '''
    if type(dicionario) is str:
        return dicionario.lower()
    if type(dicionario) is set or type(dicionario) is list or type(dicionario) is tuple:
        novo_dicionario = []
        for valor in dicionario:
            novo_dicionario += [convertToLower(valor)]
        return novo_dicionario
    if type(dicionario) is not dict:
        return dicionario
    novo_dicionario = {}
    for key in dicionario:
        nova_key = key.lower() if type(key) is str else key
        novo_dicionario[nova_key] = convertToLower(dicionario[key])
    return novo_dicionario

# ------------------------------------------------------------------------
def findDocModelFile(kind):
    '''
    Dado um tipo de documento (kind), por exemplo kind='cnh', a função findDocModelFile
    procura o arquivo json referente a esse tipo (no caso do nosso exemplo, 'cnh.json')
    nos dois diretórios definidos pelas variáveis DOC_MODELS_DIR e DOC_MODELS_DIR_ALTERNATIVE,
    e a procura é feita nessa ordem.
    '''
    global DOC_MODELS_DIR
    global DOC_MODELS_DIR_ALTERNATIVE
    kindlower = kind.lower()
    nome_arquivo = f'{kindlower}.json'
    caminho_arquivo = os.path.join(DOC_MODELS_DIR,nome_arquivo)
    if not os.path.exists(caminho_arquivo):
        caminho_arquivo_alternativo = os.path.join(DOC_MODELS_DIR_ALTERNATIVE,nome_arquivo)
    if os.path.exists(caminho_arquivo_alternativo):
        caminho_arquivo = caminho_arquivo_alternativo
    else:
        return None
    # -----
    try:
        dadosJson = open(caminho_arquivo,'r')
        modelJson = convertToLower( json.load(dadosJson) )
        dadosJson.close()
        kindInterno = vc_utils.dictValue(modelJson,"kind",as_type="str")
        if kindInterno != kindlower:
            vc_utils.printLog(f"AVISO: arquivo de modelo de documento '{nome_arquivo}' possui o atributo 'kind' diferente (='{kindInterno}'). Será considerado o valor '{kindlower}'")
            modelJson['kind'] = kindlower
        return modelJson
    except:
        raise Exception(f"Erro ao carregar modelo de documento {nome_arquivo}")

# ------------------------------------------------------------------------
def adjustAlignedFields(alignedList):
    '''
    Parametro:
        alignedList = lista contendo sequências de campos (pode ter mais de dois) que tem que estar alinhados
    Retorno (list, set):
        retorna uma tupla contendo:
        - lista com pares de campos que tem que estar alinhados
        - conjunto com todos os campos presentes na lista acima

    A função adjustAlignedFields recebe como parâmetro uma lista contendo sequências de campos que tem que estar
    alinhados (visualmente) no modelo de documento corrente. Esta lista deve estar definida no campo "aligned" do
    arquivo de modelo.
    Abaixo temos um exemplo do campo "aligned" do modelo "cnh.json":
    ----
    ARQUIVO cnh.json
    ----
    {
        "kind": "cnh",
        "aligned": [
            ["cpf_cnh","nascimento_cnh"],
            ["registro_cnh","validade_cnh","pri_habilitacao_cnh"],
            ["local_emissao_cnh","data_emissao_cnh"]
        ],

        "fields": {
    ...

    Supondo que tenhamos uma variável umaLista com o valor abaixo:
    umaLista = [
        ["cpf_cnh","nascimento_cnh"],
        ["registro_cnh","validade_cnh","pri_habilitacao_cnh"],
        ["local_emissao_cnh","data_emissao_cnh"]
    ]
    
    Se executarmos a função adjustAlignedFields(umaLista), teremos como retorno a tupla abaixo:
    (
        [
            ['cpf_cnh', 'nascimento_cnh'],
            ['registro_cnh', 'pri_habilitacao_cnh'],
            ['registro_cnh', 'validade_cnh'],
            ['validade_cnh', 'pri_habilitacao_cnh'],
            ['local_emissao_cnh', 'data_emissao_cnh']
        ], 
        
        {
            'nascimento_cnh', 
            'cpf_cnh', 
            'registro_cnh', 
            'validade_cnh', 
            'data_emissao_cnh', 
            'pri_habilitacao_cnh', 
            'local_emissao_cnh'
        }
    )
    '''
    newList = []
    newFields = set()
    for n in range(len(alignedList)):
        fields = alignedList[n]
        sizeFields = len(fields)
        if sizeFields < 2: continue
        newFields = newFields.union(set(fields))
        if sizeFields == 2:
            newList += [list(fields)]
            continue
        pairs = sorted([[-abs(p[0]-p[1]),p[0],p[1]] for p in combinations(range(sizeFields),2)])
        for _,i,j in pairs:
            newList += [[fields[i],fields[j]]]
    return newList, newFields

