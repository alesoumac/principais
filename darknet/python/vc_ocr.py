# Reestruturação da aplicacao, divisao em classes, separação do darknet_server.py 30% performance

import os

import vc_constants
import vc_img_process
import vc_strings
import vc_utils

# As variáveis de ambiente VCDOC_USE_EASYOCR e VCDOC_USE_TESSERACT
# definem quais bibliotecas estarão disponíveis para realizar OCR.
# Se essas variáveis não estiverem definidas no ambiente, elas têm
# o valor padrão True. 
# Aimportação das bibliotecas easyocr e pytesseract depende do 
# valor dessas variáveis.
#   True:  é feito o import e qualquer processamento relacionado
#          ao OCR da biblioteca em questão é ativado;
#   False: não é feito o import e são desativados todos os
#          processamentos da biblioteca.
# Ao importar cada uma das bibliotecas (EasyOCR e Tesseract),
# é feito um try/except, para verificar se a biblioteca em questão
# está realmente instalada no ambiente. Caso não esteja, a
# variável VCDOC_USE_EASYOCR ou VCDOC_USE_TESSERACT relativa receberá o
# valor False, indicando que a biblioteca não está disponível.

# Biblioteca EasyOCR
EASY = None
READER_EASYOCR = None
try:
    VCDOC_USE_EASYOCR = os.environ['VCDOC_USE_EASYOCR']
except:
    VCDOC_USE_EASYOCR = True

try:
    if VCDOC_USE_EASYOCR:
        import easyocr
        EASY = easyocr
except:
    VCDOC_USE_EASYOCR = False

#Biblioteca PyTesseract
TESS = None
try:
    VCDOC_USE_TESSERACT = os.environ['VCDOC_USE_TESSERACT']
except:
    VCDOC_USE_TESSERACT = True

try:
    if VCDOC_USE_TESSERACT:
        import pytesseract
        TESS = pytesseract
except:
    VCDOC_USE_TESSERACT = False

#############################

def availableOcrEngine(ocr_kind):
    global VCDOC_USE_TESSERACT
    global VCDOC_USE_EASYOCR
    
    availOcrEngine = vc_constants.OCR_KIND_NONE

    if not VCDOC_USE_EASYOCR and not VCDOC_USE_TESSERACT:
        return availOcrEngine, "Não há OCR Engine disponível"
    
    if ocr_kind == vc_constants.OCR_KIND_EASYOCR:
        availOcrEngine = vc_constants.OCR_KIND_EASYOCR if VCDOC_USE_EASYOCR else vc_constants.OCR_KIND_TESSERACT
    else:
        availOcrEngine = vc_constants.OCR_KIND_TESSERACT if VCDOC_USE_TESSERACT else vc_constants.OCR_KIND_EASYOCR
    
    return (availOcrEngine, "Tesseract" if availOcrEngine == vc_constants.OCR_KIND_TESSERACT else "EasyOCR")

def run_ocr_engine(img, ocr_kind, try_tesseract, ident = None):
    global READER_EASYOCR
    global VCDOC_USE_TESSERACT
    global VCDOC_USE_EASYOCR

    if ocr_kind == vc_constants.OCR_KIND_NONE: # nenhum OCR está disponível
        return "" if ident is None else (ident,"")
    #---
    if VCDOC_USE_EASYOCR and VCDOC_USE_TESSERACT:                    # se os dois tipos de OCR estão disponíveis
        if (ocr_kind == vc_constants.OCR_KIND_EASYOCR and not try_tesseract): # então forço o uso do Tesseract para o OCR
            kind_final = ocr_kind                                    # em campos do tipo CPF.
        else:
            kind_final = vc_constants.OCR_KIND_TESSERACT
    else:
        kind_final = ocr_kind
    #---
    if kind_final == vc_constants.OCR_KIND_EASYOCR:
        r = READER_EASYOCR.readtext(img)
        texto =' '.join([ r[i][1] for i in range(len(r)) ])
    else:
        texto = TESS.image_to_string(img, lang="por")
    #---
    return texto if ident is None else (ident, texto)

def divide_linhas_metade(img_original):
    '''
    Objetivo: Encontrar a metade do campo Filiação, evitando cortar uma possível linha.
    Entrada: a imagem - crop da Filiação
    Saída: duas imagens (imgA e imgB) contendo respectivamente a parte de cima e a parte de baixo da imagem original.
    '''
    faixas_de_linhas = vc_img_process.find_possible_lines(img_original)
    alt_original,larg_original = img_original.shape[:2]
    metade_alt_original = alt_original // 2
    maxy2_imagem_superior = None
    miny1_imagem_inferior = None
    for (y1,y2,ymeio) in faixas_de_linhas:
        if ymeio >= metade_alt_original: # estou decidindo se a faixa vai ficar na imagem superior (pai) ou na imagem inferior (mãe)
            if miny1_imagem_inferior is None or miny1_imagem_inferior > y1-2: 
                miny1_imagem_inferior = y1-2
        else:
            if maxy2_imagem_superior is None or maxy2_imagem_superior < y2+2:
                maxy2_imagem_superior = y2+2

    #Nesse ponto, podemos ter os seguintes cenários em relação a maxy2_imagem_superior e miny1_imagem_inferior 
    
    # Primeiro cenário: não foi encontrada nenhuma faixa de linha nem na parte superior nem na parte inferior 
    # da imagem original.
    # Daí tanto maxy2_imagem_superior quanto miny1_imagem_inferior estão com valor None
    # e então é feito a divisão simples da imagem, ou seja, na metade da altura original
    if maxy2_imagem_superior is None and miny1_imagem_inferior is None:
        maxy2_imagem_superior = miny1_imagem_inferior = metade_alt_original

    # Segundo cenário: não foi encontrada nenhuma faixa de linha na parte superior da imagem original.
    # Nesse caso, maxy2_imagem_superior vem com o valor None, e então simplesmente fazemos
    # maxy2_imagem_superior receber o valor de miny1_imagem_inferior.
    elif maxy2_imagem_superior is None:
        maxy2_imagem_superior = miny1_imagem_inferior

    # Terceiro cenário: semelhante ao segundo, mas agora não foi encontrada nenhuma faixa de linha 
    # na parte inferior da imagem original.
    # Nesse caso, miny1_imagem_inferior vem com o valor None, e então simplesmente fazemos
    # miny1_imagem_inferior receber o valor de maxy2_imagem_superior.
    elif miny1_imagem_inferior is None:
        miny1_imagem_inferior = maxy2_imagem_superior

    # Quarto cenário: foram encontradas linhas de texto tanto na parte superior quanto na inferior,
    # mas com um gap entre maxy2_imagem_superior e miny1_imagem_inferior.
    # Para evitar que se perca esse gap da imagem, que se exclua essa parte da imagem,
    # verificamos qual ponto (maxy2_imagem_superior ou miny1_imagem_inferior) está
    # mais próximo (distância menor) da metade da imagem original. O ponto que estiver mais próximo será
    # o escolhido para fazer a divisão da imagem.
    elif miny1_imagem_inferior > maxy2_imagem_superior:
        dist_a = abs(maxy2_imagem_superior - metade_alt_original)
        dist_b = abs(miny1_imagem_inferior - metade_alt_original)
        if dist_a < dist_b:
            miny1_imagem_inferior = maxy2_imagem_superior
        else:
            maxy2_imagem_superior = miny1_imagem_inferior

    # Quinto cenário (que não está explícito no IF...ELIF...ELSE): foram encontradas linhas de texto
    # tanto na parte superior quanto na inferior da imagem, e:
    #   1) maxy2_imagem_superior == miny1_imagem_inferior
    #      parte de cima colada na parte de baixo
    #   2) maxy2_imagem_superior > miny1_imagem_inferior
    #      parte de cima fica "encavalada" com a parte de baixo,
    #      e isso pode acontecer quando há uma certa inclinação 
    #      no texto da imagem
    # Nesses casos, não é preciso fazer nada, e os valores de maxy2_imagem_superior 
    # e miny1_imagem_inferior continuam os mesmos.

    # Então, depois de definido os pontos de corte da imagem, é feito o crop,
    # e a função retorna as duas subimagens (metade superior e inferior).
    pedaco_superior = vc_img_process.crop_image(img_original, 0, 0,                     larg_original, maxy2_imagem_superior)
    pedaco_inferior = vc_img_process.crop_image(img_original, 0, miny1_imagem_inferior, larg_original, alt_original)
    return pedaco_superior, pedaco_inferior

def corta_palavra_unica_na_primeira_linha(s):
    linhas = s.split('\n')
    if len(linhas) > 1 and ' ' not in linhas[0]:
        return '\n'.join(linhas[1:])
    return s

def run_ocr(img_np_rgb, ocr_kind, try_tesseract = False, try_divided = False):
    lin_res = ""
    try:
        if try_divided:
            img_pai,img_mae = divide_linhas_metade(img_np_rgb)
            nome_pai = run_ocr(img_pai,ocr_kind,try_tesseract,False)
            nome_mae = run_ocr(img_mae,ocr_kind,try_tesseract,False)
            return f"{nome_pai}\n{nome_mae}" 
        st = run_ocr_engine(img_np_rgb, ocr_kind, try_tesseract)
        st = vc_strings.strip_spaces(st.replace('\n',' ').replace('\r',' '))
        lin_res = st.strip()
    except Exception as err:
        vc_utils.printLog(f"Não foi possível executar o OCR. Erro: {err}")
        return ""
    return lin_res

def inicializa_easyocr(path_prog):
    global READER_EASYOCR
    if not VCDOC_USE_EASYOCR: return

    models_dir = os.getenv('VCDOC_API_OCR_CAMINHO')
    if models_dir is None:
        models_dir = os.path.join(path_prog, "models")
    #print(f'Arquivos de modelo do EasyOCR em: "{models_dir}"')
    
    user_dir = os.path.expanduser('~')
    pasta_easyocr_model = os.path.join(user_dir, '.EasyOCR/model')
    vc_utils.forceDirectory(pasta_easyocr_model)
    vc_utils.copyFile(os.path.join(models_dir,'craft_mlt_25k.pth'), pasta_easyocr_model)
    vc_utils.copyFile(os.path.join(models_dir,'latin_g2.pth'), pasta_easyocr_model)
    if READER_EASYOCR is None:
        READER_EASYOCR = EASY.Reader(['en','pt','es','it'],gpu=True)
