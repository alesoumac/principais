# Reestruturação da aplicacao
# -*- coding: utf-8 -*-

import os
import subprocess
from tempfile import NamedTemporaryFile

import vc_constants
import vc_utils

# from pdf2image import convert_from_bytes
# pra usar essa função é só chamar:
# pilimg_list = convert_from_bytes('nome_arquivo.pdf')
# for pilimg in pilimg_list:
#     faz qualquer coisa com a pilimg

class PdfConversionError(Exception): pass

def convertPdfToImages(pdf_name_or_bytes, img_base_file=None, raise_error = True):
    '''
    A função convertPdfToImages converte um PDF em uma sequência de imagens.
    Para isso, utiliza o programa "pdftoppm" (que é instalado com o pacote 
    poppler-utils), que gera uma imagem jpeg para cada página do PDF.
    
    Parâmetros da função:
    :param pdf_name_or_bytes: str | bytes
        Nome do arquivo PDF ou conteúdo (bytes) do PDF
    :param img_base_file: str | None
        Nome de arquivo base para as imagens a serem criadas
    :param raise_error: bool (default = True)
        Em caso de erro, esse parâmetro indica se será gerada uma exceção ou não.
        Se não for gerada a exceção, o erro pode ser tratado através do 
        'returncode', contido no retorno da função.
    :return dict:
        A função retorna o dicionário com o seguinte conteúdo:
        {
            'returncode': código de retorno (int) do comando "pdftoppm"
            'basepath':   diretório das imagens (str)
            'basefile':   nome de arquivo base das imagens (str)
            'suffixes':   lista de strings, contendo os sufixos dos arquivos (números de página)
        }
    ----
    # Exemplo de uso da função convertPdfToImages:
    
    import os
    
    from vc_pdf2img import convertPdfToImages
    
    ...
    result = convertPdfToImages('~/Documents/test.pdf', '/tmp/test_img')
    if result['returncode'] == 0:
        for suffix in result['suffixes']:
            image_name = os.path.join(result['basepath], result[basefile] + suffix) # image_name contêm o nome do arquivo com a página atual
    ...
    '''
    # Verifico o nome base para as imagens, criando um nome temporário se necessário
    baseImgFile = img_base_file
    if baseImgFile is None:
        basePath = os.path.join( os.path.expanduser('~'), '.tmp', 'vcd' )
        vc_utils.forceDirectory(basePath)
        baseImgFile,_ = vc_utils.makeTemporaryFile(basePath)
    # Verifico se o que foi passado para a função é um nome de PDF ou um conjunto 
    # de bytes (o próprio conteúdo do PDF). Se for um conjunto de bytes, crio um 
    # arquivo PDF temporário, que será deletado ao final da conversão.
    filename = pdf_name_or_bytes
    tem_que_deletar = False
    if type(pdf_name_or_bytes) is bytes:
        filetmp = NamedTemporaryFile(prefix='vc_tmp', suffix='.pdf', delete=False)
        filename = filetmp.name
        filetmp.write(pdf_name_or_bytes)
        filetmp.close()
        tem_que_deletar = True

    # if type(filename) is bytes:
    #     return convert_from_bytes(filename)

    # Preparo as variáveis para o subprocesso pdftoppm
    basePath,baseFile = os.path.split(baseImgFile)
    listSuffixes = []

    # Rodo o subprocesso para converter o PDF para imagens
    cp = subprocess.run(['pdftoppm', '-jpeg', '-scale-to-x',str(vc_constants.PDF_WIDTH_DEFAULT),filename, baseImgFile])

    # Deleto, se necessário, o arquivo PDF temporário
    if tem_que_deletar:
        os.remove(filename)

    # Em caso de sucesso, preencho a lista com os sufixos dos 
    # nomes das imagens geradas.
    # Exemplo:
    #   Se foram gerados os arquivos 'pagina_01.jpg' e 'pagina_02.jpg',
    #   então teremos na lista de sufixos ('suffixes') os valores:
    #   ['-01.jpg', '-02.jpg']
    if cp.returncode == 0:
        for root,_,files in os.walk(basePath):
            if root != basePath: continue
            for file in files:
                if not file.startswith(baseFile): continue
                listSuffixes += [file[len(baseFile):]]
    # Em caso de erro, e se for para gerar exceção, então gero a exceção.
    elif raise_error:
        raise PdfConversionError(f'Erro ao executar pdftoppm: Return Code = {cp.returncode}')
    # Quando o processo ocorre com sucesso ou quando não é pra gerar exceção,
    # a função retorna o dicionário abaixo.
    return {'returncode': cp.returncode, 'basepath': basePath, 'basefile': baseFile, 'suffixes': listSuffixes}
