import argparse
import os
import sys
import threading

import boto3
import botocore
import pandas as pd
from boto3.s3.transfer import TransferConfig

'''
Para fazer a transferência de um ou mais arquivos/diretórios, basta preparar no 
seu computador local uma pasta com toda a estrutura de diretório e seus respectivos
arquivos e rodar o programa move2ceph indicando que essa pasta é a raiz (equivalente 
à raiz do bucket).
É necessário colocar no arquivo ceph_credenciais.csv (esse arquivo tem que estar na mesma pasta do programa move2ceph.py)
os dados da chave de acesso ao bucket.

Por exemplo:
No computador local temos a pasta:
~/temp
    |- /dados
    |  |- xpto.data
    |
    |- /config
    |  |- yolo.cfg
    |- dados.txt

E temos o bucket "treinocompras"

Ao executar os comandos..
    $ cd pasta_do_move2ceph/
    $ python move2ceph.py treinocompras --localroot ~/temp

.. os arquivos na pasta temp e todos os subdiretórios serão passados para o bucket "treinocompras"
'''

LOCAL_ROOT = ''

S3_ENDPOINT_URL = None
S3_ACCESS_KEY_ID = None
S3_SECRET_ACCESS_KEY = None
S3_BUCKET = None
MOVE_QUIET = True

def printLogMove(s='', end='\n'):
    if MOVE_QUIET: return
    print(s, end=end)

# get estaleiro credentials
def getCredentials(id):
    global S3_ENDPOINT_URL
    global S3_ACCESS_KEY_ID
    global S3_SECRET_ACCESS_KEY
    global S3_BUCKET

    try:
        endPoint = os.getenv('S3_ESTALEIRO_ENDPOINT_URL')
        if endPoint != '':
            S3_ENDPOINT_URL = endPoint
    except:
        pass
    path_prog = sys.path[0]
    ceph_csv = os.path.join(path_prog,'ceph_credenciais.csv')
    if os.path.exists(ceph_csv):
        df = pd.read_csv(ceph_csv,index_col=False)
        dfid = df[df['id'] == id]
        ident = 'Identificador'
        if len(dfid) == 0:
            dfid = df[df['bucket_name'] == id]
            ident = 'Bucket'
        if len(dfid) > 1:
            raise Exception(f'{ident} ambíguo: "{id}". Verificar o arquivo {ceph_csv}.')
        if len(dfid) == 0:
            raise Exception(f'Identificador ou bucket inexistente: "{id}". Verificar o arquivo {ceph_csv}.')

        if S3_ENDPOINT_URL is None:
            S3_ENDPOINT_URL = list(dfid['s3_endpoint_url'])[0]

        S3_ACCESS_KEY_ID     = list(dfid['s3_access_key_id'])[0]
        S3_SECRET_ACCESS_KEY = list(dfid['s3_secret_access_key'])[0]
        S3_BUCKET            = list(dfid['bucket_name'])[0]

def forceDirectory(direc):
    if not os.path.exists(direc):
        path,_ = os.path.split(direc)
        if forceDirectory(path):
            try:
                os.mkdir(direc)
            except:
                return False
    return True

def s3_upload_file(buck,filePath):
    global S3_ENDPOINT_URL
    global S3_ACCESS_KEY_ID
    global S3_SECRET_ACCESS_KEY
    global LOCAL_ROOT

    arquivo = os.path.abspath(filePath)
    printLogMove(LOCAL_ROOT)
    printLogMove(arquivo)
    if not arquivo.startswith(LOCAL_ROOT): return
    chave = arquivo[len(LOCAL_ROOT):]
    if chave.startswith('/'): chave = chave[1:]

    try:
        aws_akid = S3_ACCESS_KEY_ID
        aws_sak = S3_SECRET_ACCESS_KEY

        s3 = boto3.resource('s3',
                    '',
                    use_ssl = True,
                    verify = True,
                    endpoint_url = S3_ENDPOINT_URL,
                    aws_access_key_id = aws_akid,
                    aws_secret_access_key = aws_sak,
                )
        
        GB = 1024 ** 3
        config = TransferConfig(multipart_threshold=5 * GB, max_concurrency=10, use_threads=True)

        s3.meta.client.upload_file(arquivo, buck, chave,
                                                Config=config,
                                                Callback=ProgressPercentage(arquivo))
        printLogMove("\033[KUploaded {}".format(arquivo), end='\r')

    except botocore.exceptions.EndpointConnectionError:
        printLogMove("Network Error: Please Check your Internet Connection")

def s3_upload_dir(buck,dir):
    #file_paths = get_files()
    printLogMove(f'Moving {dir}')
    if not os.path.isdir(dir):
        s3_upload_file(buck,dir)
        return
    
    for root, _, files in os.walk(dir):
        for arq in files:
            s3_upload_file(buck,os.path.join(root,arq))

    return

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

def get_bucket_files(buck, file = ''):
    global S3_ENDPOINT_URL
    global S3_ACCESS_KEY_ID
    global S3_SECRET_ACCESS_KEY
    global LOCAL_ROOT

    use_asterisco = False
    if file.endswith('*'):
        file = file[:-1]
        use_asterisco = True

    try:
        aws_akid = S3_ACCESS_KEY_ID
        aws_sak = S3_SECRET_ACCESS_KEY
        s3 = boto3.resource('s3',
                    '',
                    use_ssl = True,
                    verify = True,
                    endpoint_url = S3_ENDPOINT_URL,
                    aws_access_key_id = aws_akid,
                    aws_secret_access_key = aws_sak,
                )
        
        my_bucket = s3.Bucket(buck)

        for obj in my_bucket.objects.all():
            printLogMove(f'Testando {obj.key}', end='\r')
            if file == '' \
            or (file != '' and use_asterisco and obj.key.startswith(file)) \
            or (file != '' and not use_asterisco and obj.key == file):
                printLogMove(f"{obj.key}              ")
                objPath,objName = os.path.split(obj.key)
                targetDir = os.path.join(LOCAL_ROOT,objPath)
                forceDirectory(targetDir)
                targetFile = os.path.join(targetDir,objName)
                
                # Ensure that multipart uploads only happen if the size of a transfer
                # is larger than S3's size limit for nonmultipart uploads, which is 5 GB.
                s3.meta.client.download_file(buck, obj.key, targetFile)
            else:
                printLogMove()

    except botocore.exceptions.EndpointConnectionError:
        printLogMove("Network Error: Please Check your Internet Connection")

def move2ceph_main(bucket,download=False,localroot='.',file='', quiet=True):
    global LOCAL_ROOT
    global MOVE_QUIET
    global S3_BUCKET

    MOVE_QUIET = quiet
    getCredentials(bucket)
    LOCAL_ROOT = os.path.abspath(localroot)
    if not os.path.isdir(LOCAL_ROOT):
        if download and not os.path.exists(LOCAL_ROOT):
            forceDirectory(LOCAL_ROOT)
        else:
            raise Exception('Err: localroot must be a directory')

    fileOrPath = file
    if not download and fileOrPath == '': fileOrPath = LOCAL_ROOT

    printLogMove(f'Local Root = {LOCAL_ROOT}')
    printLogMove(f'Local File/Path = {fileOrPath}')
    printLogMove(f'Bucket = {S3_BUCKET}')

    if download:
        get_bucket_files(S3_BUCKET, fileOrPath)
    else:
        s3_upload_dir(S3_BUCKET, fileOrPath)

    printLogMove("\nEnd Transfer!   ")

def main():
    parser = argparse.ArgumentParser(description='UPLOAD A FILE TO CEPH')
    parser.add_argument('bucket',            metavar='BUCKET_NAME', type=str, help='Name of the bucket to upload/download')
    parser.add_argument('-d', '--download',  action='store_true',             help='Indicates downloading operation')
    parser.add_argument('-r', '--localroot', type=str,  default='.',          help='Local path corresponding to S3 Bucket root')
    parser.add_argument('-f', '--file',      type=str,  default='',           help='File or path to be uploaded or downloaded')

    args = parser.parse_args()
    move2ceph_main(args.bucket, args.download, args.localroot, args.file, False)

if __name__ == '__main__':
    main()
'''
https://storagegw.estaleiro.serpro.gov.br Region: us-east-1		
QDV7PCJX5A4YA6XWEAC2	
66H6mKSIxLMLn0F97SeAx9RDXnOXDpQWEN1EmYPR
'''
