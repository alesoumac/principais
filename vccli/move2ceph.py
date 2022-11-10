import argparse
import boto3
import botocore
import os
import base64
import sys
import threading
from boto3.s3.transfer import TransferConfig
import pandas as pd

# get estaleiro credentials
df = pd.read_csv('ceph_credenciais.csv',index_col=False)

s3_endpoint_url = df['s3_endpoint_url'][0]
s3_access_key_id = df['s3_access_key_id'][0]
s3_secret_access_key = df['s3_secret_access_key'][0]

def s3_upload_file(buck,arquivo):
    try:
        # Configure an S3 Resource 
        # Higher level object oriented API
        
        #aws_akid = base64.decodebytes(bytes(s3_access_key_id,'utf-8')).decode('utf-8')
        #aws_sak = base64.decodebytes(bytes(s3_secret_access_key, 'utf-8')).decode('utf-8')
        #print(aws_akid)
        #print(aws_sak)

        aws_akid = s3_access_key_id
        aws_sak = s3_secret_access_key

        s3 = boto3.resource('s3',
                    '',
                    use_ssl = False,
                    verify = False,
                    endpoint_url = s3_endpoint_url,
                    aws_access_key_id = aws_akid,
                    aws_secret_access_key = aws_sak,
                )
        
        GB = 1024 ** 3
        
            # Ensure that multipart uploads only happen if the size of a transfer
            # is larger than S3's size limit for nonmultipart uploads, which is 5 GB.
        config = TransferConfig(multipart_threshold=5 * GB, max_concurrency=10, use_threads=True)

        s3.meta.client.upload_file(arquivo, buck, os.path.basename(arquivo),
                                                Config=config,
                                                Callback=ProgressPercentage(arquivo))
        print("\033[KUploaded {}".format(arquivo), end='\r')

    except botocore.exceptions.EndpointConnectionError:
        print("Network Error: Please Check your Internet Connection")

def s3_upload_dir(buck,dir):     
    #file_paths = get_files()
    if not os.path.isdir(dir):
        s3_upload_file(buck,dir)
        return
    
    for root, _, files in os.walk(dir):
        if dir != root:
            continue
        for arq in files:
            _,ex = os.path.splitext(arq)
            if ex not in ['.png', '.txt']:
                continue
            s3_upload_file(buck,os.path.join(dir,arq))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UPLOAD A FILE TO CEPH')
    parser.add_argument('bucket', metavar='BUCKET_NAME', type=str,
            help='Name of the bucket to which files has to be uploaded')
    parser.add_argument('path', type=str, help='Path containing files to be upload')
     
    args = parser.parse_args()
    s3_upload_dir(args.bucket, args.path)
    print("Fim   ")

'''
https://storagegw.estaleiro.serpro.gov.br Region: us-east-1		
QDV7PCJX5A4YA6XWEAC2	
66H6mKSIxLMLn0F97SeAx9RDXnOXDpQWEN1EmYPR
'''