# Reestruturação da aplicação, divisão em classes, separação do darknet_server.py 30% performance

import os
from datetime import datetime

import vc_constants
import vc_utils


def get_client_host(req):
    try:
        return getattr(req,'_starlette').client.host
    except:
        vc_utils.printErr('Não foi possível obter o host da requisição')
        raise

def inicializa_audit(path_base):
    path_audit = os.path.join(path_base,"static","audit")
    path_upload = os.path.join(path_audit,"images")
    vc_utils.forceDirectory(path_upload)
    return path_audit, path_upload

def clear_image_fields(req_media, resp_media):
    if vc_constants.FIELD_IMAGE    in req_media:  del req_media[vc_constants.FIELD_IMAGE]
    
    if vc_constants.FIELD_PRED_IMG in resp_media: del resp_media[vc_constants.FIELD_PRED_IMG]
    if vc_constants.FIELD_WORK_IMG in resp_media: del resp_media[vc_constants.FIELD_WORK_IMG]
