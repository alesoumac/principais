#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import urllib.request as reqst
from datetime import datetime

import jwt
from responder import status_codes as sc

import vc_audit
import vc_constants
import vc_excepts
import vc_utils

PUBLIC_KEY_CACHE = None
PUBLIC_KEY_TIMESTAMP = None
JWKS_ADDRESS = None

def verify_and_decode_jwt(jwt_str):
    global PUBLIC_KEY_CACHE
    global PUBLIC_KEY_TIMESTAMP
    global JWKS_ADDRESS

    if vc_constants.IGNORE_AUTENTIKUS:
        return True,""

    if PUBLIC_KEY_TIMESTAMP is None or (datetime.now() - PUBLIC_KEY_TIMESTAMP).total_seconds() > 3600:
        initialize_public_key_autentikus(JWKS_ADDRESS)
    public_key = PUBLIC_KEY_CACHE
    if public_key == "":
        return False, "Chave pública do Autentikus inválida"

    try:
        # pegar a audiência do payload
        b64_header, b64_payload, _ = jwt_str.split('.')
        payload = json.loads(jwt.utils.base64url_decode(b64_payload))
        header = json.loads(jwt.utils.base64url_decode(b64_header))
        algoritmo = header['alg'] if 'alg' in header else ""
        audiencia = payload['aud'] if 'aud' in payload else ""

        # verificar Roles e Permissions
        if 'cliente' not in payload['roles']:
            return False, 'Role error'
        if 'post validacao_de_documentos' not in payload['permissions']:
            return False, "Permission error"

        # decodificar o bearer token
        jwt.decode(jwt_str, public_key, algorithms=algoritmo, audience=[audiencia])

        # se chegar até aqui, significa que a verificação foi Ok
        # e retorna True (autorizado) e mensagem de erro vazia.
        return True,""

    except jwt.exceptions.ExpiredSignatureError:
        return False, "Expired token signature"
    except (jwt.exceptions.InvalidSignatureError, jwt.exceptions.InvalidTokenError, jwt.exceptions.ImmatureSignatureError):
        return False, "Invalid token signature"
    except Exception as e:
        return False, "Exception {}: {}".format(str(e.__class__), str(e))

def verify_bearer_token(req_header,resp):
    if vc_constants.IGNORE_AUTENTIKUS:
        return True

    hora_inicial = datetime.now()

    msg_erro = "Authorization error"
    autorizado = False
    status = sc.HTTP_403

    try:
        if "Authorization" not in req_header:
            raise vc_excepts.AuthenticationError("Erro de autenticação: Header não possui campo \"Authorization\"")
        #if "alg" not in req_header:
        #    raise jwt.exceptions.InvalidAlgorithmError("Authentication error")

        auth_str = req_header["Authorization"] # auth_str receberá a string "Bearer ..."

        auth_bearer_token = auth_str[7:] if auth_str.startswith("Bearer ") else ""
        #algoritmo = req_header['alg']
        autorizado, msg_erro = verify_and_decode_jwt(auth_bearer_token)
    except vc_excepts.AuthenticationError as e:
        status = sc.HTTP_401
        msg_erro = str(e)
    except Exception as e:
        msg_erro = str(e)

    if not autorizado:
        resp.status_code,resp.media = vc_utils.responseError(status,msg_erro)

    hora_final = datetime.now()
    return autorizado

def initialize_public_key_autentikus(jwks_address):
    global PUBLIC_KEY_CACHE
    global JWKS_ADDRESS
    global PUBLIC_KEY_TIMESTAMP

    if vc_constants.IGNORE_AUTENTIKUS:
        PUBLIC_KEY_CACHE = ""
        return
    
    # para usar chave pública fixa, descomentar as 2 linhas abaixo
    #PUBLIC_KEY_CACHE = '-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvu4V+C+1qlaCAKlfZb60JM5emAsplhsJQ69RpwpLlsTo4hHI0n4HJWWSKiXTmxjbkS8lOt1CfYwm10uMpWf5syUBqmzDZi9iwjGbOG1E41cKlhX3dNkKinKlYZthhyfe9LMxRJ5TllrCCPx2uED6XLJXl4aMnFKTre09EcHjj0nWvYiFhe/lHCjWWmDNDzCxHqLjfmkMfvKgkIkV/1mK5V0JG5QPqjQsgaMc0LbhJZEGIHQV+TWMgVb9Xsb8VBp5DeOBgMKlSABL3x14tP3/5AyjiILjJTQeR4Q2oAaERVJ5SiItVjaaMuSPDl6PwYzriLBLGSKBMcLrugsq0xFhQQIDAQAB\n-----END PUBLIC KEY-----'.encode()
    #return
    
    # pegar a chave pública do Autentikus

    JWKS_ADDRESS = jwks_address

    try:
        vc_utils.printLog(f"Obtendo chave pública de {jwks_address}")
        respStream = reqst.urlopen(jwks_address)
        resp_key = json.loads(respStream.read())
        if 'n' in resp_key:
            PUBLIC_KEY_CACHE = f'-----BEGIN PUBLIC KEY-----\n{resp_key["n"]}\n-----END PUBLIC KEY-----'.encode()
        PUBLIC_KEY_TIMESTAMP = datetime.now()
        vc_utils.printLog(f"Timestamp da obtenção da chave pública: {PUBLIC_KEY_TIMESTAMP}")

    except Exception as e:
        vc_utils.printLog(f"Erro ao carregar o JSON do Authentikus. Classe da exceção = {e.__class__}")
        vc_utils.printLog(f"Msg: { str(e) }")

def securityVerification(request,response):
    host_cliente = vc_audit.get_client_host(request)
    vc_utils.printLog(f"Requisição originada de {host_cliente}")
    vc_utils.printLog("Iniciando protocolo de autenticação")
    req_header = request.headers

    if not verify_bearer_token(req_header, response): 
        vc_utils.printErr("Falha na autenticação")
        return False
    
    vc_utils.printLog("Autorizado!")
    return True
