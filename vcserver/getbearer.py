# -*- coding: utf-8 -*-
import sys
import requests
import json

try:
    v = sys.argv
    ambiente = "P" if len(v) < 2 else v[1].upper()
    if ambiente not in ['D','H','P']:
        raise Exception("Ambiente não reconhecido")

    if ambiente in "HD":
        url_inicial = "valautentikus"
        auth_basic = "dWhqZmxxYTYwZWVjN29maGU4NDNvOXBjbTA6bTFzZ3B1ZzUwczlnbDJrdGc1Y2NnaDVobnI="
    else:
        url_inicial = "autentikus"
        auth_basic = "M2MzZWVtMjE0dGk3Zmw1MTJvNzFjcGVrMTE6OXZmNWg5aDRocGk4ZGFmc3Rua2psaXFiN2Q="
    auth_str = "Basic {}".format(auth_basic)
    header_autk = {"content-type": "application/x-www-form-urlencoded", "Authorization": auth_str}
    data_autk   = "grant_type=client_credentials&scope=escopo_vcdoc"
    url = "https://{}.estaleiro.serpro.gov.br/autentikus-authn/api/v1/token".format(url_inicial)
    resp_token  = requests.post(url, data=data_autk, headers=header_autk)
    if resp_token.status_code != 200:
        raise Exception("Erro de conexão Autentikus - {}".format(resp_token.content.decode()))
    else:
        print(json.dumps(resp_token.json(), indent=2))
except Exception as err:
    msg = str(err).replace('"','\'')
    print('{"access_token":"{}","token_type":"erro","expires_in":0}'.format(msg))
    exit(-1)
