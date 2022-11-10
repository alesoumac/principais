# -*- coding: utf-8 -*-
import argparse
import io
import json
import os
from datetime import datetime, timedelta

def ler_json(nome_json):
    try:
        valfile = open(nome_json, "r")
        expected_values = json.loads(valfile.read())
        valfile.close()
    except:
        expected_values = {}
    return expected_values

def lista_valores(pasta,campo):
    for root, _, files in os.walk(pasta):
        if pasta != root: continue
        filessrt = sorted(files)
        for fi in filessrt:
            _,ex = os.path.splitext(fi)
            if ex.lower() == ".json":
                vals = ler_json(os.path.join(root,fi))
                if campo in vals:
                    print(vals[campo])

def main():
    # Defino argumentos da linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Paths with jsons to be processed")
    parser.add_argument("-f","--field", type=str, help="Field to be processed")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"\"{args.path}\" não é um diretório")
        exit(-1)

    lista_valores(args.path,args.field)
if __name__ == "__main__":
    main()
