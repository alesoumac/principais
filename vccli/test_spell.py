# -*- coding: utf-8 -*-
import io
import os
import argparse
from spellchecker import SpellChecker

SPELL = SpellChecker(language="pt", distance=1)
path_prog = ""

dict_transchar = {}
def init_transchar():
    global dict_transchar
    if dict_transchar == {}:
        chrf = open(os.path.join(path_prog,"extchars.txt"), "r")
        for linha in chrf:
            lstr = linha.strip()
            if len(lstr) >= 2:
                c1 = lstr[0]
                c2 = lstr[1]
                dict_transchar.update({c1 : c2})
        chrf.close()

def translate_extchar(s):
    global dict_transchar
    r = ''.join([s])
    r = r.replace(' \'', ' ').replace('\' ',' ').replace('\'','e ')
    for i in range(len(r)):
        aChar = r[i]
        if aChar in dict_transchar:
            aChar = dict_transchar[aChar]
        if aChar < 'a' or aChar > 'z':
            aChar = ' '
        r[i] = aChar
    while '  ' in r:
        r = r.replace('  ',' ')
    return r.strip().lower()

def trata_palavra(p):
    return p
    s = p.replace(',',' ').replace('.',' ').replace('-',' ').replace('_',' ').replace('\t',' ').replace('\r',' ').replace('\n',' ')
    s = s.replace(' \'', ' ').replace('\' ',' ')
    s = s.replace('\'','e ')
    while '  ' in s:
        s = s.replace('  ',' ')
    return s.strip().upper()

def main():
    global SPELL
    global path_prog

    path_prog = sys.path[0]

    parser = argparse.ArgumentParser()
    parser.add_argument("words", nargs='+', type=str, help="Palavras a corrigir (se necessário)")
    args = parser.parse_args()

    palavras = [trata_palavra(p) for p in args.words]
    palavras = (' '.join(palavras)).split(' ')
    print(palavras)
    desconhecidas = [p.upper() for p in SPELL.unknown(palavras)]
    print(' '.join(desconhecidas))
    corrigidas = [SPELL.candidates(p) for p in palavras if p in desconhecidas]
    print(corrigidas)
    #print(' '.join(corrigidas))

if __name__ == "__main__":
    main()    
#oi tô aqui tmb
