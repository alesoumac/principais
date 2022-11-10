# -*- coding: utf-8 -*-
import io
import os
import sys
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
    rs = ''
    for i in range(len(r)):
        aChar = r[i]
        if aChar in dict_transchar:
            aChar = dict_transchar[aChar]
        if (aChar >= 'a' and aChar <= 'z') or (aChar >= 'A' and aChar <= 'Z'):
            aChar = aChar.lower()
        else:
            aChar = ' '
        rs += aChar
    while '  ' in rs:
        rs = rs.replace('  ',' ')
    return rs.strip()

def unknown_words(s):
    global SPELL
    if isinstance(s, str):
        palavras = s.split(' ')
    else:
        palavras = s
    palavras = [p.lower() for p in palavras]
    desconhecidas = SPELL.unknown(palavras)
    if len(desconhecidas) == 0:
        return [], palavras

    desconhecidas = [p.lower() for p in desconhecidas]
    junta = []
    juntou = -1
    tam = len(palavras)
    for i in range(tam):
        if i == juntou: continue
        p = palavras[i]
        if p in desconhecidas:
            if i-1 > juntou:
                if len(SPELL.unknown([palavras[i-1] + p])) == 0:
                    junta += [(i-1,i)]
                    juntou = i
                    continue
            elif i < tam-1:
                if len(SPELL.unknown([p + palavras[i+1]])) == 0:
                    junta += [(i,i+1)]
                    juntou = i+1
                    continue
    for i in range(tam):
        if (i-1,i) in junta: continue
        if (i,i+1) in junta:
            palavras[i] = palavras[i] + palavras[i+1]
            palavras[i+1] = ''
    desconhecidas = SPELL.unknown(palavras)
    desconhecidas = [] if len(desconhecidas) == 0 else [p.lower() for p in desconhecidas if p != '']
    palavras = [p for p in palavras if p != '']
    return desconhecidas,palavras

def str_repeat(s,expand=None,ntimes=None):
    ex = 1 if expand is None else expand
    nt = 1 if ntimes is None else ntimes
    if ex == 0 or nt == 0: return ''
    if s == '': return ' ' * int(ex*nt)
    q,r = divmod(int(ex),len(s))
    return (s * q + s[:r]) * nt

def main():
    global SPELL
    global path_prog

    path_prog = sys.path[0]
    init_transchar()

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', type=str, help="Arquivos com palavras")
    args = parser.parse_args()
    conjunto = []

    filenames = sorted(args.files)
    maxlen = 0
    for filename in filenames:
        print("Lendo arquivo {}".format(filename))
        f = open(filename,"r")
        for linha in f:
            palavras = translate_extchar(linha)
            desconhecidas,_ = unknown_words(palavras)
            if len(desconhecidas) != 0:
                conjunto += desconhecidas
                conjunto = list(set(conjunto))
                maxlen = max(maxlen,max([len(p) for p in desconhecidas]))
                #print(' '.join(desconhecidas))
                print(str_repeat('.',len(desconhecidas)),end=' ')

        f.close()
    
    print("Número de palavras novas: {}".format(len(conjunto)))
    print("Maior tamanho: {}".format(maxlen))
    f = open(os.path.join(path_prog,"custom_spell.txt"),"w")
    for t in range(maxlen+1):
        if t == 0: continue
        for d in sorted([p for p in conjunto if len(p) == t]):
            f.write(d+'\n')
    f.close()

    #palavras = [trata_palavra(p) for p in args.words]
    #palavras = (' '.join(palavras)).split(' ')
    #print(palavras)
    #desconhecidas = [p.upper() for p in SPELL.unknown(palavras)]
    #print(' '.join(desconhecidas))
    #corrigidas = [SPELL.candidates(p) for p in palavras if p in desconhecidas]
    #print(corrigidas)
    #print(' '.join(corrigidas))

if __name__ == "__main__":
    main()    
#oi tô aqui tmb
