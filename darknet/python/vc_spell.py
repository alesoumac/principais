# Reestruturação da aplicacao
# -*- coding: utf-8 -*-

import os
import re

import rapidfuzz.fuzz as fuzz
from spellchecker import SpellChecker

import vc_constants
import vc_strings
import vc_utils

SPELL = SpellChecker(language="pt")
MUNICIPIOS = None
ORG_EXPED = None
UFS = None

SPECIAL_CITY = {
    'sao paulo sp': ('paulo',),
    'rio de janeiro rj': ('janeiro','rio de',),
    'belo horizonte mg': ('horizonte',),
    'brasilia df': ('distrito','federal','brasilia',),
    'curitiba pr': ('curitiba',),
    'florianopolis sc': ('florianopolis',),
    'porto alegre rs': ('alegre rs',),
    'fortaleza ce': ('fortaleza',),
    }

def unknown_words(s):
    """Com base no dicionário definido para o processo de SpellChecker, 
    a função verifica quais palavras são desconhecidas em *s*.
    
    O parâmetro *s* pode ser uma string ou uma lista de strings.
    
    Antes de declarar que uma palavra é desconhecida, são feitas
    tentativas de juntar tal palavra com as palavras posicionadas antes 
    e depois para verificar se há algum espaço inserido numa palavra
    existente. 
    
    >>> Ex. "Paralelep ipedo"
    ...     Aqui temos duas palavras desconhecidas, mas que juntas formam
    ...     uma palavra conhecida.
    >>>

    A função retorna duas listas, onde a primeira lista são as palavras 
    desconhecidas, e a segunda lista são todas as palavras reconhecidas
    (já juntadas se necessário).
    """

    global SPELL

    if isinstance(s, str):
        palavras = s.split(' ')
    else:
        palavras = s
    palavras = [p.upper() for p in palavras]
    if vc_constants.IGNORE_SPELL:
        return [], palavras
    desconhecidas = [p.upper() for p in SPELL.unknown(palavras)]
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
    desconhecidas = [] if len(desconhecidas) == 0 else [p.upper() for p in desconhecidas if p != '']
    palavras = [p for p in palavras if p != '']
    return desconhecidas,palavras

def spellCorrection(palavra):
    correto = SPELL.correction(palavra)
    return palavra if correto is None else correto

def spellCandidates(palavra):
    candidatas = SPELL.candidates(palavra)
    return set() if candidatas is None else candidatas

def SpellAdjust(s, correcting=True, strip_short_noums=False, first_word_is_numeric=False, filter=[]):
    global SPELL

    if s == '': return s
    filtro = set([vc_strings.translate_extchar(f.lower()) for f in filter])
    pedacos = vc_strings.separate_phrases(s)

    rs = ''
    if first_word_is_numeric:
        achei = False
        for (palavras,_) in pedacos:
            if len(vc_strings.KeepChars(palavras,"0123456789")) > 0:
                achei = True
                break
        if not achei:
            first_word_is_numeric = False
            strip_short_noums = True

    for (palavras,separador) in pedacos:
        if first_word_is_numeric and rs == "" and len(vc_strings.KeepChars(palavras,"0123456789")) == 0:
            continue
    
        if palavras != "":
            desconhecidas,novas = unknown_words(palavras)
            espaco = False
            for p in novas:
                if p in desconhecidas:
                    candidatos = spellCandidates(p)
                    candidatos = {vc_strings.translate_extchar(c) for c in candidatos}
                    if len(filtro) > 0 and len(filtro.intersection(candidatos)) > 0:
                        p = ''
                    else:
                        p = spellCorrection(p).upper() if correcting else p
                elif p.lower() in filtro:
                    p = ''

                if strip_short_noums and len(p) <= 2 and rs == '':
                    p = ''
                if espaco and rs != '':
                    rs = rs + ' '
                else:
                    espaco = True
                rs = rs + p

        if separador != "":
            if rs != '' or not strip_short_noums:
                rs = rs + separador

    return rs.strip()

def fuzz_ratio(s,t):
    score = fuzz.ratio(s,t)
    return score, int(round((1 - score/100.0) * (len(s) + len(t))))
    #score, erro = basicCompareRate(s,t)
    #return score * 100.0, erro

def AdjustCity(s):
    global MUNICIPIOS
    global SPECIAL_CITY
    global UFS

    # Separo esse municipio em pedaços, tirando os acentos e outros caracteres.
    r = ' '.join([pal for pal in vc_strings.translate_extchar(s,2).split(" ") if len(pal) > 0])

    special = None
    for p in SPECIAL_CITY: # tratamento especial para algumas cidades mais populosas. Não vai ser usado no órgão expedidor
        for pedaco in SPECIAL_CITY[p]:
            if pedaco in r:
                if special is None:
                    special = p
                elif special != p:
                    special = ""

    max_score = min_erro = melhor_escolha = None
    for municipio in MUNICIPIOS:
        score,erro = fuzz_ratio(r,municipio)
        m2 = municipio[:-3]
        uf = municipio[-2:]
        uf = UFS[ uf ] if uf in UFS else ""
        m3 = f"{m2} {uf}"
        score2,erro2 = fuzz_ratio(r,m2)
        score3,erro3 = fuzz_ratio(r,m3)

        if score2 > score:
            score,erro = score2,erro2
        if score3 > score:
            score,erro = score3,erro3

        if score < 0.25: continue
        if max_score is None \
        or score == 100 \
        or erro < min_erro \
        or (erro == min_erro and score > max_score):
            (max_score,min_erro,melhor_escolha) = (score,erro,municipio)
            if score == 100: break
    
    if melhor_escolha is None:
        return s.strip().upper()

    if max_score < 100 and special is not None and special != "":
        score,erro = fuzz_ratio(special, r)
        if max_score <= score or min_erro >= erro:
            (max_score,min_erro,melhor_escolha) = (score,erro,special)

    r = melhor_escolha.upper()
    r = r[:-3] + ',' + r[-3:]
    if r == 'BRASILIA, DF' or r == 'BRASILIA DISTRITO FEDERAL, DF':
        r = 'BRASILIA-DISTRITO FEDERAL, DF'
    
    return r

letras = "abcdefghijklmnopqrstuvwxyz"
digitos = "0123456789"

def separar_str_doc(s):
    '''Retorna uma lista contendo pedaços separados da string s. As regras pra separar os pedaços são as seguintes:
    * Uma única barra (/) é um pedaço
    * Um único traço (-) é um pedaço
    * Uma sequencia consecutiva de dígitos (0-9) é um pedaço
    * Uma sequencia consecutiva de letras (a-z) é um pedaço
    * Uma sequencia consecutiva de qualquer outro caracter (todos exceto -, /, 0-9 e a-z) é um pedaço
    '''
    if not isinstance(s,str):
        return s
    s = vc_strings.translate_extchar(s) # transformo caracteres acentuados em letras sem acento e já coloco em lowercase
    pedacos = []
    while s != "":
        if s[0] in '/-': # uma única barra ou traço já forma um pedaço
            pedacos += [s[0]]
            s = s[1:]
            continue

        pedaco = re.search(r"^\d+",s) # digitos
        if pedaco is None: pedaco = re.search(r"^[a-z]+",s) # letras
        if pedaco is None: pedaco = re.search(r"^[^0-9^a-z^\-^\/]+",s) # quaisquer outros caracteres
        fim = pedaco.end() if pedaco is not None else len(s)
        pedacos += [s[:fim]]
        s = s[fim:]
    return pedacos

def aproxima_orgao_exp(s):
    global ORG_EXPED
    max_score = min_erro = melhor_escolha = None
    for orgao in ORG_EXPED:
        score,erro = fuzz_ratio(s,orgao)
        if erro > 4: continue
        if max_score is None \
        or score == 100 \
        or erro < min_erro \
        or (erro == min_erro and score > max_score):
            (max_score,min_erro,melhor_escolha) = (score,erro,orgao)

    return melhor_escolha if melhor_escolha is not None else s

padrao_numero = '#'
padrao_letra  = 'x'
padrao_letra1 = '.'
padrao_letra2 = ':'
padrao_orgao  = 'O'
padrao_uf     = 'U'
padrao_barra  = '/'
padrao_resto  = '_'
padroes_letras = 'x.:OU'

PADROES_IMEDIATOS = [ 
    "#",       "O",       "U",       "#O",      "O#",      "U#",      "#_U",     "O_#",     \
    "#_O",     "O/U",     ".#O",     "U#_O",    ".#_O",    ":#_O",    "#_O/U",   "#_O_U",   \
    "#_O/:_U", "#_O/O_U", ]

ARQ_ORG = None
def procurar_padroes(s):
    global PADROES_IMEDIATOS
    global ORG_EXPED
    global UFS
    global ARQ_ORG

    pedacos = s.lower().split(" ")
    pedacos = ["ssp" + pedaco[3:] if pedaco[:3].replace("8","s") == "ssp" else pedaco for pedaco in pedacos if pedaco]
    s = " ".join(pedacos)
    pedacos = separar_str_doc(s)
    padroes = ""
    for pedaco in pedacos:
        if pedaco[0] in digitos:
            padroes += padrao_numero
        elif pedaco == "/":
            padroes += padrao_barra
        elif pedaco in ORG_EXPED:
            padroes += padrao_orgao
        elif pedaco in UFS:
            padroes += padrao_uf
        elif pedaco[0] in letras:
            if len(pedaco) == 1:
                padroes += padrao_letra1
            elif len(pedaco) == 2:
                padroes += padrao_letra2
            else:
                padroes += padrao_letra
        else:
            padroes += padrao_resto
    
    indN = set(vc_strings.indexes(padroes,padrao_numero))
    indO = set(vc_strings.indexes(padroes,padrao_orgao))
    while True:
        mudei = False
        r = ''.join(pedacos)
        if padroes == '' or padroes in PADROES_IMEDIATOS:
            return r

        if padroes[-1] == '_' and padroes[:-1] in PADROES_IMEDIATOS:
            return ''.join(pedacos[:-1])
        if padroes[0] == '_' and padroes[1:] in PADROES_IMEDIATOS:
            return ''.join(pedacos[1:])
        if padroes[-2:] == '_.' and padroes[:-2] in PADROES_IMEDIATOS:
            return ''.join(pedacos[:-2])

        if vc_strings.indexes(padroes,padrao_orgao) == []:
            p = None
            novo_org = ''
            for p in vc_strings.indexes(padroes,padrao_letra)[::-1] :
                org = pedacos[p]
                novo_org = aproxima_orgao_exp(org)
                if novo_org != org:
                    break
            else:
                p = None
            
            if p is not None:
                pedacos[p] = novo_org
                padroes = padroes[:p] + padrao_orgao + padroes[p+1:]
                mudei = True
        
        if not mudei: break
        indO = set(vc_strings.indexes(padroes,padrao_orgao))
        if len(indO) == 0: break

    setN = set([] if len(indN) == 0 else range(min(indN), max(indN)+1))
    setO = set([] if len(indO) == 0 else range(min(indO), max(indO)+1))
    inicio,fim = 0,len(padroes) - 1
    iniN = fimN = None

    if len(setN) == 0 and len(setO) == 0:
        return ''.join(pedacos)

    if len(setO) != 0 and len(setN) != 0:
        setN2 = set(range(0,max(setN)+1))
        if len(setO - setN2) != 0: # existe pelo menos um órgão à direita do último número
            setO -= setN2
        elif len(setO - setN) != 0: # existe pelo menos um órgão à esquerda do primeiro número
            setO -= setN
        else:                       # todos os órgãos (um ou mais) que existem estão no meio dos números
            setN -= set(range(min(setO),max(setN)+1)) # aqui faço com que os números que serão considerados sejam apenas os que estão à esquerda do primeiro órgão
        iniN = min(setN)
        fimN = max(setN)
        inicio = min(iniN,min(setO))
        fim = max(fimN,max(setO))

    elif len(setN) != 0:
        iniN,fimN = inicio,fim = (min(setN), max(setN))

    else:
        inicio,fim = (min(setO), max(setO))

    if iniN is not None and iniN == inicio and iniN > 0 and padroes[iniN-1] in padroes_letras:
        inicio = iniN-1
        pedacos[inicio] = pedacos[inicio][-2:]
    indU = vc_strings.indexes(padroes,padrao_uf,fim)
    if indU == []: indU = vc_strings.indexes(padroes,padrao_letra2,fim)
    indU = fim if indU == [] else indU[-1]
    fim = max(fim,indU)
    for p in setN:
        pedaco = pedacos[p]
        if pedaco[0] in digitos or pedaco[0] in letras: continue
        pedacos[p] = vc_strings.KeepChars(pedaco,'-/')
    
    return ''.join(pedacos[inicio:fim+1])

def AdjustIdentity(s):
    return procurar_padroes(s).upper()

def define_custom_spell_dicitionary(arquivo_texto):
    global SPELL
    try:
        SPELL.word_frequency.load_text_file(arquivo_texto)
    except:
        pass

def read_list(arquivo):
    lista = []
    try:
        fl = open(arquivo)
        lista = [elemento.strip() for elemento in fl]
        fl.close()
    except:
        vc_utils.printErr(f"Erro ao ler arquivo \"{arquivo}\"")
    return lista

def inicializa_spellchecker(path_prog):
    global MUNICIPIOS
    global ORG_EXPED
    global UFS

    define_custom_spell_dicitionary(os.path.join(path_prog,'vcd_spell.dic'))
    MUNICIPIOS = read_list(os.path.join(path_prog,'vcd_cities.dic'))
    ORG_EXPED = read_list(os.path.join(path_prog,'vcd_org_exped.dic'))
    UFS = read_list(os.path.join(path_prog,'vcd_ufs.dic'))
    # transformo minha lista de UF's em um dicionário
    dic = {}
    lista_orgaos = ORG_EXPED.copy()
    for uf in UFS:
        sg = uf[:2]
        dic[ sg ] = uf[3:]
        for orgao in ORG_EXPED:
            if (sg == 'ex' and orgao[-2:] == 'ex') or (sg != 'ex' and orgao[-2:] != 'ex'):
                novo_orgao = f'{orgao}{sg}'
                lista_orgaos += [novo_orgao]
    UFS = dic
    ORG_EXPED = lista_orgaos
