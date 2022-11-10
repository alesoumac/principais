import os
import pytesseract as tess
from spellchecker import SpellChecker
import modulo_env as ME

SPELL = SpellChecker(language="pt")
SPELL.word_frequency.load_text_file(os.path.join(ME.PATH_PROG,'modulo_ocr.dic'))

def cmp_rate(s,t,principal = None):
    if principal is not None and principal == 1 and len(s) == 0: return 1.0, 0
    if principal is not None and principal == 2 and len(t) == 0: return 1.0, 0
    packt = ''.join(set(t))
    positions = {c : [i for i in range(len(s)) if s[i] == c] for c in packt}
    #
    maiores = [ [] for i in t]
    mais_maior = 0
    #
    for i in range(len(t))[::-1]:
        c = t[i]
        for m in positions[c]:
            maior = 0
            r = ''
            for k in range(i+1,len(t)):
                ck = t[k]
                for l,n in enumerate(positions[ck]):
                    if n > m and len(maiores[k][l]) > maior:
                        r = maiores[k][l]
                        maior = len(r)
            r = f"{c}{r}"
            maiores[i] += [r]
            if len(r) > mais_maior:
                mais_maior = len(r)
    #print(positions)
    #print(maiores)
    numerador = denominador = 0
    if principal is None or principal == 1:
        numerador += mais_maior
        denominador += len(s)
    
    if principal is None or principal == 2:
        numerador += mais_maior
        denominador += len(t)
        
    erro = len(t) + len(s) - 2*mais_maior

    return float(numerador) / float(denominador), erro

def cmp_rate_2(s,t,principal = None):
    if principal is not None and principal == 1 and len(s) == 0: return 1.0, 0
    if principal is not None and principal == 2 and len(t) == 0: return 1.0, 0
    packt = ''.join(set(t))
    positions = {c : [i for i in range(len(s)) if s[i] == c] for c in packt}
    #
    maiores = [ [] for i in t]
    mais_maior = 0
    #
    for i in range(len(t)):
        c = t[i]
        for m in positions[c]:
            maior = 0
            r = ''
            for k in range(0,i):
                ck = t[k]
                for l,n in enumerate(positions[ck]):
                    if n < m and len(maiores[k][l]) > maior:
                        r = maiores[k][l]
                        maior = len(r)
            r += c
            maiores[i] += [r]
            if len(r) > mais_maior:
                mais_maior = len(r)
    #print(positions)
    #print(maiores)
    numerador = denominador = 0
    if principal is None or principal == 1:
        numerador += mais_maior
        denominador += len(s)
    
    if principal is None or principal == 2:
        numerador += mais_maior
        denominador += len(t)
        
    erro = len(t) + len(s) - 2*mais_maior

    return float(numerador) / float(denominador), erro

def unknown_words(s):
    global SPELL
    if isinstance(s, str):
        palavras = s.split(' ')
    else:
        palavras = s
    palavras = [p.upper() for p in palavras]
    desconhecidas = SPELL.unknown(palavras)
    if len(desconhecidas) == 0:
        return [], palavras

    desconhecidas = [p.upper() for p in desconhecidas]
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

DIC_EXT = {
    225: 'a', 224: 'a', 228: 'a', 227: 'a', 226: 'a', 192: 'a', 193: 'a', 196: 'a', 195: 'a', 194: 'a', 233: 'e', 232: 'e', 235: 'e', 7869: 'e', 234: 'e', 
    203: 'e', 200: 'e', 201: 'e', 7868: 'e', 202: 'e', 239: 'i', 237: 'i', 297: 'i', 238: 'i', 236: 'i', 296: 'i', 205: 'i', 206: 'i', 204: 'i', 207: 'i', 
    245: 'o', 243: 'o', 244: 'o', 242: 'o', 246: 'o', 213: 'o', 211: 'o', 212: 'o', 210: 'o', 214: 'o', 361: 'u', 250: 'u', 251: 'u', 249: 'u', 252: 'u', 
    360: 'u', 218: 'u', 219: 'u', 217: 'u', 220: 'u', 231: 'c', 199: 'c', 241: 'n', 209: 'n', 253: 'y', 221: 'y', 7923: 'y', 7922: 'y', 124: 'l', 161: 'i', 
    162: 'c', 163: 'e', 164: 'o', 165: 'y', 170: 'a', 176: 'o', 178: '2', 179: '3', 181: 'u', 185: '1', 186: 'o', 197: 'a', 198: 'e', 208: 'd', 215: 'x', 
    216: '0', 222: 'd', 223: 'b', 229: 'a', 230: 'e', 240: 'o', 248: '0', 254: 'b', 255: 'y', 256: 'a', 257: 'a', 258: 'a', 259: 'a', 260: 'a', 261: 'a', 
    262: 'c', 263: 'c', 264: 'c', 265: 'c', 266: 'c', 267: 'c', 268: 'c', 269: 'c', 270: 'd', 271: 'd', 272: 'd', 273: 'd', 274: 'e', 275: 'e', 276: 'e', 
    277: 'e', 278: 'e', 279: 'e', 280: 'e', 281: 'e', 282: 'e', 283: 'e', 284: 'g', 285: 'g', 286: 'g', 287: 'g', 288: 'g', 289: 'g', 290: 'g', 291: 'g', 
    292: 'h', 293: 'h', 294: 'h', 295: 'h', 298: 'i', 299: 'i', 300: 'i', 301: 'i', 302: 'i', 303: 'i', 304: 'i', 305: 'i', 306: 'u', 307: 'j', 308: 'j', 
    309: 'j', 310: 'k', 311: 'k', 312: 'k', 313: 'l', 314: 'l', 315: 'l', 316: 'l', 317: 'l', 318: 'l', 319: 'l', 320: 'l', 321: 'l', 322: 'l', 323: 'n', 
    324: 'n', 325: 'n', 326: 'n', 327: 'n', 328: 'n', 329: 'n', 330: 'n', 331: 'n', 332: 'o', 333: 'o', 334: 'o', 335: 'o', 336: 'o', 337: 'o', 338: 'o', 
    339: 'o', 340: 'r', 341: 'r', 342: 'r', 343: 'r', 344: 'r', 345: 'r', 346: 's', 347: 's', 348: 's', 349: 's', 571: 'e', 572: 'e', 573: 't', 575: 's', 
    576: 'z', 577: '7', 578: '7', 579: 'b', 582: 'e', 583: 'e', 584: 'j', 585: 'j', 586: 'q', 587: 'q', 588: 'r', 589: 'f', 590: 'y', 591: 'y', 7682: 'b', 
    7683: 'b', 7690: 'd', 7691: 'd', 7710: 'f', 7711: 'f', 7744: 'm', 7745: 'm', 7766: 'p', 7767: 'p', 7776: 's', 7777: 's', 7786: 't', 7787: 't', 7808: 'w', 
    7809: 'w', 7810: 'w', 7811: 'w', 7812: 'w', 7813: 'w', 7835: 'f', 350: 's', 351: 's', 352: 's', 353: 's', 354: 't', 355: 't', 356: 't', 357: 't', 358: 'f', 
    359: 'e', 362: 'u', 363: 'u', 364: 'u', 365: 'u', 366: 'u', 367: 'u', 368: 'u', 369: 'u', 370: 'u', 371: 'u', 372: 'w', 373: 'w', 374: 'y', 375: 'y', 
    376: 'y', 377: 'z', 378: 'z', 379: 'z', 380: 'z', 381: 'z', 382: 'z', 384: 'b', 385: 'b', 386: 'b', 387: 'b', 388: 'b', 389: 'b', 390: 'o', 391: 'c', 
    392: 'c', 393: 'd', 394: 'd', 395: 'd', 396: 'd', 397: 'g', 398: '3', 399: 'd', 400: 'e', 401: 'f', 402: 'f', 403: 'g', 404: 'y', 408: 'k', 409: 'k', 
    412: 'w', 413: 'n', 414: 'n', 415: 'o', 416: 'o', 417: 'o', 420: 'p', 421: 'b', 422: 'r', 423: '2', 461: 'a', 462: 'a', 463: 'i', 464: 'i', 465: 'o', 
    466: 'o', 467: 'u', 468: 'u', 469: 'u', 470: 'u', 471: 'u', 472: 'u', 473: 'u', 474: 'u', 475: 'u', 476: 'u', 477: 'd', 478: 'a', 479: 'a', 480: 'a', 
    481: 'a', 482: 'e', 483: 'e', 484: 'g', 485: 'g', 486: 'g', 487: 'g', 488: 'k', 489: 'k', 490: 'q', 491: 'q', 492: 'q', 493: 'q', 494: '3', 495: '3', 
    496: 'j', 500: 'g', 501: 'g', 503: 'p', 504: 'n', 505: 'n', 506: 'a', 507: 'a', 508: 'e', 509: 'e', 510: '0', 511: '0', 512: 'a', 513: 'a', 514: 'a', 
    515: 'a', 516: 'e', 517: 'e', 518: 'e', 519: 'e', 520: 'i', 521: 'i', 522: 'i', 523: 'i', 524: 'o', 525: 'o', 526: 'o', 527: 'o', 528: 'r', 529: 'r', 
    530: 'r', 531: 'r', 532: 'u', 533: 'u', 534: 'u', 535: 'u', 536: 's', 537: 's', 538: 't', 539: 't', 540: '3', 541: '3', 542: 'h', 543: 'h', 544: 'n', 
    548: 'z', 549: 'z', 550: 'a', 551: 'a', 552: 'e', 553: 'e', 554: 'o', 555: 'o', 556: 'o', 557: 'o', 558: 'o', 559: 'o', 560: 'o', 561: 'o', 562: 'y', 
    563: 'y', 570: 'a', 424: '2', 425: 'e', 426: 'l', 427: 't', 428: 't', 429: 'f', 430: 't', 431: 'u', 432: 'u', 435: 'y', 436: 'y', 437: 'z', 438: 'z', 
    439: '3', 440: 'e', 441: 'e', 444: '5', 445: '5', 447: 'p', 658: '3', 688: 'h', 689: 'h', 690: 'j', 691: 'r', 695: 'w', 696: 'y', 730: 'o', 738: 's', 
    739: 'x', 755: 'o', 902: 'a', 904: 'e', 905: 'h', 906: 'i', 908: 'o', 910: 'y', 913: 'a', 914: 'b', 916: 'a', 917: 'e', 918: 'z', 919: 'h', 920: 'o', 
    921: 'i', 922: 'k', 923: 'a', 924: 'm', 925: 'n', 927: 'o', 929: 'p', 931: 'e', 932: 't', 933: 'y', 935: 'x', 938: 'i', 939: 'y', 940: 'a', 941: 'e', 
    942: 'n', 943: 'i', 944: 'u', 945: 'a', 946: 'b', 947: 'y', 948: 'o', 949: 'e', 951: 'n', 952: 'o', 953: 'i', 954: 'k', 956: 'u', 957: 'v', 958: 'e', 
    959: 'o', 960: 'n', 961: 'p', 962: 'c', 963: 'o', 964: 't', 965: 'u', 967: 'x', 969: 'w', 970: 'i', 971: 'u', 972: 'o', 973: 'u', 974: 'w', 975: 'k', 
    976: 'b', 977: 'o', 978: 'y', 979: 'y', 980: 'y', 983: 'x', 984: 'q', 985: 'q', 986: 'c', 987: 'c', 988: 'f', 989: 'f', 994: 'w', 995: 'w', 1010: 'c', 
    1011: 'j', 1013: 'e', 1014: '3', 1017: 'c', 1018: 'm', 1019: 'm', 1020: 'p', 7936: 'a', 7937: 'a', 7938: 'a', 7939: 'a', 7940: 'a', 7941: 'a', 7942: 'a', 
    7943: 'a', 7944: 'a', 7945: 'a', 7946: 'a', 7947: 'a', 7948: 'a', 7949: 'a', 7950: 'a', 7951: 'a', 7952: 'e', 7953: 'e', 7954: 'e', 7955: 'e', 7956: 'e', 
    7957: 'e', 7960: 'e', 7961: 'e', 7962: 'e', 7963: 'e', 7964: 'e', 7965: 'e', 7968: 'n', 7969: 'n', 7970: 'n', 7971: 'n', 7972: 'n', 7973: 'n', 7974: 'n', 
    7975: 'n', 7976: 'h', 7977: 'h', 7978: 'h', 7979: 'h', 7980: 'h', 7981: 'h', 7982: 'h', 7983: 'h', 7984: 'i', 7985: 'i', 7986: 'i', 7987: 'i', 7988: 'i', 
    7989: 'i', 7990: 'i', 7991: 'i', 7992: 'i', 7993: 'i', 7994: 'i', 7995: 'i', 7996: 'i', 7997: 'i', 7998: 'i', 7999: 'i', 8000: 'o', 8001: 'o', 8002: 'o', 
    8003: 'o', 8004: 'o', 8005: 'o', 8008: 'o', 8009: 'o', 8010: 'o', 8011: 'o', 8012: 'o', 8013: 'o', 8016: 'u', 8017: 'u', 8018: 'u', 8019: 'u', 8020: 'u', 
    8021: 'u', 8022: 'u', 8023: 'u', 8025: 'y', 8027: 'y', 8029: 'y', 8031: 'y', 8032: 'w', 8033: 'w', 8034: 'w', 8035: 'w', 8036: 'w', 8037: 'w', 8038: 'w', 
    8039: 'w', 8048: 'a', 8049: 'a', 8050: 'e', 8051: 'e', 8052: 'n', 8053: 'n', 8054: 'i', 8055: 'i', 8056: 'o', 8057: 'o', 8058: 'p', 8059: 'p', 8060: 'w', 
    8061: 'w', 8064: 'a', 8065: 'a', 8066: 'a', 8067: 'a', 8068: 'a', 8069: 'a', 8070: 'a', 8071: 'a', 8072: 'a', 8073: 'a', 8074: 'a', 8075: 'a', 8076: 'a', 
    8077: 'a', 8078: 'a', 8079: 'a', 8080: 'n', 8081: 'n', 8082: 'n', 8083: 'n', 8084: 'n', 8085: 'n', 8086: 'n', 8087: 'n', 8088: 'h', 8089: 'h', 8090: 'h', 
    8091: 'h', 8092: 'h', 8093: 'h', 8094: 'h', 8095: 'h', 8096: 'w', 8097: 'w', 8098: 'w', 8099: 'w', 8100: 'w', 8101: 'w', 8102: 'w', 8103: 'w', 8112: 'a', 
    8113: 'a', 8114: 'a', 8115: 'a', 8116: 'a', 8118: 'a', 8119: 'a', 8120: 'a', 8121: 'a', 8122: 'a', 8123: 'a', 8124: 'a', 8130: 'n', 8131: 'n', 8132: 'n', 
    8134: 'n', 8135: 'n', 8136: 'e', 8137: 'e', 8138: 'h', 8139: 'h', 8140: 'h', 8144: 'i', 8145: 'i', 8146: 'i', 8147: 'i', 8150: 'i', 8151: 'i', 8152: 'i', 
    8153: 'i', 8154: 'i', 8155: 'i', 8160: 'u', 8161: 'u', 8162: 'u', 8163: 'u', 8164: 'p', 8165: 'p', 8166: 'u', 8167: 'u', 8168: 'y', 8169: 'y', 8170: 'y', 
    8171: 'y', 8172: 'p', 8178: 'w', 8179: 'w', 8180: 'w', 8182: 'w', 8183: 'w', 8184: 'o', 8185: 'o', 1084: 'm', 1085: 'h', 1086: 'o', 1088: 'p', 1089: 'c', 
    1090: 't', 1091: 'y', 1093: 'x', 1094: '4', 1095: '4', 1096: 'w', 1097: 'w', 1098: 'b', 1100: 'b', 1101: '3', 1104: 'e', 1105: 'e', 1108: 'e', 1109: 's', 
    1110: 'i', 1111: 'i', 1112: 'j', 1116: 'k', 1118: 'y', 1120: 'w', 1121: 'w', 1122: 'b', 1123: 'b', 1138: 'o', 1139: 'o', 1140: 'v', 1141: 'v', 1142: 'v', 
    1143: 'v', 1150: 'w', 1151: 'w', 1152: 'c', 1153: 'c', 1164: 'b', 1165: 'b', 1166: 'p', 1167: 'p', 1024: 'e', 1025: 'e', 1028: 'e', 1029: 's', 1030: 'i', 
    1031: 'i', 1032: 'j', 1035: 'h', 1036: 'k', 1038: 'y', 1040: 'a', 1041: 'b', 1042: 'b', 1044: 'a', 1045: 'e', 1047: '3', 1050: 'k', 1052: 'm', 1053: 'h', 
    1054: 'o', 1056: 'p', 1057: 'c', 1058: 't', 1059: 'y', 1060: 'o', 1061: 'x', 1062: '4', 1063: '4', 1064: 'w', 1065: 'w', 1066: 'b', 1068: 'b', 1069: '3', 
    1072: 'a', 1073: 'b', 1074: 'b', 1076: 'a', 1077: 'e', 1079: '3', 1082: 'k', 1170: 'f', 1171: 'f', 1176: '3', 1177: '3', 1178: 'k', 1179: 'k', 1180: 'k', 
    1181: 'k', 1182: 'k', 1183: 'k', 1184: 'k', 1185: 'k', 1186: 'h', 1187: 'h', 1188: 'h', 1189: 'h', 1194: 'c', 1195: 'c', 1196: 't', 1197: 't', 1198: 'y', 
    1199: 'y', 1202: 'x', 1203: 'x', 1206: '4', 1207: '4', 1210: 'h', 1211: 'h', 1212: 'e', 1213: 'e', 1214: 'e', 1215: 'e', 1216: 'i', 1219: 'k', 1220: 'k', 
    1223: 'h', 1224: 'h', 1225: 'h', 1226: 'h', 1227: '4', 1228: '4', 1229: 'm', 1230: 'm', 1231: 'i', 1232: 'a', 1233: 'a', 1234: 'a', 1235: 'a', 1236: 'e', 
    1237: 'e', 1238: 'e', 1239: 'e', 1240: 'd', 1241: 'd', 1242: 'd', 1243: 'd', 1246: '3', 1247: '3', 1248: '3', 1249: '3', 1254: 'o', 1255: 'o', 1256: 'o', 
    1257: 'o', 1258: 'o', 1259: 'o', 1260: '3', 1261: '3', 1262: 'y', 1263: 'y', 1264: 'y', 1265: 'y', 1266: 'y', 1267: 'y', 1268: '4', 1269: '4', 1276: 'x', 
    1277: 'x', 8304: '0', 8305: 'i', 8308: '4', 8309: '5', 8310: '6', 8311: '7', 8312: '8', 8313: '9', 8319: 'n', 8320: '0', 8321: '1', 8322: '2', 8323: '3', 
    8324: '4', 8325: '5', 8326: '6', 8327: '7', 8328: '8', 8329: '9', 8336: 'a', 8337: 'e', 8338: 'o', 8339: 'x', 8340: 'd', 8341: 'h', 8342: 'k', 8343: 'l', 
    8344: 'm', 8345: 'n', 8346: 'p', 8347: 's', 8348: 't', 8212: '-', 8208: '-', 8209: '-', 8210: '-', 8211: '-', 8213: '-', 8315: '-', 8331: '-', 8722: '-', 
    9472: '-'}

def translate_extchar(s):
    global DIC_EXT
    rs = ''
    for i in range(len(s)):
        aChar = s[i]
        if ord(aChar) in DIC_EXT:
            aChar = DIC_EXT[ord(aChar)]
        elif ord(aChar) in [9,10,13]:
            aChar = ' '
        rs += aChar
    return rs

def separa_frases(s,trans=False):
    def tira_branco_repetido(ss):
        mr = ''.join(ss)
        while "  " in mr:
            mr = mr.replace("  "," ")
        return mr
    set_caracter_palavra = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    sem_acento = translate_extchar(s)
    ult = 0
    pedacos = []
    tam = len(sem_acento)
    for i in range(tam):
        if i < ult: continue
        u = i
        while u < tam and (sem_acento[u] == ' ' or sem_acento[u] in set_caracter_palavra):
            u += 1
        ult = u
        while ult < tam and (sem_acento[ult] == ' ' or sem_acento[ult] not in set_caracter_palavra):
            ult += 1
        while u > i and sem_acento[u-1] == ' ': u -= 1
        p1 = tira_branco_repetido(sem_acento[i:u] if trans else s[i:u])
        p2 = tira_branco_repetido(sem_acento[u:ult] if trans else s[u:ult])
        pedacos += [(p1,p2)]
    return pedacos

def verifica_spell(s):
    global SPELL
    if s == '': return s
    pedacos = separa_frases(s)
    rs = ''
    for (palavras,separador) in pedacos:
        if palavras != "":
            desconhecidas,novas = unknown_words(palavras)
            espaco = ""
            for p in novas:
                if espaco == "":
                    espaco = " "
                else:
                    rs = rs + ' '
                if p in desconhecidas:
                    rs = rs + SPELL.correction(p).upper()
                else:
                    rs = rs + p
        if separador != "":
            rs = rs + separador
    return rs.strip()

def get_spell_candidates(s):
    global SPELL
    if s == '': return []
    pedacos = separa_frases(s)
    resultado = []
    for (palavras,separador) in pedacos:
        if palavras != "":
            desconhecidas,novas = unknown_words(palavras)
            espaco = ""
            for p in novas:
                if espaco == "":
                    espaco = " "
                else:
                    resultado += [[espaco]]
                if p in desconhecidas:
                    resultado += [[pd.upper() for pd in SPELL.candidates(p)]]
                else:
                    resultado += [[p]]
        if separador != "":
            resultado += [[separador]]
    ncn = 1
    for i in range(len(resultado)):
        if i > 0 and len(resultado[i]) == 1 and len(resultado[i-1]) == 1:
            resultado[i][0] = resultado[i-1][0] + resultado[i][0]
            resultado[i-1] = None
        ncn *= len(resultado[i])
    resultado = [cn for cn in resultado if cn is not None]
    return resultado

def run_tesseract(img):
    return tess.image_to_string(img)

def run_ocr(imgOrFile, numerico = False):
    lin_res = ""
    try:
        if isinstance(imgOrFile, str):
            img = cv2.imread(imgOrFile)
        else:
            img = imgOrFile
        if USE_CALAMARI_OCR:
            st = run_calamari(img)
        else:
            st = run_tesseract(img)
        st = st.replace('\n',' ').replace('\r',' ')
        while '  ' in st:
            st = st.replace('  ',' ')
        lin_res = st.strip()
    except:
        print("Não foi possível executar o OCR")
        return ""
    return lin_res
