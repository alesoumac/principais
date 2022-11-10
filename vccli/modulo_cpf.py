def resto11(n):
    r = n % 11
    if r < 2: return 0
    else:     return 11 - r

def dv_ch(s,inicio,fim,max):
    try:
        ss = s[inicio:fim+1][::-1]
        return str(resto11(sum([(y % (max-1) + 2) * int(ss[y]) for y in range(len(ss))])))
    except:
        return ""

def dv_cpf(s):
    if len(s) != 9: return ""
    dv1 = dv_ch(s,0,8,11)
    if dv1 == "": return ""
    return dv1+dv_ch(s+dv1,0,9,11)

def dv_cnpj(s):
    if len(s) != 12: return ""
    dv1 = dv_ch(s,0,11,9)
    if dv1 == "": return ""
    return dv1+dv_ch(s+dv1,0,12,9)

def cpf_ok(s):
    if len(s) != 11: return False
    return dv_cpf(s[:-2]) == s[-2:]

def cnpj_ok(s):
    if len(s) != 14: return False
    return dv_cnpj(s[:-2]) == s[-2:]

def find_cpf(s):
    cpfs = []
    for i in range(len(s) - 10):
        cpf_candidato = s[i:i+11]
        if cpf_ok(cpf_candidato):
            if cpf_candidato not in cpfs:
                cpfs += [cpf_candidato]
    return cpfs
