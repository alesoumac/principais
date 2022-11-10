# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:44:08 2019

@author: 00248173766
"""

import os
import sys

if __name__ == '__main__':
    for i in range(1,len(sys.argv)):
        nomearq = sys.argv[i]
        if os.path.isdir(nomearq): continue
        _, ext = os.path.splitext(nomearq)
        if ext == '':
            print(nomearq, '-->', nomearq+'.jpg')
            os.rename(nomearq,nomearq+'.jpg')
    print("Fim")
