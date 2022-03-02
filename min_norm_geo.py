# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:35:54 2022

@author: gmostafa
"""
import numpy as np

from ls_with_qr import ls_with_qr as ls
from min_norm import min_norm

def proj(C,d,v):
    x = min_norm(C,d)
    print(x)
    w = ls(C.T,v)
    print(w)
    m = np.matmul(C.T,w)
    P = v - m
    z = x + P
    return z
   
    
C =  np.array([[1.,1.,1.]])
d=np.array([[1.]])
v = np.array([[1.],[1.],[2.]])

# solution 
z = proj(C,d,v)
print(z)