# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:13:54 2021

@author: Gabriel
"""
import numpy as np
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

def harmonicfy(vector, k = 200):
    
    def positiv_sin(x):
        return 0.5*(1.0 + np.sin(x))    
   
    #interpolation
    x = np.arange(len(vector))
    f = interp1d(x, vector, kind = 'cubic')    
    xnew = np.linspace(0, len(vector)-1,k*len(vector))
    vector_interp = f(xnew)
    
    #seletor harmonico
    seletor = np.round((len(xnew) - 1)*positiv_sin(np.linspace(-np.pi/2,np.pi/2,len(xnew))))
    seletor = seletor.astype(int)
    
    #vector harmonizado
    vector_harmonized = vector_interp[seletor[::k]]
    
    # ============================ testes ==============================
    # plt.plot(x,vector.real,'r.', xnew,vector_interp.real,'k--', x, vector_harmonized.real, '.g')
    # plt.plot(x,vector.imag,'r.', xnew,vector_interp.imag,'k--', x, vector_harmonized.imag, '.g')
    # plt.plot(x,np.abs(vector),'r.', xnew,np.abs(vector_interp),'k--', x, np.abs(vector_harmonized), '.g')
    # plt.plot(x,np.angle(vector),'r.', xnew,np.angle(vector_interp),'k--', x, np.angle(vector_harmonized), '.g')
    # plt.plot(vector.real, vector.imag, 'r.', vector_interp.real, vector_interp.imag, 'k--', vector_harmonized.real, vector_harmonized.imag, '.g')
    # plt.plot(vector.real, vector.imag, 'r.', vector_harmonized.real, vector_harmonized.imag, '.g')
    
    return vector_harmonized

def logistic_like(v0, vf, pts):
    return v0 + (vf-v0)*(1 / ( 1 + np.exp(-(np.linspace(-5,5,pts)))))

def positiv_sin_like(v0, vf, pts):
    return v0 + (vf-v0) * (0.5 + np.sin(np.linspace(-np.pi/2, np.pi/2, pts))/2)

def create_complex_circle(centro, raio, pts):
    t = np.linspace(0, 2*np.pi, pts)
    list_c = np.zeros(pts).astype(complex)
    for c_index in range(pts):
        list_c[c_index] = centro + raio*np.exp(complex(0,t[c_index]))
    return list_c

