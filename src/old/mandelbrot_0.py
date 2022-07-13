# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 10:13:53 2021

@author: Gabriel
"""
import numba as nb
import numpy as np

parallel = False

@nb.njit(parallel=parallel, fastmath=True)
def mandelbrot(c, max_inter):
    """retorna o número de iterações necessárias para que |Z(k+1)| = (Re_z(k) + i.Im_z(k))^2  + Re_c + i.Im_c >= 2
    Aqui z_0 = 0 e cada ponto do grid representa c"""
    
    z = 0.0j #definição do conjunto de mandelbrot, para diferentes pontos iniciais veja Julia Sets
     
    if(c.real*c.real + c.imag*c.imag) >= 4:
        return 0 
    else:
        for i in nb.prange(max_inter):
            z = z*z + c #processo iterativo para gerar o mandelbrot set
            if(z.real*z.real + z.imag*z.imag) >= 4:
                return i + 1
        return max_inter + 1
    
@nb.njit(parallel=parallel, fastmath=True)
def julia(z, c, max_inter):
    """ retorna o número de iterações necessárias para que |Z(k+1)| = (Re_z(k) + i.Im_z(k))^2  + Re_c + i.Im_c >= 2
    aqui c é contante e cada ponto do grid representa z_0"""
      
    if z.real*z.real + z.imag*z.imag >= 4:
        return 0
    else:
        for i in nb.prange(max_inter):
            z = z*z + c #processo iterativo para gerar o julia set
            if(z.real*z.real + z.imag*z.imag) >= 4:
                return i + 1
        return max_inter + 1

# @nb.jit(parallel=parallel, fastmath=True)    
# def create_mandelbrot_image_old(n_Re, n_Im, max_inter, Re_range = (-2.0,1.0), Im_range = (-1.2, 1.2)):
#     """cria uma imagem (numpay array n_Re x n_Im)"""
#     Re = np.linspace(Re_range[0], Re_range[1], n_Re)
#     Im = np.linspace(Im_range[0], Im_range[1], n_Im)
    
#     resultado = np.zeros((n_Re, n_Im))
#     Re_index = 0
#     for Re_index in nb.prange(n_Re):
#         Im_index = 0
#         for Im_index in nb.prange(n_Im): 
#             resultado[Re_index, Im_index] = mandelbrot(Re[Re_index], Im[Im_index], 
#                                                        max_inter)
#     return resultado.T

@nb.jit(parallel=parallel, fastmath=True)    
def create_mandelbrot_image(n_k_resolution = 1, xpixels = 960,  
                            centro = -1 + 0j,  
                            max_inter = 300):
   
    n_Re = resolucao_min * n_k_resolution
    n_Im = int(n_Re * n_y/n_x)
    Im_range = (centro.imag - Im_max        , centro.imag + Im_max        )
    Re_range = (centro.real - Im_max*n_x/n_y, centro.real + Im_max*n_x/n_y)
   
    Re = np.linspace(Re_range[0], Re_range[1], n_Re)
    Im = np.linspace(Im_range[0], Im_range[1], n_Im)
    
    resultado = np.zeros((n_Im, n_Re))
    Re_index = 0
    for Re_index in nb.prange(n_Re):
        Im_index = 0
        for Im_index in nb.prange(n_Im): 
            resultado[Im_index, Re_index] = mandelbrot(Re[Re_index], Im[Im_index], 
                                                       max_inter)
    return resultado


# @nb.jit(parallel=parallel, fastmath=True)    
# def create_julia_image_old(n_Re, n_Im, c, max_inter, Re_range = (2.0,-2.0), Im_range = (-2.0,2.0)):
#     """cria uma imagem (numpay array n_Re x n_Im)"""
    
#     Re = np.linspace(Re_range[0], Re_range[1], n_Re)
#     Im = np.linspace(Im_range[0], Im_range[1], n_Im)
    
#     resultado = np.zeros((n_Re, n_Im))
#     Re_index = 0
#     for Re_index in nb.prange(n_Re):
#         Im_index = 0
#         for Im_index in nb.prange(n_Im): 
#             resultado[Re_index, Im_index] = julia(Re[Re_index], Im[Im_index], 
#                                                   c,
#                                                   max_inter)
#     return resultado.T

@nb.jit(parallel=parallel, fastmath=True)    
def create_julia_image(c, n_k_resolution = 1, n_x = 16, n_y = 9, centro = 0 + 0j, 
                       Im_max = 1.2, max_inter = 100, resolucao_min = 960):
    """cria uma imagem (numpay array n_Re x n_Im)"""
    
    n_Re = resolucao_min * n_k_resolution
    n_Im = int(n_Re * n_y/n_x)
    Im_range = (centro.imag - Im_max        , centro.imag + Im_max        )
    Re_range = (centro.real - Im_max*n_x/n_y, centro.real + Im_max*n_x/n_y)   
    
    Re = np.linspace(Re_range[0], Re_range[1], n_Re)
    Im = np.linspace(Im_range[0], Im_range[1], n_Im)
    
    resultado = np.zeros((n_Re, n_Im))
    Re_index = 0
    for Re_index in nb.prange(n_Re):
        Im_index = 0
        for Im_index in nb.prange(n_Im): 
            resultado[Re_index, Im_index] = julia(Re[Re_index], Im[Im_index], 
                                                  c,
                                                  max_inter)
    return resultado.T


@nb.jit(parallel=parallel, fastmath=True)
def create_julia_images(n_k_resolution, n_x, n_y, centro, Im_max, list_c, max_inter = 100, resolucao_min = 960):
    """Cria varia imagens de julia, para cada complexo 'c' em list_c"""
    n_images = len(list_c)
    images = list()
    for index_c in nb.prange(n_images):
        c = list_c[index_c]
        julia_image = create_julia_image(n_k_resolution, n_x, n_y, 
                                         centro, Im_max,
                                         c, max_inter, resolucao_min)
 
        images.append(julia_image)
    return images
        

    
    