# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:11:39 2021

@author: Gabriel
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from IPython.display import HTML, Image

def imshow(data, cmap = 'jet', size = (16,9), n_k_resolution = 1, interpolation = 'bilinear', resolucao_min = 960):    
    dpi = resolucao_min*n_k_resolution/size[0]    
    fig = plt.figure(figsize = size, dpi = dpi, 
                     clear = True, tight_layout = True,
                     facecolor=None, edgecolor=None)
    ax = plt.gca()    
    ax.imshow(data, cmap = cmap, aspect = 'auto', interpolation = interpolation)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.set_frameon(False)
    return fig


def anim_mandelbrot_julia(mandelbrot_image, julia_images, list_c, 
                          mandelbrot_Re_range, mandelbrot_Im_range,
                          julia_Re_range, julia_Im_range,
                          dpi = 120, cmap = 'jet', interval = 50):
    
    fig = plt.figure(dpi=dpi)
    
    def update(n):
        
        plt.clf()

        #mandelbrot===========
        ext_mandelbrot = (mandelbrot_Re_range[0], mandelbrot_Re_range[1],
                          mandelbrot_Im_range[0], mandelbrot_Im_range[1])        
        plt.subplot(2,1,1)
        plt.imshow(mandelbrot_image, cmap = cmap, alpha = 0.5, 
                   interpolation = 'bilinear',
                   extent = ext_mandelbrot)
        plt.plot(np.real(list_c), np.imag(list_c), 'k--')
        plt.plot(list_c[n].real, list_c[n].imag, 'yo',mec = None, mew = 1, ms = 3)
        plt.axis('off')
        plt.title(f'${list_c[n]}$')
        
        #julia================
        ext_julia = (julia_Re_range[0], julia_Re_range[1],
                     julia_Im_range[0], julia_Im_range[1])
        plt.subplot(2,1,2)
        plt.imshow(julia_images[n], cmap = cmap, 
                   interpolation = 'bilinear',
                   extent = ext_julia)
        plt.axis('off')
        
        
        #fig.suptitle('c = ' + f'${list_c[n]}$')
        fig.tight_layout()
    
    anim = mpl.animation.FuncAnimation(fig,update,frames=len(list_c),interval=interval)
    
    return anim
    
    # HTML(anim.to_jshtml())
    
def animacao(images, n_k_resolution = 1, size = (16,9), cmap = 'jet', interval = 50, interpolation = 'bilinear', resolucao_min = 960):
    dpi = resolucao_min*n_k_resolution/size[0]
    fig = plt.figure(figsize = size, dpi = dpi)   
    
    def update(n):
        plt.clf()
        plt.imshow(images[n], cmap = cmap, 
                   interpolation = interpolation, aspect = 'auto')
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.set_frameon(False)
        fig.tight_layout()
        
    anim = mpl.animation.FuncAnimation(fig, update, frames = len(images), interval = interval)
    return anim