# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 10:14:55 2021

@author: Gabriel
"""

#cd C:\Users\Gabriel\Documents\Python\fractaland
#%matplotlib qt
import numpy as np
import src.mandelbrot as mdb
import pandas as pd
import plot.plot_2D as plot
import matplotlib.pyplot as plt
import logging
#import palettable
import os
from output import carregar_pickle, salvar_pickle
from IPython.display import HTML, Image


FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"    
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)
#%%mandelbrot e julia

#setup==================================================
# cmap = palettable.cmocean.sequential.Deep_5.mpl_colormap
# cmap = palettable.cmocean.sequential.Ice_8.mpl_colormap
#cmap = palettable.cmocean.sequential.Dense_8.mpl_colormap
cmap = 'jet'
# 1K: 960x540
# 2K: 1920x1080 (essa tela aqui!)
# 4K: 3840x2160
# 8K: 7680x4320
resolucao_min = int(960/2)
n_k_resolution = 1
n_x = 16
n_y = 9
max_inter = int(300)

#circulo=================================================
#centro = -0.75 + 0.125j
centro = [0.2805 +0.5303j, -1.482386 + 0j, -0.21706 + 0.64466j]
raio = [0.047, 0.005, 0.004]
pts = 36*5
list_c = mdb.create_complex_circle(centro[2], raio[2], pts)


#mandelbrot==============================================
centro_mandelbrot = centro[2]
Im_max_mandelbrot = 1.5*raio[2]
mandelbrot_image = mdb.create_mandelbrot_image()#(n_k_resolution, n_x, n_y,                                                 
                                               centro_mandelbrot, Im_max_mandelbrot, 
                                               max_inter = max_inter, resolucao_min = resolucao_min)



#julias==================================================
centro_julia = 0 + 0j
Im_max_julia = 1
julia_images = mdb.create_julia_images(n_k_resolution, n_x, n_y, centro_julia, Im_max_julia, list_c, max_inter, resolucao_min)



#animacao================================================

mandelbrot_Im_range = (centro_mandelbrot.imag - Im_max_mandelbrot        , centro_mandelbrot.imag + Im_max_mandelbrot        )
mandelbrot_Re_range = (centro_mandelbrot.real - Im_max_mandelbrot*n_x/n_y, centro_mandelbrot.real + Im_max_mandelbrot*n_x/n_y)
julia_Im_range = (centro_julia.imag - Im_max_julia        , centro_julia.imag + Im_max_julia        )
julia_Re_range = (centro_julia.real - Im_max_julia*n_x/n_y, centro_julia.real + Im_max_julia*n_x/n_y)

fps = 30
interval = 1/fps * 1000
animacao = plot.anim_mandelbrot_julia(mandelbrot_image, julia_images, list_c, 
                                      mandelbrot_Re_range, mandelbrot_Im_range,
                                      julia_Re_range, julia_Im_range,
                                      dpi = 120, cmap = cmap, interval = interval)

#%matplotlib qt
animacao.save(os.path.join('output','mandel_julia.gif'))
# HTML(animacao.to_jshtml())

#%%julia movie
#setup===================================================
cmap = palettable.cmocean.sequential.Dense_8.mpl_colormap
resolucao_min = int(960/1)
n_k_resolution = 1
n_x = 16
n_y = 9
max_inter = 100

#circulo=================================================
centro = -1 + 0j
#raio = 0.275
raio = 0.275
pts = 420
list_c = mdb.create_complex_circle(centro, raio, pts)

#julias==================================================
centro_julia = 0 + 0j
Im_max_julia = 1
julia_images = mdb.create_julia_images(n_k_resolution, n_x, n_y, centro_julia, Im_max_julia, list_c, max_inter, resolucao_min)

#figura==================================================
# frame = 188
# figura = plot.imshow(julia_images[frame], cmap = cmap, size = (n_x,n_y), n_k_resolution = n_k_resolution, interpolation = 'bilinear')
# figura.savefig(os.path.join('temp',f'julia_{n_k_resolution}K_frame_{frame}.jpg'),
#                dpi = 'figure', bbox_inches = 'tight', pad_inches = 0)

#video===================================================
fps = 60
interval = 1/fps * 1000
anim = plot.animacao(julia_images, n_k_resolution = n_k_resolution,
                     size = (n_x, n_y), cmap = cmap, 
                     interval = interval, interpolation = 'bilinear', 
                     resolucao_min = resolucao_min)

dpi = resolucao_min*n_k_resolution/n_x 
anim.save(os.path.join('temp',f'julia_{cmap.name}_{n_k_resolution}K_{fps}fps_{len(list_c)}_frames.gif'), 
          dpi = dpi, fps = fps)


#%%mandelbrot zoom
tic = pd.Timestamp.now()
#cmap = palettable.cmocean.sequential.Dense_8.mpl_colormap
#cmap = palettable.cmocean.sequential.Oxy_10.mpl_colormap
#cmap = palettable.cmocean.diverging.Balance_19.mpl_colormap
#cmap = palettable.colorbrewer.diverging.PuOr_11.mpl_colormap
cmap = palettable.colorbrewer.diverging.RdYlBu_11.mpl_colormap


#cmap = palettable.scientific.diverging.Berlin_20.mpl_colormap
resolucao_min = int(960/4)
n_k_resolution = 1 #essa tela é 2K
n_x = 16
n_y = 9
max_inter = int(100)


n_frames = 105*20
zoom_speed = 2
#centro_mandelbrot = -0.75 + 0.125j
#centro_mandelbrot = complex(-np.e/7, -np.e/20)
centro_mandelbrot = complex(-0.743643887037158704752191506114774, 0.131825904205311970493132056385139) 
Im_max_mandelbrot_0 = 1.2

mandelbrot_zoom_images = list()
Im_max_mandelbrot = Im_max_mandelbrot_0

#video===================================================
fps = 40
interval = 1/fps * 1000


movie_name = f'mandelbrot_zoom6_{cmap.name}_{n_k_resolution}K_{fps}fps_{n_frames}_frames.mp4'
if os.path.isfile(os.path.join('input_output','output',movie_name)): 
             os.remove(os.path.join('input_output','output',movie_name))

t = list()
for frame in range(n_frames):
    mandelbrot_image = mdb.create_mandelbrot_image(n_k_resolution, n_x, n_y, 
                                                   centro_mandelbrot, 
                                                   Im_max_mandelbrot, max_inter, 
                                                   resolucao_min)    
    Im_max_mandelbrot = Im_max_mandelbrot / (1 + zoom_speed/100)
    max_inter = max(max_inter, int(-np.log2(Im_max_mandelbrot)*200))
    #max_inter = min(max_inter + zoom_speed, max_max_inter)
    mandelbrot_zoom_images.append(mandelbrot_image)
   # salvar_pickle(mandelbrot_zoom_images, os.path.join('input_output','temp', movie_name +'.pickle'))
    if frame%20 == 0:
        logger.info(f'fazendo filme zoom de mandelbrot, progresso: {frame/n_frames*100:.2f}%')
        t.append(pd.Timestamp.now())  
    


anim = plot.animacao(mandelbrot_zoom_images, n_k_resolution = n_k_resolution,
                     size = (n_x, n_y), cmap = cmap, 
                     interval = interval, interpolation = 'bilinear', 
                     resolucao_min = resolucao_min)

dpi = resolucao_min*n_k_resolution/n_x 
anim.save(os.path.join('input_output','output', movie_name), 
          dpi = dpi, fps = fps)

del mandelbrot_zoom_images

toc = pd.Timestamp.now()
logger.info(f'mandlbrot zoom concluido, tempo decorrido: {toc-tic}')
#%%test_max_inter
n_frames = 105*20 #limite ~105*18
zoom_speed = 2
Im_max_mandelbrot_0 = 1.2
Im_max_mandelbrot = Im_max_mandelbrot_0
max_inter = int(60)
#max_max_inter = 1000
plot_max_inter = [max_inter]
plot_Im_max_mandelbrot = [Im_max_mandelbrot]
for frame in range(n_frames):
    
    Im_max_mandelbrot = Im_max_mandelbrot / (1 + zoom_speed/100)
    plot_Im_max_mandelbrot.append(Im_max_mandelbrot)
    max_inter = max(max_inter, int(-np.log2(Im_max_mandelbrot)*200))
   
    #max_inter = min(max_inter + zoom_speed, max_max_inter)
    plot_max_inter.append(max_inter)

frames = np.arange(n_frames+ 1)

plt.plot(frames, plot_max_inter, 'g-')
plt.plot(frames, plot_Im_max_mandelbrot,'r')
#%%cardioid
a = 0.25
phi = np.linspace(0,2*np.pi,200)
f = 2*a*(1 - np.cos(phi))*np.exp(phi*complex(0,1))
f = f + 0.25
plt.plot(f.real, f.imag)
plt.grid()

