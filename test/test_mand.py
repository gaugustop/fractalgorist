# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:21:58 2021

@author: Gabriel
"""

#%%IMPORTS
from src.mand import Mandelbrot, sin_colortable
from src.tools import logistic_like, harmonicfy, positiv_sin_like
from src.mand_julia import cd_explorer
from output import carregar_pickle, salvar_pickle
import numpy as np
#cd C:\Users\Gabriel\Documents\Python\fractalgorist
#%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"    
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)


#%%INICIATE

mand = Mandelbrot(oversampling = 1, maxiter = 100, xpixels = 600, expoente = 2, gpu = True)
mand.os = 1
mand.xpixels = 600
mand.ypixels = 600
mand.set_aspect(ratio_x = 1, ratio_y = 1)
mand.explore()

#%%BLACK AND WHITE
path_output = os.path.join('output','black_and_white')
fig_name = 'darkbrot #01'
fig_name = 'test1'

#=== salva e imprime === 
#dpi = 350 para impressao
mand.os = 2
#mand.maxiter = 600
#25/2.54*dpi
mand.xpixels = 5000 #50/2.54*dpi
mand.ypixels = 5000
mand.update_set()
mand.draw(os.path.join(path_output,f'{fig_name}.jpg'), signature = '@fractalgorist')

#%%MOVIE
pts = 40
mand.xpixels = int(460)
mand.ypixels = int(460)

#===========light=============
light = mand.light.copy()
lights = list()
v0, vf = 90, 180
for i in positiv_sin_like(v0,vf,int(pts)):
    light[0] = i
    #light[3] = 0.4 + 0.5*(i - v0)/(vf-v0)
    lights.append(light.copy())


# lights += lights[::-1]  
len(lights)

#=========maxiters=========
v0, vf = 90,600
maxiters = positiv_sin_like(v0,vf,int(pts))
# maxiters = np.concatenate((maxiters,maxiters[::-1]), axis=0)

len(maxiters)
#movie===============
dict_to_walk = dict()
dict_to_walk['light'] = lights
dict_to_walk['maxiter'] = maxiters

file_out_gif = os.path.join(path_output,f'{fig_name}.gif')
mand.animate_param(dict_to_walk, file_out_gif, reverse = True)

#%%SAVE DICT TO REPRODUCE
#dict_black_and_white = dict()
pickle_path = os.path.join(path_output,'pickles','dict_bw.pickle')
dict_black_and_white = carregar_pickle(pickle_path)
dict_black_and_white[fig_name] = {'xpixels': mand.xpixels, 'maxiter': mand.maxiter,
                                  'coord': mand.coord, 'gpu':True, 'ncycle': mand.ncycle,
                                  'rgb_amp': mand.rgb_amp, 'rgb_thetas': mand.rgb_thetas, 
                                  'global_phase': mand.global_phase, 'oversampling': mand.os, 
                                  'stripe_s':mand.stripe_s, 'stripe_sig': mand.stripe_sig, 
                                  'step_s': mand.step_s, 'light': mand.light,
                                  'julia_c': mand.julia_c 
                                  }
salvar_pickle(dict_black_and_white, pickle_path)

#%%mandjulia
mandjulia = cd_explorer()
mandjulia.explore()

