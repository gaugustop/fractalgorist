# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 18:54:23 2021

@author: Gabriel
"""

from src.mandelbrot_jlesuffleur import Mandelbrot
from src.tools import logistic_like, harmonicfy, positiv_sin_like
from src.mandelbrot_jlesuffleur import jlesuffleur_pics
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


#%%
#%matplotlib qt
x = np.linspace(0,np.pi*4,200)
y = np.sin(x)
plt.plot(x,y)
plt.show()
#%%natal
fig_name = 'natal_13'
mand = carregar_pickle(os.path.join(path_out_png,f'{fig_name}.pickle'))
path_out_png = os.path.join('output','natal')
julia_c = -0.2 + 0j
mand = Mandelbrot(**jlesuffleur_pics['web'], oversampling =1, maxiter = 1000)
mand = Mandelbrot(oversampling =1, maxiter = 1000, julia_c = julia_c)
mand = Mandelbrot(oversampling = 1, maxiter = 1000)
mand.os = 1
mand.xpixels = 700
mand.ypixels = 700
mand.set_aspect(ratio_x = 1, ratio_y = 1)
mand.update_set()
mand.explore()
del mand.explorer

#%%=== salva e imprime === 
dpi = 350
mand.os = 3
mand.maxiter = 300
#25/2.54*dpi
mand.xpixels = 3000 #25/2.54*dpi
mand.ypixels = 3000
mand.update_set()
salvar_pickle(mand,os.path.join(path_out_png,f'{fig_name}.pickle'))
mand.draw(os.path.join(path_out_png,f'{fig_name}.jpg'), signature = '@fractalgorist')
#%%jlesuffleur_pics
mand_crow = Mandelbrot(**jlesuffleur_pics['crow'], oversampling =1, maxiter = 1000)
mand_crow.explore()
mand_crow.draw()

for pic in jlesuffleur_pics.keys():
    mand = Mandelbrot(**jlesuffleur_pics[pic])
    outputfile = os.path.join('output','jlesuffleur',f'{pic}.png')
    mand.draw(outputfile)



#%% julia movie
julia = carregar_pickle(pickle_path)
file_out_png = os.path.join('output','cylic_paradigm','cyclic_paradigm_tease_006.png')

#circulo============
centro = [0.2805 +0.5303j, -1.482386 + 0j, -0.21706 + 0.64466j]
raio = [0.047, 0.005, 0.004]

pts = 36*5
list_c = mdb.create_complex_circle(centro[0], raio[0], pts)
#um pedaco =======
list_c = list_c[110:160]
list_c = harmonicfy(list_c)
len(list_c)

#reta =========
pts = 40
pi = [-1.14533 + 0.28102j, -0.7514314 + 0.0309663j, -0.7556595 + 0.0412344j, -0.0138453 + 0.6866048j, -0.879076 + 0.2487507j, 0.2357794 + 0.5333962j]
pf = [-1.15168 + 0.26912j, -0.7621021 + 0.0945879j, -0.7480087 + 0.0362010j, -0.0772848 + 0.7144291j, -0.748691 + 0.2840497j, 0.2357794 + 0.6608076j]
list_c = positiv_sin_like(pi[5],pf[5],pts)
len(list_c)

plt.figure()
plt.plot(list_c.real, list_c.imag, '.')
#list_c = list_c[:60]

julia = Mandelbrot(gpu = True, maxiter = 100, 
                   rgb_thetas = [0, 30 ,270], step_s = 0,
                   coord=[-1., 1., -1., 1.],
                   julia_c = list_c[0])

julia.xpixels = int(460)
julia.ypixels = int(460)
julia.julia_c = list_c[-1]
julia.update_set()
julia.explore()



#light=============
light = julia.light.copy()
list_light = list()
v0 = 90
vf = 180
for i in positiv_sin_like(v0,vf,int(pts)):
    light[0] = i
    #light[3] = 0.4 + 0.5*(i - v0)/(vf-v0)
    list_light.append(light.copy())


list_light += list_light[::-1]  
len(list_light)

#rgb_thetas
rgb_theta = julia.rgb_thetas.copy()
rgb_thetas = list()
v0 = 0
vf = 180
for i in positiv_sin_like(v0,vf,int(pts)):
    rgb_theta[0] = i
    rgb_thetas.append(rgb_theta.copy())

len(rgb_thetas)    
    
#nclycle
ncycles = positiv_sin_like(10,60,int(n_frames))
ncycles = np.concatenate((ncycle,ncycle[::-1]), axis=0)
len(ncycles)

#global_phase
global_phases = positiv_sin_like(0,45,int(pts))
len(global_phases)

#step_s
step_ss = positiv_sin_like(13,19,int(pts))
len(step_ss)

#stripe_s
stripe_ss = positiv_sin_like(3,12,int(pts))
len(stripe_ss)

julia.explore()
julia.draw(file_out_png)

#movie===============
dict_to_walk = dict()
dict_to_walk['julia_c'] = list_c
dict_to_walk['light'] = list_light
dict_to_walk['rgb_thetas'] = rgb_thetas
dict_to_walk['global_phase'] = global_phases
dict_to_walk['ncycle'] = ncycles
dict_to_walk['step_s'] = step_ss
dict_to_walk['stripe_s'] = stripe_ss
dict_to_walk.keys()
file_out_gif = os.path.join('output','cylic_paradigm','cyclic_paradigm_008.gif')
julia.animate_param(dict_to_walk, file_out_gif, reverse = True)
pickle_path = os.path.join('output','cylic_paradigm','pickles', 'cyclic_paradigm_006.pickle')
del julia.explorer
salvar_pickle(julia, pickle_path)


#%%test 16:9 gif, azimuth and ncycle
filepath = os.path.join('output','avatar.png')
mand = Mandelbrot()
mand.set_aspect(ratio_x = 1, ratio_y = 1)
mand.xpixels = 512
mand.ypixels = 512
mand.update_set()
mand.explore()
mand.
light = mand.light.copy()
n_frames = 40


list_light = list()
for i in logisticfy(0,90,n_frames):
    light[0] = i
    list_light.append(light.copy())

dict_to_walk = {'light':list_light}
# dict_to_walk['ncycle'] = np.linspace(15,20,30)
dict_to_walk['ncycle'] = logisticfy(15,20,n_frames)
mand.animate2(dict_to_walk, filepath, reverse = True)
mand.ncycle
mand.explore()

# from pygifsicle import optimize
# optimize(filepath, os.path.join('input_output','output','mand_zoom_az_test-opt.gif'))

#%%square spyder 
filepath = os.path.join('input_output','output','square_spider_anim.gif')
mand = Mandelbrot()
mand.xpixels = 360
mand.set_aspect(ratio_x = 1, ratio_y = 1)
mand.draw(filepath)
dict_to_walk = dict()
dict_to_walk['stripe_s'] = logisticfy(0,0.5,30)
dict_to_walk['ncycle'] = logisticfy(30,15,30)
mand.animate_param(dict_to_walk, filepath, reverse = True)
mand.explore()
mand.step_s = 0
mand.update_set()
mand.set.shape

#%%lightining
filepath = os.path.join('input_output','output','spiral_forest.gif')
mand = Mandelbrot(xpixels = 720)
mand.set_aspect(ratio_x = 1, ratio_y = 1)
mand.explore()
mand.draw(filepath)
mand.xpixels = 360*2
mand.ypixels = 360*2
mand.update_set()

mand.xpixels = 1920*2
mand.yp

mand.set_aspect(ratio_x = 1, ratio_y = 1)
mand.draw(filepath)

n_frames = 40
dict_to_walk = dict()
dict_to_walk['global_phase'] = logisticfy(130,170,n_frames)
light = mand.light.copy()
list_light = list()
for i in logisticfy(60,90,n_frames):
    light[0] = i
    list_light.append(light.copy())
    
dict_to_walk['light'] = list_light
dict_to_walk['step_s'] = logisticfy(20,40,n_frames)

mand.animate_param(dict_to_walk, filepath, reverse = True)
mand.explore()
