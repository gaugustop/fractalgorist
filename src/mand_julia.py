# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:41:03 2021

@author: Gabriel
"""

import numpy as np

import matplotlib.pyplot as plt
import numba as nb
import math

from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Slider
from numba import cuda
from PIL import Image
import imageio
import logging
import copy

logger = logging.getLogger(__name__)
#%%functions

@nb.njit
def mandelbrot(z, c):
    return z*z + c

@nb.njit
def glynn(z,c):
    return z**1.5 + c

@nb.njit
def mitose(z,c):
    if z == 0:
        return 1    
    return (z*z*z+ c)/z

@nb.njit
def logist(z,c):
    return (z - c)*(1-z)*np.exp(z)

@nb.njit
def calc_niter(z0, c, func, maxiter, esc_radius):
    z = z0
    for niter in nb.prange(maxiter):
        z = func(z,c)
        if abs(z) > esc_radius or niter > maxiter:
            break
    return niter    
    
@nb.njit(parallel = True)
def create_c_set(c_coord, xpixels, ypixels, func, maxiter, esc_radius):    
    c_set = np.zeros((ypixels,xpixels), dtype = np.uint8)
    reais = np.linspace(c_coord[0],c_coord[1],xpixels)
    imags = np.linspace(c_coord[2],c_coord[3],ypixels)
    for r in nb.prange(xpixels):
        for i in nb.prange(ypixels):
            real = reais[r]
            imag = imags[i]            
            niter = calc_niter(0, complex (real, imag), func, maxiter, esc_radius)                           
            c_set[i,r] = niter
    return c_set

@nb.njit(parallel = True)
def create_z_set(z_coord, c, xpixels, ypixels, func, maxiter, esc_radius):    
    z_set = np.zeros((ypixels,xpixels), dtype = np.uint8)
    reais = np.linspace(z_coord[0],z_coord[1],xpixels)
    imags = np.linspace(z_coord[2],z_coord[3],ypixels)
    for r in nb.prange(xpixels):
        for i in nb.prange(ypixels):
            real = reais[r]
            imag = imags[i]            
            niter = calc_niter(complex (real, imag), c, func, maxiter, esc_radius)                           
            z_set[i,r] = niter
    return z_set

@nb.cuda.jit
def c_plane_gpu(c_set, xmin, xmax, ymin, ymax, xpixels, ypixels, expoent, maxiter, esc_radius):
    # Retrieve x and y from CUDA grid coordinates
    index = nb.cuda.grid(1)
    x, y = index % c_set.shape[1], index // c_set.shape[1]    
    
    
    # x, y = nb.cuda.grid(2)
    #xmin, xmax, ymin, ymax = c_coord

    # Check if x and y are not out of mat bounds
    if (y < c_set.shape[0]) and (x < c_set.shape[1]):
        creal = xmin + x / (c_set.shape[1] - 1) * (xmax - xmin)
        cim = ymin + y / (c_set.shape[0] - 1) * (ymax - ymin)
        # Initialization of c
        c = complex(creal, cim)
        # c_set[y,x] = calc_niter(0, c, func, maxiter, esc_radius)
        #c_set[y,x] = 2
        
        niter = calc_niter(0, c, expoent, maxiter, esc_radius)
        c_set[y,x] = niter
        #c_set[y,x] = calc_niter(0, c, func, maxiter, esc_radius)
        #color_pixel(c_set[y,x], niter)

@nb.njit
def color_pixel(c_setxy, niter):
    c_setxy = niter
    

#%%class
class cd_explorer():
    def __init__(self, 
                 c_coord = np.array([-2,1.,-1.5,1.5], dtype = np.float32), 
                 z_coord = np.array([-1.5,1.5,-1.5,1.5], dtype = np.float32), 
                 c = 0j, xpixels = 400, ypixels = 400, func = mandelbrot,
                 maxiter = 200, esc_radius = 2, blit = True, dpi = 10):
        
        self.c_coord = c_coord
        self.z_coord = z_coord
        self.c = c
        self.xpixels = xpixels
        self.ypixels = ypixels
        self.func = func
        self.maxiter = maxiter
        self.esc_radius = esc_radius
        self.blit = blit
        self.c_set = create_c_set(c_coord, xpixels, ypixels, func, maxiter, esc_radius)
        self.z_set = create_z_set(z_coord, c, xpixels, ypixels, func, maxiter, esc_radius)
        
        self.dpi = dpi   
        self.fig, self.axs = plt.subplots(nrows = 1, ncols = 2, 
                                          figsize = (self.xpixels*2/dpi + 1, 
                                                     self.ypixels/dpi + 2))
        
        #plot c_set
        plt.sca(self.axs[0])
        self.c_image = plt.imshow(self.c_set, extent = self.c_coord, 
                                  interpolation="None", cmap = 'jet')    
        self.c_position = plt.scatter(c.real, c.imag, 
                                      color = 'white', edgecolors = 'yellow')
        self.axs[0].set_title('c_plane')
        plt.axis('off')
        
        #plot z_set
        plt.sca(self.axs[1])
        self.z_image = plt.imshow(self.z_set, extent = self.z_coord, 
                                  interpolation="None", cmap = 'viridis')
        self.axs[1].set_title(f'z_plane \n $c = {self.c}$')
        plt.axis('off')
        
        plt.suptitle(f'function = {func.__qualname__}', fontsize = 30)
        
        self.change_c = False
        
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.on_click)        
        self.cid2 = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid3 = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    def update_z_set(self):
        self.z_set = create_z_set(self.z_coord, self.c, self.xpixels, self.ypixels, 
                                  self.func, self.maxiter, self.esc_radius)    
    
    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            if event.inaxes == self.axs[0]:  
                self.change_c = True
                self.list_c = list()
                
                self.c = complex(event.xdata, event.ydata)
                self.list_c.append(self.c)
                self.c_position.set_offsets((event.xdata, event.ydata))
                self.update_z_set()
                self.z_image.set_data(self.z_set)
                self.axs[1].set_title(f'z_plane \n $c = {self.c}$')
                self.fig.canvas.draw()
                
                if self.blit:
                    # cache the background
                    self.ax0background = self.fig.canvas.copy_from_bbox(self.axs[0].bbox)
                    self.ax1background = self.fig.canvas.copy_from_bbox(self.axs[1].bbox)
                
    def on_move(self, event):
        if self.change_c == True:
            if event.inaxes == self.axs[0]:
                self.c = complex(event.xdata, event.ydata)
                self.list_c.append(self.c)
                self.c_position.set_offsets((event.xdata, event.ydata))
                self.update_z_set()
                self.z_image.set_data(self.z_set)
                self.axs[1].set_title(f'z_plane \n $c = {self.c}$')
                
                #self.fig.canvas.flush_events()
                #plt.draw()  
                if self. blit:
                    # restore background
                    self.fig.canvas.restore_region(self.ax0background)
                    self.fig.canvas.restore_region(self.ax1background)
        
                    # redraw just the points                    
                    self.axs[0].draw_artist(self.c_position)
                    self.axs[1].draw_artist(self.z_image)
        
                    # fill in the axes rectangle
                    self.fig.canvas.blit(self.axs[0].bbox)
                    self.fig.canvas.blit(self.axs[1].bbox)
        
                    # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
                    # it is mentionned that blit causes strong memory leakage. 
                    # however, I did not observe that.
        
                else:
                    # redraw everything
                    self.fig.canvas.draw()
               
               # self.fig.canvas.flush_events()     
    
    def on_release(self,event):
        self.change_c = False
        self.axs[1].set_title(f'z_plane \n $c = {self.c}$')
        self.fig.canvas.draw()
                
    
        
#%%TESTE

cd = cd_explorer( maxiter = 100, esc_radius = 2)

#len(cd_teste.list_c)
