#!/usr/bin/env python3

"""Compute and draw/explore/animate the Mandelbrot set.

Fast computation of the Mandelbrot set using Numba on CPU or GPU. The set is
smoothly colored with custom colortables.

  mand = Mandelbrot()
  mand.explore()
"""

import math
import random
import os
import numpy as np

import matplotlib.pyplot as plt
from numba import  njit
from matplotlib.widgets import Slider#,TextBox, Button
from numba import cuda
from PIL import Image, ImageFont, ImageDraw 

import imageio
import logging


logger = logging.getLogger(__name__)

def star_interior(size, p):
    np.random.seed(111)
    mat = np.zeros(size**2)
    n_stars = int(p*len(mat))
    mat[:n_stars] = np.random.rand(n_stars)
    np.random.shuffle(mat)
    mat = mat.reshape((size, size))
    return mat
    

def sin_colortable(rgb_amp = [1.,1.,1], rgb_thetas=[0., 250., 270.], global_phase = 0, ncol=2**8):
    """ Sinusoidal color table
   
    Cyclic and smooth color table made with a sinus function for each color
    channel
   
    Args:
        rgb_amp: [float, float, float]
            amplitude for each color channel, in [0,1]
        rgb_thetas: [float, float, float]
            phase in degrees for each color channel, in [0,360]
        global_phase: float
            global phase, between 0 and 360
        ncol: int
            number of color in the output table

    Returns:
        ndarray(dtype=float, ndim=2): color table
    """
    global_phase *= math.pi/180
    rgb_thetas = np.array(rgb_thetas).astype(float)
    rgb_thetas *= math.pi/180
    def colormap(x, rgb_amp, rgb_thetas):
        # x in [0,1]
        # Compute the amplitude and phase of each channel
        y = np.column_stack(((global_phase + x + rgb_thetas[0]),
                             (global_phase + x + rgb_thetas[1]),
                             (global_phase + x + rgb_thetas[2])))
        
        # Set amplitude between 0 and rgb_amp
        val = rgb_amp*(0.5 + 0.5*np.sin(y))
        return val
    return colormap(np.linspace(0, 2*math.pi, ncol),rgb_amp, rgb_thetas)

@njit
def blinn_phong(normal, light):
    ## Lambert normal shading (diffuse light)
    normal = normal / abs(normal)    
    
    # theta: light azimuth; phi: light angle
    # light vector: [cos(theta)cos(phi), sin(theta)cos(phi), sin(phi)]
    # normal vector: [normal.real, normal.imag, 1]
    # Diffuse light = dot product(light, normal)
    theta = light[0]/180*math.pi
    phi = light[1]/180*math.pi
    
    ldiff = (normal.real*math.cos(theta)*math.cos(phi) +
               normal.imag*math.sin(theta)*math.cos(phi) + 
               1*math.sin(phi))
    # Normalization
    ldiff = ldiff/(1+1*math.sin(phi))
    
    ## Specular light: Blinn Phong shading
    # Phi half: average between phi and pi/2 (viewer angle)
    # Specular light = dot product(phi_half, normal)
    phi_half = (math.pi/2 + phi)/2
    lspec = (normal.real*math.cos(theta)*math.sin(phi_half) +
             normal.imag*math.sin(theta)*math.sin(phi_half) +
             1*math.cos(phi_half))
    # Normalization
    lspec = lspec/(1+1*math.cos(phi_half))
    #spec_angle = max(0, spec_angle)
    lspec = lspec ** light[6] # shininess
    
    ## Brightness = ambiant + diffuse + specular
    bright = light[3] + light[4]*ldiff + light[5]*lspec
    ## Add intensity
    bright = bright * light[2] + (1-light[2])/2 
    return(bright)

# @njit
# def newton(expoente, c, p):
#     def f(z, z0):
#         f = z0
#         for n in range(p):
#             f = f**expoente + c
        

@njit
def smooth_iter(c, maxiter, stripe_s, stripe_sig, 
                julia_c, center, rot, expoente, z_memo):
    """ Smooth number of iteration in the Mandelbrot set for given c
   
    Args:
        c: complex
            point of the complex plane
        maxiter: int
            maximal number of iterations
        stripe_s:
            frequency parameter of stripe average coloring
        stripe_sig:
            memory parameter of stripe average coloring

    Returns: (float, float, float, complex)
        - smooth iteration count at escape, 0 if maxiter is reached
        - stripe average coloring value, in [0,1]
        - dem: estimate of distance to the nearest point of the set
        - normal, used for shading
    """
    #rotation
    c = rotate_point(c,center,rot)
    
    # Escape radius squared: 2**2 is enough, but using a higher radius yields
    # better estimate of the smooth iteration count and the stripes
    esc_radius = 10**6
 
    #inicial z  
    #z = complex(0, 0)  
    z = c
    
    if julia_c:
        # z = c #the inicial z becomes the point in canvas
        c = julia_c #
   
    # Stripe average coloring if parameters are given
    stripe = (stripe_s > 0) and (stripe_sig > 0)
    stripe_a =  0
    # z derivative
    dz = 1+0j    
    # if  julia_c: 
    #     dc = 0 
    # else: 
    #     dc = 0
    dc = 0
    # Mandelbrot iteration 
    
    # z_memo[:] = 0 + 0j
    # n_memo = len(z_memo)
    for n in range(maxiter):
        # memory_id = n % n_memo
        #saving last iteration        
        # z_memo[memory_id] = z        
        z_old = z
        dz_old = dz
        # derivative update        
        dz = expoente*z_old**(expoente - 1)*dz_old + dc         
        # z update
        z = z_old**expoente + c
        
        modz = abs(z)
        moddz = abs(dz)
        
        #interior detection
        eps = 1/esc_radius
       
        #delta_z = z - z_old
        #mod_delta_z = abs(delta_z)
        # for i in range(min(n_memo,n)):
        #     delta_z_new = z - z_memo[i]
        #     mod_delta_z_new = abs(delta_z_new)
        #     if mod_delta_z_new < mod_delta_z:
        #         delta_z = delta_z_new  
        #         mod_delta_z = mod_delta_z_new
                
        if moddz < eps:
           
            # Milton's distance estimator
            dem =  expoente * math.log(moddz)
            
            log_ratio = math.log(eps)/math.log(moddz)
            smooth_i = math.log(log_ratio)*24*expoente#math.log(log_ratio)#/math.log(expoente) #so that 0 < smooth_i < 1
            #smooth_i = 1 - math.log(modz)*(expoente - 1)/math.log(expoente)
            
            # Normal vector for lighting            
            normal = dz #not good
            
            return (n + smooth_i,0,dem,normal)
        
        if stripe:
            # Stripe Average Coloring
            # See: Jussi Harkonen On Smooth Fractal Coloring Techniques
            # cos instead of sin for symmetry
            # np.angle inavailable in CUDA
            # np.angle(z) = math.atan2(z.imag, z.real)
            stripe_t = (math.sin(stripe_s*math.atan2(z.imag, z.real)) + 1) / 2
        
        # If escape: save (smooth) iteration count
        # Equivalent to abs(z) > esc_radius
        if modz > esc_radius:
            
            # Smooth iteration count: equals n when abs(z) = esc_radius
            log_ratio = math.log(modz)/math.log(esc_radius)
            smooth_i = 1 - math.log(log_ratio)/math.log(expoente) #so that 1 > smooth_i > 0
            
            if stripe:
                # Stripe average coloring
                # Smoothing + linear interpolation
                # spline interpolation does not improve
                stripe_a = (stripe_a * (1 + smooth_i * (stripe_sig-1)) +
                            stripe_t * smooth_i * (1 - stripe_sig))
                # Same as 2 following lines:
                #a2 = a * stripe_sig + stripe_t * (1-stripe_sig)
                #a = a * (1 - smooth_i) + a2 * smooth_i            
                # Init correction, init weight is now:
                # stripe_sig**n * (1 + smooth_i * (stripe_sig-1))
                # thus, a's weight is 1 - init_weight. We rescale
                stripe_a = stripe_a / (1 - stripe_sig**n *
                                       (1 + smooth_i * (stripe_sig-1)))

            # Normal vector for lighting
            # u = z/dz
            #u = u/abs(u)
            # normal = u # 3D vector (u.real, u.imag. 1)
            normal = z/dz

            # Milton's distance estimator
            dem = modz * math.log(modz) / moddz / 2

            # real smoothiter: n+smooth_i (1 > smooth_i > 0)
            # so smoothiter <= niter, in particular: smoothiter <= maxiter
            return (n + smooth_i, stripe_a, dem, normal)
       
        if stripe:
            stripe_a = stripe_a * stripe_sig + stripe_t * (1-stripe_sig)
           
    # Otherwise: set parameters to 0
    return (0,0,0,0)
           
@njit
def color_pixel(matxy, niter, stripe_a, step_s, dem, normal, colortable,
                ncycle, light):
    """ Colors given pixel, in-place
   
    Coloring is based on the smooth iteration count niter which cycles through
    the colortable (every ncycle). Then, shading is added using the stripe
    average coloring, distance estimate and normal for lambert shading.
   
    Args:
        matxy: ndarray(dtype=float, ndim=1)
            pixel to color, 3 values in [0,1]
        niter: float
            smooth iteration count
        stripe_a: float
            stripe average coloring value
        dem: float
            boundary distance estimate
        normal: complex
            normal
        colortable: ndarray(dtype=uint8, ndim=2)
            cyclic RGB colortable
        ncycle: float
            number of iteration before cycling the colortable
           

    Returns: None
    """

    ncol = colortable.shape[0] - 1
    
    # Apply power post-transform to ncycle
    ncycle = math.sqrt(ncycle)
    
    # Power post-transform and mapping to [0,1]
    niter = math.sqrt(niter) % ncycle / ncycle
    
    # Cycle through colortable
    col_i = round(niter * ncol)

    def overlay(x, y, gamma):
        """x, y  and gamma floats in [0,1]. Returns float in [0,1]"""
        if (2*y) < 1:
            out = 2*x*y
        else:
            out = 1 - 2 * (1 - x) * (1 - y)
        return out * gamma + x * (1-gamma)
    
    # brightness with Blinn Phong shading
    bright = blinn_phong(normal, light)
    
    # dem: log transform and sigmoid on [0,1] => [0,1]
    dem = -math.log(dem)/12
    dem = 1/(1+math.exp(-10*((2*dem-1)/2)))

    # Shaders: steps and/or stripes
    nshader = 0
    shader = 0
    # Stripe shading
    if stripe_a > 0:
        #bright = overlay(bright, stripe_a, 1) * (1-dem) + dem * bright
        nshader += 1
        shader = shader + stripe_a
    # Step shading
    if step_s > 0:
        # Color update: constant color on each major step
        step_s = 1/step_s
        col_i = round((niter - niter % step_s)* ncol)
        # Major step: step_s frequency
        x = niter % step_s / step_s
        light_step = 6*(1-x**5-(1-x)**100)/10
        # Minor step: n for each major step
        step_s = step_s/8
        x = niter % step_s / step_s
        light_step2 = 6*(1-x**5-(1-x)**30)/10
        # Overlay merge between major and minor steps
        light_step = overlay(light_step2, light_step, 1)
        nshader += 1
        shader = shader + light_step
    # Applying shaders to brightness
    if nshader > 0:
        bright = overlay(bright, shader/nshader, 1) * (1-dem) + dem * bright
    # Set pixel color with brightness
    for i in range(3):
        # Pixel color
        matxy[i] = colortable[col_i,i]
        # Brightness with overlay mode
        matxy[i] = overlay(matxy[i], bright, 1)
        # Clipping to [0,1]
        matxy[i] = max(0,min(1, matxy[i]))
        
@njit
def compute_set(creal, cim, maxiter, colortable, ncycle, stripe_s, stripe_sig,
                step_s, diag, light, julia_c, center, rot, expoente, z_memo):
    """ Compute and color the Mandelbrot set (CPU version)
   
    Args:
        creal: ndarray(dtype=float, ndim=1)
            vector of real coordinates
        cim: ndarray(dtype=float, ndim=1)
            vector of imaginary coordinates
        maxiter: int
            maximal number of iterations
        colortable: ndarray(dtype=uint8, ndim=2)
            cyclic RGB colortable
        ncycle: float
            number of iteration before cycling the colortable
        stripe_s:
            frequency parameter of stripe average coloring
        stripe_sig:
            memory parameter of stripe average coloring

    Returns:
        ndarray(dtype=uint8, ndim=3): image of the Mandelbrot set
    """
    xpixels = len(creal)
    ypixels = len(cim)

    # Output initialization
    mat = np.zeros((ypixels, xpixels, 3))    

    # Looping through pixels
    for x in range(xpixels):
        for y in range(ypixels):
            
            # Initialization of c
            c = complex(creal[x], cim[y])
            # Get smooth iteration count
            niter, stripe_a, dem, normal = smooth_iter(c, maxiter, stripe_s,
                                                      stripe_sig, julia_c,
                                                      center, rot, expoente,
                                                      z_memo)
            # If escaped: color the set
            if niter > 0:
                # dem normalization by diag
                color_pixel(mat[y,x,], niter, stripe_a, step_s, dem/diag,
                            normal, colortable,
                            ncycle, light)
    return mat

@cuda.jit
def compute_set_gpu(mat, xmin, xmax, ymin, ymax, maxiter, colortable, ncycle,
                    stripe_s, stripe_sig, step_s, diag, light, 
                    julia_c, center, rot, expoente, z_memo):
    """ Compute and color the Mandelbrot set (GPU version)
   
    Uses a 1D-grid with blocks of 32 threads.
   
    Args:
        mat: ndarray(dtype=uint8, ndim=3)
            shared data to write the output image of the set
        xmin, xmax, ymin, ymax: float
            coordinates of the set
        maxiter: int
            maximal number of iterations
        colortable: ndarray(dtype=uint8, ndim=2)
            cyclic RGB colortable
        ncycle: float
            number of iteration before cycling the colortable
        stripe_s:
            frequency parameter of stripe average coloring
        stripe_sig:
            memory parameter of stripe average coloring

    Returns:
        mat: ndarray(dtype=uint8, ndim=3)
            shared data to write the output image of the set
    """
    # Retrieve x and y from CUDA grid coordinates
    index = cuda.grid(1)
    x, y = index % mat.shape[1], index // mat.shape[1]
    #ncol = colortable.shape[0] - 1
   
    # Check if x and y are not out of mat bounds
    if (y < mat.shape[0]) and (x < mat.shape[1]):
        # Mapping pixel to C
        creal = xmin + x / (mat.shape[1] - 1) * (xmax - xmin)
        cim = ymin + y / (mat.shape[0] - 1) * (ymax - ymin)
        # Initialization of c
        c = complex(creal, cim)
        # Get smooth iteration count
        niter, stripe_a, dem, normal = smooth_iter(c, maxiter, stripe_s,
                                                   stripe_sig, julia_c,
                                                   center, rot, expoente,
                                                   z_memo)
        # If escaped: color the set
        if niter > 0:
            color_pixel(mat[y,x,], niter, stripe_a, step_s, dem/diag, normal,
                        colortable, ncycle, light)

@njit
def rotate_point(c, center, rot): 
    #from gradius to rad
    rot = np.pi*rot/180 
    #translate
    c -= center      
    #rotate
    c_new = complex(c.real*math.cos(rot) - c.imag*math.sin(rot),
                    c.real*math.sin(rot) + c.imag*math.cos(rot))                    
    #translate back
    c_new += center
    return c_new

#não funciona, precisaria de 4 pontos ao invez de só os da diagonal
# def rotate_coord(coord, rot):
    
#     #to facilitate
#     def rotate_point(x,y,angle):        
#            x_new = x*math.cos(angle) - y*math.sin(angle)
#            y_new = x*math.sin(angle) + y*math.cos(angle)
#            return x_new,y_new
       
#     x_left, x_right = coord[0], coord[1]
#     y_bottom, y_top = coord[2], coord[3]             
          
#     #getting center
#     x_center = 0.5*(x_left + x_right)
#     y_center = 0.5*(y_bottom + y_top)
    
#     #from gradius to rad
#     rot = np.pi*rot/180 
    
#     #translation
#     x_left -= x_center
#     x_right -= x_center
#     y_bottom -= y_center
#     y_top -= y_center
    
#     #rotation       
#     x_left_new, y_bottom_new = rotate_point(x_left, y_bottom, rot)
#     # x_left_new, y_top_new = rotate_point(x_left, y_top, rot)
#     x_right_new, y_top_new = rotate_point(x_right, y_top, rot)
#     # x_right_new, y_bottom_new = rotate_point(x_right, y_bottom, rot)
    
#     #translation back
#     x_left_new += x_center
#     x_right_new += x_center
#     y_bottom_new += y_center
#     y_top_new += y_center
        
#     return [x_left_new, x_right_new, y_bottom_new, y_top_new]

@njit
def sample_colortable(colortable, ncycle):
     
    #n colors on colortable
    ncol = colortable.shape[0] - 1
         
    colortable_sample = np.zeros((1,ncol,3)) 
    
    # Apply power post-transform to ncycle
    ncycle = math.sqrt(ncycle)    
    
    for x in range(ncol):
        # Power post-transform and mapping to [0,1]
        #niter <- x
        niter = math.sqrt(x) % ncycle / ncycle            
        # Cycle through colortable
        col_i = round(niter * ncol)            
        for i in range(3):
            # Pixel color
            colortable_sample[0,x,i] = colortable[col_i,i]
            # Clipping to [0,1]
            #colortable_sample[0,x,i] = max(0,min(1, colortable_sample[0,x,i]))       
    
    colortable_sample = (255*colortable_sample).astype(np.uint8)
    
    return colortable_sample     
    
class Mandelbrot():
    """Mandelbrot set object"""
    def __init__(self, xpixels=1280, maxiter=5000,
                 coord=[-2.600125, 1.844125, -1.25, 1.25], gpu=True, ncycle=32,
                 rgb_amp = [1., 1., 1.], rgb_thetas=[0., 200., 270.], global_phase = 0,
                 oversampling=3, stripe_s=0,
                 stripe_sig=.9, step_s=0,
                 light = [.125*180/math.pi, .5*180/math.pi, .75, .2, .5, .5, 20],
                 julia_c = False, rot = 0, expoente = 2, max_memo = 2000, **kwargs):
        """Mandelbrot set object
   
        Args:
            xpixels: int
                image width (in pixels)
            maxiter: int
                maximal number of iterations
            coord: (float, float, float, float)
                coordinates of the frame in the complex space. Default to the
                main view of the Set, with a 16:9 ratio.
            gpu: boolean
                use CUDA on GPU to compute the set
            ncycle: float
                number of iteration before cycling the colortable
            rgb_amp: [float, float, float]
                amplitude for each color channel, in [0,1]
            rgb_thetas: [float, float, float]
                phase in degrees for each color channel, in [0,360]
            global_phase: float
                global phase of the colormap
            oversampling: int
                for each pixel, a [n, n] grid is computed where n is the
                oversampling_size. Then, the average color of the n*n pixels
                is taken. Set to 1 for no oversampling.
            stripe_s:
                stripe density: frequency parameter of stripe average coloring.
                Set to 0 for no stripes.
            stripe_sig:
                memory parameter of stripe average coloring
            step_s:
                step density: frequency parameter of step coloring. Set to 0
                for no steps.
            light: [float, float, float, float, float, float, float]
                light vector: azimuth [0-1], angle [0-1],
                opacity [0,1], k_ambiant, k_diffuse, k_spectral, shininess
           
        """
        self.xpixels = xpixels
        self.maxiter = maxiter
        self.coord = coord
        self.gpu = gpu
        self.ncycle = ncycle
        self.os = oversampling
        self.rgb_amp = rgb_amp
        self.rgb_thetas = rgb_thetas
        self.global_phase = global_phase
        self.stripe_s = stripe_s
        self.stripe_sig = stripe_sig
        self.step_s = step_s
        # Light angles mapping
        self.light = np.array(light)
        self.light[0] = self.light[0]
        self.light[1] = self.light[1]
        
        #expoente
        self.expoente = expoente
        
        self.max_memo = max_memo
        
        #julia sets
        self.julia_c = julia_c     
        
        #rotation
        self.rot = rot
        self.center = 0.5*complex(coord[0] + coord[1], 
                                  coord[2] + coord[3])
        
        # Compute ypixels so the image is not stretched (1:1 ratio)
        self.ypixels = round(self.xpixels / (self.coord[1]-self.coord[0]) *
                             (self.coord[3]-self.coord[2]))
         # Compute the set
        self.update_set()        
    
    def set_aspect(self, ratio_x = 16, ratio_y = 9):
        """keep x, adjust y"""
        y_center = (self.coord[2] + self.coord[3])/2
        width = (self.coord[1] - self.coord[0])
        new_height = width*(ratio_y/ratio_x)
        self.coord[2] = y_center - (new_height/2)
        self.coord[3] = y_center + (new_height/2)
        
        # Compute ypixels so the image is not stretched (1:1 ratio)
        self.ypixels = round(self.xpixels / width * new_height)        
        self.update_set()            
        
    def update_set(self):
        """Updates the set
   
        Compute and color the Mandelbrot set, using CPU or GPU
        """
        #for rotatation
        self.center = 0.5*complex(self.coord[0] + self.coord[1], 
                                  self.coord[2] + self.coord[3])
        
        # Initialization of colortable
        self.colortable = sin_colortable(self.rgb_amp, self.rgb_thetas, self.global_phase)
        
        #update colortable_sample
        self.colortable_sample = sample_colortable(self.colortable, self.ncycle) 
        
        diag = math.sqrt((self.coord[1]-self.coord[0])**2 +
                  (self.coord[3]-self.coord[2])**2)
        
        z_memo = np.zeros(self.max_memo, dtype = np.complex128) 
        
        # Oversampling: rescaling by os
        xp = self.xpixels*self.os
        yp = self.ypixels*self.os
       
        if(self.gpu):
            # Pixel mapping is done in compute_self_gpu
            self.set = np.zeros((yp, xp, 3))
            # Compute set with GPU:
            # 1D grid, with n blocks of 32 threads
            npixels = xp * yp
            nthread = 32
            nblock = math.ceil(npixels / nthread)
            compute_set_gpu[nblock,
                            nthread](self.set, *self.coord, self.maxiter,
                                     self.colortable, self.ncycle, self.stripe_s,
                                     self.stripe_sig, self.step_s, diag,
                                     self.light, self.julia_c, self.center, 
                                     self.rot, self.expoente, z_memo)
        else:
            # Mapping pixels to C
            creal = np.linspace(self.coord[0], self.coord[1], xp)
            cim = np.linspace(self.coord[2], self.coord[3], yp)
            # Compute set with CPU
            self.set = compute_set(creal, cim, self.maxiter,
                                   self.colortable, self.ncycle, self.stripe_s,
                                   self.stripe_sig, self.step_s, diag,
                                   self.light, self.julia_c, self.center, 
                                   self.rot, self.expoente, z_memo)
        self.set = (255*self.set).astype(np.uint8)
        
        # Oversampling: reshaping to (ypixels, xpixels, 3)
        if self.os > 1:
            self.set = (self.set
                        .reshape((self.ypixels, self.os,
                                  self.xpixels, self.os, 3))
                        .mean(3).mean(1).astype(np.uint8))
   
    def draw(self, filename = None, signature = False):
        """Draw or save, using PIL"""
        # Reverse x-axis (equivalent to matplotlib's origin='lower')
        image_array = self.set[::-1,:,:]
        image_array = image_array.astype(np.uint8)
        img = Image.fromarray(image_array, 'RGB')
        
        if signature:
            #font_path = os.path.join('fonts','Sofia','Sofia-Regular.ttf')
            #font_path = os.path.join('fonts','Shalimar','Shalimar-Regular.ttf')
            font_path = os.path.join('fonts','ReggaeOne','ReggaeOne-Regular.ttf')
            font_size = int((self.xpixels + self.ypixels)/80)
            signature_font = ImageFont.truetype(font_path, font_size)
            #img.putalpha(255)
            image_editable = ImageDraw.Draw(img)
            x0, y0 = int(self.xpixels*0.77), int(self.ypixels*0.94)
            RGB = 200*np.ones(3)# - image_array[y0:,x0:,:].mean(axis = 0).mean(axis = 0)
            R,G,B = RGB.astype(int)
            image_editable.text((x0, y0), signature, (R,G,B) , font=signature_font)        
        
        if filename is not None:
            img.save(filename) # fast (save in jpg) (compare reading as well)
        else:
            img.show() # slow
        return img
           
    def draw_mpl(self, filename=None, dpi=72):
        """Draw or save, using Matplotlib"""
        fig, ax = plt.subplots(figsize=(self.xpixels/dpi, self.ypixels/dpi))
        plt.imshow(self.set, extent=self.coord, origin='lower')
        # Remove axis and margins
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        # Write figure to file
        if filename is not None:
            plt.savefig(filename, dpi=dpi)
        else:
            plt.show()
        return fig
       
    def zoom_at(self, x, y, s):
        """Zoom at (x,y): center at (x,y) and scale by s"""
        xrange = (self.coord[1] - self.coord[0])/2
        yrange = (self.coord[3] - self.coord[2])/2
        self.coord = [x - xrange * s,
                      x + xrange * s,
                      y - yrange * s,
                      y + yrange * s]
       
    def szoom_at(self, x, y, s):
        """Soft zoom (continuous) at (x,y): partial centering"""
        xrange = (self.coord[1] - self.coord[0])/2
        yrange = (self.coord[3] - self.coord[2])/2
        x = x * (1-s**2) + (self.coord[1] + self.coord[0])/2 * s**2
        y = y * (1-s**2) + (self.coord[3] + self.coord[2])/2 * s**2
        self.coord = [x - xrange * s,
                      x + xrange * s,
                      y - yrange * s,
                      y + yrange * s]      
          
    def animate_zoom(self, x, y, file_out, n_frames=150, loop=True):
        """Animated zoom to GIF file
   
        Note that the Mandelbrot object is modified by this function
       
        Args:
            x: float
                real part of point to zoom at
            y: float
                imaginary part of point to zoom at
            file_out: str
                filename to save the GIF output
            n_frames: int
                number of frames in the output file
            loop: boolean
                loop back to original coordinates
        """        
        # Zoom scale: gaussian shape, from 0% (s=1) to 30% (s=0.7)
        # => zoom scale (i.e. speed) is increasing, then decreasing
        def gaussian(n, sig = 1):
            x = np.linspace(-1, 1, n)
            return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))
        s = 1 - gaussian(n_frames, 1/2)*.3
       
        # Update in case it was not up to date (e.g. parameters changed)
        self.update_set()
        images = [self.set]
        # Making list of images
        for i in range(1, n_frames):
            # Zoom at (x,y)
            self.szoom_at(x,y,s[i])
            # Update the set
            self.update_set()
            images.append(self.set)
           
        # Go backward, one image in two (i.e. 2x speed)
        if(loop):
            images += images[::-2]
        # Make GIF
        imageio.mimsave(file_out, images)  
    
    def animate_param(self, dict_to_walk, file_out, fps = 30, reverse = False):
        
        # Update in case it was not up to date (e.g. parameters changed)
        self.update_set()
        
        #create a list o images
        images = list()
        
        #set a frame for each change
        n_frames = len(list(dict_to_walk.values())[0])   
        
        for key in dict_to_walk.keys():
            if len(dict_to_walk[key]) != n_frames:
                logger.error(f'tamanho de {key} em dict_to_walk diferente de {n_frames}')
        
        for ind in range(n_frames):           
            for key in dict_to_walk.keys():
                self.__setattr__(key,dict_to_walk[key][ind])
            self.update_set()
            images.append(self.set)
        
        if reverse:
            images += images[::-1]
            
        # Make GIF
        imageio.mimsave(file_out, images, fps = fps)              
            
    def explore(self, dpi=72):
        """Run the Mandelbrot explorer: a Matplotlib GUI"""
        # It is important to keep track of the object in a variable, so the
        # slider and button are responsive
        self.explorer = Mandelbrot_explorer(self, dpi)


class Mandelbrot_explorer():
    """A Matplotlib GUI to explore the Mandelbrot set"""
    def __init__(self, mand, dpi=72):
        self.mand = mand
        # Update in case it was not up to date (e.g. parameters changed)
        self.mand.update_set()
        # Plot the set
        perc_menu = 1.5
        figsize = (perc_menu*mand.xpixels/dpi, mand.ypixels/dpi)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.graph = plt.imshow(mand.set,
                                extent=mand.coord, origin='lower')
        plt.subplots_adjust(left= 0.95 - 1/perc_menu, right=0.99, 
                            bottom=0, top=1)
        plt.axis('off')
        
        #zoom and move scale
        self.zoom_scale = 1.5
        self.move_scale = 0.05
        
        #colortable_sample
        self.colortable_sample = plt.axes([0.1,0.92,0.15,0.04]).imshow(self.mand.colortable_sample,
                                                                     extent = [0,30,0,4])
        plt.axis('off')        
        
        ## Sliders
        slider_pos = [[0.1, 0.88 - x, 0.15, 0.02] for x in np.arange(0,0.88,0.02)]
        #========== maxiter ==========
        self.sld_expoente = Slider(plt.axes(slider_pos[0]),'expoente',
                                   1.25, 9, mand.expoente, valstep=0.25)
        self.sld_expoente.on_changed(self.update_val)
        
        self.sld_maxit = Slider(plt.axes(slider_pos[1]),'Iterations',
                                0, 5000, mand.maxiter, valstep=5)
        self.sld_maxit.on_changed(self.update_val)
        
        #=====global phase and ncycle========
        self.sld_n = Slider(plt.axes(slider_pos[3]), 'ncycle',
                            0, 200, mand.ncycle, valstep=1)
        self.sld_n.on_changed(self.update_val)
        self.sld_p = Slider(plt.axes(slider_pos[4]), 'global phase',
                            0, 360, 0, valstep=1)
        self.sld_p.on_changed(self.update_val)
        
        #========== rgb amplitudes =========
        self.sld_r_amp = Slider(plt.axes(slider_pos[5]), 'R amp',
                            0, 1, mand.rgb_amp[0], valstep=.01)
        self.sld_r_amp.on_changed(self.update_val)
        self.sld_g_amp = Slider(plt.axes(slider_pos[6]), 'G amp',
                            0, 1, mand.rgb_amp[1], valstep=.01)
        self.sld_g_amp.on_changed(self.update_val)
        self.sld_b_amp = Slider(plt.axes(slider_pos[7]), 'B amp',
                            0, 1, mand.rgb_amp[2], valstep=.01)
        self.sld_b_amp.on_changed(self.update_val)
        
        #=========== rgb phases ============
        self.sld_r = Slider(plt.axes(slider_pos[8]), 'R phase',
                            0, 360, mand.rgb_thetas[0], valstep=1)
        self.sld_r.on_changed(self.update_val)
        self.sld_g = Slider(plt.axes(slider_pos[9]), 'G phase',
                            0, 360, mand.rgb_thetas[1], valstep=1)
        self.sld_g.on_changed(self.update_val)
        self.sld_b = Slider(plt.axes(slider_pos[10]), 'B phase',
                            0, 360, mand.rgb_thetas[2], valstep=1)
        self.sld_b.on_changed(self.update_val)
       
        #========== paint modes=================
        paint_mode_position = 12
        self.sld_st = Slider(plt.axes(slider_pos[paint_mode_position]), 'step_s',
                             0, 30, mand.step_s, valstep=1)
        self.sld_st.on_changed(self.update_val)
        self.sld_s = Slider(plt.axes(slider_pos[paint_mode_position+1]), 'stripe_s',
                            0, 30, mand.stripe_s, valstep=1)
        self.sld_s.on_changed(self.update_val)
        self.sld_s2 = Slider(plt.axes(slider_pos[paint_mode_position+2]), 'stripe_sig',
                            0, 0.9, mand.stripe_sig, valstep=0.05)
        self.sld_s2.on_changed(self.update_val)
        
        #=========== light ============
        light_position = 16
        self.sld_li1 = Slider(plt.axes(slider_pos[light_position]), 'light_azimuth',
                              0, 360, mand.light[0], valstep=.01)
        self.sld_li1.on_changed(self.update_val)
        self.sld_li2 = Slider(plt.axes(slider_pos[light_position+1]), 'light_angle',
                              0, 90, mand.light[1], valstep=.01)
        self.sld_li2.on_changed(self.update_val)
        self.sld_li3 = Slider(plt.axes(slider_pos[light_position+2]), 'opacity',
                              0, 1, mand.light[2], valstep=.01)
        self.sld_li3.on_changed(self.update_val)
        self.sld_li4 = Slider(plt.axes(slider_pos[light_position+3]), 'k_ambiant',
                              0, 1, mand.light[3], valstep=.01)
        self.sld_li4.on_changed(self.update_val)
        self.sld_li5 = Slider(plt.axes(slider_pos[light_position+4]), 'k_diffuse',
                              0, 1, mand.light[4], valstep=.01)
        self.sld_li5.on_changed(self.update_val)
        self.sld_li6 = Slider(plt.axes(slider_pos[light_position+5]), 'k_specular',
                              0, 1, mand.light[5], valstep=.01)
        self.sld_li6.on_changed(self.update_val)
        self.sld_li7 = Slider(plt.axes(slider_pos[light_position+6]), 'shininess',
                              1, 100, mand.light[6], valstep=1)
        self.sld_li7.on_changed(self.update_val)
        
        #=========== zoom and move scale ============
        self.sld_li8 = Slider(plt.axes(slider_pos[23]), 'zoom scale',
                              1, 10,self.zoom_scale, valstep=0.1)
        self.sld_li9 = Slider(plt.axes(slider_pos[24]), 'move scale',
                              0,0.5, self.move_scale, valstep = 0.01)
        
        #========= rotation ==========
        self.sld_li10 = Slider(plt.axes(slider_pos[25]), 'rotation',
                               -180,180, mand.rot, valstep = 5)
        self.sld_li10.on_changed(self.update_val)
        
        #========== Zoom events =========
        plt.sca(self.ax)
        # Note that it is mandatory to keep track of those objects so they are
        # not deleted by Matplotlib, and callbacks can be used
        # Responsiveness for any click or scroll
        
        #self.cid1 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        #========= move events ==========
        self.kp = self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        
        plt.show()
       
    def update_val(self, val):
        """Slider interactivity: update object values"""
        rgb_thetas = [self.sld_r.val, self.sld_g.val, self.sld_b.val]
        global_phase = self.sld_p.val        
        rgb_amp = [self.sld_r_amp.val, self.sld_g_amp.val, self.sld_b_amp.val]
        
        self.mand.global_phase = global_phase
        self.mand.rgb_thetas = rgb_thetas
        self.mand.rgb_amp = rgb_amp
        #self.mand.colortable = sin_colortable(rgb_amp, rgb_thetas, global_phase)
        self.mand.expoente = self.sld_expoente.val
        self.mand.maxiter = self.sld_maxit.val
        self.mand.ncycle = self.sld_n.val
        self.mand.stripe_s = self.sld_s.val
        self.mand.stripe_sig = self.sld_s2.val
        self.mand.step_s = self.sld_st.val
        self.mand.light = np.array([self.sld_li1.val,
                           self.sld_li2.val, self.sld_li3.val,
                           self.sld_li4.val, self.sld_li5.val, 
                           self.sld_li6.val, self.sld_li7.val])
        
        
        # coord_rot0 = rotate_coord(self.mand.coord, -self.mand.rot)
        self.mand.rot = self.sld_li10.val
        # self.mand.coord = rotate_coord(coord_rot0,self.mand.rot)
        
        self.mand.update_set()
        
        self.graph.set_data(self.mand.set)
        self.colortable_sample.set_data(self.mand.colortable_sample)
        
        plt.draw()      
        plt.show()
       
    def onclick(self, event):
        """Event interactivity function"""
        
        # This function is called by any click/scroll
        self.zoom_scale = self.sld_li8.val
        if event.inaxes == self.ax:
            # Click or scroll in the main axe: zoom event
            # Default: zoom in
            zoom = 1/self.zoom_scale
            #zoom = 1/1.5
            if event.button in ('down', 3):
                # If right click or scroll down: zoom out
                zoom = 1/zoom
            # Zoom and update
            c_zoom = rotate_point(complex(event.xdata,event.ydata),
                                          self.mand.center,self.mand.rot)
            self.mand.zoom_at(c_zoom.real,c_zoom.imag, zoom)
            self.mand.update_set()
            # Updating the graph
            self.graph.set_data(self.mand.set)
            self.graph.set_extent(self.mand.coord)
            plt.draw()      
            plt.show()
    def onpress(self, event):
        # press arrow keys to move in the fractal
        coord = self.mand.coord
        self.move_scale = self.sld_li9.val
        
        width = coord[1] - coord[0]
        height = coord[3] - coord[2]
        if event.key == 'left':
            coord[0] = coord[0] - self.move_scale*width
            coord[1] = coord[1] - self.move_scale*width
        elif event.key == 'right':
            coord[0] = coord[0] + self.move_scale*width
            coord[1] = coord[1] + self.move_scale*width
        elif event.key == 'down':
            coord[2] = coord[2] - self.move_scale*height
            coord[3] = coord[3] - self.move_scale*height
        elif event.key == 'up':
            coord[2] = coord[2] + self.move_scale*height
            coord[3] = coord[3] + self.move_scale*height
        
        # Updating set
        self.mand.coord = coord
        self.mand.update_set()
        
        # Updating the graph
        self.graph.set_data(self.mand.set)
        self.graph.set_extent(self.mand.coord)
        plt.draw()      
        plt.show()