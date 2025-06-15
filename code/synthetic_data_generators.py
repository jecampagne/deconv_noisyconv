import numpy as np
import matplotlib
import matplotlib.pylab as plt
import torch.nn as nn
import torch
import os
import random
 

######################################################################################
################################ data generators  ################################
######################################################################################


################ c alpha ################
def make_fft_filter_1d(N):
    filt = abs(torch.fft.fftshift(torch.fft.fftfreq(N)))
    filt = 1/filt/N
    i = torch.where(filt == torch.inf)[0]
    filt[i.item()] = filt[i.item()+1]

    return filt



def make_fft_filter_2d(N):
    filt = torch.fft.fftshift(torch.fft.fftfreq(N))
    X, Y = torch.meshgrid(filt, filt)
    filt2 = (X**2 + Y**2).sqrt()
    filt2 = 1/filt2/N
    i, j = torch.where(filt2 == torch.inf)
    filt2[i.item(), j.item()] = filt2[i.item()+1, j.item()]
    return filt2


def make_fft_filter_2d_seprable(N):  
    filt = torch.fft.fftshift(torch.fft.fftfreq(N))
    i = torch.where(filt == 0)[0]
    filt[i.item()] = filt[i.item()+1]
    filt2 = abs(torch.matmul(filt.reshape(-1,1), filt.reshape(1,-1))).sqrt()
    filt2 = 1/filt2/N
    return filt2 

def make_C_alpha_contour(alpha, filt, N = 43): 
    ders = torch.rand(size = (N,), dtype=torch.float32)*2-1
    integrated = torch.fft.ifft(torch.fft.ifftshift( ( torch.fft.fftshift(torch.fft.fft(ders)) *( filt** (alpha))) )).real
    return integrated


def make_C_beta_background(beta, filt2, N = 43): 
    ders2 = torch.rand(size = (N,N), dtype=torch.float32)*2-1
    integrated =torch.fft.ifft2(torch.fft.ifftshift( ( torch.fft.fftshift(torch.fft.fft2(ders2)) * (filt2**(beta))) )).real 
    return integrated
 
    

def make_C_alpha_images(alpha, beta, separable=False, im_size=43,
                        num_samples=1, constant_background=False,
                        factor=(1,1)):
    '''
    im_size: image size. 
    num_samples: number of images
    for vertical blurring: (factor, 1)
    '''
    all_im = []

    filt = make_fft_filter_1d(im_size)
    
    if separable:
        filt2 = make_fft_filter_2d_seprable(im_size)
    else: 
        filt2 = make_fft_filter_2d(im_size)

    ave_filt = nn.Conv2d(1,1,(factor[0],factor[1]), stride = (1,1), padding = (int(factor[0]/2),int(factor[1]/2)) , padding_mode='reflect', bias = None)
    ave_filt.weight = torch.nn.Parameter(torch.ones((1,1,factor[0],factor[1]))/(factor[0]* factor[1]))
    while len(all_im)<num_samples:
        contour = make_C_alpha_contour(alpha ,filt, im_size )
        contour = torch.round(contour*im_size).type(torch.int) + int((im_size)/2)    
        
        if constant_background: 
            background1 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))
            background2 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))

        else: 
            background1 = make_C_beta_background(beta,filt2, im_size)
            background2 = make_C_beta_background(beta,filt2, im_size)        

            ## rescale background to create an intensity difference between the two parts 
            # thresh = torch.rand(1).item() # change to this to enfornce a larger gap between the mean of the two backgrounds 
            if torch.randint(low=0, high=2, size=(1,)).item() == 0:
                background1 = rescale_image_range(background1,max_I=1, min_I=torch.rand(1).item())
                background2 = rescale_image_range(background2,max_I=torch.rand(1).item(), min_I=0)
            else:
                background2 = rescale_image_range(background2,max_I=1, min_I=torch.rand(1).item())
                background1 = rescale_image_range(background1,max_I=torch.rand(1).item(), min_I=0)
        #### mask and replace         
        mask = torch.ones((im_size , im_size))
        for i in range(im_size):
            mask[0:contour[i],i]=0
        if factor[0] >1 or factor[1]>1:
            mask_down = ave_filt(mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
            if factor[0]%2==0: 
                mask_down = mask_down[0:-1, :]
            if factor[1]%2==0: 
                mask_down = mask_down[:,0:-1]
                
            im = background1 * mask_down + background2 * (1-mask_down) 
        else: 
            im = background1 * mask + background2 * (1-mask) 

        im = rescale_image_range(im,max_I=1.0,min_I=-1.0)
        all_im.append(im)

    all_im = torch.stack(all_im).unsqueeze(1) 

    
    return all_im 

def rescale_image_range(im,  max_I, min_I=0):

    temp = (im - im.min())  /((im.max() - im.min()))
    return temp *(max_I-min_I) + min_I


################################## original code ###############################
def make_C_alpha_images_original(alpha, beta, separable=False, im_size=43,num_samples=1, constant_background=False, factor=(1,1) , antialiasing=0, wavelet="db2", mode="reflect"):
    '''
    im_size: image size. 
    num_samples: number of images
    for vertical blurring: (factor, 1)
    '''
    all_im = []


    if antialiasing > 0: 
        im_size = im_size * (2**antialiasing) - 2 ** antialiasing

    filt = make_fft_filter_1d(im_size)
    
    if separable:
        filt2 = make_fft_filter_2d_seprable(im_size)
    else: 
        filt2 = make_fft_filter_2d(im_size)

    ave_filt = nn.Conv2d(1,1,(factor[0],factor[1]), stride = (1,1), padding = (int(factor[0]/2),int(factor[1]/2)) , padding_mode='reflect', bias = None)
    ave_filt.weight = torch.nn.Parameter(torch.ones((1,1,factor[0],factor[1]))/(factor[0]* factor[1]))
    while len(all_im)<num_samples:
        contour = make_C_alpha_contour(alpha ,filt, im_size )
        contour = torch.round(contour*im_size).type(torch.int) + int((im_size)/2)    
        
        if constant_background: 
            background1 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))
            background2 = np.random.rand(1)[0] * torch.ones(size = (im_size,im_size))

        else: 
            background1 = make_C_beta_background(beta,filt2, im_size)
            background2 = make_C_beta_background(beta,filt2, im_size)        

            ## rescale background to create an intensity difference between the two parts 
            # thresh = torch.rand(1).item() # change to this to enfornce a larger gap between the mean of the two backgrounds 
            if torch.randint(low=0, high=2, size=(1,)).item() == 0:
                background1 = rescale_image_range(background1,max_I=1, min_I=torch.rand(1).item())
                background2 = rescale_image_range(background2,max_I=torch.rand(1).item(), min_I=0)
            else:
                background2 = rescale_image_range(background2,max_I=1, min_I=torch.rand(1).item())
                background1 = rescale_image_range(background1,max_I=torch.rand(1).item(), min_I=0)
        #### mask and replace         
        mask = torch.ones((im_size , im_size))
        for i in range(im_size):
            mask[0:contour[i],i]=0
        if factor[0] >1 or factor[1]>1:
            mask_down = ave_filt(mask.unsqueeze(0).unsqueeze(0)).squeeze().detach()
            if factor[0]%2==0: 
                mask_down = mask_down[0:-1, :]
            if factor[1]%2==0: 
                mask_down = mask_down[:,0:-1]
                
            im = background1 * mask_down + background2 * (1-mask_down) 
        else: 
            im = background1 * mask + background2 * (1-mask) 
            
        all_im.append(im)

    all_im = torch.stack(all_im).unsqueeze(1) 
    all_im = downsample(all_im.detach(), num_times=antialiasing, wavelet=wavelet, mode=mode)# (N, 1, H/2^j, W/2^j)
    if antialiasing >0: 
        all_im = rescale_image_range(all_im, max_I=1, min_I=0)
    return all_im 



def downsample(x, num_times, wavelet="db2", mode="periodization"):
    """ Downsample an (*, H, W) image `num_times` times using the given wavelet filter. """
    transform = OneLevelWaveletTransform(wavelet=wavelet, mode=mode)

    for _ in range(num_times):
        x = transform.decompose(x)[..., 0, :, :]  # (*, H/2^j, W/2^j)

    return x




