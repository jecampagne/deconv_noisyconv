import os
import numpy as np
import cv2
import time
import argparse
import yaml
from types import SimpleNamespace



def make_fft_filter_1d(N):
    filt = abs(np.fft.fftshift(np.fft.fftfreq(N)))
    i = np.where(filt == 0.)[0]
    filt[i.item()] = filt[i.item()+1]
    filt = 1/filt/N

    return filt
def make_fft_filter_2d(N):
    filt = np.fft.fftshift(np.fft.fftfreq(N))
    X, Y = np.meshgrid(filt, filt)
    filt2 = np.sqrt(X**2 + Y**2)
    i, j = np.where(filt2 == 0.)
    filt2[i.item(), j.item()] = filt2[i.item()+1, j.item()]    
    filt2 = 1/filt2/N
    return filt2

def make_C_alpha_contour_1d(rng, alpha, filt, N = 43):
    ders = rng.uniform(size = (N,))*2-1
    integrated = np.fft.ifft(np.fft.ifftshift( ( np.fft.fftshift(np.fft.fft(ders)) *( filt** (alpha))) )).real
    return integrated


def make_C_beta_background(rng, beta, filt2, N = 43):
    ders2 = rng.uniform(size = (N,N))*2-1
    integrated =np.fft.ifft2(np.fft.ifftshift( ( np.fft.fftshift(np.fft.fft2(ders2)) * (filt2**(beta))) )).real
    return integrated

def transform(pt,pt0,pt1,ptp0,ptp1):
  u = pt1-pt0
  a,b = u[0],u[1]
  v = ptp1-ptp0
  c,d = v[0],v[1]
  x,y = pt[0],pt[1]
  den = np.sqrt((a**2+b**2)*(c**2+d**2))
  c1 = a*c+b*d
  c2 = (b*c-a*d)
  xp = c1*x + c2*y
  yp = -c2*x + c1*y
  return np.array([xp,yp])/den

def norm(u):
  return np.sqrt(u[0]**2+u[1]**2)

def make_C_alpha_mask_2d(rng, alpha,image_size, Nc=3, phi0=0., rho=1.0):
    """ Produce a closed contour with C^alpha regularity except at nodes
    rng: random generator
    alpha: regularity factor
    Ns: number of sampling points of the 1D contour
    Nc: number of nodes and edges of the polygon
    phi0: rotation angle of the polygon
    rh0: downsizing factor of the polygon
    """
    rho = np.atleast_1d(rho)
    if len(rho) == 1: # symmetric downsizing
      rho = np.repeat(rho,Nc)
    assert len(rho) == Nc, "make sure to give ax many rhos as Nc or choose a single value"
    
    gamma = 2*np.pi/Nc
    Ns = 10*image_size

    #define a C_alpha 1d contour
    filt = make_fft_filter_1d(Ns)
    xts = make_C_alpha_contour_1d(rng, alpha, filt, N = Ns)
    xts[-1]=xts[0] # garanty the periodicity
    xts -= xts[0]  # garaanty end points (0,0) & (1,0)
    xts = 0.1*xts/np.max(np.abs(xts))
    ts = np.linspace(0,1,Ns)

    #make regular polygonal nodes
    ptps = np.zeros((Nc+1,2))
    for i in range(Nc):
      ptps[i] = rho[i] * np.array([np.cos(i*gamma + phi0),np.sin(i*gamma + phi0)])
    ptps[-1] = ptps[0] 

    # transfom the C_alpha contour as edges of the regular polygone -> 2d C_alpha contour
    ptsInit = np.zeros((Ns,2))
    ptsInit[:,0]=ts
    ptsInit[:,1]=xts

    pt0 = ptsInit[0]
    pt1 = ptsInit[-1]

    s10 = norm(pt1-pt0)
    
    ptsOut = np.zeros((Nc,Ns,2))

    for j in range(Nc):
      for i in range(Ns):
        ptp0= ptps[j]
        ptp1= ptps[j+1]
        ptsOut[j,i]=norm(ptp1-ptp0)/s10 * transform(ptsInit[i],pt0,pt1,ptp0,ptp1) + ptps[j]
    ptsOut = ptsOut.reshape((-1,2))

    # contour binarization
    H, xedges, yedges = np.histogram2d(ptsOut[:,1], ptsOut[:,0], bins=image_size, range=[[-1.2,1.2],[-1.2,1.2]])
    H[H>0]=255
    H = H.astype(np.uint8)

    # make a small dilation of the contour to easy the filling procedure (Flood-fil)
    _, binary = cv2.threshold(H, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    inverted = cv2.bitwise_not(dilated)
    # Flood-fill from external point (0,0)
    h, w = inverted.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood_filled = inverted.copy()
    cv2.floodFill(flood_filled, mask, (0, 0), 255)
    # Invert the result to get the inside
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    #Combine inside and contour
    final_result = cv2.bitwise_or(dilated, flood_filled_inv)

    # make the final mask basde on the inside (outside) of the 2d C_alpha contour: 1->inside, 0->outside
    contours, hierarchy = cv2.findContours(final_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(H.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contours[0]], -1,1, thickness=cv2.FILLED)

    return mask


def make_image(rng, alpha, beta_in, beta_out, seed, image_size, Nc, phi0, rho):

    filt2 = make_fft_filter_2d(image_size)

    background1 = make_C_beta_background(rng,beta_out,filt2, image_size)
    background2 = make_C_beta_background(rng,beta_in,filt2, image_size)

    mask = make_C_alpha_mask_2d(rng, alpha, image_size, Nc, phi0, rho)
    
    return (1-mask) * background1 + mask * background2


  
################################
if __name__ == '__main__':

    root_dir = "/lustre/fsn1/projects/rech/ixh/ufd72rp"
    #root_dir  = "/lustre/fswork/projects/rech/ixh/ufd72rp/datasets"
    
    tag_dataset = "dataset_calphabeta"

    out_dir = root_dir + "/" + tag_dataset + "/"
    
    try:
        os.makedirs(
            out_dir, exist_ok=False
        )  # avoid erase the existing directories (exit_ok=False)
    except OSError:
        pass


    parser = argparse.ArgumentParser(description="Calpha,beta images")
    parser.add_argument("--file", help="Config file")
    args0 = parser.parse_args()


    ## Load yaml configuration file
    with open(args0.file, 'r') as config:
        settings_dict = yaml.safe_load(config)
    args = SimpleNamespace(**settings_dict)

    
    N_data = args.n_data  # 100_000 train/ 10_000 test / 10_000 valid
    image_size = args.image_size   # H=W
    Nc = args.n_edges # number of discontinuities in the contour
    # regularity ranges: contour: alpha,  background: beta
    alpha_min = args.alpha[0]
    alpha_max = args.alpha[1]
    assert alpha_max >= alpha_min
    assert alpha_min>0.0
    betai_min  = args.beta_in[0]
    betai_max  = args.beta_in[1]
    assert betai_max >= betai_min
    assert betai_min>0.0
    betao_min  = args.beta_out[0]
    betao_max  = args.beta_out[1]
    assert betao_max >= betao_min
    assert betao_min>0.0

    #global rotation range
    phi0_min = args.phi[0]
    phi0_max = np.clip(args.phi[1], 0., np.pi)
    assert phi0_max >= phi0_min
    assert phi0_min>=0.0
    
    #asymetric contraction
    rho_min = np.clip(args.phi[0], 0.2, 1.0)
    rho_max = np.clip(args.phi[1], 0.2, 1.0)
    assert rho_max >= rho_min
    
    seed = 117052025  #xDDMMAAAA

    
    print("settings: ", {
        "N_data":N_data,
        "image_size": image_size,
        "Nc":Nc,
        "alpha":[alpha_min, alpha_max],
        "beta_in":[betai_min,betai_max],
        "beta_out":[betao_min,betao_max],
        "phi":[phi0_min, phi0_max],
        "rho":[rho_min,rho_max],
        "seed":seed
    })

    rng = np.random.default_rng(seed)

    t0 = time.time()
    tprev=t0
    for i in range(N_data):
        if i%1000 == 0:
            print("i:",i,"time=",time.time()-tprev)
        
        alpha = rng.uniform(low=alpha_min, high=alpha_max)
        beta_in  = rng.uniform(low=betai_min, high=betai_max)
        beta_out  = rng.uniform(low=betao_min, high=betao_max)
        phi0  = rng.uniform(low=phi0_min, high=phi0_max)
        rho   = rng.uniform(low=rho_min, high=rho_max, size=(Nc,))
        
        img= make_image(rng, alpha, beta_in, beta_out, seed, image_size, Nc, phi0, rho)

        data = {
            "img": img,
            "alpha":alpha,
            "betai":beta_in,
            "betao":beta_out,
            }

        f_name = "d_" + str(i) + ".npz"
        np.savez(out_dir + "/" + f_name, **data)
        tprev = time.time()
    # end
    tf = time.time()
    print("all done!", tf - t0)
