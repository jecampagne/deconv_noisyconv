import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
from tqdm import trange
import time
import regex as re
import glob
import argparse
import yaml
from types import SimpleNamespace



file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(os.path.basename(file))
    if not match:
        return math.inf
    return int(match.groups()[-1])


def get_list(dir, pattern):
    # dir = pathlib.Path(dir)
    a = list(glob.glob(dir + "/" + pattern))
    return sorted(a, key=get_order)



def make_psf(fwhm):
    """model de PSF"""
    # The FWHM of a Gaussian is 2 sqrt(2 ln2) sigma
    _fwhm_factor = 2.3548200450309493

    sigma = fwhm / _fwhm_factor
    _inv_sigsq = 1.0 / sigma**2
    W = np.linspace(-5 * sigma, 5 * sigma, int(10 * sigma), endpoint=True)
    H = np.linspace(-5 * sigma, 5 * sigma, int(10 * sigma), endpoint=True)

    X, Y = np.meshgrid(W, H)
    rsq = X**2 + Y**2

    y =  np.exp(-0.5 * rsq * _inv_sigsq)

    #normalization to 1
    y /= y.sum()

    return y

def rescale_image_range(im,  max_I, min_I):
    temp = (im - im.min())  /((im.max() - im.min()))
    return temp *(max_I-min_I) + min_I


class Simul:
    def __init__(self, seed=42,
                 img_H=128, img_W=128,
                 fwhm_min=15, fwhm_max=25,
                 sigma_min=0.0, sigma_max=0.5,
                 with_img_conv=False):

        #init random generator
        self.rng = np.random.default_rng(seed)

        # Height/Width final image
        self.img_H = img_H
        self.img_W = img_W

        # FWHM of PSF
        self.fwhm_min = fwhm_min
        self.fwhm_max = fwhm_max

        # sigma of noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get(self,img_clean, with_img_conv=False):
        """
        return:
        - img_noisy : final image
        - psf_fwhm: FWHM of the PSF
        - sigma of the noise
        - img_conv : convolved image (optional: with_img_conv) 
        """

        
        # convolution
        psf_fwhm = self.rng.uniform(low=self.fwhm_min,high=self.fwhm_max)
        psf = make_psf(fwhm=psf_fwhm)
        img_conv = signal.convolve2d(img_clean, psf, mode="same")

        # add noise
        sigma_noise =  self.rng.uniform(low=self.sigma_min,high=self.sigma_max)
        noise = sigma_noise * self.rng.normal(size=(self.img_H, self.img_W))
        img_noisy = img_conv + noise

        # return
        if with_img_conv:
            return img_noisy, img_conv,  psf_fwhm, sigma_noise
        else:
            return img_noisy, psf_fwhm, sigma_noise


################################
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description="Cnov and Noisy Calpha,beta images")
    parser.add_argument("--file", help="Config file")
    args0 = parser.parse_args()


    ## Load yaml configuration file
    with open(args0.file, 'r') as config:
        settings_dict = yaml.safe_load(config)
    args = SimpleNamespace(**settings_dict)



    input_dir  = args.input_root_dir + args.input_dataset
    output_root_dir = args.output_root_dir
    tag_dataset = args.output_dataset_tag

    train_dir = output_root_dir + tag_dataset + "/train/"
    test_dir = output_root_dir + tag_dataset + "/test/"
    val_dir = output_root_dir + tag_dataset + "/val/"    
    n_dirs  = [train_dir, test_dir, val_dir]
    
    data_names = ["train", "test", "val"]

    for dname in n_dirs:
        try:
            os.makedirs(
                dname, exist_ok=False
            )  # avoid erase the existing directories (exit_ok=False)
        except OSError:
            pass

    #clean image dataset
    data_clean = get_list(input_dir, "d*.npz")

    n_train = args.n_train
    n_test = args.n_test
    n_val = args.n_val
    assert n_train+n_test+n_val <= len(data_clean), "check sizes of train/test/val"

    fwhm_min = args.fwhm[0]
    fwhm_max = args.fwhm[1]
    sigma_min = args.sigma[0]
    sigma_max = args.sigma[1]
    seed = args.seed # JEC 24/5/25
    mysimu = Simul(seed=seed,
                   fwhm_min=fwhm_min, fwhm_max=fwhm_max,
                   sigma_min=sigma_min, sigma_max=sigma_max)
    

    # Go!
    t0 = time.time()
    for data_name in data_names:
        print("do ", data_name, "dataset")
        if data_name == "train":
            out_dir = train_dir
            Ndata = n_train
            idx_clean_start = 0     # included
            idx_clean_end = n_train # excluded
            with_img_conv = False
        elif data_name == "test":
            out_dir = test_dir
            Ndata = n_test
            idx_clean_start = n_train # included
            idx_clean_end = idx_clean_start + n_test # excluded
            with_img_conv = False
        else:
            out_dir = val_dir
            Ndata  = n_val
            idx_clean_start = n_train+n_test  # included
            idx_clean_end = idx_clean_start + n_val # excluded
            with_img_conv = True
            

        for i in range(idx_clean_start,idx_clean_end):
            if i%1000 == 0:
                print(data_name,"i=",i,"time=",time.time()-t0)

            #keep the index of the original clean image for tracability
            fout_name = "d_" + str(i) + ".npz"
            img_clean = np.load(data_clean[i])['img']
            # rescale pixel value in range
            img_clean =  rescale_image_range(img_clean, max_I=1.0, min_I=-1.0)


            out = mysimu.get(img_clean, with_img_conv)
            
            if with_img_conv:
                img_noisy, img_conv,  psf_fwhm, sigma_noise = out
                data = {
                    "img_noisy":img_noisy.astype(np.float32),
                    "img_clean":img_clean.astype(np.float32),
                    "img_conv":img_conv.astype(np.float32),
                    "psf_fwhm":psf_fwhm,
                    "sigma_noise":sigma_noise
                }
            else:
                img_noisy, psf_fwhm, sigma_noise = out
                data = {
                    "img_noisy":img_noisy.astype(np.float32),
                    "img_clean":img_clean.astype(np.float32),
                    "psf_fwhm":psf_fwhm,
                    "sigma_noise":sigma_noise
                }

            np.savez(out_dir + "/" + fout_name, **data)

    # end
    tf = time.time()
    print("all done!", tf - t0)


    
    
