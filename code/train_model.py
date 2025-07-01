import argparse
import yaml
import json
import os
import random
import pathlib
import pickle
from types import SimpleNamespace
import multiprocessing
import time
import regex as re
import glob

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift

import numpy as np

import galsim
from model import *


#################
# Training model for COSMOS galaxy with LSST features dataset
################


################
# Utils
################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)




class MultiScaleLoss(nn.Module):
	def __init__(self, scales=3, norm='L1'):
		super(MultiScaleLoss, self).__init__()
		self.scales = scales
		if norm == 'L1':
			self.loss = nn.L1Loss()
		if norm == 'L2':
			self.loss = nn.MSELoss()

		self.weights = torch.FloatTensor([1/(2**scale) for scale in range(self.scales)])
		self.multiscales = [nn.AvgPool2d(2**scale, 2**scale) for scale in range(self.scales)]

	def forward(self, output, target):
		loss = 0
		for i in range(self.scales):
			output_i, target_i = self.multiscales[i](output), self.multiscales[i](target)
			loss += self.weights[i]*self.loss(output_i, target_i)
		return loss


##############
# Galaxy & Image utils
##############


def get_flux(ab_magnitude, exp_time, zero_point, gain, qe):
    """Calculate flux (ADU/arcsec^2) from magnitude.

    Args:
        ab_magnitude (`float`): Absolute magnitude.
        exp_time (`float`): Exposure time (s).
        zero_point (`float`): Instrumental zero point, i.e. absolute magnitude that would produce one e- per second.
        gain (`float`): Gain (e-/ADU) of the CCD.
        qe (`float`): Quantum efficiency of CCD.

    Returns:
        `float`: (Flux ADU/arcsec^2).
    """
    return exp_time * zero_point * 10 ** (-0.4 * (ab_magnitude - 24)) * qe / gain


def down_sample(input, rate=4):
    """Downsample the input image with a factor of 4 using an average filter.

    Args:
        input (`torch.Tensor`): The input image with shape `[H, W]`.
        rate (`int`, optional): Downsampling rate. Defaults to `4`.

    Returns:
        `torch.Tensor`: The downsampled image.
    """
    weight = torch.ones([1, 1, rate, rate]) / (rate**2)  # Average filter.
    input = input.unsqueeze(0).unsqueeze(0)
    output = F.conv2d(input=input, weight=weight, stride=rate).squeeze(0).squeeze(0)

    return output


def get_LSST_PSF(
    lam_over_diam,
    opt_defocus,
    opt_c1,
    opt_c2,
    opt_a1,
    opt_a2,
    opt_obscuration,
    atmos_fwhm,
    atmos_e,
    atmos_beta,
    spher,
    trefoil1,
    trefoil2,
    g1_err=0,
    g2_err=0,
    fov_pixels=48,
    pixel_scale=0.2,
    upsample=4,
):
    """Simulate a PSF from a ground-based observation (typically LSST). The PSF consists of an optical component and an atmospheric component.

    Args:
        lam_over_diam (float): Wavelength over diameter of the telescope.
        opt_defocus (float): Defocus in units of incident light wavelength.
        opt_c1 (float): Coma along y in units of incident light wavelength.
        opt_c2 (float): Coma along x in units of incident light wavelength.
        opt_a1 (float): Astigmatism (like e2) in units of incident light wavelength.
        opt_a2 (float): Astigmatism (like e1) in units of incident light wavelength.
        opt_obscuration (float): Linear dimension of central obscuration as fraction of pupil linear dimension, [0., 1.).
        atmos_fwhm (float): The full width at half maximum of the Kolmogorov function for atmospheric PSF.
        atmos_e (float): Ellipticity of the shear to apply to the atmospheric component.
        atmos_beta (float): Position angle (in radians) of the shear to apply to the atmospheric component, twice the phase of a complex valued shear.
        spher (float): Spherical aberration in units of incident light wavelength.
        trefoil1 (float): Trefoil along y axis in units of incident light wavelength.
        trefoil2 (float): Trefoil along x axis in units of incident light wavelength.
        g1_err (float, optional): The first component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        g2_err (float, optional): The second component of extra shear applied to the overall PSF to simulated a erroneously estimated PSF. Defaults to `0`.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for the PSF image. Defaults to `4`.

    Returns:
        `torch.Tensor`: Simulated PSF image with shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """

    # Atmospheric PSF
    atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1)
    atmos = atmos.shear(e=atmos_e, beta=atmos_beta * galsim.radians)

    # Optical PSF
    optics = galsim.OpticalPSF(
        lam_over_diam,
        defocus=opt_defocus,
        coma1=opt_c1,
        coma2=opt_c2,
        astig1=opt_a1,
        astig2=opt_a2,
        spher=spher,
        trefoil1=trefoil1,
        trefoil2=trefoil2,
        obscuration=opt_obscuration,
        flux=1,
    )

    # Convolve the two components.
    psf = galsim.Convolve([atmos, optics])

    # Shear the overall PSF to simulate a erroneously estimated PSF when necessary.
    psf = psf.shear(g1=g1_err, g2=g2_err)

    # Draw PSF images.
    psf_image = galsim.ImageF(fov_pixels * upsample, fov_pixels * upsample)
    psf.drawImage(psf_image, scale=pixel_scale / upsample, method="auto")
    psf_image = torch.from_numpy(psf_image.array)

    return psf_image


def get_COSMOS_Galaxy(
    gal_orig,
    psf_hst,
    gal_g,
    gal_beta,
    gal_mu,
    dx,
    dy,
    fov_pixels=48,
    pixel_scale=0.2,
    upsample=4,
    theta=0.0,
):
    """Simulate a background galaxy with data from COSMOS Catalog.

    Args:
        gal_orig: original COSMOS galaxy (type galsim.real.RealGalaxy)
        psf_hst: associated HST PSF to reconvolved the gal_orig (type  galsim.interpolatedimage.InterpolatedImage)
        gal_g (float): The shear to apply.
        gal_beta (float): Position angle (in radians) of the shear to apply, twice the phase of a complex valued shear.
        gal_mu (float): The lensing magnification to apply.
        fov_pixels (int, optional): Width of the simulated images in pixels. Defaults to `48`.
        pixel_scale (float, optional): Pixel scale of the simulated image determining the resolution. Defaults to `0.2`.
        upsample (int, optional): Upsampling factor for galaxy image. Defaults to `4`.
        theta (float): Rotation angle of the galaxy (in radians, positive means anticlockwise). Defaults 0].

    Returns:
        `torch.Tensor`: Simulated galaxy image of shape `(fov_pixels*upsample, fov_pixels*upsample)`.
    """
    # Add random rotation, shear, and magnification.
    gal = gal_orig.rotate(theta * galsim.radians)  # Rotate by a random angle
    gal = gal.shear(g=gal_g, beta=gal_beta * galsim.radians)  # Apply the desired shear
    gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.

    # Draw galaxy image.
    gal_image = galsim.ImageF(fov_pixels * upsample, fov_pixels * upsample)
    gal = galsim.Convolve([psf_hst, gal])  # Concolve wth original PSF of HST.
    gal.drawImage(
        gal_image, scale=pixel_scale / upsample, offset=(dx, dy), method="auto"
    )

    gal_image = torch.from_numpy(gal_image.array)  # Convert to PyTorch.Tensor.
    gal_image = torch.max(gal_image, torch.zeros_like(gal_image))

    return gal_image


###########
# dataset utils
###########

file_pattern = re.compile(r".*?(\d+).*?")


def get_order(file):
    match = file_pattern.match(os.path.basename(file))
    if not match:
        return math.inf
    return int(match.groups()[-1])


def get_list(loc, pattern):
    # dir = pathlib.Path(dir)
    a = list(glob.glob(loc + "/" + pattern))
    return sorted(a, key=get_order)

def generator(args, gal_orig, psf_hst):
    """
    args: settings
    gal_orig: original Real Galaxy extracted from catalog
    psf_hst: HST PSF associated to gal_orig
    """
    #print("generator:",args)

    # Atmospheric PSF
    atmos_fwhm = 0.6 # args.rng_fwhm() # Atmospheric seeing (arcsec), the FWHM of the Kolmogorov function.
    atmos_e = 0.02  #0.01 + 0.02 * args.rng() # Ellipticity of atmospheric PSF (magnitude of the shear in the “distortion” definition), U(0.01, 0.03).
    atmos_beta = 0.0 # 2.0 * np.pi * args.rng()  # Shear position angle (radians), N(0,2*pi).

    # Optical PSF
    opt_defocus = 0.36 #args.rng_defocus()  # Defocus (wavelength), N(0.0.36).
    opt_a1 = 0.01 # args.rng_gaussian()  # Astigmatism (like e2) (wavelength), N(0.0.07).
    opt_a2 = -0.01 # args.rng_gaussian()  # Astigmatism (like e1) (wavelength), N(0.0.07).
    opt_c1 = 0.02  #args.rng_gaussian()  # Coma along y axis (wavelength), N(0.0.07).
    opt_c2 = -0.01 # args.rng_gaussian()  # Coma along x axis (wavelength), N(0.0.07).
    spher = 0.01 # args.rng_gaussian()  # Spherical aberration (wavelength), N(0.0.07).
    trefoil1 = 0.1 #args.rng_gaussian()  # Trefoil along y axis (wavelength), N(0.0.07).
    trefoil2 = -0.1 # args.rng_gaussian()  # Trefoil along x axis (wavelength), N(0.0.07).
    opt_obscuration = 0.3 #0.1 + 0.4 * args.rng() # Linear dimension of central obscuration as fraction of pupil linear dimension, U(0.1, 0.5).
    lam_over_diam = 0.019 # 0.017 + 0.007 * args.rng() # Wavelength over diameter (arcsec), U(0.017, 0.024).

    psf_image = get_LSST_PSF(
        lam_over_diam,
        opt_defocus,
        opt_c1,
        opt_c2,
        opt_a1,
        opt_a2,
        opt_obscuration,
        atmos_fwhm,
        atmos_e,
        atmos_beta,
        spher,
        trefoil1,
        trefoil2,
        0,
        0,
        args.fov_pixels,
        args.pixel_scale,
        args.upsample,
    )

    # Galaxy parameters .
    gal_g = 0.01 # args.rng_gal_shear() # Shear of the galaxy (magnitude of the shear in the "reduced shear" definition), U(0.01, 0.05).
    gal_beta = np.pi # 2.0 * np.pi * args.rng()  # Shear position angle (radians), N(0,2*pi).
    gal_mu = 1.01 # 1 + args.rng() * 0.1  # Magnification, U(1.,1.1).
    theta = 0. # 2.0 * np.pi * args.rng()  # Rotation angle (radians), U(0,2*pi).
    dx = 0. # 2 * args.rng() - 1  # Offset along x axis, U(-1,1).
    dy = 0. # 2 * args.rng() - 1  # Offset along y axis, U(-1,1).
    gal_image = get_COSMOS_Galaxy(
        gal_orig,
        psf_hst,
        gal_g=gal_g,
        gal_beta=gal_beta,
        theta=theta,
        gal_mu=gal_mu,
        dx=dx,
        dy=dy,
        fov_pixels=args.fov_pixels,
        pixel_scale=args.pixel_scale,
        upsample=args.upsample,
    )

    # Scaling by SNR
    snr = 100. # args.rng_snr()
    gal_image_down = down_sample(
        gal_image.clone(), args.upsample
    )  # Downsample galaxy image for SNR calculation.
    alpha = snr * args.sigma / torch.sqrt((gal_image_down**2).sum()) # Scale the flux of galaxy to meet SNR.
    gt = alpha * gal_image  # ground truth image

    # Convolution with the new PSF
    conv = ifftshift(ifft2(fft2(psf_image.clone()) * fft2(gt.clone()))).real

    # Downsample images to desired pixel scale.
    conv = down_sample(conv.clone(), args.upsample)
    psf = down_sample(psf_image.clone(), args.upsample)
    gt = down_sample(gt.clone(), args.upsample)

    # Add CCD noise
    conv = torch.max(torch.zeros_like(conv), conv)  # Set negative pixels to zero.
    obs = conv + torch.normal(
        mean=torch.zeros_like(conv), std=args.sigma * torch.ones_like(conv)
    )  # observed imave

    return gt, obs


class GalaxyDataset(Dataset):

    def __init__(
        self,
        settings,
        all_gal,
        all_psf,
        all_noise,
        all_info,
        sequence
    ):
        """
        settings: run settings
        all_gal: list of galaxy FITS files
        all_psf: list of HST PSF files
        all_noise: list of noise files 
        all_info: list of additional information files        
        sequence: indexes of galaxies in the dataset
        """
        self.settings = settings
        #
        self.all_gal   = all_gal
        self.all_psf   = all_psf
        self.all_noise = all_noise
        self.all_info  = all_info
        #
        n_gal = len(self.all_gal)
        self.seq = sequence
        
        assert n_gal >= len(self.seq), "pb n_gal < sequence length" 

        print("GalaxyDataset: size",len(self.seq),' among ',n_gal,'galaxies')

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        """
        Build ground truth and observation from HST galaxy
        idx : index < len(self.seq)
        """
        
        i = self.seq[idx]

        #t0 = time.time()
        #print(f"Get gal {idx}/{i}")

        # Read out real galaxy from the catalog and the correspondig HST PSF
        gal_orig = galsim.fits.read(self.all_gal[i]) # original galaxy image
        psf      = galsim.fits.read(self.all_psf[i]) # original HST PSF
        noise    = galsim.fits.read(self.all_noise[i]) # original noise image
        with open(self.all_info[i], "r") as f:
            info = json.load(f)
            pixel_scale = info["pixel_scale"]  # original pixel_scale
            var = info['var']                  # original noise variance
            
        #t1 = time.time()
        #print(f"time to load {idx}",t1-t0)

        # Genetare a couple of ground truth and obersation with new PSF and snr
        psf.array = psf.array/psf.array.sum() # adjust flux to 1.0 for HST PSF
        psf_hst   = galsim.InterpolatedImage(psf)
        gal_rg    = galsim.RealGalaxy((gal_orig,psf,noise,pixel_scale,var))
        
        gt, obs = generator(self.settings, gal_rg, psf_hst)

        #t2 = time.time()
        #print(f"time to generate {idx}",t2-t1)
        
        
        # transform to CHW with C=1
        gt  = gt.unsqueeze(0)
        obs = obs.unsqueeze(0)
        
        return gt, obs


################
# train/test 1-epoch
################
def train(args, model, criterion, train_loader, transforms, optimizer, epoch):

    # train mode
    model.train()

    loss_sum = 0  # to get the mean loss over the dataset
    print("Train.... start")

    t0 = time.time()
    for i_batch, imgs in enumerate(train_loader):

        #if i_batch==0 or i_batch%10 ==0:
        #    print("train batch:",i_batch,"...start at",time.time()-t0)
        
        gt, obs = imgs # ground truth and observaton
        
        gt = gt.to(args.device)
        obs = obs.to(args.device)

        # train step
        optimizer.zero_grad()
        output = model(obs)

        loss = criterion(output, gt)
        loss_sum += loss.item()
        # backprop to compute the gradients
        loss.backward()
        # perform an optimizer step to modify the weights
        optimizer.step()

    print(f"train epoch ...stop= {time.time()-t0:.2f}")

    return loss_sum / (i_batch + 1)


def test(args, model, criterion, test_loader, transforms, epoch):

    # test mode
    model.eval()

    t0 = time.time()

    loss_sum = 0  # to get the mean loss over the dataset
    with torch.no_grad():
        for i_batch, imgs in enumerate(test_loader):

            #if i_batch==0 or i_batch%10 ==0:
            #    print("test batch:",i_batch,"...start at",time.time()-t0)

            gt, obs = imgs # ground truth and observaton
        
            gt = gt.to(args.device)
            obs = obs.to(args.device)

            output = model(obs)

            loss = criterion(output, gt)
            loss_sum += loss.item()

    print(f"test epoch ...stop= {time.time()-t0:.2f}")
    return loss_sum / (i_batch + 1)


################
# Main: init & loop on epochs
################


def main_dev():
    print("#############################")
    print("####### DEV Training ########")
    print("#############################")
    t0 = time.time()

    # Training config
    parser = argparse.ArgumentParser(
        description="Deconvolve varibale noisy and convolved images"
    )
    parser.add_argument("--file", help="Config file")
    args0 = parser.parse_args()

    ## Load yaml configuration file
    with open(args0.file, "r") as config:
        settings_dict = yaml.safe_load(config)
    args = SimpleNamespace(**settings_dict)

    # check number of num_workers
    NUM_CORES = multiprocessing.cpu_count()
    if args.num_workers >= NUM_CORES:
        print("Info: # workers set to", NUM_CORES // 2)
        args.num_workers = NUM_CORES // 2

    # where to put all model training stuff
    args.out_root_dir = args.out_root_dir + "/" + args.run_tag + "/"

    try:
        os.makedirs(
            args.out_root_dir, exist_ok=False
        )  # avoid erase the existing directories (exit_ok=False)
    except OSError:
        pass

    print("Info: outdir is ", args.out_root_dir)

    # device cpu/gpu...
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # seeding random generator others than Galsim
    set_seed(args.seed)

    # update settings for galaxy simulator (used in GalaxyDataset)
    args.sky_level_pixel = (
        get_flux(
            ab_magnitude=args.sky_brightness,
            exp_time=args.exp_time,
            zero_point=args.zero_point,
            gain=args.gain,
            qe=args.qe,
        )
        * args.pixel_scale**2
    )  # Sky level (ADU/pixel).
    args.sigma = np.sqrt(
        args.sky_level_pixel + (args.read_noise * args.qe / args.gain) ** 2
    )  # Standard deviation of total noise (ADU/pixel).
    args.rng_base = galsim.BaseDeviate(seed=args.seed)
    args.rng = galsim.UniformDeviate(seed=args.seed)  # U(0,1).
    args.rng_defocus = galsim.GaussianDeviate(
        args.rng_base, mean=0.0, sigma=args.sigma_defocus
    )  # Default N(0,0.36).
    args.rng_gaussian = galsim.GaussianDeviate(
        args.rng_base, mean=0.0, sigma=args.sigma_opt_psf
    )  # Default N(0,0.07).
    fwhm_table = galsim.LookupTable(x=args.fwhms, f=args.freqs, interpolant="spline")
    fwhms = np.linspace(
        args.fwhms[0], args.fwhms[-1], 100
    )  # Upsample the distribution.
    freqs = (
        np.array([fwhm_table(fwhm) for fwhm in fwhms]) / fwhm_table.integrate()
    )  # Normalization.
    args.rng_fwhm = galsim.DistDeviate(
        seed=args.rng_base,
        function=galsim.LookupTable(x=fwhms, f=freqs, interpolant="spline"),
    )
    args.rng_gal_shear = galsim.DistDeviate(
        seed=args.rng, function=lambda x: x, x_min=args.min_shear, x_max=args.max_shear
    )  # Uniform
    args.rng_snr = galsim.DistDeviate(
        seed=args.rng,
        function=lambda x: 1 / (x**0.7),
        x_min=args.min_snr,
        x_max=args.max_snr,
        npoints=1000,
    )

    # dataset & dataloader
    ds_path = args.input_root_dir + "/" + args.input_dataset + "/"
    all_gal   = get_list(ds_path,"gal_*.fits")
    all_psf   = get_list(ds_path,"psf_*.fits")
    all_noise = get_list(ds_path,"noise_*.fits")
    all_info  = get_list(ds_path,"info*.json")
    n_total = len(all_gal)
    assert n_total == len(all_psf), "pb n_total neq n_psf"
    assert n_total == len(all_noise), "pb n_total neq n_noise"
    assert n_total == len(all_info), "pb n_total neq n_info"

    n_train = args.n_train
    n_val = args.n_val  # never used dureing trainig only to make the final plots
    assert n_train + n_val < n_total, "check sizes failed"
    n_test = n_total - (n_train + n_val)

    sequence = np.arange(0, n_total)  # Generate random sequence for dataset.
    np.random.shuffle(sequence)
    train_seq = sequence[0:n_train]
    test_seq = sequence[n_train : n_train + n_test]
    assert len(train_seq) == n_train, "check training sequence size"
    assert len(test_seq)  == n_test, "check testing sequence size"

    print(f"{n_total} gals splitted in {n_train}/{n_test}/{n_val} for train/test/val ")

    #print("args:",type(args),"\n",args)
    
    ds_train = GalaxyDataset(args,
                             all_gal,
                             all_psf,
                             all_noise,
                             all_info,
                             train_seq)

    ds_test  = GalaxyDataset(args,
                             all_gal,
                             all_psf,
                             all_noise,
                             all_info,
                             test_seq)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=ds_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # transformation: data augmentation and to torch tensor
    # nb: at least end with ToTensor()
    train_transforms = None
    test_transforms = None

    # model instantiation
    if args.archi == "Unet-Full":
        model = UNet(args)
    else:
        print("Error: ", args.archi, "unknown")
        return

    # check ouptut of model is ok. Allow to determine the model config done at run tile
    fake_input = torch.rand(1,1,args.fov_pixels,args.fov_pixels)
    out = model(fake_input)
    assert out.shape == fake_input.shape
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters:{n_params/10**6:.1f} millions")

    # put model to device before loading scheduler/optimizer parameters
    model.to(args.device)

    # optimizer & scheduler

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_init,
        # amsgrad=True,
        # eps=1e-8, # by default is 1e-8
        # weight_decay=1e-3   # default is 0
    )

    if args.use_scheduler:

        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.num_epochs, eta_min=1e-5
            )
        elif args.scheduler == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.lr_decay,
                patience=args.patience,
                min_lr=1e-5,
            )
        elif args.scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.patience, gamma=args.lr_decay
            )
        else:
            print("FATAL: not known scheduler...")
            return

    # check for resume session: load model/optim/scheduler dictionnaries
    start_epoch = 0

    train_loss_history = []
    test_loss_history = []

    if args.resume:
        args.checkpoint_file = args.out_root_dir + args.checkpoint_file
        args.history_loss_cpt_file = args.out_root_dir + args.history_loss_cpt_file

        # load checkpoint of model/scheduler/optimizer
        if os.path.isfile(args.checkpoint_file):
            print("=> loading checkpoint '{}'".format(args.checkpoint_file))
            checkpoint = torch.load(args.checkpoint_file)
            # the first epoch for the new training
            start_epoch = checkpoint["epoch"]
            # model update state
            model.load_state_dict(checkpoint["model_state_dict"])
            if args.resume_scheduler:
                # optizimer update state
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # scheduler update state
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print("=>>> scheduler not resumed")
                if args.resume_optimizer:
                    # optizimer update state
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                else:
                    print("=>>> optimizer not resumed")
            print("=> loaded checkpoint")
        else:
            print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
            return

        # load previous history of losses
        if os.path.isfile(args.history_loss_cpt_file):
            loss_history = np.load(args.history_loss_cpt_file, allow_pickle=True)
            train_loss_history = loss_history[0].tolist()
            test_loss_history = loss_history[1].tolist()

        else:
            print(
                "=> FATAL no history loss checkpoint '{}'".format(
                    args.history_loss_cpt_file
                )
            )
            return

    else:
        print("=> no checkpoints then Go as fresh start")

    # loss
    if args.loss == "MSE":
        criterion = nn.MSELoss(reduction="mean")
    elif args.loss == "L1":
        criterion = nn.L1Loss(reduction="mean")
    elif args.loss == "MultiScale":
        criterion = MultiScaleLoss(scales=args.multiscale_scale, norm=args.multScale_norm)
    else:
        print("FATAL loss criterion not well defined")
        return
    
    # loop on epochs
    t0 = time.time()
    best_test_loss = np.inf

    print("The current args:", args)

    for epoch in range(start_epoch, args.num_epochs + 1):

        print("Start epoch",epoch)
        
        # training
        train_loss = train(
            args, model, criterion, train_loader, train_transforms, optimizer, epoch
        )
        # test
        test_loss = test(
            args, model, criterion, test_loader, test_transforms, epoch
        )

        # print & book keeping
        print(
            f"Epoch {epoch}, Losses train: {train_loss:.6f}",
            f"test {test_loss:.6f}, LR= {scheduler.get_last_lr()[0]:.2e}",
            f"time {time.time()-t0:.2f}",
        )
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # update scheduler
        if args.use_scheduler:
            if args.scheduler == "reduce":
                # Warning ReduceLROnPlateau needs a metric
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # save state at each epoch to be able to reload and continue the optimization
        if args.use_scheduler:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        else:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_last_state.pth",
        )
        # save intermediate history
        np.save(
            args.out_root_dir + "/" + args.archi + "_last_history.npy",
            np.array((train_loss_history, test_loss_history)),
        )

        # if better loss update best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_best_state.pth",
        )

    # Bye
    tf = time.time()
    print("all done!", tf - t0)

"""
def main_orig():

    # Training config
    parser = argparse.ArgumentParser(
        description="Deconvolve varibale noisy and convolved images"
    )
    parser.add_argument("--file", help="Config file")
    args0 = parser.parse_args()

    ## Load yaml configuration file
    with open(args0.file, "r") as config:
        settings_dict = yaml.safe_load(config)
    args = SimpleNamespace(**settings_dict)

    # check number of num_workers
    NUM_CORES = multiprocessing.cpu_count()
    if args.num_workers >= NUM_CORES:
        print("Info: # workers set to", NUM_CORES // 2)
        args.num_workers = NUM_CORES // 2

    # where to put all model training stuff
    args.out_root_dir = args.out_root_dir + "/" + args.run_tag + "/"

    try:
        os.makedirs(
            args.out_root_dir, exist_ok=False
        )  # avoid erase the existing directories (exit_ok=False)
    except OSError:
        pass

    print("Info: outdir is ", args.out_root_dir)

    # device cpu/gpu...
    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # seeding
    set_seed(args.seed)
    args.rng = np.random.default_rng(args.seed)  # JEC 26 may 25

    # dataset & dataloader
    # where to find the dataset "clean" Calpha images
    input_dir = args.input_root_dir + args.input_dataset
    data_clean = get_list(input_dir, "d*.npz")
    print(f"Clean Dataset: {len(data_clean)} data loaded")
    n_train = args.n_train
    n_test = args.n_test
    n_val = args.n_val  # not used in the training but we should keep them unused
    assert n_train + n_test + n_val <= len(data_clean), "check sizes of train/test/val"
    data_train = data_clean[:n_train]
    data_test = data_clean[n_train : n_train + n_test]

    # normalization (13June25)
    ds_train = CustumDataset(data_train, min_I=args.min_I, max_I=args.max_I)
    ds_test = CustumDataset(data_test, min_I=args.min_I, max_I=args.max_I)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset=ds_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # transformation: data augmentation and to torch tensor
    # nb: at least end with ToTensor()
    train_transforms = None
    test_transforms = None

    # get a batch to determin the image sizes
    train_img = next(iter(train_loader))
    img_H = train_img.shape[-2]
    img_W = train_img.shape[-1]
    print("image sizes: HxW", img_H, img_W)

    # model instantiation
    if args.archi == "Unet-Full":
        model = UNet(args)
    else:
        print("Error: ", args.archi, "unknown")
        return

    # check ouptut of model is ok. Allow to determine the model config done at run tile
    out = model(train_img)
    assert out.shape == train_img.shape
    print(
        "number of parameters is ",
        sum(p.numel() for p in model.parameters() if p.requires_grad) // 10**6,
        "millions",
    )
    # for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)

    # put model to device before loading scheduler/optimizer parameters
    model.to(args.device)

    # optimizer & scheduler

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_init,
        # amsgrad=True,
        # eps=1e-8, # by default is 1e-8
        # weight_decay=1e-3   # default is 0
    )

    if args.use_scheduler:

        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.num_epochs, eta_min=1e-5
            )
        elif args.scheduler == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.lr_decay,
                patience=args.patience,
                min_lr=1e-5,
            )
        elif args.scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.patience, gamma=args.lr_decay
            )
        else:
            print("FATAL: not known scheduler...")
            return

    # check for resume session: load model/optim/scheduler dictionnaries
    start_epoch = 0

    train_loss_history = []
    test_loss_history = []

    if args.resume:
        args.checkpoint_file = args.out_root_dir + args.checkpoint_file
        args.history_loss_cpt_file = args.out_root_dir + args.history_loss_cpt_file

        # load checkpoint of model/scheduler/optimizer
        if os.path.isfile(args.checkpoint_file):
            print("=> loading checkpoint '{}'".format(args.checkpoint_file))
            checkpoint = torch.load(args.checkpoint_file)
            # the first epoch for the new training
            start_epoch = checkpoint["epoch"]
            # model update state
            model.load_state_dict(checkpoint["model_state_dict"])
            if args.resume_scheduler:
                # optizimer update state
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # scheduler update state
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                print("=>>> scheduler not resumed")
                if args.resume_optimizer:
                    # optizimer update state
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                else:
                    print("=>>> optimizer not resumed")
            print("=> loaded checkpoint")
        else:
            print("=> FATAL no  checkpoint '{}'".format(args.checkpoint_file))
            return

        # load previous history of losses
        if os.path.isfile(args.history_loss_cpt_file):
            loss_history = np.load(args.history_loss_cpt_file, allow_pickle=True)
            train_loss_history = loss_history[0].tolist()
            test_loss_history = loss_history[1].tolist()

        else:
            print(
                "=> FATAL no history loss checkpoint '{}'".format(
                    args.history_loss_cpt_file
                )
            )
            return

    else:
        print("=> no checkpoints then Go as fresh start")

    # loss
    if args.loss == "MSE":
        criterion = nn.MSELoss(reduction="mean")
    elif args.loss == "L1":
        criterion = nn.L1Loss(reduction="mean")
    elif args.loss == "MultiScale":
        criterion = MultiScaleLoss(scales=args.multiscale_scale, norm=args.multScale_norm)
    else:
        print("FATAL loss criterion not well defined")
        return
    
    # loop on epochs
    t0 = time.time()
    best_test_loss = np.inf

    print("The current args:", args)

    for epoch in range(start_epoch, args.num_epochs + 1):
        # training
        train_loss = train(
            args, model, criterion, train_loader, train_transforms, optimizer, epoch
        )
        # test
        test_loss = test(args, model, criterion, test_loader, test_transforms, epoch)

        # print & book keeping
        print(
            f"Epoch {epoch}, Losses train: {train_loss:.6f}",
            f"test {test_loss:.6f}, LR= {scheduler.get_last_lr()[0]:.2e}",
            f"time {time.time()-t0:.2f}",
        )
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # update scheduler
        if args.use_scheduler:
            if args.scheduler == "reduce":
                # Warning ReduceLROnPlateau needs a metric
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # save state at each epoch to be able to reload and continue the optimization
        if args.use_scheduler:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
        else:
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_last_state.pth",
        )
        # save intermediate history
        np.save(
            args.out_root_dir + "/" + args.archi + "_last_history.npy",
            np.array((train_loss_history, test_loss_history)),
        )

        # if better loss update best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        torch.save(
            state,
            args.out_root_dir + "/" + args.archi + "_best_state.pth",
        )

    # Bye
    tf = time.time()
    print("all done!", tf - t0)
"""

################################
if __name__ == "__main__":
    main_dev()
