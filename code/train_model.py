import argparse
import yaml
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
    gal = gal_ori.rotate(theta * galsim.radians)  # Rotate by a random angle
    gal = gal.shear(g=gal_g, beta=gal_beta * galsim.radians)  # Apply the desired shear
    gal = gal.magnify(
        gal_mu
    )  # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.

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


class GalaxyDataset(Dataset):

    def __init__(
        self,
        real_galaxy_catalog,
        sequence
    ):
        """
        real_galaxy_catalog: Galsim real galaxy catalog
        sequence: indexes of galaxies in the catalog
        """
        self.rgc = real_galaxy_catalog
        self.seq = sequence
        print("GalaxyDataset: seq",self.seq,len(self.seq))
        
    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        """
        Read out real galaxy from the catalog and the correspondig HST PSF
        """
        print("get idx:",idx,self.seq[idx])
        gal_ori = galsim.RealGalaxy(self.rgc, index = self.seq[idx])
        psf_hst = self.rgc.getPSF(self.seq[idx])
        
        return gal_ori, psf_hst


################
# train/test 1-epoch
################
def train(args, model, criterion, train_loader, transforms, optimizer, epoch):
    # convolution & noise params
    fwhm_min = args.fwhm[0]
    fwhm_max = args.fwhm[1]
    sigma_min = args.sigma[0]
    sigma_max = args.sigma[1]

    # train mode
    model.train()

    loss_sum = 0  # to get the mean loss over the dataset
    for i_batch, imgs_clean in enumerate(train_loader):

        # cleans  (N C H W)
        bs = imgs_clean.shape[0]
        imgs_clean = imgs_clean.to(args.device)

        # convolution
        psf_fwhm = args.rng.uniform(low=fwhm_min, high=fwhm_max, size=(bs,))
        psf = [make_psf(f) for f in psf_fwhm]  # nb. all psf haven't the same HW shapes
        imgs_conv = torch.stack(
            [
                F.conv2d(
                    imgs_clean[i],
                    torch.from_numpy(psf[i][None, None, :, :])
                    .to(torch.float32)
                    .to(args.device),
                    padding="same",
                )
                for i in range(bs)
            ]
        )
        # add noise
        sigma_noise = torch.rand(size=(imgs_conv.shape[0], 1, 1, 1), device=args.device)

        if args.quadratic:  # JEC 17June25
            sigma_noise = (
                sigma_noise * (sigma_max**0.5 - sigma_min**0.5) + sigma_min**0.5
            )
            sigma_noise = sigma_noise**2
        else:
            sigma_noise = sigma_noise * (sigma_max - sigma_min) + sigma_min

        noise = sigma_noise * torch.randn(size=imgs_conv.shape, device=args.device)
        imgs = imgs_conv + noise

        # train step
        optimizer.zero_grad()
        output = model(imgs)

        #####JEC(10 june 25) TEST ONLY FOT FWHM=0 => denoiser residual
        #####if args.residual:
        #####    output = imgs - output # densoised
        #####

        loss = criterion(output, imgs_clean)
        loss_sum += loss.item()
        # backprop to compute the gradients
        loss.backward()
        # perform an optimizer step to modify the weights
        optimizer.step()

    return loss_sum / (i_batch + 1)


def test(args, model, criterion, test_loader, transforms, epoch):
    # convolution & noise params
    fwhm_min = args.fwhm[0]
    fwhm_max = args.fwhm[1]
    sigma_min = args.sigma[0]
    sigma_max = args.sigma[1]

    # test mode
    model.eval()

    loss_sum = 0  # to get the mean loss over the dataset
    with torch.no_grad():
        for i_batch, imgs_clean in enumerate(test_loader):

            # cleans  (N C H W)
            bs = imgs_clean.shape[0]
            imgs_clean = imgs_clean.to(args.device)

            # convolution
            psf_fwhm = args.rng.uniform(low=fwhm_min, high=fwhm_max, size=(bs,))
            psf = [
                make_psf(f) for f in psf_fwhm
            ]  # nb. all psf haven't the same HW shapes
            imgs_conv = torch.stack(
                [
                    F.conv2d(
                        imgs_clean[i],
                        torch.from_numpy(psf[i][None, None, :, :])
                        .to(torch.float32)
                        .to(args.device),
                        padding="same",
                    )
                    for i in range(bs)
                ]
            )
            # add noise
            sigma_noise = torch.rand(
                size=(imgs_conv.shape[0], 1, 1, 1), device=args.device
            )

            if args.quadratic:  # JEC 17June25
                sigma_noise = (
                    sigma_noise * (sigma_max**0.5 - sigma_min**0.5) + sigma_min**0.5
                )
                sigma_noise = sigma_noise**2
            else:
                sigma_noise = sigma_noise * (sigma_max - sigma_min) + sigma_min

            noise = sigma_noise * torch.randn(size=imgs_conv.shape, device=args.device)
            imgs = imgs_conv + noise

            #
            output = model(imgs)

            #####JEC(10 june 25) TEST ONLY FOT FWHM=0 => denoiser residual
            #####if args.residual:
            #####    output = imgs - output # densoising by residual
            #####

            loss = criterion(output, imgs_clean)
            loss_sum += loss.item()

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

    # seeding
    set_seed(args.seed)

    # dataset & dataloader
    try:
        sample = str(args.dataset_type)
        real_galaxy_catalog = galsim.RealGalaxyCatalog(
            dir=args.input_root_dir + args.input_dataset, sample=sample
        )
        n_total = real_galaxy_catalog.nobjects  # - 56062
        print(f" Successfully read in {n_total} sample={sample} galaxies.")
    except:
        raise Exception(f" Failed reading in {sample}  galaxies.")

    n_train = args.n_train
    n_val = args.n_val  # never used dureing trainig only to make the final plots
    assert n_train+n_val < n_total, "check sizes failed"
    n_test = n_total - (n_train+n_val)

    sequence = np.arange(0, n_total) # Generate random sequence for dataset.
    np.random.shuffle(sequence)
    train_seq = sequence[0:n_train]
    test_seq  = sequence[n_train:n_train+n_test]
    assert len(train_seq)==n_train, "check training sequence size"
    assert len(test_seq)==n_test, "check testing sequence size"
    
    print(f"{n_total} gals splitted in {n_train}/{n_test}/{n_val} for train/test/val ")

    ds_train = GalaxyDataset(real_galaxy_catalog,train_seq)
    ds_test  = GalaxyDataset(real_galaxy_catalog,test_seq)

    train_loader = DataLoader(
        dataset=ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print("len train_loader",len(train_loader))
    
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

    # Bye
    tf = time.time()
    print("all done!", tf - t0)


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
    criterion = nn.MSELoss(reduction="mean")  # nn.L1Loss(reduction="mean")

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


################################
if __name__ == "__main__":
    main_dev()
