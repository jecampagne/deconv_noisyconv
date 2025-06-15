import argparse
import yaml
import os
import random
import pathlib
import pickle
from types import SimpleNamespace
import multiprocessing
import time

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import regex as re
import glob


from model import *
from image_manip import *


################
# Utils
################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


################
# dataloader
################

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


# 13Juin25: JEC add min_I, max_I rescaling
class CustumDataset(Dataset):
    """Load the data set which is supposed to be a Numpy structured array"""

    def __init__(self, dataset, min_I, max_I):
        """
        dataset: list of files
        """
        self.datas = dataset
        self.min_I = min_I
        self.max_I = max_I
        print(
            f"CustumDataset: {len(self.datas)} data loaded, rescaling:[{self.min_I},{self.max_I}]",
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        """
        rescale image
        transform from numpy HW  to torch CWH (C=1) float32
        """
        img_clean = np.load(self.datas[index])["img"]
        # rescale pixel value in range
        img_clean = rescale_image_range(img_clean, max_I=self.max_I, min_I=self.min_I)

        # float32 array
        img_clean = np.float32(img_clean)

        # torch convention CHW
        img_clean = np.expand_dims(img_clean, axis=0)  # 1xHxW

        # to torch tensor
        clean = torch.from_numpy(img_clean)

        return clean


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


def main():

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

    #normalization (13June25)
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
    main()
