import numpy as np
import galsim
import os
import time
import json


def main():
    """ Extract from COSMOS Catalog ready for galsim.RealGalaxyCatalog
    the HSF galaxy and PSF images and them as fits files to be
    afterwards read by galsim.fits.read(...) to get galsim. Image as
 
    galsim.Image(bounds=galsim.BoundsI(xmin=1, xmax=137, ymin=1, ymax=138), array=
    array([[ 3.77889839e-03,  3.39048356e-03, -1.36750680e-03, ...,
        -8.69793075e-05,  1.61084044e-03,  4.23315959e-03],
       [ 5.58069255e-03,  3.19484761e-03, -1.05097797e-03, ...,
        -5.40580694e-03, -1.79140863e-03, -1.38120085e-03],
       [ 6.62140315e-03,  3.07191606e-03,  7.28662126e-04, ...,
        -2.47776206e-03,  1.10220059e-03, -6.02471118e-04],
       ...,
       [ 1.48496195e-03,  1.00102881e-03, -5.14127873e-03, ...,
        -8.85593705e-04,  2.21097912e-03,  5.48594398e-03],
       [ 8.92920129e-04, -1.54932996e-03, -1.46600022e-03, ...,
        -1.71994267e-03, -2.92305776e-04,  2.36457703e-03],
       [-1.04477664e-03, -4.52986266e-03, -2.89684930e-03, ...,
         2.82304944e-03,  1.47912535e-03, -4.29218228e-04]],
      shape=(138, 137)), wcs=galsim.PixelScale(0.029999999329447746))
    """

    #Type of samples
    sample = "23.5"
    
    # location of the output catalog directory
    out_catalog_dir = "/lustre/fsn1/projects/rech/ixh/ufd72rp/COSMOS_"+sample+"_dataset"
    try:
        os.makedirs(
            out_catalog_dir, exist_ok=False
        )  # avoid erase the existing directories (exit_ok=False)
    except OSError:
        pass

    
    # location of the original COSMOS catalog
    path_catalog = "/lustre/fsn1/projects/rech/ixh/ufd72rp/COSMOS_"+sample+"_training_sample"

    try:
        real_galaxy_catalog = galsim.RealGalaxyCatalog(
            dir=path_catalog, sample=sample
        )
        n_total = real_galaxy_catalog.nobjects  # - 56062
        print(f" Successfully read in {n_total} sample={sample} galaxies.")
    except:
        raise Exception(f" Failed reading in {sample}  galaxies.")

    # Generate random sequence for dataset.
    start = 36231
    sequence = np.arange(start, n_total) 
    #np.random.shuffle(sequence)

    # Loop....
    t0 = time.time()
    for gal_idx in sequence:
        if  gal_idx==0 or  gal_idx%1000==0:
            print(f"Process {gal_idx} at t:{time.time()-t0:.1f}sec")
        # read HST galaxy and PSF
        #gal_ori = galsim.RealGalaxy(real_galaxy_catalog, index=gal_idx)
        #psf_hst = real_galaxy_catalog.getPSF(gal_idx)

        #get images and information
        gal_ori_image = real_galaxy_catalog.getGalImage(gal_idx)
        psf_ori_image = real_galaxy_catalog.getPSFImage(gal_idx)
        noise_ori_image, pixel_scale, var = real_galaxy_catalog.getNoiseProperties(gal_idx)
        
        # save images to fits files and varaibles to a  json file
        fname =  out_catalog_dir +"/"+"gal_"+str(gal_idx)+".fits"
        gal_ori_image.write(fname)
        
        fname =  out_catalog_dir +"/"+"psf_"+str(gal_idx)+".fits"
        psf_ori_image.write(fname)

        fname =  out_catalog_dir +"/"+"noise_"+str(gal_idx)+".fits"
        noise_ori_image.write(fname)

        info = {"pixel_scale":pixel_scale, "var":var}
        fname = out_catalog_dir +"/"+"info_"+str(gal_idx)+".json"
        with open(fname, 'w') as f:
            json.dump(info,f)


    # end
    print(f"All done at t:{time.time()-t0:.1f}sec")


################################
if __name__ == "__main__":
    main()

        
