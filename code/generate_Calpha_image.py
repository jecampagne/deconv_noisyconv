import time
import torch
from synthetic_data_generators import make_C_alpha_images
import argparse
import os
import random
import numpy as np
import yaml
from types import SimpleNamespace

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def rescale_image_range(im, max_I=1.0, min_I=-1.0):
    temp = (im - im.min()) / ((im.max() - im.min()))
    return temp * (max_I - min_I) + min_I


def main():
    parser = argparse.ArgumentParser(description="Calpha original images")
    parser.add_argument("--file", help="Config file")
    #parser.add_argument('--alpha',type=float )
    #parser.add_argument('--seed', type=int)
    #parser.add_argument('--size',type=int , default=128) #image size   
    #parser.add_argument('--num_samples',type=int , default=1)   
    #parser.add_argument('--factor', default=(2,2)) # default blur factor
    args0 = parser.parse_args()

    
    ## Load yaml configuration file
    with open(args0.file, 'r') as config:
        settings_dict = yaml.safe_load(config)
    args = SimpleNamespace(**settings_dict)


    print("dbg:<",args,">")
    
    set_seed(args.seed)


    root_dir = args.root_dir
    if args.size == 128:
        path = root_dir + '/dataset_ca_'+str(args.alpha)+"/"
    else:
        path = root_dir + '/dataset_ca_'+str(args.alpha)+"_"+str(args.size)+"/"
    

    if os.path.exists(path): #to avoid over-writing the existing dataset 
        pass
    else:
        os.makedirs(path)


    start_time_total = time.time()

    for i in range(args.num_samples):
        if i%(args.num_samples//10)==0:
            print("i:",i)
        img = make_C_alpha_images(alpha = args.alpha, 
                                  beta = args.alpha, 
                                  im_size=args.size, 
                                  num_samples=1, 
                                  factor=args.factor,
                                  )[0]
        
        #store
        #print(img.shape,type(img),img.flatten()[0])
        img = img.squeeze().numpy()
        img = rescale_image_range(img)
        data = {
            "img": img,
            "alpha":args.alpha
        }
        
        f_name = "d_" + str(i) + ".npz"
        np.savez(path + "/" + f_name, **data)
        
        
    print("--- %s seconds ---" % (time.time() - start_time_total))

        

if __name__ == "__main__":
    main()
