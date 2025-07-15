import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io

    
class ThreeClassHE(Dataset):
    def __init__(self, 
                 root_dir,
                 dataset_name,
                 norm_method,
                 mode: str ="train", 
                 img_ids: np.array = None,
                 transform=None,
                 all_transform = None):
        
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.img_ids = img_ids
        self.norm_method = norm_method
        self.all_transform = all_transform
        if mode == "train":
            self.image_folder = f"{root_dir}\{dataset_name}\Train"
            self.mask_folder = f"{root_dir}\{dataset_name}\y\y_train/"
            self.gradmask_folder =  f"{root_dir}\{dataset_name}\y\Gradients\y_train/"
            self.wm_folder = f"{root_dir}\{dataset_name}\y\Weight_Maps\y_train/"
            self.dm_folder = f"{root_dir}\{dataset_name}\y\Distance_Maps\y_train/"
            
            tempdir = f"{root_dir}\{dataset_name}\Train"
            self.img_ids = pd.DataFrame(os.listdir(tempdir + r"\RGB\UN"))
                        
        elif mode == "valid":
            self.image_folder = f"{root_dir}\{dataset_name}\Val"
            self.mask_folder = f"{root_dir}\{dataset_name}\y\y_val/"
            self.gradmask_folder =  f"{root_dir}\{dataset_name}\y\Gradients\y_val/"
            self.wm_folder = f"{root_dir}\{dataset_name}\y\Weight_Maps\y_val/"
            self.dm_folder = f"{root_dir}\{dataset_name}\y\Distance_Maps\y_val/"


            tempdir = f"{root_dir}\{dataset_name}\Val"
            self.img_ids = pd.DataFrame(os.listdir(tempdir + r"\RGB\UN"))
            
        elif mode == "test":
            self.image_folder = f"{root_dir}\{dataset_name}\Test"
            self.mask_folder = f"{root_dir}\{dataset_name}\y\y_test/"
            self.gradmask_folder =  f"{root_dir}\{dataset_name}\y\Gradients\y_test/"
            self.wm_folder = f"{root_dir}\{dataset_name}\y\Weight_Maps\y_test/"
            self.dm_folder = f"{root_dir}\{dataset_name}\y\Distance_Maps\y_test/"

            
            tempdir = f"{root_dir}\{dataset_name}\Test"
            self.img_ids = pd.DataFrame(os.listdir(tempdir + r"\RGB\UN"))
        
        elif mode == 'TNBC':
            self.image_folder = f"{root_dir}\TNBC\Test"
            self.mask_folder = f"{root_dir}\TNBC\y\y_test/"
            self.gradmask_folder = f"{root_dir}\TNBC\y\Gradients\y_test"
            self.wm_folder = f"{root_dir}\TNBC\y\Weight_Maps\y_test"

            tempdir = f"{root_dir}\TNBC\Test"
            self.img_ids = pd.DataFrame(os.listdir(tempdir + r"\RGB\UN"))
        
        elif mode == 'TCGA':
            self.image_folder = f"{root_dir}\TCGA\Test"
            self.mask_folder = f"{root_dir}\TCGA\y\y_test/"
            self.gradmask_folder =f"{root_dir}\TCGA\y\Gradients\y_test"
            self.wm_folder =f"{root_dir}\TCGA\y\Weight_Maps\y_test"

            tempdir = f"{root_dir}\TCGA\Test"
            self.img_ids = pd.DataFrame(os.listdir(tempdir + r"\RGB\UN"))
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image_name = self.img_ids[0][idx]
        
        image_dir = os.path.join(self.image_folder, "RGB", self.norm_method, image_name)
        mask_dir = os.path.join(self.mask_folder, image_name)
        gradient_dir = os.path.join(self.image_folder, "Gradients", self.norm_method, image_name)
        wm_dir = os.path.join(self.wm_folder, image_name)
        grad_mask = os.path.join(self.gradmask_folder, image_name)
            
        image = Image.open(image_dir)
        mask = Image.open(mask_dir)
        gradient = Image.open(gradient_dir)
        gradmask = Image.open(grad_mask)
        wm = io.read_image(wm_dir).float()/255
        
        if self.mode == 'train':
            if random.random() >0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                gradient = TF.hflip(gradient)
                gradmask = TF.hflip(gradmask)
                wm = TF.hflip(wm)
        
            if random.random() >0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                gradient = TF.vflip(gradient)
                gradmask = TF.vflip(gradmask)
                wm = TF.vflip(wm)    

        if self.transform:
            image = self.transform(image)
            
        if self.all_transform:
            
            image = self.all_transform(image)
            mask = self.all_transform(mask)
            gradient = self.all_transform(gradient)
            gradmask = self.all_transform(gradmask)
            
        sample = {'image':image,'mask':mask,'gradient':gradient,'gradmask':gradmask,'wm':wm}

        return sample 
 
    