import os
import cv2
import numpy as np

from PIL import Image
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T


class WoodblockDataset(Dataset):
    def __init__(
        self,
        opt,
        log,
        train
    ):
        super().__init__()
        self.dataset_dir = Path(opt.dataset_dir) / ('train' if train else 'valid')
        self.print_dir = self.dataset_dir / 'print_512'
        self.depth_dir = self.dataset_dir / 'np_depth_512'

        self.print_img_path_list = list(Path(self.print_dir).glob("*.png"))
        self.depth_img_path_list = [Path(os.path.join(self.depth_dir, print_img_path.stem + ".npy")) for print_img_path in self.print_img_path_list]        
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)
        ])

        log.info(f"[Dataset] Built Woodblock dataset {self.print_dir=}, size={len(self.print_img_path_list)}!")
        log.info(f"[Dataset] Built Woodblock dataset {self.depth_dir=}, size={len(self.depth_img_path_list)}!")

    
    def __len__(self):
        return len(self.print_img_path_list)

    def preprocess_image(self, image_path):
        """Utility function that load an image an convert to torch."""
        # open image using OpenCV (HxWxC)
        img = Image.open(image_path).convert('L')
        # convert image to torch tensor (CxHxW)
        img_t: torch.Tensor = self.transform(img)
        return img_t

    def preprocess_depth(self, depth_path):
        """Utility function that load an image an convert to torch."""
        # open image using OpenCV (HxWxC)
        img: np.ndarray = np.load(depth_path)
        # unsqueeze to make it 1xHxW
        img = np.expand_dims(img, axis=0)
        # cast type as np.float32
        img = img.astype(np.float32)
        # convert image to torch tensor (CxHxW)
        img_t: torch.Tensor = torch.from_numpy(img)
        return img_t * 2 - 1
    
    def __getitem__(self, index):
        print_path = self.print_img_path_list[index]
        depth_path = self.depth_img_path_list[index]

        t_print = self.preprocess_image(print_path)
        t_depth = self.preprocess_depth(depth_path)
        
        mask = t_depth != torch.max(t_depth)

        return t_depth, t_print, mask