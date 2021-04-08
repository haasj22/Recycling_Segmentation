from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import ConcatDataset
from IIC.code.datasets.segmentation.cocostuff import _Coco
import os
import cv2

class _RecyclingDataset(_Coco):

    def __init__(self, **kwargs):
        super(_RecyclingDataset, self).__init__(**kwargs)
        self._set_files()

    def _set_files(self):
        file_list = [filename for filename in os.listdir(self.root + "/small_johns_images_for_IIC")]
        self.files = file_list

    def _load_data(self, image_id):
        image_path = str(self.root) + "/small_johns_images_for_IIC/" + str(image_id)
        print("file path: " + str(image_path))
        image = cv2.imread(image_path).astype(np.uint8)
        print("Image size: " + str(image.shape))

        label = np.array([[[-1]]])
        return image, label

"""
def create_Recycling_Dataloaders(config):
    DATA_DIR = "/mnt/storage2/METRO_recycling/imgds.npy"
    X_train = np.load(DATA_DIR)
    print(f"Shape of training data: {X_train.shape}")
    print(f"Data type: {type(X_train)}")
    
    all_partitions = X_train
    img_list = []
    for partition in all_partitions:
        img_curr = X_train(
                **{"config:": config,
                    "split": partition,
                    "purpose": "test"}
                
        )
        img_list +=im
    recycling_dataset = ConcatDataset(X_train)
    dataloaders = []
    
    do_shuffle = (config.num_dataloaders == 1)
    for loader in range(config.num_dataloaders):
        #creating a dataloader containing the training images
        train_dataloader = torch.utils.data.DataLoader(recycling_dataset,
                                        batch_size=config.dataloader_batch_sz,
                                        shuffle = do_shuffle,
                                        num_workers = 0,
                                        drop_last = False)

        if loader > 0:
            assert len(dataloaders[loader]) == lwn(dataloaders[loader])

        dataloaders.append(train_dataloader)

    return dataloaders, dataloaders, dataloaders
    """
