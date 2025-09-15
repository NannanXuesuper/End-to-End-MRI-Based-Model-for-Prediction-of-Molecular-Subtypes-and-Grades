import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
# from tqdm import tqdm
import numpy as np
import time
import os
import h5py
import torchvision.models as models
import warnings
from model.model import IClassifier
warnings.filterwarnings('ignore')
from datasets.dataset import MyDataset
from train_utils.train import training_onefold, validate_one_epoch  


def train_allfolds(image_type ): 
    # folds = (0,1,2,3,4)   
    # for fold in folds:
    fold =1
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(30),
                                    # transforms.RandomAutocontrast(),
                                    # transforms.ElasticTransform(alpha=250.0),
                                    # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                                    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
                                    ]),
        "val": transforms.Compose([transforms.ToTensor()]),
        "zzu1": transforms.Compose([transforms.ToTensor()]),
        "zzu3": transforms.Compose([transforms.ToTensor()]),
        "xy_gs_nm_hb": transforms.Compose([transforms.ToTensor()]),
        "xj": transforms.Compose([transforms.ToTensor()]),
        "xy_gs_nm": transforms.Compose([transforms.ToTensor()]),
        "hb": transforms.Compose([transforms.ToTensor()]),
        "upenn": transforms.Compose([transforms.ToTensor()]),
        "tcga": transforms.Compose([transforms.ToTensor()]),}

    data_root = r'/local/CTimages/brain/code_5mm/data/tumor_slices_2mm' # get data root path
    weight_path =os.path.join(data_root, "zzu2/03_h5data")# flower data set path
    train_image_path = os.path.join(data_root, "zzu2/03_h5data",'train_images_fold'+str(fold)+'.hdf5')# flower data set path
    train_dataset =  MyDataset(train_image_path,image_type,transform=data_transform["train"])
    train_num = len(train_dataset)
    batch_size =256
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 8
  
    
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=nw, pin_memory=True)

    val_image_path = os.path.join(data_root, "zzu2/03_h5data",'test_images_fold'+str(fold)+'.hdf5')  
    validate_dataset = MyDataset(val_image_path,image_type,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                    batch_size=batch_size, shuffle=False,
                                                    num_workers=nw, pin_memory=True)
    print("val finished")
    test_image_paths = {
        "zzu1": os.path.join(data_root, "zzu1/03_h5data/noADC/test_images_zzu1.hdf5"),
        "zzu3": os.path.join(data_root, "zzu3/03_h5data/test_images_zzu3.hdf5"),
        "xy_gs_nm": os.path.join(data_root, "xy_gs_nm/03_h5data/test_images_xy_gs_nm.hdf5"),
        "hb": os.path.join(data_root, "hb/03_h5data/noADC/test_images_hb.hdf5"),
        "xj": os.path.join(data_root, "xj/03_h5data/noADC/test_images_xj.hdf5"),
        "tcga": os.path.join(data_root, "tcga/03_h5data/noADC/test_images_tcga.hdf5"),
        "upenn": os.path.join(data_root, "upenn/03_h5data/noADC/test_images_upenn.hdf5")
    }
    test_loaders = {}
    for center, path in test_image_paths.items():
        dataset = MyDataset(path, image_type, transform=data_transform[center])
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        test_loaders[center] = loader
        print(f"{center} dataloader is ready")

    print(f"using {train_num} images for training, {val_num} images for validation")
    for center, loader in test_loaders.items():
        print(f"{len(loader.dataset)} images for {center}")
        

    start = time.perf_counter()
    training_onefold(train_loader, validate_loader, test_loaders, image_type, fold,weight_path)
    run_time = round(time.perf_counter()-start)
    # 计算时分秒
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    print("共用时 {}h{}m{}s".format(hour,minute,second))
    

    return None

if __name__ == '__main__':
    train_allfolds(image_type = "T2_CET1_Flair")

