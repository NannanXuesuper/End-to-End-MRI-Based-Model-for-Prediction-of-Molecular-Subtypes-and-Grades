from torchvision import transforms
import numpy as np
import gc
import time
import os
import h5py
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from model.model import IClassifier
from datasets.dataset import MyDataset
from extract_feature.extract_feature import get_feature,get_feature_one_epoch
device   = torch.device("cpu")
print("using {} device.".format(device))
for fold in (0,1,2,3,4):

    image_type =['T2', 'Flair', 'CET1']
    start =time.time()
    print("################第{}折#################".format(fold))
    B = torch.tensor([2])
    norm=nn.BatchNorm2d
    resnet = models.resnet34(pretrained='resnet34', norm_layer=norm)
    for param in resnet.parameters():
        param.requires_grad = True
    resnet.fc = nn.Identity()

    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    num_feats = 512      
    num_classes = 1
    batch_size = 256
    nw = 8
    net1 = IClassifier(resnet, num_feats, output_class=num_classes).to(device)

    h5py_path = f"/local/CTimages/brain/code_5mm/data/tumor_slices_2mm/zzu2/03_h5data/features/{image_type}"
    if not os.path.exists(h5py_path):
        os.makedirs(h5py_path)
    data_transform = {
                "train": transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomRotation(30),
                                            transforms.RandomAutocontrast(),
                                            transforms.ElasticTransform(alpha=250.0),
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

    train_image_path = os.path.join(data_root, "zzu2/03_h5data",'train_images_fold'+str(fold)+'.hdf5')# flower data set path
    train_dataset =  MyDataset(train_image_path,image_type,transform=data_transform["val"])
    train_num = len(train_dataset)

    # # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=nw, pin_memory=True)
    center = "train"
    get_feature_one_epoch(train_loader,center,net1,device,fold,image_type,h5py_path)

    val_image_path = os.path.join(data_root, "zzu2/03_h5data",'test_images_fold'+str(fold)+'.hdf5')  
    validate_dataset = MyDataset(val_image_path,image_type,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                    batch_size=batch_size, shuffle=False,
                                                    num_workers=nw, pin_memory=True)
    print("val finished")
    print(f"using {train_num} images for training, {val_num} images for validation")

    center = "val"
    get_feature_one_epoch(validate_loader,center,net1,device,fold,image_type,h5py_path)





    test_image_paths = {
        "zzu1": os.path.join(data_root, "zzu1/03_h5data/noADC/test_images_zzu1.hdf5"),
        "zzu3": os.path.join(data_root, "zzu3/03_h5data/test_images_zzu3.hdf5"),
        "xy_gs_nm": os.path.join(data_root, "xy_gs_nm/03_h5data/test_images_xy_gs_nm.hdf5"),
        "hb": os.path.join(data_root, "hb/03_h5data/noADC/test_images_hb.hdf5"),
        "xj": os.path.join(data_root, "xj/03_h5data/noADC/test_images_xj.hdf5"),
        # "tcga": os.path.join(data_root, "tcga/03_h5data/noADC/test_images_tcga.hdf5"),
        "upenn": os.path.join(data_root, "upenn/03_h5data/noADC/test_images_upenn.hdf5")
    }

    test_loaders = {}
    for center, path in test_image_paths.items():
        dataset = MyDataset(path, image_type, transform=data_transform[center])
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        test_loaders[center] = loader
        print(f"{center} dataloader is ready")
    for center, loader in test_loaders.items():
        print(f"{len(loader.dataset)} images for {center}")
    for center, loader in test_loaders.items():
        print(f"{len(loader.dataset)} images for {center}")
        get_feature_one_epoch(loader,center,net1,device,fold,image_type,h5py_path)