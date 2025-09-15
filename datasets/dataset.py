import os
import torch
import numpy as np
import time
from torch.utils.data import Dataset
import h5py
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import warnings
from torch.autograd import Variable

class MyDataset(Dataset):
    def __init__(self, path, image_type, transform=None):
        self.IDH_idx = {b'Mutation': 0, b'Wild': 1}
        self.grade_idx = {b'GBM': 0, b'LGG': 1}
        self.p119q_idx = {b'Codeleted': 0, b'Non_codeleted': 1, b'Mask': 2}
        
        self.hdf5_file = h5py.File(path, 'r')
        self.image_type = image_type
        
        self.ID = self.hdf5_file['ids'][:]
        self.IDH = self.label_to_class(self.hdf5_file['IDH'][:], "IDH")
        self.P119Q = self.label_to_class(self.hdf5_file['p1_19qs'][:], "p119q")
        self.Grade = self.label_to_class(self.hdf5_file['Grade'][:], "grade")
        self.len = len(self.ID)
        
        self.transform = transform
    
    def __getitem__(self, index):
        if self.image_type == ["T1"]:
            img = self.hdf5_file['input'][index, :, :, 0]
        elif self.image_type == ["T2"]:
            img = self.hdf5_file['input'][index, :, :, 1]
        elif self.image_type == ["Flair"]:
            img = self.hdf5_file['input'][index, :, :, 2]
        elif self.image_type == ["CET1"]:
            img = self.hdf5_file['input'][index, :, :, 3]
        elif self.image_type == ["ADC"]:
            img = self.hdf5_file['input'][index, :, :, 4]
        elif self.image_type == ["T2", "CET1"]:
            img = self.hdf5_file['input'][index, :, :, (1, 3)]
        elif self.image_type == "T2_Flair":
            img = self.hdf5_file['input'][index, :, :, (1, 2)]
        elif self.image_type == "T2_CET1_Flair":
            img = self.hdf5_file['input'][index, :, :, (1, 2, 3)]
        # else:
        #     img = self.hdf5_file['input'][index, :, :, :]
        
        # print(self.image_type)
        if self.transform is not None:
            img = self.transform(img)
        
        id = self.ID[index]
        IDH = self.IDH[index]
        p1_19q = self.P119Q[index]
        grade = self.Grade[index]

        return img, id, IDH, p1_19q, grade
    
    def label_to_class(self,labels,phonetype):
        if phonetype == "IDH":
            class_to_idx = self.IDH_idx
        if phonetype == "p119q":
            class_to_idx = self.p119q_idx
        if phonetype == "grade":
            class_to_idx = self.grade_idx
        label_nums =[]
        for label in labels:
            label_num = class_to_idx[label]
            label_nums.append(label_num)
        return  label_nums
    
    def __len__(self):
        return self.len
    
    def __del__(self):
        self.hdf5_file.close()
        
class My_features_Dataset(Dataset):
    def __init__(self, path,feature_forselection = False):
        with h5py.File(path, 'r') as hdf5_file:
            self.Grade_Features_type0 = torch.from_numpy(hdf5_file['features_grade_type0'][:])
            self.IDH_Features_type0 = torch.from_numpy(hdf5_file['features_idh_type0'][:])
            self.P119q_Features_type0 = torch.from_numpy(hdf5_file['features_p119q_type0'][:])

            self.Grade_Features_type1 = torch.from_numpy(hdf5_file['features_grade_type1'][:])
            self.IDH_Features_type1 = torch.from_numpy(hdf5_file['features_idh_type1'][:])
            self.P119q_Features_type1 = torch.from_numpy(hdf5_file['features_p119q_type1'][:])

            self.Grade_Features_type2 = torch.from_numpy(hdf5_file['features_grade_type2'][:])
            self.IDH_Features_type2 = torch.from_numpy(hdf5_file['features_idh_type2'][:])
            self.P119q_Features_type2 = torch.from_numpy(hdf5_file['features_p119q_type2'][:])
            
            self.IDH = torch.from_numpy(hdf5_file['labels_idh'][:].astype(int))
            self.P119Q =torch.from_numpy(hdf5_file['labels_p119q'][:].astype(int))
            self.Grade =torch.from_numpy(hdf5_file['labels_grade'][:].astype(int))
            self.ID = hdf5_file['ids'][:]
            
        self.grade_Feature_type0_list, self.IDH_Feature_type0_list, self.P119Q_Feature_type0_list,self.grade_Feature_type1_list, self.IDH_Feature_type1_list, self.P119Q_Feature_type1_list,self.grade_Feature_type2_list, self.IDH_Feature_type2_list, self.P119Q_Feature_type2_list, self.id_list, self.grade_list, self.idh_list, self.p119q_list = self.get_bag_data(
            self.ID, self.Grade_Features_type0, self.IDH_Features_type0, self.P119q_Features_type0, self.Grade_Features_type1, self.IDH_Features_type1, self.P119q_Features_type1,self.Grade_Features_type2, self.IDH_Features_type2, self.P119q_Features_type2,self.Grade, self.IDH, self.P119Q)
        self.len = len(self.grade_Feature_type0_list)
        self.feature_selection = feature_forselection
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.feature_selection:
            Grade_Features_type0= self.Grade_Features_type0[index].detach().numpy()
            IDH_Features_type0= self.IDH_Features_type0[index].detach().numpy()
            p1_19q_Features_type0= self.P119q_Features_type0[index].detach().numpy()

            Grade_Features_type1= self.Grade_Features_type1[index].detach().numpy()
            IDH_Features_type1= self.IDH_Features_type1[index].detach().numpy()
            p1_19q_Features_type1= self.P119q_Features_type1[index].detach().numpy()

            Grade_Features_type2= self.Grade_Features_type2[index].detach().numpy()
            IDH_Features_type2= self.IDH_Features_type2[index].detach().numpy()
            p1_19q_Features_type2= self.P119q_Features_type2[index].detach().numpy()
            # if self.transform is not None:
            #     Grade_Features = self.transform(Grade_Features)
            #     IDH_Features = self.transform(IDH_Features)
            #     p1_19q_Features = self.transform(p1_19q_Features)

            id = self.ID[index]
            IDH = self.IDH[index]
            p1_19q = self.P119Q[index]
            grade = self.Grade[index]

            return Grade_Features_type0,IDH_Features_type0,p1_19q_Features_type0,Grade_Features_type1,IDH_Features_type1,p1_19q_Features_type1,Grade_Features_type2,IDH_Features_type2,p1_19q_Features_type2,id,IDH,p1_19q,grade
        else:
            Grade_Features_type0 = self.grade_Feature_type0_list[index].detach().numpy()
            IDH_Features_type0 = self.IDH_Feature_type0_list[index].detach().numpy()
            p1_19q_Features_type0 = self.P119Q_Feature_type0_list[index].detach().numpy()
            

            Grade_Features_type1 = self.grade_Feature_type1_list[index].detach().numpy()
            IDH_Features_type1 = self.IDH_Feature_type1_list[index].detach().numpy()
            p1_19q_Features_type1 = self.P119Q_Feature_type1_list[index].detach().numpy()

            Grade_Features_type2 = self.grade_Feature_type2_list[index].detach().numpy()
            IDH_Features_type2 = self.IDH_Feature_type2_list[index].detach().numpy()
            p1_19q_Features_type2 = self.P119Q_Feature_type2_list[index].detach().numpy()

            id = self.id_list[index]
            IDH = self.idh_list[index]
            p1_19q = self.p119q_list[index]
            grade = self.grade_list[index]

            return Grade_Features_type0,IDH_Features_type0,p1_19q_Features_type0,Grade_Features_type1,IDH_Features_type1,p1_19q_Features_type1,Grade_Features_type2,IDH_Features_type2,p1_19q_Features_type2,id,IDH,p1_19q,grade


    def get_bag_data(self, IDs, Grade_Features_type0, IDH_Features_type0, P119q_Features_type0,Grade_Features_type1, IDH_Features_type1, P119q_Features_type1, Grade_Features_type2, IDH_Features_type2, P119q_Features_type2, Grade, IDH, P119Q):
            current_id = IDs[0]
            start = 0
            grade_Feature_type0_list = []
            IDH_Feature_type0_list = []
            P119Q_Feature_type0_list = []

            grade_Feature_type1_list = []
            IDH_Feature_type1_list = []
            P119Q_Feature_type1_list = []

            grade_Feature_type2_list = []
            IDH_Feature_type2_list = []
            P119Q_Feature_type2_list = []

            id_list = []
            grade_list, idh_list, p119q_list = [], [], []

            for inx, new_id in enumerate(IDs):
                if current_id != new_id:
                    grade_Feature_type0_list.append(Grade_Features_type0[start:inx])
                    IDH_Feature_type0_list.append(IDH_Features_type0[start:inx])
                    P119Q_Feature_type0_list.append(P119q_Features_type0[start:inx])

                    grade_Feature_type1_list.append(Grade_Features_type1[start:inx])
                    IDH_Feature_type1_list.append(IDH_Features_type1[start:inx])
                    P119Q_Feature_type1_list.append(P119q_Features_type1[start:inx])

                    grade_Feature_type2_list.append(Grade_Features_type2[start:inx])
                    IDH_Feature_type2_list.append(IDH_Features_type2[start:inx])
                    P119Q_Feature_type2_list.append(P119q_Features_type2[start:inx])

                    id_list.append(IDs[inx - 1])
                    grade_list.append(Grade[inx - 1])
                    idh_list.append(IDH[inx - 1])
                    p119q_list.append(P119Q[inx - 1])

                    current_id = new_id
                    start = inx

            # Add the last group
            grade_Feature_type0_list.append(Grade_Features_type0[start:])
            IDH_Feature_type0_list.append(IDH_Features_type0[start:])
            P119Q_Feature_type0_list.append(P119q_Features_type0[start:])

            grade_Feature_type1_list.append(Grade_Features_type1[start:])
            IDH_Feature_type1_list.append(IDH_Features_type1[start:])
            P119Q_Feature_type1_list.append(P119q_Features_type1[start:])

            grade_Feature_type2_list.append(Grade_Features_type2[start:])
            IDH_Feature_type2_list.append(IDH_Features_type2[start:])
            P119Q_Feature_type2_list.append(P119q_Features_type2[start:])

            id_list.append(IDs[-1])
            grade_list.append(Grade[-1])
            idh_list.append(IDH[-1])
            p119q_list.append(P119Q[-1])
            
            return grade_Feature_type0_list, IDH_Feature_type0_list, P119Q_Feature_type0_list,grade_Feature_type1_list, IDH_Feature_type1_list, P119Q_Feature_type1_list,grade_Feature_type2_list, IDH_Feature_type2_list, P119Q_Feature_type2_list, id_list, grade_list, idh_list, p119q_list        