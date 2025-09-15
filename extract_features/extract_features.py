import torch
import numpy as np
import gc
import time
import os
import h5py
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from model.model import IClassifier
from datasets.dataset import MyDataset


def get_feature(genetype,image_type,fold,images,ids,grade,net):
    weights_root = f"/local/CTimages/brain/code_5mm/data/tumor_slices_2mm/zzu2/03_h5data/{genetype}/{image_type}/save_fold{fold}/best_weights"
    weights_path1 = os.path.join(weights_root, os.listdir(weights_root).pop())  
    net.load_state_dict(torch.load(weights_path1, map_location='cpu'))
    features_grade, logits = net(images.to(device))
    features_grade = features_grade.cpu().squeeze().numpy()
    ids_grade = ids.numpy()
    labels_grade = grade.numpy()
    logits = torch.squeeze(logits)
    predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
    return ids_grade,labels_grade,predict_y,features_grade

def get_feature_one_epoch(test_loader, center, net1,  device, fold, image_type,h5py_path):
    net1.eval()

    B = torch.tensor([2])
    output_file = os.path.join(h5py_path,f'features_fold{fold}_{center}.h5')
    with torch.no_grad():
        test_acc_grade_type0 = 0
        test_acc_grade_type1 = 0
        test_acc_grade_type2 = 0

        test_acc_IDH_type0 = 0
        test_acc_p1_19q_type0 = 0
        test_acc_IDH_type1 = 0
        test_acc_p1_19q_type1 = 0
        test_acc_IDH_type2 = 0
        test_acc_p1_19q_type2 = 0
        test_num_grade = 0
        test_num_IDH = 0
        test_num_p1_19q = 0

        # Create or open the h5py file
        
        with h5py.File(output_file, 'a') as h5f:
            # Create datasets if they do not exist
            if 'features_grade' not in h5f:
                h5f.create_dataset('features_grade_type0', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('features_grade_type1', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('features_grade_type2', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('labels_grade', (0,), maxshape=(None,), dtype='int32')

                h5f.create_dataset('features_idh_type0', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('features_idh_type1', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('features_idh_type2', (0, 512), maxshape=(None, 512), dtype='float32')
                # h5f.create_dataset('ids_idh', (0,), maxshape=(None,), dtype='int32')
                h5f.create_dataset('labels_idh', (0,), maxshape=(None,), dtype='int32')

                h5f.create_dataset('features_p119q_type0', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('features_p119q_type1', (0, 512), maxshape=(None, 512), dtype='float32')
                h5f.create_dataset('features_p119q_type2', (0, 512), maxshape=(None, 512), dtype='float32')
                # h5f.create_dataset('ids_p119q', (0,), maxshape=(None,), dtype='int32')
                h5f.create_dataset('labels_p119q', (0,), maxshape=(None,), dtype='int32')

                h5f.create_dataset('ids', (0,), maxshape=(None,), dtype='int32')

       
            for test_data in test_loader:
                images, ids, IDH, p1_19q_, grade = test_data

                genetype = "grade"
                ids_grade,labels_grade,predict_y,features_grade_type0 = get_feature(genetype,image_type[0],fold,images[:,0,:,:].unsqueeze(1),ids,grade,net1)  
                test_acc_grade_type0 += torch.eq(predict_y, grade.float().to(device)).sum().item()
                test_num_grade += len(grade)

                _,_,predict_y,features_grade_type1 = get_feature(genetype,image_type[1],fold,images[:,1,:,:].unsqueeze(1),ids,grade,net1)
                test_acc_grade_type1 += torch.eq(predict_y, grade.float().to(device)).sum().item()
                
                _,_,predict_y,features_grade_type2 = get_feature(genetype,image_type[2],fold,images[:,2,:,:].unsqueeze(1),ids,grade,net1)
                test_acc_grade_type2 += torch.eq(predict_y, grade.float().to(device)).sum().item()
                h5f['features_grade_type0'].resize((h5f['features_grade_type0'].shape[0] + features_grade_type0.shape[0]), axis=0)
                h5f['features_grade_type0'][-features_grade_type0.shape[0]:] = features_grade_type0

                h5f['features_grade_type1'].resize((h5f['features_grade_type1'].shape[0] + features_grade_type1.shape[0]), axis=0)
                h5f['features_grade_type1'][-features_grade_type1.shape[0]:] = features_grade_type1

                h5f['features_grade_type2'].resize((h5f['features_grade_type2'].shape[0] + features_grade_type2.shape[0]), axis=0)
                h5f['features_grade_type2'][-features_grade_type2.shape[0]:] = features_grade_type2

                h5f['labels_grade'].resize((h5f['labels_grade'].shape[0] + labels_grade.shape[0]), axis=0)
                h5f['labels_grade'][-labels_grade.shape[0]:] = labels_grade

                genetype = "IDH"
                ids_idh,labels_idh,predict_y,features_idh_type0 = get_feature(genetype,image_type[0],fold,images[:,0,:,:].unsqueeze(1),ids,IDH,net1)
                test_acc_IDH_type0 += torch.eq(predict_y, IDH.float().to(device)).sum().item()

                _,_,predict_y,features_idh_type1 = get_feature(genetype,image_type[1],fold,images[:,1,:,:].unsqueeze(1),ids,IDH,net1)
                test_acc_IDH_type1 += torch.eq(predict_y, IDH.float().to(device)).sum().item()

                _,_,predict_y,features_idh_type2 = get_feature(genetype,image_type[2],fold,images[:,2,:,:].unsqueeze(1),ids,IDH,net1)
                test_acc_IDH_type2 += torch.eq(predict_y, IDH.float().to(device)).sum().item()

                
                test_num_IDH += len(IDH)
             


                # Append data to h5py datasets
                h5f['features_idh_type0'].resize((h5f['features_idh_type0'].shape[0] + features_idh_type0.shape[0]), axis=0)
                h5f['features_idh_type0'][-features_idh_type0.shape[0]:] = features_idh_type0

                h5f['features_idh_type1'].resize((h5f['features_idh_type1'].shape[0] + features_idh_type1.shape[0]), axis=0)
                h5f['features_idh_type1'][-features_idh_type1.shape[0]:] = features_idh_type1
                
                h5f['features_idh_type2'].resize((h5f['features_idh_type2'].shape[0] + features_idh_type2.shape[0]), axis=0)
                h5f['features_idh_type2'][-features_idh_type2.shape[0]:] = features_idh_type2

                h5f['labels_idh'].resize((h5f['labels_idh'].shape[0] + labels_idh.shape[0]), axis=0)
                h5f['labels_idh'][-labels_idh.shape[0]:] = labels_idh

                genetype = "p119q"
                _,labels_p119q,predict_y,features_p119q_type0 = get_feature(genetype,image_type[0],fold,images[:,0,:,:].unsqueeze(1),ids,p1_19q_,net1)
                predict_y_p119q = predict_y[~np.isin(p1_19q_, B)]
                p1_19qs = p1_19q_[~np.isin(p1_19q_, B)]
                if len(predict_y) > 1:
                    test_acc_p1_19q_type0 += torch.eq(predict_y_p119q, p1_19qs.float().to(device)).sum().item()
                   

                _,_,predict_y,features_p119q_type1 = get_feature(genetype,image_type[1],fold,images[:,1,:,:].unsqueeze(1),ids,p1_19q_,net1)
                predict_y_p119q = predict_y[~np.isin(p1_19q_, B)]
                p1_19qs = p1_19q_[~np.isin(p1_19q_, B)]
                if len(predict_y) > 1:
                    test_acc_p1_19q_type1 += torch.eq(predict_y_p119q, p1_19qs.float().to(device)).sum().item()
                    test_num_p1_19q += len(p1_19qs)
                
                _,_,predict_y,features_p119q_type2 = get_feature(genetype,image_type[2],fold,images[:,2,:,:].unsqueeze(1),ids,p1_19q_,net1)
                predict_y_p119q = predict_y[~np.isin(p1_19q_, B)]
                p1_19qs = p1_19q_[~np.isin(p1_19q_, B)]
                if len(predict_y) > 1:
                    test_acc_p1_19q_type2 += torch.eq(predict_y_p119q, p1_19qs.float().to(device)).sum().item()
                   
                #  Append data to h5py datasets
                h5f['features_p119q_type0'].resize((h5f['features_p119q_type0'].shape[0] + features_p119q_type0.shape[0]), axis=0)
                h5f['features_p119q_type0'][-features_p119q_type0.shape[0]:] = features_p119q_type0

                h5f['features_p119q_type1'].resize((h5f['features_p119q_type1'].shape[0] + features_p119q_type1.shape[0]), axis=0)
                h5f['features_p119q_type1'][-features_p119q_type1.shape[0]:] = features_p119q_type1
                
                h5f['features_p119q_type2'].resize((h5f['features_p119q_type2'].shape[0] + features_p119q_type2.shape[0]), axis=0)
                h5f['features_p119q_type2'][-features_p119q_type2.shape[0]:] = features_p119q_type2

                h5f['labels_p119q'].resize((h5f['labels_p119q'].shape[0] + labels_p119q.shape[0]), axis=0)
                h5f['labels_p119q'][-labels_p119q.shape[0]:] = labels_p119q
                
                h5f['ids'].resize((h5f['ids'].shape[0] + ids.numpy().shape[0]), axis=0)
                h5f['ids'][-ids.numpy().shape[0]:] = ids.numpy()           

    test_accurate_grade_type0 = test_acc_grade_type0 / test_num_grade
    test_accurate_grade_type1 = test_acc_grade_type1 / test_num_grade
    test_accurate_grade_type2 = test_acc_grade_type2 / test_num_grade
    test_accurate_IDH_type0 = test_acc_IDH_type0 / test_num_IDH
    test_accurate_IDH_type1 = test_acc_IDH_type1 / test_num_IDH
    test_accurate_IDH_type2 = test_acc_IDH_type2 / test_num_IDH
    gc.collect()

    if center != "upenn":
        test_accurate_1p_19q_type0 = test_acc_p1_19q_type0 / test_num_p1_19q
        test_accurate_1p_19q_type1 = test_acc_p1_19q_type1 / test_num_p1_19q
        test_accurate_1p_19q_type2 = test_acc_p1_19q_type2 / test_num_p1_19q
        print(f'........{center}: {image_type[0]} , test_accuracy_grade: {test_accurate_grade_type0:.3f}, test_accuracy_IDH:{test_accurate_IDH_type0:.3f}, test_accuracy_1p_19q: {test_accurate_1p_19q_type0:.3f}')
        print(f'      : {image_type[1]} , test_accuracy_grade: {test_accurate_grade_type1:.3f}, test_accuracy_IDH:{test_accurate_IDH_type1:.3f}, test_accuracy_1p_19q: {test_accurate_1p_19q_type1:.3f}')
        print(f'      : {image_type[2]} , test_accuracy_grade: {test_accurate_grade_type2:.3f}, test_accuracy_IDH:{test_accurate_IDH_type2:.3f}, test_accuracy_1p_19q: {test_accurate_1p_19q_type2:.3f}')
        return  None
    else:
        print(f'........{center}: {image_type[0]} , test_accuracy_grade: {test_accurate_grade_type0:.3f}, test_accuracy_IDH:{test_accurate_IDH_type0:.3f}')
        print(f'........{center}: {image_type[1]} , test_accuracy_grade: {test_accurate_grade_type1:.3f}, test_accuracy_IDH:{test_accurate_IDH_type1:.3f}')
        print(f'........{center}: {image_type[2]} , test_accuracy_grade: {test_accurate_grade_type2:.3f}, test_accuracy_IDH:{test_accurate_IDH_type2:.3f}')
        return None