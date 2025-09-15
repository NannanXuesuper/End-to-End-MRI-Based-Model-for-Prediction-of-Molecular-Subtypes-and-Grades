import os
import torch.optim as optim
import numpy as np
import time
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
from train_utils.train import validate_mil_one_epoch
from dataset.dataset import My_features_Dataset
from utils.calculate import calculate_metrics
from model.model import GatedAttention,fusionmodel

batch_size = 1
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
data_root = "/local/CTimages/brain/code_5mm/data/tumor_slices_2mm/zzu2/03_h5data/features/['T2', 'Flair', 'CET1']"

# fold = 3
for fold in (0,1,2,3,4):
    internal_test_image_paths = {
        "zzu1": os.path.join(data_root, f"features_fold{fold}_zzu1.h5")}
    prospective_test_image_paths = {
        "zzu3": os.path.join(data_root, f"features_fold{fold}_zzu3.h5")}
    external_test_image_paths ={
        "xy_gs_nm": os.path.join(data_root, f"features_fold{fold}_xy_gs_nm.h5"),
        "hb": os.path.join(data_root, f"features_fold{fold}_hb.h5"),
        "xj": os.path.join(data_root, f"features_fold{fold}_xj.h5"),
        # "tcga": os.path.join(data_root, f"features_fold{fold}_tcga.h5"),
        "upenn": os.path.join(data_root, f"features_fold{fold}_upenn.h5")
    }
    internal_test_loaders = {}
    for center, path in internal_test_image_paths.items():
        dataset = My_features_Dataset(path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        internal_test_loaders[center] = loader
        print(f"{center} dataloader is ready")
    prospective_test_loaders = {}
    for center, path in prospective_test_image_paths.items():
        dataset = My_features_Dataset(path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        prospective_test_loaders[center] = loader
        print(f"{center} dataloader is ready")

    external_test_loaders = {}
    for center, path in external_test_image_paths.items():
        dataset = My_features_Dataset(path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        external_test_loaders[center] = loader
        print(f"{center} dataloader is ready")
    train_image_path = os.path.join(data_root, f"features_fold{fold}_train.h5")
    train_dataset = My_features_Dataset(train_image_path)
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, shuffle=True,
                                                    num_workers=nw, pin_memory=True)


        

    train_dataset_selectfeature = My_features_Dataset(train_image_path,feature_forselection=True)

    train_loader_selectfeature = torch.utils.data.DataLoader(train_dataset_selectfeature,
                                                    batch_size=len(train_dataset_selectfeature), shuffle=False,
                                                num_workers=8)
    val_image_paths = {
        "val": os.path.join(data_root, f"features_fold{fold}_val.h5")}
    val_test_loaders = {}
    for center, path in val_image_paths.items():
        dataset = My_features_Dataset(path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
        val_test_loaders[center] = loader
        print(f"{center} dataloader is ready")
    device = "cuda:0"

    top_k = 512
    fusion_layres = 'concat'
    relu = True
    print(f"######################################Fold:{fold},top_k: {top_k},  relu:{relu}######################\n")
    net1 = GatedAttention(feat= top_k,relu=relu).to(device)
    params1 = [p for p in net1.parameters() if p.requires_grad]
    optimizer1 = optim.Adam(params1, lr=0.0001)
    net2 = GatedAttention(feat= top_k,relu=relu).to(device)
    params2 = [p for p in net2.parameters() if p.requires_grad]
    optimizer2 = optim.Adam(params2, lr=0.0001)
    net3 = GatedAttention(feat= top_k,relu=relu).to(device)
    params3 = [p for p in net3.parameters() if p.requires_grad]
    optimizer3 = optim.Adam(params3, lr=0.0001)

    net4 = GatedAttention(feat= top_k,relu=relu).to(device)
    params4 = [p for p in net4.parameters() if p.requires_grad]
    optimizer4 = optim.Adam(params4, lr=0.0001)
    net5 = GatedAttention(feat= top_k,relu=relu).to(device)
    params5 = [p for p in net5.parameters() if p.requires_grad]
    optimizer5 = optim.Adam(params5, lr=0.0001)
    net6 = GatedAttention(feat= top_k,relu=relu).to(device)
    params6 = [p for p in net6.parameters() if p.requires_grad]
    optimizer6 = optim.Adam(params6, lr=0.0001)

    net10 = GatedAttention(feat= top_k,relu=relu).to(device)
    params10 = [p for p in net10.parameters() if p.requires_grad]
    optimizer10 = optim.Adam(params10, lr=0.0001)
    net11 = GatedAttention(feat= top_k,relu=relu).to(device)
    params11 = [p for p in net11.parameters() if p.requires_grad]
    optimizer11 = optim.Adam(params11, lr=0.0001)
    net12 = GatedAttention(feat= top_k,relu=relu).to(device)
    params12 = [p for p in net12.parameters() if p.requires_grad]
    optimizer12 = optim.Adam(params12, lr=0.0001)

    net7 = fusionmodel(n_classes= 1,fusion_layres = fusion_layres,dropout=0.25,scale_dim1=8,gate_path=1,scale_dim2=8,gate_omic=1,skip=True,relu = relu,top_k =top_k).to(device)
    params7 = [p for p in net7.parameters() if p.requires_grad]
    optimizer7 = optim.Adam(params7, lr=0.0001)
    net8 = fusionmodel(n_classes= 1,fusion_layres = fusion_layres,dropout=0.25,scale_dim1=8,gate_path=1,scale_dim2=8,gate_omic=1,skip=True,relu = relu,top_k =top_k).to(device)
    params8 = [p for p in net8.parameters() if p.requires_grad]
    optimizer8 = optim.Adam(params8, lr=0.0001)
    net9 = fusionmodel(n_classes= 1,fusion_layres = fusion_layres,dropout=0.25,scale_dim1=8,gate_path=1,scale_dim2=8,gate_omic=1,skip=True,relu = relu,top_k =top_k).to(device)
    params9 = [p for p in net9.parameters() if p.requires_grad]
    optimizer9 = optim.Adam(params9, lr=0.0001)

    B = torch.tensor([2])
    loss_function = nn.BCEWithLogitsLoss()
    epochs = 50
    train_steps = len(train_loader)



    for epoch in range(epochs):
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        net5.train()
        net6.train()
        net7.train()
        net8.train()
        net9.train()
        net10.train()
        net11.train()
        net12.train()
        running_loss = 0.0
        train_acc_grade_type0 = 0.0
        train_acc_IDH_type0 = 0.0
        train_acc_p1_19q_type0= 0.0
        train_acc_grade_type1 = 0.0
        train_acc_IDH_type1 = 0.0
        train_acc_p1_19q_type1= 0.0
        train_acc_grade_type2 = 0.0
        train_acc_IDH_type2 = 0.0
        train_acc_p1_19q_type2= 0.0
        train_acc_grade_type3 = 0.0
        train_acc_IDH_type3 = 0.0
        train_acc_p1_19q_type3= 0.0
        train_num_grade = 0
        train_num_IDH = 0
        train_num_p1_19q = 0
        t1 = time.perf_counter()
        for train_data in train_loader:
                # images,id,IDH,p1_19q_,grade = test_data
                Grade_Features_type0,IDH_Features_type0,p1_19q_Features_type0_,Grade_Features_type1,IDH_Features_type1,p1_19q_Features_type1_,Grade_Features_type2,IDH_Features_type2,p1_19q_Features_type2_,id,IDH,p1_19q_,grade= train_data
                
                
                grade_features_type0 = torch.squeeze(Grade_Features_type0)
                optimizer1.zero_grad()
        
                logits_type0 = net1(grade_features_type0.to(device))
                logits_type0 = torch.squeeze(logits_type0)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net1.parameters())
                grade = torch.squeeze(grade)
                loss_grade_type0 = loss_function(logits_type0,grade.float().to(device))+0.01 * l2_norm
                loss_grade_type0.backward()
                optimizer1.step()
                
                predict_y_type0 = torch.ge(torch.sigmoid(logits_type0), 0.5).float()
                train_acc_grade_type0 += torch.eq(predict_y_type0, grade.float().to(device)).sum().item()

                grade_features_type1 = torch.squeeze(Grade_Features_type1)
                optimizer2.zero_grad()
                # grade_features_index = get_important_features(grade_features,genetype= "grade",labels = grade,top_k = top_k,fold= fold,device= device)
                #  7. 使用选出的特征进行训练
                # grade_features_type1 = grade_features_type1[:, grade_features_index_type1]
                # print(grade_features_type1.shape)
                logits_type1 = net2(grade_features_type1.to(device))
                logits_type1 = torch.squeeze(logits_type1)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net2.parameters())
                loss_grade_type1 = loss_function(logits_type1,grade.float().to(device))+0.01 * l2_norm
                loss_grade_type1.backward()
                optimizer2.step()
                predict_y_type1 = torch.ge(torch.sigmoid(logits_type1), 0.5).float()
                train_acc_grade_type1 += torch.eq(predict_y_type1, grade.float().to(device)).sum().item()


                grade_features_type2 = torch.squeeze(Grade_Features_type2)
                optimizer10.zero_grad()
                # grade_features_index = get_important_features(grade_features,genetype= "grade",labels = grade,top_k = top_k,fold= fold,device= device)
                #  7. 使用选出的特征进行训练
                # grade_features_type1 = grade_features_type1[:, grade_features_index_type1]
                # print(grade_features_type1.shape)
                logits_type2 = net10(grade_features_type2.to(device))
                logits_type2 = torch.squeeze(logits_type2)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net10.parameters())
                loss_grade_type2 = loss_function(logits_type2,grade.float().to(device))+0.01 * l2_norm
                loss_grade_type2.backward()
                optimizer10.step()
                predict_y_type2 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
                train_acc_grade_type2 += torch.eq(predict_y_type2, grade.float().to(device)).sum().item()
                # grade_features_type2 = torch.cat([grade_features_type0, grade_features_type1], axis=1)


                optimizer7.zero_grad()
                logits_type3 = net7(grade_features_type0.to(device), grade_features_type1.to(device), grade_features_type2.to(device))
                logits_type3 = torch.squeeze(logits_type3)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net7.parameters())
                loss_grade_type3 = loss_function(logits_type3,grade.float().to(device))+0.01 * l2_norm
                loss_grade_type3.backward()
                optimizer7.step()
                predict_y_type3 = torch.ge(torch.sigmoid(logits_type3), 0.5).float()
                train_acc_grade_type3 += torch.eq(predict_y_type3, grade.float().to(device)).sum().item()

                train_num_grade += 1
        

                IDH = torch.squeeze(IDH)
                IDH_features_type0 = torch.squeeze(IDH_Features_type0)
                optimizer3.zero_grad()
                # IDH_features_index = get_important_features(IDH_features,genetype= "IDH", labels = IDH ,top_k = top_k,fold= fold,device= device)
                # 7. 使用选出的特征进行训练
                # IDH_features_type0 = IDH_features_type0[:, IDH_features_index_type0]
                logits_type0 = net3(IDH_features_type0.to(device))
                logits_type0 = torch.squeeze(logits_type0)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net3.parameters())
                loss_IDH_type0 = loss_function(logits_type0,IDH.float().to(device))+0.01 * l2_norm
                loss_IDH_type0.backward()
                optimizer3.step()
                predict_y_type0 = torch.ge(torch.sigmoid(logits_type0), 0.5).float()
                train_acc_IDH_type0 += torch.eq(predict_y_type0, IDH.float().to(device)).sum().item()

                IDH_features_type1 = torch.squeeze(IDH_Features_type1)
                optimizer4.zero_grad()
                # IDH_features_index = get_important_features(IDH_features,genetype= "IDH", labels = IDH ,top_k = top_k,fold= fold,device= device)
                # 7. 使用选出的特征进行训练
                # IDH_features_type1 = IDH_features_type1[:, IDH_features_index_type1]
                logits_type1 = net4(IDH_features_type1.to(device))
                logits_type1 = torch.squeeze(logits_type1)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net4.parameters())
                loss_IDH_type1 = loss_function(logits_type1,IDH.float().to(device))+0.01 * l2_norm
                loss_IDH_type1.backward()
                optimizer4.step()
                predict_y_type1 = torch.ge(torch.sigmoid(logits_type1), 0.5).float()
                train_acc_IDH_type1 += torch.eq(predict_y_type1, IDH.float().to(device)).sum().item()

                IDH_features_type2 = torch.squeeze(IDH_Features_type2)
                optimizer11.zero_grad()
                # IDH_features_index = get_important_features(IDH_features,genetype= "IDH", labels = IDH ,top_k = top_k,fold= fold,device= device)
                # 7. 使用选出的特征进行训练
                # IDH_features_type1 = IDH_features_type1[:, IDH_features_index_type1]
                logits_type2 = net11(IDH_features_type2.to(device))
                logits_type2 = torch.squeeze(logits_type2)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net11.parameters())
                loss_IDH_type2 = loss_function(logits_type2,IDH.float().to(device))+0.01 * l2_norm
                loss_IDH_type2.backward()
                optimizer11.step()
                predict_y_type2 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
                train_acc_IDH_type2 += torch.eq(predict_y_type2, IDH.float().to(device)).sum().item()
                # IDH_features_type2 = torch.cat([IDH_features_type0, IDH_features_type1], axis=1)
                optimizer8.zero_grad()
                # IDH_features_index = get_important_features(IDH_features,genetype= "IDH", labels = IDH ,top_k = top_k,fold= fold,device= device)
                # 7. 使用选出的特征进行训练
                # IDH_features = IDH_features[:, IDH_features_index]
                logits_type3 = net8(IDH_features_type0.to(device), IDH_features_type1.to(device),IDH_features_type2.to(device))
                logits_type3 = torch.squeeze(logits_type3)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                l2_norm = sum(p.pow(2.0).sum() for p in net8.parameters())
                loss_IDH_type3 = loss_function(logits_type3,IDH.float().to(device))+0.01 * l2_norm
                loss_IDH_type3.backward()
                optimizer8.step()
                predict_y_type3 = torch.ge(torch.sigmoid(logits_type3), 0.5).float()
                train_acc_IDH_type3 += torch.eq(predict_y_type3, IDH.float().to(device)).sum().item()
                train_num_IDH += 1
            

                p1_19qs = torch.squeeze(p1_19q_[~np.isin(p1_19q_, B)])
                p1_19qs_features_type0 = torch.squeeze(p1_19q_Features_type0_[~np.isin(p1_19q_, B)])
                p1_19qs_features_type1 = torch.squeeze(p1_19q_Features_type1_[~np.isin(p1_19q_, B)])
                p1_19qs_features_type2 = torch.squeeze(p1_19q_Features_type2_[~np.isin(p1_19q_, B)])
                
                if p1_19q_ != B :
                    optimizer5.zero_grad()
                    # p1_19qs_features_index = get_important_features(p1_19qs_features,genetype= "p119q",labels = p1_19qs,top_k = top_k,fold= fold,device= device)
                    # p1_19qs_features_type0 = p1_19qs_features_type0[:, p1_19qs_features_index_type0]
                    logits_type0 = net5(p1_19qs_features_type0.to(device))
                    logits_type0 = torch.squeeze(logits_type0)
                    # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                    l2_norm = sum(p.pow(2.0).sum() for p in net5.parameters())
                    loss_p1_19q_type0 = loss_function(logits_type0,p1_19qs.float().to(device))+0.01 * l2_norm
                    loss_p1_19q_type0.backward()
                    optimizer5.step()
                    predict_y_type0 = torch.ge(torch.sigmoid(logits_type0), 0.5).float()
                    train_acc_p1_19q_type0 += torch.eq(predict_y_type0, p1_19qs.float().to(device)).sum().item()

                    optimizer6.zero_grad()
                    # p1_19qs_features_index = get_important_features(p1_19qs_features,genetype= "p119q",labels = p1_19qs,top_k = top_k,fold= fold,device= device)
                    # p1_19qs_features_type1 = p1_19qs_features_type1[:, p1_19qs_features_index_type1]
                    logits_type1 = net6(p1_19qs_features_type1.to(device))
                    logits_type1 = torch.squeeze(logits_type1)
                    # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                    l2_norm = sum(p.pow(2.0).sum() for p in net6.parameters())
                    loss_p1_19q_type1 = loss_function(logits_type1,p1_19qs.float().to(device))+0.01 * l2_norm
                    loss_p1_19q_type1.backward()
                    optimizer6.step()
                    predict_y_type1 = torch.ge(torch.sigmoid(logits_type1), 0.5).float()
                    train_acc_p1_19q_type1 += torch.eq(predict_y_type1, p1_19qs.float().to(device)).sum().item()

                    optimizer12.zero_grad()
                    # p1_19qs_features_index = get_important_features(p1_19qs_features,genetype= "p119q",labels = p1_19qs,top_k = top_k,fold= fold,device= device)
                    # p1_19qs_features_type1 = p1_19qs_features_type1[:, p1_19qs_features_index_type1]
                    logits_type2 = net12(p1_19qs_features_type2.to(device))
                    logits_type2 = torch.squeeze(logits_type2)
                    # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                    l2_norm = sum(p.pow(2.0).sum() for p in net12.parameters())
                    loss_p1_19q_type2 = loss_function(logits_type2,p1_19qs.float().to(device))+0.01 * l2_norm
                    loss_p1_19q_type2.backward()
                    optimizer12.step()
                    predict_y_type2 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
                    train_acc_p1_19q_type2 += torch.eq(predict_y_type2, p1_19qs.float().to(device)).sum().item()

                    # p1_19qs_features_type2 = torch.cat([p1_19qs_features_type0, p1_19qs_features_type1], axis=1)
                    optimizer9.zero_grad()
                    # p1_19qs_features_index = get_important_features(p1_19qs_features,genetype= "p119q",labels = p1_19qs,top_k = top_k,fold= fold,device= device)
                    # p1_19qs_features = p1_19qs_features[:, p1_19qs_features_index]
                    logits_type3 = net9(p1_19qs_features_type0.to(device), p1_19qs_features_type1.to(device), p1_19qs_features_type2.to(device))
                    logits_type3 = torch.squeeze(logits_type3)
                    # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                    l2_norm = sum(p.pow(2.0).sum() for p in net9.parameters())
                    loss_p1_19q_type3 = loss_function(logits_type3,p1_19qs.float().to(device))+0.01 * l2_norm
                    loss_p1_19q_type3.backward()
                    optimizer9.step()
                    predict_y_type3 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
                    train_acc_p1_19q_type3 += torch.eq(predict_y_type2, p1_19qs.float().to(device)).sum().item()
                    train_num_p1_19q += 1
                
                else:
                    loss_p1_19q = torch.tensor([0])
                    
            
        # print(train_acc_grade_type0,train_num_grade)
        train_accurate_grade_type0 = train_acc_grade_type0 / train_num_grade
        train_accurate_IDH_type0 = train_acc_IDH_type0 / train_num_IDH
        train_accurate_1p_19q_type0 = train_acc_p1_19q_type0 / train_num_p1_19q

        train_accurate_grade_type1 = train_acc_grade_type1 / train_num_grade
        train_accurate_IDH_type1 = train_acc_IDH_type1 / train_num_IDH
        train_accurate_1p_19q_type1 = train_acc_p1_19q_type1 / train_num_p1_19q

        train_accurate_grade_type2 = train_acc_grade_type2 / train_num_grade
        train_accurate_IDH_type2 = train_acc_IDH_type2 / train_num_IDH
        train_accurate_1p_19q_type2 = train_acc_p1_19q_type2 / train_num_p1_19q

        train_accurate_grade_type3 = train_acc_grade_type3 / train_num_grade
        train_accurate_IDH_type3 = train_acc_IDH_type3 / train_num_IDH
        train_accurate_1p_19q_type3 = train_acc_p1_19q_type3 / train_num_p1_19q

        print('[epoch %d] T2: train_accuracy_grade: %.3f,train_accuracy_IDH: %.3f,train_accuracy_1p_19q: %.3f' %
                (epoch + 1,train_accurate_grade_type0, train_accurate_IDH_type0,train_accurate_1p_19q_type0))
        print('[epoch %d] Flair: train_accuracy_grade: %.3f,train_accuracy_IDH: %.3f,train_accuracy_1p_19q: %.3f' %
                (epoch + 1, train_accurate_grade_type1, train_accurate_IDH_type1,train_accurate_1p_19q_type1))
        print('[epoch %d] CET1: train_accuracy_grade: %.3f,train_accuracy_IDH: %.3f,train_accuracy_1p_19q: %.3f' %
                (epoch + 1, train_accurate_grade_type2, train_accurate_IDH_type2,train_accurate_1p_19q_type2))
        print('[epoch %d] Combine: train_accuracy_grade: %.3f,train_accuracy_IDH: %.3f,train_accuracy_1p_19q: %.3f' %
                (epoch + 1, train_accurate_grade_type3, train_accurate_IDH_type3,train_accurate_1p_19q_type3))
        val_accurate_grade_type2, val_accurate_IDH_type2, val_accurate_1p_19q_type2, val_grade_labels, val_grade_preds0, val_grade_probs0,val_grade_preds1, val_grade_probs1,val_grade_preds2, val_grade_probs2,val_grade_preds3, val_grade_probs3, val_IDH_labels, val_IDH_preds0, val_IDH_probs0,val_IDH_preds1, val_IDH_probs1,val_IDH_preds2, val_IDH_probs2,val_IDH_preds3, val_IDH_probs3, val_p1_19q_labels, val_p1_19q_preds0, val_p1_19q_probs0, val_p1_19q_preds1, val_p1_19q_probs1, val_p1_19q_preds2, val_p1_19q_probs2, val_p1_19q_preds3, val_p1_19q_probs3= validate_mil_one_epoch(test_loader=val_test_loaders,center= "validation",net1=net1,net2=net2,net3=net3,net4=net4,net5=net5,net6=net6,net7=net7,net8=net8,net9=net9,net10=net10,net11=net11,net12=net12,device= device)
        # print("\n··················································································································")
        # for center, loader in test_loaders[:2].items():
        intest_accurate_grade_type2, intest_accurate_IDH_type2, intest_accurate_1p_19q_type2, int_grade_labels, int_grade_preds0, int_grade_probs0,int_grade_preds1, int_grade_probs1,int_grade_preds2, int_grade_probs2, int_grade_preds3, int_grade_probs3, int_IDH_labels, int_IDH_preds0, int_IDH_probs0,int_IDH_preds1, int_IDH_probs1,int_IDH_preds2, int_IDH_probs2,int_IDH_preds3, int_IDH_probs3, int_p1_19q_labels, int_p1_19q_preds0, int_p1_19q_probs0, int_p1_19q_preds1, int_p1_19q_probs1, int_p1_19q_preds2, int_p1_19q_probs2, int_p1_19q_preds3, int_p1_19q_probs3 = validate_mil_one_epoch(test_loader=internal_test_loaders, center="internal", net1=net1, net2=net2, net3=net3, net4=net4, net5=net5, net6=net6, net7=net7, net8=net8, net9=net9, net10=net10,net11=net11,net12=net12,device=device)
        extest_accurate_grade_type2, extest_accurate_IDH_type2, extest_accurate_1p_19q_type2,  ext_grade_labels, ext_grade_preds0, ext_grade_probs0,ext_grade_preds1, ext_grade_probs1,ext_grade_preds2, ext_grade_probs2,ext_grade_preds3, ext_grade_probs3, ext_IDH_labels, ext_IDH_preds0, ext_IDH_probs0,ext_IDH_preds1, ext_IDH_probs1,ext_IDH_preds2, ext_IDH_probs2,ext_IDH_preds3, ext_IDH_probs3, ext_p1_19q_labels, ext_p1_19q_preds0, ext_p1_19q_probs0, ext_p1_19q_preds1, ext_p1_19q_probs1, ext_p1_19q_preds2, ext_p1_19q_probs2,ext_p1_19q_preds3, ext_p1_19q_probs3 = validate_mil_one_epoch(test_loader=external_test_loaders, center="external", net1=net1, net2=net2, net3=net3, net4=net4, net5=net5, net6=net6, net7=net7, net8=net8, net9=net9, net10=net10,net11=net11,net12=net12,device=device)
        protest_accurate_grade_type2, protest_accurate_IDH_type2, protest_accurate_1p_19q_type2,  pro_grade_labels, pro_grade_preds0, pro_grade_probs0,pro_grade_preds1, pro_grade_probs1,pro_grade_preds2, pro_grade_probs2,pro_grade_preds3, pro_grade_probs3, pro_IDH_labels, pro_IDH_preds0, pro_IDH_probs0,pro_IDH_preds1, pro_IDH_probs1,pro_IDH_preds2, pro_IDH_probs2,pro_IDH_preds3, pro_IDH_probs3, pro_p1_19q_labels, pro_p1_19q_preds0, pro_p1_19q_probs0, pro_p1_19q_preds1, pro_p1_19q_probs1, pro_p1_19q_preds2, pro_p1_19q_probs2 ,pro_p1_19q_preds3, pro_p1_19q_probs3 = validate_mil_one_epoch(test_loader=prospective_test_loaders, center="prospective", net1=net1, net2=net2, net3=net3, net4=net4, net5=net5, net6=net6, net7=net7, net8=net8, net9=net9, net10=net10,net11=net11,net12=net12, device=device)
            # print("\n··················································································································")
        grade_path = os.path.join(data_root, f"save_{fusion_layres}_relu{relu}/fold{fold}/grade")
        if not os.path.exists(grade_path):
            os.makedirs(grade_path)
        IDH_path = os.path.join(data_root, f"save_{fusion_layres}_relu{relu}/fold{fold}/IDH")
        if not os.path.exists(IDH_path):
            os.makedirs(IDH_path)
        
        if val_accurate_grade_type2 >0.795 and intest_accurate_grade_type2>0.798 and extest_accurate_grade_type2>0.8 and protest_accurate_grade_type2>0.8:
            print("save grade weight")
            name = f'Epoch{epoch + 1}_resNet34_pre_Grade_val{round(val_accurate_grade_type2, 3)}_internal{round(intest_accurate_grade_type2, 3)}_external{round(extest_accurate_grade_type2, 3)}_prospective{round(protest_accurate_grade_type2, 3)}.pth'
            save_path = os.path.join(grade_path, name)
            torch.save(net7.state_dict(), save_path)
            for k in (0,1,2,3):
                if k ==0:
                    print("T2 performance")
                    val_grade_preds, val_grade_probs = val_grade_preds0, val_grade_probs0
                    int_grade_preds, int_grade_probs = int_grade_preds0, int_grade_probs0
                    ext_grade_preds, ext_grade_probs = ext_grade_preds0, ext_grade_probs0
                    pro_grade_preds, pro_grade_probs = pro_grade_preds0, pro_grade_probs0
                elif k ==1:
                    print("Flair performance")
                    val_grade_preds, val_grade_probs = val_grade_preds1, val_grade_probs1
                    int_grade_preds, int_grade_probs = int_grade_preds1, int_grade_probs1
                    ext_grade_preds, ext_grade_probs = ext_grade_preds1, ext_grade_probs1
                    pro_grade_preds, pro_grade_probs = pro_grade_preds1, pro_grade_probs1
                elif k ==2:
                    print("CET1 performance")
                    val_grade_preds, val_grade_probs = val_grade_preds2, val_grade_probs2
                    int_grade_preds, int_grade_probs = int_grade_preds2, int_grade_probs2
                    ext_grade_preds, ext_grade_probs = ext_grade_preds2, ext_grade_probs2
                    pro_grade_preds, pro_grade_probs = pro_grade_preds2, pro_grade_probs2
                elif k ==3:
                    print("Combine performance")
                    val_grade_preds, val_grade_probs = val_grade_preds3, val_grade_probs3
                    int_grade_preds, int_grade_probs = int_grade_preds3, int_grade_probs3
                    ext_grade_preds, ext_grade_probs = ext_grade_preds3, ext_grade_probs3
                    pro_grade_preds, pro_grade_probs = pro_grade_preds3, pro_grade_probs3
                
                val_metrics = calculate_metrics(val_grade_labels, val_grade_preds, val_grade_probs)
                int_metrics = calculate_metrics(int_grade_labels, int_grade_preds, int_grade_probs)
                ext_metrics = calculate_metrics(ext_grade_labels, ext_grade_preds, ext_grade_probs)
                pro_metrics = calculate_metrics(pro_grade_labels, pro_grade_preds, pro_grade_probs)
                print(f'val Grade Metrics: Accuracy: {val_metrics[0]:.3f}, AUC: {val_metrics[1]:.3f}, Specificity: {val_metrics[2]:.3f}, Sensitivity: {val_metrics[3]:.3f}, F1-Score: {val_metrics[4]:.3f}')
                print(f'Internal Grade Metrics: Accuracy: {int_metrics[0]:.3f}, AUC: {int_metrics[1]:.3f}, Specificity: {int_metrics[2]:.3f}, Sensitivity: {int_metrics[3]:.3f}, F1-Score: {int_metrics[4]:.3f}')
                print(f'External Grade Metrics: Accuracy: {ext_metrics[0]:.3f}, AUC: {ext_metrics[1]:.3f}, Specificity: {ext_metrics[2]:.3f}, Sensitivity: {ext_metrics[3]:.3f}, F1-Score: {ext_metrics[4]:.3f}')
                print(f'Prospective Grade Metrics: Accuracy: {pro_metrics[0]:.3f}, AUC: {pro_metrics[1]:.3f}, Specificity: {pro_metrics[2]:.3f}, Sensitivity: {pro_metrics[3]:.3f}, F1-Score: {pro_metrics[4]:.3f}')

        if val_accurate_IDH_type2 > 0.83:
            print("save idh weight")
            name = f'Epoch{epoch + 1}_resNet34_pre_IDH_val{round(val_accurate_IDH_type2, 3)}_internal{round(intest_accurate_IDH_type2, 3)}_external{round(extest_accurate_IDH_type2, 3)}_prospective{round(protest_accurate_IDH_type2, 3)}.pth'
            save_path = os.path.join(IDH_path, name)
            torch.save(net8.state_dict(), save_path)
            for k in (0,1,2,3):
                if k ==0:
                    print("T2 performance")
                    val_IDH_preds, val_IDH_probs = val_IDH_preds0, val_IDH_probs0
                    int_IDH_preds, int_IDH_probs = int_IDH_preds0, int_IDH_probs0
                    ext_IDH_preds, ext_IDH_probs = ext_IDH_preds0, ext_IDH_probs0
                    pro_IDH_preds, pro_IDH_probs = pro_IDH_preds0, pro_IDH_probs0
                elif k ==1:
                    print("Flair performance")
                    val_IDH_preds, val_IDH_probs = val_IDH_preds1, val_IDH_probs1
                    int_IDH_preds, int_IDH_probs = int_IDH_preds1, int_IDH_probs1
                    ext_IDH_preds, ext_IDH_probs = ext_IDH_preds1, ext_IDH_probs1
                    pro_IDH_preds, pro_IDH_probs = pro_IDH_preds1, pro_IDH_probs1
                elif k ==2:
                    print("CET1 performance")
                    val_IDH_preds, val_IDH_probs = val_IDH_preds2, val_IDH_probs2
                    int_IDH_preds, int_IDH_probs = int_IDH_preds2, int_IDH_probs2
                    ext_IDH_preds, ext_IDH_probs = ext_IDH_preds2, ext_IDH_probs2
                    pro_IDH_preds, pro_IDH_probs = pro_IDH_preds2, pro_IDH_probs2 
                elif k ==3:
                    print("Combined performance")
                    val_IDH_preds, val_IDH_probs = val_IDH_preds3, val_IDH_probs3
                    int_IDH_preds, int_IDH_probs = int_IDH_preds3, int_IDH_probs3
                    ext_IDH_preds, ext_IDH_probs = ext_IDH_preds3, ext_IDH_probs3
                    pro_IDH_preds, pro_IDH_probs = pro_IDH_preds3, pro_IDH_probs3 
                val_metrics = calculate_metrics(val_IDH_labels, val_IDH_preds, val_IDH_probs)
                int_metrics = calculate_metrics(int_IDH_labels, int_IDH_preds, int_IDH_probs)
                ext_metrics = calculate_metrics(ext_IDH_labels, ext_IDH_preds, ext_IDH_probs)
                pro_metrics = calculate_metrics(pro_IDH_labels, pro_IDH_preds, pro_IDH_probs)
                print(f'val IDH Metrics: Accuracy: {val_metrics[0]:.3f}, AUC: {val_metrics[1]:.3f}, Specificity: {val_metrics[2]:.3f}, Sensitivity: {val_metrics[3]:.3f}, F1-Score: {val_metrics[4]:.3f}')
                print(f'Internal IDH Metrics: Accuracy: {int_metrics[0]:.3f}, AUC: {int_metrics[1]:.3f}, Specificity: {int_metrics[2]:.3f}, Sensitivity: {int_metrics[3]:.3f}, F1-Score: {int_metrics[4]:.3f}')
                print(f'External IDH Metrics: Accuracy: {ext_metrics[0]:.3f}, AUC: {ext_metrics[1]:.3f}, Specificity: {ext_metrics[2]:.3f}, Sensitivity: {ext_metrics[3]:.3f}, F1-Score: {ext_metrics[4]:.3f}')
                print(f'Prospective IDH Metrics: Accuracy: {pro_metrics[0]:.3f}, AUC: {pro_metrics[1]:.3f}, Specificity: {pro_metrics[2]:.3f}, Sensitivity: {pro_metrics[3]:.3f}, F1-Score: {pro_metrics[4]:.3f}')

        p119q_path = os.path.join(data_root, f"save_{fusion_layres}_relu{relu}/fold{fold}/p119q")
        if not os.path.exists(p119q_path):
            os.makedirs(p119q_path)
        if protest_accurate_1p_19q_type2 > 0.80:
            print("save 1p 19q weight")
            name = f'Epoch{epoch + 1}_resNet34_pre_1p19q_val{round(val_accurate_1p_19q_type2, 3)}_internal{round(intest_accurate_1p_19q_type2, 3)}_external{round(extest_accurate_1p_19q_type2, 3)}_prospective{round(protest_accurate_1p_19q_type2, 3)}.pth'
            save_path = os.path.join(p119q_path, name)
            torch.save(net9.state_dict(), save_path)
            for k in (0,1,2,3):
                if k ==0:
                    print("T2 performance.................................")
                    val_p1_19q_preds, val_p1_19q_probs = val_p1_19q_preds0, val_p1_19q_probs0
                    int_p1_19q_preds, int_p1_19q_probs = int_p1_19q_preds0, int_p1_19q_probs0
                    ext_p1_19q_preds, ext_p1_19q_probs = ext_p1_19q_preds0, ext_p1_19q_probs0
                    pro_p1_19q_preds, pro_p1_19q_probs = pro_p1_19q_preds0, pro_p1_19q_probs0
                elif k ==1:
                    print("Flair performance..............................")
                    val_p1_19q_preds, val_p1_19q_probs = val_p1_19q_preds1, val_p1_19q_probs1
                    int_p1_19q_preds, int_p1_19q_probs = int_p1_19q_preds1, int_p1_19q_probs1
                    ext_p1_19q_preds, ext_p1_19q_probs = ext_p1_19q_preds1, ext_p1_19q_probs1
                    pro_p1_19q_preds, pro_p1_19q_probs = pro_p1_19q_preds1, pro_p1_19q_probs1
                elif k ==2:
                    print("CET1 performance...........................")
                    val_p1_19q_preds, val_p1_19q_probs = val_p1_19q_preds2, val_p1_19q_probs2
                    int_p1_19q_preds, int_p1_19q_probs = int_p1_19q_preds2, int_p1_19q_probs2
                    ext_p1_19q_preds, ext_p1_19q_probs = ext_p1_19q_preds2, ext_p1_19q_probs2
                    pro_p1_19q_preds, pro_p1_19q_probs = pro_p1_19q_preds2, pro_p1_19q_probs2 
                elif k ==3:
                    print("Combined performance...........................")
                    val_p1_19q_preds, val_p1_19q_probs = val_p1_19q_preds3, val_p1_19q_probs3
                    int_p1_19q_preds, int_p1_19q_probs = int_p1_19q_preds3, int_p1_19q_probs3
                    ext_p1_19q_preds, ext_p1_19q_probs = ext_p1_19q_preds3, ext_p1_19q_probs3
                    pro_p1_19q_preds, pro_p1_19q_probs = pro_p1_19q_preds3, pro_p1_19q_probs3 
            
                val_metrics = calculate_metrics(val_p1_19q_labels, val_p1_19q_preds, val_p1_19q_probs)
                int_metrics = calculate_metrics(int_p1_19q_labels, int_p1_19q_preds, int_p1_19q_probs)
                ext_metrics = calculate_metrics(ext_p1_19q_labels, ext_p1_19q_preds, ext_p1_19q_probs)
                pro_metrics = calculate_metrics(pro_p1_19q_labels, pro_p1_19q_preds, pro_p1_19q_probs)
                print(f'val 1p_19q Metrics: Accuracy: {val_metrics[0]:.3f}, AUC: {val_metrics[1]:.3f}, Specificity: {val_metrics[2]:.3f}, Sensitivity: {val_metrics[3]:.3f}, F1-Score: {val_metrics[4]:.3f}')
                print(f'Internal 1p_19q Metrics: Accuracy: {int_metrics[0]:.3f}, AUC: {int_metrics[1]:.3f}, Specificity: {int_metrics[2]:.3f}, Sensitivity: {int_metrics[3]:.3f}, F1-Score: {int_metrics[4]:.3f}')
                print(f'External 1p_19q Metrics: Accuracy: {ext_metrics[0]:.3f}, AUC: {ext_metrics[1]:.3f}, Specificity: {ext_metrics[2]:.3f}, Sensitivity: {ext_metrics[3]:.3f}, F1-Score: {ext_metrics[4]:.3f}')
                print(f'Prospective 1p_19q Metrics: Accuracy: {pro_metrics[0]:.3f}, AUC: {pro_metrics[1]:.3f}, Specificity: {pro_metrics[2]:.3f}, Sensitivity: {pro_metrics[3]:.3f}, F1-Score: {pro_metrics[4]:.3f}')
        print("\n··················································································································")
        print(time.perf_counter()-t1)