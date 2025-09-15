import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torch
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import warnings
from torch.autograd import Variable
def validate_one_epoch(test_loader,center,net1,net2,net3,device): 
    net1.eval()
    net2.eval()
    net3.eval()
    B = torch.tensor([2])
    with torch.no_grad():
        test_acc_grade = 0
        test_acc_IDH= 0
        test_acc_p1_19q = 0
        test_num_grade = 0
        test_num_IDH = 0
        test_num_p1_19q = 0
        # val_bar = tqdm(validate_loader, file=sys.stdout)
        for test_data in test_loader:
            images,id,IDH,p1_19q_,grade = test_data
    
            _,logits = net1(images.to(device))
            logits = torch.squeeze(logits)
            predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
            test_acc_grade += torch.eq(predict_y, grade.float().to(device)).sum().item()
            test_num_grade += len(grade)

            
            _,logits = net2(images.to(device))
            logits = torch.squeeze(logits)
            predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
            
            test_acc_IDH += torch.eq(predict_y, IDH.float().to(device)).sum().item()
            test_num_IDH += len(IDH)


            p1_19qs = p1_19q_[~np.isin(p1_19q_, B)]
            p1_19qs_images = images[~np.isin(p1_19q_, B)]
            if len(p1_19qs_images) >1:
                _,logits = net3(p1_19qs_images.to(device))
                logits = torch.squeeze(logits)
                predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
                test_acc_p1_19q += torch.eq(predict_y, p1_19qs.float().to(device)).sum().item()
                test_num_p1_19q += len(p1_19qs)
            else:
                continue
            # del test_data, logits, images, p1_19qs_images

    test_accurate_grade = test_acc_grade / test_num_grade
    test_accurate_IDH = test_acc_IDH / test_num_IDH
    gc.collect()
    if center != "upenn":
        test_accurate_1p_19q = test_acc_p1_19q / test_num_p1_19q
        print(f'........{center}:  , test_accuracy_grade: {test_accurate_grade:.3f},test_accuracy_IDH:{test_accurate_IDH:.3f},test_accuracy_1p_19q: {test_accurate_1p_19q:.3f}')
        return test_accurate_grade, test_accurate_IDH,test_accurate_1p_19q
    else:
        
        print(f'........{center}:  , test_accuracy_grade: {test_accurate_grade:.3f},test_accuracy_IDH:{test_accurate_IDH:.3f}')
        return test_accurate_grade, test_accurate_IDH


def training_onefold(train_loader,validate_loader,test_loaders,image_type,fold,weight_path):
    print("###############################################using {} images Fold{} to train######################################".format(image_type,fold))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    loss_function = nn.BCEWithLogitsLoss()
    epochs = 100
    B = torch.tensor([2])
    norm=nn.BatchNorm2d
    resnet = models.resnet34(pretrained='resnet34', norm_layer=norm)
    for param in resnet.parameters():
        param.requires_grad = True
    resnet.fc = nn.Identity()
    if len(image_type) ==1:
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif image_type == "T2_Flair":
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif image_type == "T2_CET1_Flair":
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_feats = 512      
    num_classes = 1
    net1 = IClassifier(resnet, num_feats, output_class=num_classes).cuda()
     # construct an optimizer
    params1 = [p for p in net1.parameters() if p.requires_grad]
    optimizer1 = optim.Adam(params1, lr=0.0001)

    print(net1)
   
    net2 = IClassifier(resnet, num_feats, output_class=num_classes).cuda()
     # construct an optimizer
    params2 = [p for p in net2.parameters() if p.requires_grad]
    optimizer2 = optim.Adam(params2, lr=0.0001)

    net3 = IClassifier(resnet, num_feats, output_class=num_classes).cuda()
     # construct an optimizer
    params3 = [p for p in net3.parameters() if p.requires_grad]
    optimizer3 = optim.Adam(params3, lr=0.0001)
    # Train
    train_steps = len(train_loader)
  
    for epoch in range(epochs):
    
        # train
        t1 = time.perf_counter()
        net1.train()
        net2.train()
        net3.train()
        running_loss = 0.0
        train_acc_grade = 0.0
        train_acc_IDH = 0.0
        train_acc_p1_19q= 0.0
        train_num_grade = 0
        train_num_IDH = 0
        train_num_p1_19q = 0
        # train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            images,id,IDH,p1_19q_,grade = data
            
            optimizer1.zero_grad()
            _,logits = net1(images.to(device))
            logits = torch.squeeze(logits)
            loss_grade = loss_function(logits,grade.float().to(device))
            loss_grade.backward()
            optimizer1.step()
            predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
            train_acc_grade += torch.eq(predict_y, grade.float().to(device)).sum().item()
            train_num_grade += len(grade)

            optimizer2.zero_grad()
            _,logits = net2(images.to(device))
            logits = torch.squeeze(logits)
            loss_IDH = loss_function(logits,IDH.float().to(device))
            loss_IDH.backward()
            optimizer2.step()
            predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
            train_acc_IDH += torch.eq(predict_y, IDH.float().to(device)).sum().item()
            train_num_IDH += len(IDH)


            p1_19qs = p1_19q_[~np.isin(p1_19q_, B)]
            p1_19qs_images = images[~np.isin(p1_19q_, B)]
            if len(p1_19qs_images) >1:
                optimizer3.zero_grad()
                _,logits = net3(p1_19qs_images.to(device))
                logits = torch.squeeze(logits)
                # print("logist:",logits.shape,"p1_19qs:",p1_19qs.shape)
                loss_p1_19q = loss_function(logits,p1_19qs.float().to(device))
                loss_p1_19q.backward()
                optimizer3.step()
                predict_y = torch.ge(torch.sigmoid(logits), 0.5).float()
                train_acc_p1_19q += torch.eq(predict_y, p1_19qs.float().to(device)).sum().item()
                train_num_p1_19q += len(p1_19qs)
            else:
                continue
            # print("loss grade,idh,1p 19q:",loss_grade.item(),loss_IDH.item(),loss_p1_19q.item())
            # del data, logits, images, p1_19qs_images
            loss = loss_grade.item()+loss_IDH.item()+loss_p1_19q.item()
            running_loss += loss
            torch.cuda.empty_cache()
            gc.collect()
            

            
     

        train_accurate_grade = train_acc_grade / train_num_grade
        train_accurate_IDH = train_acc_IDH / train_num_IDH
        train_accurate_1p_19q = train_acc_p1_19q / train_num_p1_19q
        print('[epoch %d] train_loss: %.3f,train_accuracy_grade: %.3f,train_accuracy_IDH: %.3f,train_accuracy_1p_19q: %.3f' %
                (epoch + 1, running_loss / train_steps,train_accurate_grade, train_accurate_IDH,train_accurate_1p_19q))

        val_accurate_grade, val_accurate_IDH,val_accurate_1p_19q = validate_one_epoch(test_loader=validate_loader,center="val",net1=net1,net2=net2,net3=net3,device= device)
        
        results = {}
        for center, loader in test_loaders.items():
            results[center] = validate_one_epoch(test_loader=loader, center=center, net1=net1, net2=net2, net3=net3, device=device)
        print(time.perf_counter()-t1)
        grade_path = os.path.join(weight_path, "Grade", str(image_type), "save_fold" + str(fold),"best_weights")
        if not os.path.exists(grade_path):
            os.makedirs(grade_path)
        IDH_path = os.path.join(weight_path, "IDH", str(image_type), "save_fold" + str(fold),"best_weights")
        if not os.path.exists(IDH_path):
            os.makedirs(IDH_path)
        
        # Save the best weights for grade
        if 'best_val_accurate_grade' not in locals():
            best_val_accurate_grade = 0
        if val_accurate_grade > best_val_accurate_grade:
            best_val_accurate_grade = val_accurate_grade
            print("save best grade weight")
            name = f'Epoch{epoch + 1}_resNet34_pre_Grade_val{round(val_accurate_grade, 3)}_zzu1{round(results["zzu1"][0], 3)}_zzu3{round(results["zzu3"][0], 3)}_xygsnm{round(results["xy_gs_nm"][0], 3)}_hb{round(results["hb"][0], 3)}_xj{round(results["xj"][0], 3)}_TCGA{round(results["tcga"][0], 3)}_upenn{round(results["upenn"][0], 3)}.pth'
            save_path = os.path.join(grade_path, name)
            torch.save(net1.state_dict(), save_path)

        # Save the best weights for IDH
        if 'best_val_accurate_IDH' not in locals():
            best_val_accurate_IDH = 0
        if val_accurate_IDH > best_val_accurate_IDH:
            best_val_accurate_IDH = val_accurate_IDH
            print("save best IDH weight")
            name = f'Epoch{epoch + 1}_resNet34_pre_IDH_val{round(val_accurate_IDH, 3)}_zzu1{round(results["zzu1"][1], 3)}_zzu3{round(results["zzu3"][1], 3)}_xygsnm{round(results["xy_gs_nm"][1], 3)}_hb{round(results["hb"][1], 3)}_xj{round(results["xj"][1], 3)}_TCGA{round(results["tcga"][1], 3)}_upenn{round(results["upenn"][1], 3)}.pth'
            save_path = os.path.join(IDH_path, name)
            torch.save(net2.state_dict(), save_path)

        # Save the best weights for 1p19q
        p119q_path = os.path.join(weight_path, "p119q", str(image_type), "save_fold" + str(fold),"best_weights")
        if not os.path.exists(p119q_path):
            os.makedirs(p119q_path)
        if 'best_val_accurate_1p_19q' not in locals():
            best_val_accurate_1p_19q = 0
        if val_accurate_1p_19q > best_val_accurate_1p_19q:
            best_val_accurate_1p_19q = val_accurate_1p_19q
            print("save best 1p19q weight")
            name = f'Epoch{epoch + 1}_resNet34_pre_1p19q_val{round(val_accurate_1p_19q, 3)}_zzu1{round(results["zzu1"][2], 3)}_zzu3{round(results["zzu3"][2], 3)}_xygsnm{round(results["xy_gs_nm"][2], 3)}_hb{round(results["hb"][2], 3)}_xj{round(results["xj"][2], 3)}_TCGA{round(results["tcga"][2], 3)}.pth'
            save_path = os.path.join(p119q_path, name)
            torch.save(net3.state_dict(), save_path)
        torch.cuda.empty_cache()
        gc.collect()
        print("··················································································································")
    print("#####################################################Finish train ###########################################################")
    del net1,net2,net3
    # Final garbage collection
    gc.collect()
    return None

def validate_mil_one_epoch(test_loader, center, net1, net2, net3, net4, net5, net6, net7, net8, net9,net10, net11, net12, device):
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    net5.eval()
    net6.eval()
    net7.eval()
    net8.eval()
    net9.eval()
    net10.eval()
    net11.eval()
    net12.eval()
    B = torch.tensor([2])

    test_acc_grade_type0 = 0
    test_acc_IDH_type0 = 0
    test_acc_p1_19q_type0 = 0

    test_acc_grade_type1 = 0
    test_acc_IDH_type1 = 0
    test_acc_p1_19q_type1 = 0

    test_acc_grade_type2 = 0
    test_acc_IDH_type2 = 0
    test_acc_p1_19q_type2 = 0

    test_acc_grade_type3 = 0
    test_acc_IDH_type3 = 0
    test_acc_p1_19q_type3 = 0

    test_num_grade = 0
    test_num_IDH = 0
    test_num_p1_19q = 0

    grade_labels, grade_preds_type1, grade_probs_type1,grade_preds_type2, grade_probs_type2,grade_preds_type0, grade_probs_type0,grade_preds_type3, grade_probs_type3 = [], [], [],[], [], [],[], [],[]
    IDH_labels, IDH_preds_type0, IDH_probs_type0,IDH_preds_type1, IDH_probs_type1,IDH_preds_type2, IDH_probs_type2,IDH_preds_type3, IDH_probs_type3 = [], [], [], [],[], [], [],[], []
    p1_19q_labels, p1_19q_preds_type0, p1_19q_probs_type0, p1_19q_preds_type1, p1_19q_probs_type1, p1_19q_preds_type2, p1_19q_probs_type2, p1_19q_preds_type3, p1_19q_probs_type3 = [], [], [], [],[], [], [],[], []

    for center1, loader in test_loader.items():
        for test_data in loader:
            Grade_Features_type0, IDH_Features_type0, p1_19q_Features_type0_, Grade_Features_type1, IDH_Features_type1, p1_19q_Features_type1_,Grade_Features_type2, IDH_Features_type2, p1_19q_Features_type2_, id, IDH, p1_19q_, grade = test_data

            grade = torch.squeeze(grade)
            grade_features_type0 = torch.squeeze(Grade_Features_type0)
            logits_type0 = net1(grade_features_type0.to(device))
            logits_type0 = torch.squeeze(logits_type0)
            predict_y_type0 = torch.ge(torch.sigmoid(logits_type0), 0.5).float()
            test_acc_grade_type0 += torch.eq(predict_y_type0, grade.float().to(device)).sum().item()

            grade_features_type1 = torch.squeeze(Grade_Features_type1)
            logits_type1 = net2(grade_features_type1.to(device))
            logits_type1 = torch.squeeze(logits_type1)
            predict_y_type1 = torch.ge(torch.sigmoid(logits_type1), 0.5).float()
            test_acc_grade_type1 += torch.eq(predict_y_type1, grade.float().to(device)).sum().item()

            grade_features_type2 = torch.squeeze(Grade_Features_type2)
            logits_type2 = net10(grade_features_type2.to(device))
            logits_type2 = torch.squeeze(logits_type2)
            predict_y_type2 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
            test_acc_grade_type2 += torch.eq(predict_y_type2, grade.float().to(device)).sum().item()

            logits_type3 = net7(grade_features_type0.to(device), grade_features_type1.to(device), grade_features_type2.to(device))
            logits_type3 = torch.squeeze(logits_type3)
            predict_y_type3 = torch.ge(torch.sigmoid(logits_type3), 0.5).float()
            test_acc_grade_type3 += torch.eq(predict_y_type3, grade.float().to(device)).sum().item()
            # print("grade_labels:",grade,grade)
            
            grade_labels.append(grade.tolist())
            grade_preds_type0.append(predict_y_type0.tolist())
            grade_probs_type0.append(torch.sigmoid(logits_type0).tolist())
            grade_preds_type1.append(predict_y_type1.tolist())
            grade_probs_type1.append(torch.sigmoid(logits_type1).tolist())
            grade_preds_type2.append(predict_y_type2.tolist())
            grade_probs_type2.append(torch.sigmoid(logits_type2).tolist())
            grade_preds_type3.append(predict_y_type3.tolist())
            grade_probs_type3.append(torch.sigmoid(logits_type3).tolist())

            test_num_grade += 1

            IDH = torch.squeeze(IDH)
            IDH_features_type0 = torch.squeeze(IDH_Features_type0)
            logits_type0 = net3(IDH_features_type0.to(device))
            logits_type0 = torch.squeeze(logits_type0)
            predict_y_type0 = torch.ge(torch.sigmoid(logits_type0), 0.5).float()
            test_acc_IDH_type0 += torch.eq(predict_y_type0, IDH.float().to(device)).sum().item()

            IDH_features_type1 = torch.squeeze(IDH_Features_type1)
            logits_type1 = net4(IDH_features_type1.to(device))
            logits_type1 = torch.squeeze(logits_type1)
            predict_y_type1 = torch.ge(torch.sigmoid(logits_type1), 0.5).float()
            test_acc_IDH_type1 += torch.eq(predict_y_type1, IDH.float().to(device)).sum().item()

            IDH_features_type2 = torch.squeeze(IDH_Features_type2)
            logits_type2 = net11(IDH_features_type2.to(device))
            logits_type2 = torch.squeeze(logits_type2)
            predict_y_type2 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
            test_acc_IDH_type2 += torch.eq(predict_y_type2, IDH.float().to(device)).sum().item()

            logits_type3 = net8(IDH_features_type0.to(device), IDH_features_type1.to(device), IDH_features_type2.to(device))
            logits_type3 = torch.squeeze(logits_type3)
            predict_y_type3 = torch.ge(torch.sigmoid(logits_type3), 0.5).float()
            test_acc_IDH_type3 += torch.eq(predict_y_type3, IDH.float().to(device)).sum().item()

            IDH_labels.append(IDH.tolist())
            IDH_preds_type0.append(predict_y_type0.tolist())
            IDH_probs_type0.append(torch.sigmoid(logits_type0).tolist())
            IDH_preds_type1.append(predict_y_type1.tolist())
            IDH_probs_type1.append(torch.sigmoid(logits_type1).tolist())
            IDH_preds_type2.append(predict_y_type2.tolist())
            IDH_probs_type2.append(torch.sigmoid(logits_type2).tolist())
            IDH_preds_type3.append(predict_y_type3.tolist())
            IDH_probs_type3.append(torch.sigmoid(logits_type3).tolist())

            test_num_IDH += 1

            if center1 != "upenn":
                p1_19qs = torch.squeeze(p1_19q_[~np.isin(p1_19q_, B)])
                p1_19qs_features_type0 = torch.squeeze(p1_19q_Features_type0_[~np.isin(p1_19q_, B)])
                p1_19qs_features_type1 = torch.squeeze(p1_19q_Features_type1_[~np.isin(p1_19q_, B)])
                p1_19qs_features_type2 = torch.squeeze(p1_19q_Features_type2_[~np.isin(p1_19q_, B)])
                if p1_19q_ != B:
                    logits_type0 = net5(p1_19qs_features_type0.to(device))
                    logits_type0 = torch.squeeze(logits_type0)
                    predict_y_type0 = torch.ge(torch.sigmoid(logits_type0), 0.5).float()
                    test_acc_p1_19q_type0 += torch.eq(predict_y_type0, p1_19qs.float().to(device)).sum().item()

                    logits_type1 = net6(p1_19qs_features_type1.to(device))
                    logits_type1 = torch.squeeze(logits_type1)
                    predict_y_type1 = torch.ge(torch.sigmoid(logits_type1), 0.5).float()
                    test_acc_p1_19q_type1 += torch.eq(predict_y_type1, p1_19qs.float().to(device)).sum().item()

                    logits_type2 = net12(p1_19qs_features_type2.to(device))
                    logits_type2 = torch.squeeze(logits_type2)
                    predict_y_type2 = torch.ge(torch.sigmoid(logits_type2), 0.5).float()
                    test_acc_p1_19q_type2 += torch.eq(predict_y_type2, p1_19qs.float().to(device)).sum().item()

                    logits_type3 = net9(p1_19qs_features_type0.to(device), p1_19qs_features_type1.to(device), p1_19qs_features_type2.to(device))
                    logits_type3 = torch.squeeze(logits_type3)
                    predict_y_type3 = torch.ge(torch.sigmoid(logits_type3), 0.5).float()
                    test_acc_p1_19q_type3 += torch.eq(predict_y_type3, p1_19qs.float().to(device)).sum().item()

                    p1_19q_labels.append(p1_19qs.tolist())
                    p1_19q_preds_type0.append(predict_y_type0.tolist())
                    p1_19q_probs_type0.append(torch.sigmoid(logits_type0).tolist())
                    p1_19q_preds_type1.append(predict_y_type1.tolist())
                    p1_19q_probs_type1.append(torch.sigmoid(logits_type1).tolist())
                    p1_19q_preds_type2.append(predict_y_type2.tolist())
                    p1_19q_probs_type2.append(torch.sigmoid(logits_type2).tolist())
                    p1_19q_preds_type3.append(predict_y_type3.tolist())
                    p1_19q_probs_type3.append(torch.sigmoid(logits_type3).tolist())

                    test_num_p1_19q += 1

    test_accurate_grade_type0 = test_acc_grade_type0 / test_num_grade
    test_accurate_IDH_type0 = test_acc_IDH_type0 / test_num_IDH

    test_accurate_grade_type1 = test_acc_grade_type1 / test_num_grade
    test_accurate_IDH_type1 = test_acc_IDH_type1 / test_num_IDH

    test_accurate_grade_type2 = test_acc_grade_type2 / test_num_grade
    test_accurate_IDH_type2 = test_acc_IDH_type2 / test_num_IDH

    test_accurate_grade_type3 = test_acc_grade_type3 / test_num_grade
    test_accurate_IDH_type3 = test_acc_IDH_type3 / test_num_IDH

    test_accurate_1p_19q_type0 = test_acc_p1_19q_type0 / test_num_p1_19q
    test_accurate_1p_19q_type1 = test_acc_p1_19q_type1 / test_num_p1_19q
    test_accurate_1p_19q_type2 = test_acc_p1_19q_type2 / test_num_p1_19q
    test_accurate_1p_19q_type3 = test_acc_p1_19q_type3 / test_num_p1_19q

    print(f'........{center:<10}: T2, grade: {test_accurate_grade_type0:.3f}, IDH: {test_accurate_IDH_type0:.3f}, 1p_19q: {test_accurate_1p_19q_type0:.3f} \t Flair, grade: {test_accurate_grade_type1:.3f}, IDH: {test_accurate_IDH_type1:.3f}, 1p_19q: {test_accurate_1p_19q_type1:.3f} \t CET1, grade: {test_accurate_grade_type2:.3f}, IDH: {test_accurate_IDH_type2:.3f}, 1p_19q: {test_accurate_1p_19q_type2:.3f}\t Combin, grade: {test_accurate_grade_type3:.3f}, IDH: {test_accurate_IDH_type3:.3f}, 1p_19q: {test_accurate_1p_19q_type3:.3f} \n')

    return test_accurate_grade_type3, test_accurate_IDH_type3, test_accurate_1p_19q_type3, grade_labels, grade_preds_type0, grade_probs_type0, grade_preds_type1, grade_probs_type1, grade_preds_type2, grade_probs_type2,grade_preds_type3, grade_probs_type3, IDH_labels, IDH_preds_type0, IDH_probs_type0, IDH_preds_type1, IDH_probs_type1, IDH_preds_type2, IDH_probs_type2,IDH_preds_type3, IDH_probs_type3, p1_19q_labels, p1_19q_preds_type0, p1_19q_probs_type0, p1_19q_preds_type1, p1_19q_probs_type1, p1_19q_preds_type2, p1_19q_probs_type2,p1_19q_preds_type3, p1_19q_probs_type3