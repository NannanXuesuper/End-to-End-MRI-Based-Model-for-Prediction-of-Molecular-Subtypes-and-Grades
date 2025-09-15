import os
import pandas as pd
import numpy as np
import cv2 
import h5py
import nibabel as nib
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.utils import shuffle
from collections import Counter
from typing import List, Tuple
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_boxdata(images,rtol=1e-8):
    # images = size_combi_data
    background_value=0
    tolerance=150
    rtol=1e-8
    crops = []
    for h in range(images.shape[2]):
        image = images[:,:,h]
        is_foreground = np.logical_or(image < (background_value - tolerance),
                                    image> (background_value + tolerance))
        foreground = np.zeros(is_foreground.shape, dtype=np.int16)
        foreground[is_foreground] = 1
        infinity_norm = max(-foreground.min(), foreground.max())
        passes_threshold = np.logical_or(foreground < -rtol * infinity_norm,
                                        foreground > rtol * infinity_norm)  ##
        if foreground.ndim == 4:
            passes_threshold = np.any(passes_threshold, axis=-1)

        coords = np.array(np.where(passes_threshold))
        start = coords.min(axis=1)
        end = coords.max(axis=1) + 1

        # pad with one voxel to avoid resampling problems
        start = np.maximum(start - 1, 0)
        end = np.minimum(end + 1, foreground.shape[:3])

        crop = [(s, e) for s, e in zip(start, end)]
        crops.append(crop)
    valumes = []
    for crop in crops:
        valume = (crop[0][1]-crop[0][0])* (crop[1][1]-crop[1][0])
        valumes.append(valume)
    type = np.argmin(valumes)
    crop = crops[type]
    box_image = images[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
    return box_image

def load_image(filename,data_path,image_type):
        imagepath = os.path.join(data_path, image_type, filename+'_'+image_type+'.nii.gz')
        # labelpath = os.path.join(data_path, 'Seg', filename+'_seg.nii.gz')
        print(imagepath)

        
        img = nib.load(imagepath)
        image_array = img.get_fdata()
        # label_array = nib.load(labelpath).get_fdata()
        # thickness = img.affine[2, 2]
        # affine = img.affine

        return image_array

def get_combi_data(box_image0, box_image1,box_image2,box_image3):
    image0_array = box_image0.astype(np.float32)
    image1_array = box_image1.astype(np.float32)
    image2_array = box_image2.astype(np.float32)
    image3_array = box_image3.astype(np.float32)

    size_rgb_image = np.stack([image0_array,image1_array,image2_array,image3_array],axis=-1)
    
    return size_rgb_image
def normalization(image):
    for s in range(image.shape[2]):
        std,mean = np.std(image[:,:,s]),np.mean(image[:,:,s])
    
        image[:,:,s] = (image[:,:,s] - mean) /std
 
    return image
def creat_H5py(filenames, data_path,ids,IDHs,p1_19qs,h5py_path,image_type,grades):
    count = 1
    print(
        "IDHs num:",Counter(IDHs),"\n"
        "p1_19qs num:",Counter(p1_19qs),"\n"
        "grades num:",Counter(grades),"\n"
        )
    input_nums = 0
    input_size = 224
    output_file = h5py_path
    channels = 4
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(output_file, 'w') as f:
            f.create_dataset("input", (input_nums, input_size, input_size, channels),
                            maxshape=(None, input_size, input_size, channels),
                            chunks=(128, input_size, input_size, channels),
                            dtype='float32')
            f.create_dataset("ids", (input_nums,),
                            maxshape=(None,),
                            chunks=(128, ),
                            dtype='float32')
            f.create_dataset("Grade", (input_nums,),
                            maxshape=(None,),
                            chunks=(128,),
                            dtype=dt)
            f.create_dataset("IDH", (input_nums,),
                            maxshape=(None,),
                            chunks=(128,),
                            dtype=dt)
            f.create_dataset("p1_19qs", (input_nums,),
                            maxshape=(None,),
                            chunks=(128,),
                            dtype=dt)

    for i in range(len(filenames)):
        sample,id,IDH, p1_19q,grade= filenames[i],ids[i],IDHs[i],p1_19qs[i],grades[i]
        img0 = load_image(sample,data_path,image_type[0])
        img1 = load_image(sample,data_path,image_type[1])
        img2 = load_image(sample,data_path,image_type[2])
        img3 = load_image(sample,data_path,image_type[3])

        print(sample)
        print(id)
       
 
        # img,mask,_,_ = get_2_slices(img,mask)
        
        for j in range(img0.shape[2]):
            if img0[:,:,j].max() >100 and img1[:,:,j].max()>100 and img2[:,:,j].max()>100 and img3[:,:,j].max()>100:
                image0_2d, image1_2d,image2_2d,image3_2d = img0[:,:,j], img1[:,:,j],img2[:,:,j],img3[:,:,j]
            
                
                        
            
               
                # image0_2d = cv2.resize(image0_2d, (224,224), interpolation=cv2.INTER_LINEAR)
                # image1_2d = cv2.resize(image1_2d, (224,224), interpolation=cv2.INTER_LINEAR)
                # image2_2d = cv2.resize(image2_2d, (224,224), interpolation=cv2.INTER_LINEAR)
                combi_data= get_combi_data(image0_2d, image1_2d,image2_2d,image3_2d)
                size_combi_data = normalization(get_boxdata(combi_data))
                print("box shape",size_combi_data.shape)
                
                size_combi_data = cv2.resize(size_combi_data, (224,224), interpolation=cv2.INTER_LINEAR)
                
                # image = cv2.resize(size_combi_data, (224,224), interpolation=cv2.INTER_LINEAR)
                image = size_combi_data
                h5f = h5py.File(output_file, 'a')
                if count  >= h5f['input'].shape[0]:
                        input_nums = count 
                        h5f['input'].resize((input_nums, input_size, input_size, channels))
                        h5f['ids'].resize((input_nums,))
                        h5f['Grade'].resize((input_nums,))
                        h5f['IDH'].resize((input_nums,))
                        h5f['p1_19qs'].resize((input_nums,))

                h5f['input'][count-1] = image
                h5f['ids'][count-1] = id
                h5f['Grade'][count-1] = grade
                h5f['IDH'][count-1] = IDH
                h5f['p1_19qs'][count-1] = p1_19q

                count +=1
 
    return  count    

def add_train_val_split_colums(df):
    conditions = [
        (df["Grade"] =="LGG") & (df["IDH"] =="Mutation") & (df["1p_19q"] =="Codeleted"),
        (df["Grade"] =="LGG") & (df["IDH"] =="Mutation") & (df["1p_19q"] =="Non_codeleted"),
        (df["Grade"] =="LGG") & (df["IDH"] =="Wild") & (df["1p_19q"] =="Mask"),
        (df["Grade"] =="GBM") & (df["IDH"] =="Mutation") & (df["1p_19q"] =="Mask"),
        (df["Grade"] =="GBM") & (df["IDH"] =="Wild") & (df["1p_19q"] =="Mask"),
        ]

    # create a list of the values we want to assign for each condition
    values = [1,2,3,4,5]

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['train_val_split'] = np.select(conditions, values)
    return df
def oversample(data: List,
                     labels: List,
                     label_index: str,
                     perc_majority_label: float = 1) -> List:
    """Function to better balance the data. perc_majority_label is what percentage
	you wish to reach of the majority class. A value of 1 means the function will
	oversample until the classes are balanced."""

    neg_samples = []
    pos_samples = []
    mask_samples = []
    # iterate over data
    for sample,label in zip(data,labels):
        if label_index =="Grade":
           
            if label == "GBM":
                pos_samples.append(sample)
            else:
                neg_samples.append(sample)
        elif label_index =="IDH":
            
            if label == "Wild":
                pos_samples.append(sample)
            else:
                neg_samples.append(sample)
        elif label_index =="1p_19q":
            
            if label == "Codeleted":
                pos_samples.append(sample)
            elif label == "Non_codeleted" :
                neg_samples.append(sample)
            else:
                mask_samples.append(sample)

    num_neg = len(neg_samples)
    num_pos = len(pos_samples)
    print(num_neg,num_pos)
    assert num_neg < num_pos, 'Label is already majority of samples.'

    num_needed = int((num_pos - num_neg) * perc_majority_label)
    oversampled_samples = np.random.choice(neg_samples,
                                           size=num_needed,
                                           replace=True)
    neg_samples.extend(oversampled_samples)

    neg_samples.extend(pos_samples)
    neg_samples.extend(mask_samples)
    return neg_samples
def oversample_label(train_filename_over,train_filename,train_labels):
    import numpy as np
    labels = []
    for sam in train_filename_over:
        # print(sam)
        # # print()

        label = train_labels[np.argwhere(train_filename ==sam)[0,0]]
        # print(label)
        labels.append(label)
    return labels

def get_test_h5py(data_root,center):
    y_path = os.path.join(data_root, 'match_id_{}.csv'.format(center))
    df = pd.read_csv(y_path)
    if center =="zzu1":
        name = pd.read_csv("/local/CTimages/brain/code_5mm/data/tumor_slices_2mm/zzu1/CET1/name.csv")
        df = pd.merge(df,name,how="inner",on="Name")
    elif center =="xy_gs_nm_hb": 
        name = pd.read_csv("/local/CTimages/brain/code_5mm/data/tumor_slices_2mm/xy_gs_nm_hb/CET1/name.csv")
        df = pd.merge(df,name,how="inner",on="Name")
    elif center =="hb":
        name = pd.read_csv("/local/CTimages/brain/code_5mm/data/tumor_slices_2mm/hb/CET1/name.csv")
        df = pd.merge(df,name,how="inner",on="Name")
    ids,IDHs,p1_19qs,filenames,grades= np.array(df["ID"]),np.array(df["IDH"]),np.array(df["1p_19q"]),np.array(df["Name"]),np.array(df["Grade"])
    images_rotation_path = os.path.join(data_path,"03_h5data/noADC")
    if not os.path.exists(images_rotation_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(images_rotation_path)
    print(f"num of sample for {center} is {len(filenames)}" )
        # 『生成5份TFrecords
    hdf5_file = os.path.join(images_rotation_path,"test_images_{}.hdf5".format(center))  
    print(hdf5_file)
    counter = creat_H5py(filenames, data_path,ids, IDHs,p1_19qs,hdf5_file,image_type,grades)
    print("test.hdf5文件生成成功！")
    print('counter:{}'.format(counter))
    np.save(os.path.join(images_rotation_path,"foldsize.npy"),counter)
    return None

def get_train_val_split(data_root,center):
    
    y_path = os.path.join(data_root, 'match_id_{}.csv'.format(center))
    df = pd.read_csv(y_path)
    df = add_train_val_split_colums(df)
    data_path = os.path.join(data_root, "tumor_slices_2mm",center)
    image_type = ['T1','T2','Flair',"CET1","ADC"]
    ids,IDHs,p1_19qs,filenames,grades,groups= np.array(df["ID"]),np.array(df["IDH"]),np.array(df["1p_19q"]),np.array(df["Name"]),np.array(df["Grade"]),np.array(df["train_val_split"])
    #K(10)折交叉验证
    n_splits = 5
    images_rotation_path = os.path.join(data_path,"03_h5data")
    if not os.path.exists(images_rotation_path):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(images_rotation_path)
  
    skf = StratifiedKFold(n_splits=n_splits)
    split_i = 0
    foldsize = np.zeros((5,2))
    for train_index, test_index in skf.split(filenames,groups):
        # print("Train Index:", train_index, ",Test Index:",test_index)
        train_filename, test_filename = np.array(filenames[train_index]), np.array(filenames[test_index])
        train_grades, test_grades = np.array(grades[train_index]), np.array(grades[test_index])
        train_IDHs, test_IDHs = np.array(IDHs[train_index]), np.array(IDHs[test_index])
        train_p1_19q, test_p1_19q = np.array(p1_19qs[train_index]), np.array(p1_19qs[test_index])
        train_ids,test_ids = np.array(ids[train_index]), np.array(ids[test_index])

        Train_examples = len(train_filename)
        Test_examples = len(test_filename)
        print("train_filenames:",train_filename,
            "val_filenames:",test_filename,"\n"
            "train_IDHs num:",Counter(train_IDHs),
            "val_IDHs num:",Counter(test_IDHs),"\n"
            "train_p1_19qs num:",Counter(train_p1_19q),
            "val_p1_19qs num:",Counter(test_p1_19q),"\n"
            "train_grades num:",Counter(train_grades),
            "val_grades num:",Counter(test_grades),"\n"
            )
        train_filename_over = oversample(train_filename,train_p1_19q,label_index = "1p_19q")
        train_grades_over = oversample_label(train_filename_over,train_filename,train_grades)
        train_IDHs_over = oversample_label(train_filename_over,train_filename,train_IDHs)
        train_p1_19q_over = oversample_label(train_filename_over,train_filename,train_p1_19q)
        train_ids_over = oversample_label(train_filename_over,train_filename,train_ids)
        print("train_filenames:",train_filename_over,"\n"
            "train_IDHs_oversample num:",Counter(train_IDHs_over),"\n"
            "train_p1_19qs_oversample num:",Counter(train_p1_19q_over),"\n"
            "train_grades_oversample num:",Counter(train_grades_over),"\n"
            )
        # 生成5份TFrecords
        
        train_hdf5_file = os.path.join(images_rotation_path,"train_images_fold"+str(split_i)+".hdf5")
        test_hdf5_file = os.path.join(images_rotation_path,"test_images_fold"+str(split_i)+".hdf5")

        train_counter = creat_H5py(train_filename_over, data_path,train_ids_over, train_IDHs_over,train_p1_19q_over,train_hdf5_file,image_type,train_grades_over)
        test_counter = creat_H5py(test_filename, data_path, test_ids,test_IDHs,test_p1_19q,test_hdf5_file,image_type,test_grades)
        print(str(split_i)+".hdf5文件生成成功！")
        print('train_counter:{}'.format(train_counter))
        print('test_counter:{}'.format(test_counter))
        foldsize[split_i][0] = train_counter
        foldsize[split_i][1] = test_counter
        split_i +=1
    np.save(os.path.join(images_rotation_path,"foldsize.npy"),foldsize)
    return None 
####train and val split    
center = "zzu2"
data_root = "/local/CTimages/brain/code_5mm/data"
get_train_val_split(data_root,center)
#####test          
centers = ["zzu1","zzu3","xy_gs_nm","hb","xj","upenn"]
for center in centers:
    get_test_h5py(data_root,center)