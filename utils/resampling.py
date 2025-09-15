import scipy.ndimage as ndimage
import numpy as np
from skimage import morphology
import SimpleITK as sitk
import os, re
import pandas as pd
def resize_volume(img_array,img ):
    img_array = ndimage.zoom(img_array,(img.GetSpacing()[-1]/2,1,1),order= 0)## 双线性插值

    return img_array



def after_preprocess(image_path,name,preprocess_path,center,start,end):
    print(image_path)
    
    # label = sitk.ReadImage(label_path)
    # label_array = sitk.GetArrayFromImage(label)
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)[start:end,:,:]
    # print("num of slices:",end-start)
    print("tumor image shape:",img_array.shape)
    img_array = resize_volume(img_array,img)
    
    new_img = sitk.GetImageFromArray(img_array)
    new_img.SetDirection(img.GetDirection())
    new_img.SetOrigin((img.GetOrigin()[0],img.GetOrigin()[1],0))
    new_img.SetSpacing((img.GetSpacing()[0],img.GetSpacing()[1],2))
    image_type = image_path.split('_')[-1].split('.')[0]
    save_path =os.path.join(preprocess_path,"tumor_slices",center,image_type)
    if not os.path.exists(save_path):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(save_path)
  
    sitk.WriteImage(new_img,os.path.join(save_path,name+'_'+image_type+'.nii.gz'))
    return None
def get_path(data_root,center,meta_path = "/local/CTimages/brain/code_5mm/data/tumor_slices_2mm"):
    file_all = get_path(data_root)
    df = pd.read_csv(os.path.join(meta_path,"match_id_{}.csv".format(center)))
    MRs,ids,IDH_types,p1_19qs,starts,ends= np.array(df["MR"]),np.array(df["Name"]),np.array(df["IDH"]),np.array(df["1p_19q"]),np.array(df["Start"]),np.array(df["End"])
    for i in range(len(ids)):
        MR_id,id,start,end = MRs[i],ids[i],starts[i],ends[i]
        print(MR_id)
        f = []
        names = []
        for file in file_all:
            
            # print(file)
            if str(MR_id) in file:
                # print(file)
                f.append(file)
                names.append(id)
        image_paths = []
        for j in range(len(names)):
            path = f[j]
            if 'Seg' in path:
                label_path = path
            else:
                image_paths.append(path)
            name = names[j]
        print(name)
        for image_path in image_paths:
                    # print(image_path)
            after_preprocess(image_path,name,meta_path,center,start,end)
#####zzu1 and zzu2
data_root_zzu1 =  r"/media/nannxue/Seagate Basic/Glioma/datazzu/第一批/01_RAW"
data_root_zzu2 =  r"/media/nannxue/Seagate Basic/Glioma/datazzu/第二批/01_rawdata"        
get_path(data_root_zzu1,center = 'zzu1')          
get_path(data_root_zzu2,center = 'zzu2')

###xy
data_root_xy =  r"/media/nannxue/Seagate Basic/Glioma/dataxy/01_rawdata"
get_path(data_root_xy,center = 'xy')
###
  
#####upenn
center = 'upenn'
df = pd.read_csv("/media/nannxue/Seagate Basic/Glioma/code_zzu_xy_TCGA_upenn/data/match_id_{}.csv".format(center))

ids,IDH_types,p1_19qs,starts,ends= np.array(df["Name"]),np.array(df["IDH"]),np.array(df["1p_19q"]),np.array(df["Start"]),np.array(df["End"])
meta_path = "/media/nannxue/Seagate Basic/Glioma/code_zzu_xy_TCGA_upenn/data"
image_type = ["CET1","Flair","seg","T1","T2"]
for i in range(len(ids)):
    id,IDH_type,p1_19q,start,end = ids[i],IDH_types[i],p1_19qs[i],starts[i],ends[i]
    for j in image_type:
        image_path = "/media/nannxue/Seagate Basic/Glioma/code_zzu_xy_TCGA_upenn/data/all_slices/upenn/{}/{}_{}.nii.gz".format(j,id,j)
        name = id
        
        after_preprocess(image_path,name,meta_path,center,start,end)


