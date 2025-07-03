import monai
import numpy as np
import os
import SimpleITK as sitk
from monai.transforms import Spacing
import torch
from tqdm import tqdm

dataset = 'btcv'  # btcv or chaos
multi_labelPath_iso = '/root/autodl-tmp/Kim/kits23/dataset'
output_path = '/root/autodl-tmp/Kim/kits23/dataset'
multi_labels = os.listdir(multi_labelPath_iso)
label_num = 2  # 指定哪个器官

label_dict = {1:'kidney', 2:'tumor'}

for multi_label in tqdm(multi_labels):
    if 'case_' not in multi_label:
        continue
    image_path = os.path.join(multi_labelPath_iso, multi_label, 'segmentation_pCE.nii.gz')
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件
    # print("img shape:", itk_img.shape)
    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]
    # print(numpySpacing, numpyImage.shape)  ## numpy Image 是只有0和1(代表图片里这个像素点是黑还是白)
    # 需要转化成坐标的形式

    # start from all zeros
    out = np.zeros_like(numpyImage, dtype=numpyImage.dtype)

    # keep your organ as 1
    out[numpyImage == label_num] = 1

    # keep original “unlabeled” voxels as 4
    out[numpyImage == 4] = 4

    # now out has {0,1,4}
    numpyImage = out

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    # print(numpySpacing)
    save_path = os.path.join(output_path,label_dict[label_num],'labelsTr_gt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    sitk.WriteImage(sitk_img, save_path + '/{}.nii.gz'.format(multi_label))  # 存为nii
