import monai
import numpy as np
import os
import SimpleITK as sitk
from monai.transforms import Spacing
import torch

dataset = 'btcv'  # btcv or chaos
data_path = '/root/autodl-tmp/Kim/kits23/dataset'
multi_labelPath_ori = './dataset/{}/label'.format(dataset)
multi_imagePath_ori = './dataset/{}/image'.format(dataset)
multi_labelPath_iso = '/root/autodl-tmp/Kim/kits23/dataset/label_iso'
multi_imagePath_iso = '/root/autodl-tmp/Kim/kits23/dataset/image_iso'
multi_labels = os.listdir(data_path)
multi_images = os.listdir(data_path)
os.makedirs(multi_imagePath_iso)
os.makedirs(multi_labelPath_iso)

for multi_label in multi_labels:
    image_path = os.path.join(data_path, multi_label, 'segmentation_pCE')
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件

    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]
    print(numpySpacing, numpyImage.shape)  ## numpy Image 是只有0和1(代表图片里这个像素点是黑还是白)
    # 需要转化成坐标的形式

    spacing_transform = Spacing(pixdim=(1 / numpySpacing[2], 1 / numpySpacing[0], 1 / numpySpacing[1]), mode="nearest")
    numpyImage = spacing_transform(torch.tensor(np.expand_dims(numpyImage, axis=0)))[0][0]
    print(numpyImage.shape)

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    print(numpySpacing)
    sitk.WriteImage(sitk_img, multi_labelPath_iso + '/{}.nii.gz'.format(multi_label))  # 存为nii
    print('label ok')

for multi_image in multi_images:
    image_path = os.path.join(multi_imagePath_ori, multi_image)
    sitk_img = sitk.ReadImage(image_path)  # 是个单label单块的文件
    # print("img shape:", itk_img.shape)
    ''' return order [z, y, x] , numpyImage, numpyOrigin, numpySpacing '''
    # 医学图像处理 空间转换的比例，一个像素代表多少毫米。 保证前后属性一致
    numpyImage = sitk.GetArrayFromImage(sitk_img)  # # [z, y, x]
    numpyOrigin = np.array(sitk_img.GetOrigin())  # # [z, y, x]
    numpySpacing = np.array((sitk_img.GetSpacing()))  # # [z, y, x]
    print(numpySpacing, numpyImage.shape)  ## numpy Image 是只有0和1(代表图片里这个像素点是黑还是白)
    # 需要转化成坐标的形式

    spacing_transform = Spacing(pixdim=(1 / numpySpacing[2], 1 / numpySpacing[0], 1 / numpySpacing[1]), mode="bilinear")
    numpyImage = spacing_transform(torch.tensor(np.expand_dims(numpyImage, axis=0)))[0][0]
    print(numpyImage.shape)

    sitk_img = sitk.GetImageFromArray(numpyImage, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing([1, 1, 1])
    print(numpySpacing)
    sitk.WriteImage(sitk_img, multi_imagePath_iso + '/{}.nii.gz'.format(multi_image))  # 存为nii
    print('image ok')
