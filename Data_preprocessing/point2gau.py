import torch
from torch.autograd import Variable
import os
import SimpleITK as sitk
import numpy as np
import pydicom
import random
import cv2

# ---------------------------------------------------------
points = 200
delta = 10
dataset = 'btcv'  # 'btcv' or 'chaos'
organ = 'liver'
# ---------------------------------------------------------

one_labelpath = f"./dataset/{dataset}/{organ}/labelsTr_gt"
rootpath = f'./dataset/{dataset}/{organ}/p_{points}_d_{delta}'
if not os.path.exists(rootpath):
    os.mkdir(rootpath)
point_num = points


def farthest_point_sample(xyz, pnum):
    """
    Input:
        xyz: pointcloud data, shape [B, N, 3]
        pnum: number of samples
    Return:
        centroids: indices of sampled points, shape [B, pnum]
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = int(pnum)

    # allocate array for sampled point indices
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # track minimum distance from each point to any chosen centroid
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # compute the overall centroid of the cloud and pick farthest point as first sample
    barycenter = xyz.sum(dim=1) / N
    barycenter = barycenter.view(B, 1, C)
    dist = torch.sum((xyz - barycenter) ** 2, dim=-1)
    farthest = torch.max(dist, dim=1)[1]

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # compute squared distances to the newest centroid
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        # update the running minimum distances
        mask = dist < distance
        distance[mask] = dist[mask].float()
        # pick next farthest point
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


one_labels = os.listdir(one_labelpath)

for one_label in one_labels:
    image_path = os.path.join(one_labelpath, one_label)
    sitk_img = sitk.ReadImage(image_path)  # single-label volume file
    # get the 3D array [z, y, x]
    numpyImage = sitk.GetArrayFromImage(sitk_img)
    numpyOrigin = sitk_img.GetOrigin()
    numpySpacing = sitk_img.GetSpacing()
    print(numpyImage.shape)  # shows volume dimensions

    # collect coordinates where mask == 1
    points = np.where(numpyImage == 1)
    points_to_be_sampled = []
    for i in range(len(points[0])):
        z, y, x = points[0][i], points[1][i], points[2][i]
        points_to_be_sampled.append([x, y, z])

    # randomly keep half of the candidate points
    points_to_be_sampled = random.sample(points_to_be_sampled, int(len(points_to_be_sampled) * 0.5))
    points_to_be_sampled = torch.tensor(points_to_be_sampled)

    # prepare as batch of size 1
    points_to_be_sampled = points_to_be_sampled.unsqueeze(0)
    sim_data = Variable(points_to_be_sampled)
    print(points_to_be_sampled.shape)

    # run farthest-point sampling
    centroids = farthest_point_sample(sim_data, point_num)
    centroid = sim_data[0, centroids, :][0]

    # build sparse point-label volume
    point_labeled_image = np.zeros_like(numpyImage)
    for cord in centroid:
        x, y, z = cord
        print(x, y, z)
        point_labeled_image[z, y, x] = 1

    # save the point-label as NIfTI
    sitk_img = sitk.GetImageFromArray(point_labeled_image, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing(numpySpacing)
    point_label_path = os.path.join(rootpath, 'point_label')
    if not os.path.exists(point_label_path):
        os.mkdir(point_label_path)
    sitk.WriteImage(sitk_img, os.path.join(point_label_path, f'{one_label}'))


def point2GAU(img_name, root_path, save_name, delta):
    """
    Generate a Gaussian heatmap around the sampled points.
    Inputs:
        img_name: filename of sparse point-label NIfTI
        root_path: base output folder
        save_name: name for saving the Gaussian map
        delta: standard deviation for Gaussian
    """
    one_label_file = os.path.join(root_path, 'point_label', img_name)
    one_label = sitk.ReadImage(one_label_file)
    numpyImage = sitk.GetArrayFromImage(one_label)
    numpyOrigin = np.array(one_label.GetOrigin())
    numpySpacing = np.array(one_label.GetSpacing())

    # create coordinate grids
    Z, Y, X = np.mgrid[:numpyImage.shape[0], :numpyImage.shape[1], :numpyImage.shape[2]]

    gaussian_result = np.zeros(numpyImage.shape)
    for z in range(numpyImage.shape[0]):
        for y in range(numpyImage.shape[1]):
            for x in range(numpyImage.shape[2]):
                if numpyImage[z, y, x] != 0:
                    # add a 3D Gaussian centered at (z,y,x)
                    dist_sq = (Z - z)**2 + (Y - y)**2 + (X - x)**2
                    gau_func = np.exp(-dist_sq / (2 * (delta**2)))
                    gaussian_result += gau_func

    # normalize to [0,1]
    Min, Max = gaussian_result.min(), gaussian_result.max()
    gaussian_result = (gaussian_result - Min) / (Max - Min)

    # save the Gaussian map
    sitk_img = sitk.GetImageFromArray(gaussian_result, isVector=False)
    sitk_img.SetOrigin(numpyOrigin)
    sitk_img.SetSpacing(numpySpacing)
    labelsTr_path = os.path.join(root_path, 'labelsTr')
    if not os.path.exists(labelsTr_path):
        os.mkdir(labelsTr_path)
    sitk.WriteImage(sitk_img, os.path.join(labelsTr_path, save_name))


point_file_list = os.listdir(os.path.join(rootpath, 'point_label'))
for point_file in point_file_list:
    print(point_file)
    point2GAU(point_file, rootpath, point_file, delta)

print(f"{organ} delta={delta} points={points}")