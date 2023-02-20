import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms
import nibabel as nib
import torch
import itertools
import os
import json
import torch.nn.functional as F
import sys


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)/2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)/2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)/2

    return flow


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, iterations, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])
        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_validation(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, imgs, labels, norm=False):
        'Initialization'
        super(Dataset_epoch_validation, self).__init__()

        self.imgs = imgs
        self.labels = labels
        self.norm = norm
        self.imgs_pair = list(itertools.permutations(imgs, 2))
        self.labels_pair = list(itertools.permutations(labels, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.imgs_pair[step][0])
        img_B = load_4D(self.imgs_pair[step][1])

        label_A = load_4D(self.labels_pair[step][0])
        label_B = load_4D(self.labels_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=False):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list)
        moved_img = load_4D(self.move_list[index])
        fixed_label = load_4D(self.fixed_label_list)
        moved_label = load_4D(self.move_label_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output


# Ref: https://github.com/MDL-UzL/L2R/blob/main/examples/task_specific/NLST/Example_NLST.ipynb

class NLST(torch.utils.data.Dataset):
    def __init__(self, root_dir, masked=False, downsampled=False, train=True):
        """
        NLST_Dataset
        Provides FIXED_IMG, MOVING_IMG, FIXED_KEYPOINTS, MOVING_KEYPOINTS
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir,'imagesTr')
        self.keypoint_dir = os.path.join(root_dir,'keypointsTr')
        self.masked = masked
        with open(os.path.join(root_dir,'NLST_dataset.json')) as f:
            self.dataset_json = json.load(f)
        self.shape = self.dataset_json['tensorImageShape']['0']
        self.H, self.W, self.D = self.shape
        self.downsampled = downsampled
        self.train = train
        self.transforms = transforms.Resize((192, 192))
        if self.train :
            self.type_data = 'training_paired_images'
        
        else:
            self.type_data = 'registration_val'
        
    def __len__(self):
        
        if self.train:
            return self.dataset_json['numPairedTraining']
        else:
            return len(self.dataset_json['registration_val'])

    def get_shape(self):
        if self.downsampled:
            return [x//2 for x in self.shape]
        else:
            return self.shape
    
    def __getitem__(self, idx):
        fix_path=os.path.join(self.root_dir,self.dataset_json[self.type_data][idx]['fixed'])
        
        mov_path=os.path.join(self.root_dir,self.dataset_json[self.type_data][idx]['moving'])
    
        fixed_label = fix_path.replace('images', 'masks')
        move_label = mov_path.replace('images', 'masks')

        fixed_img=torch.from_numpy(imgnorm(nib.load(fix_path).get_fdata())).float().permute(1,0,2)
        moving_img=torch.from_numpy(imgnorm(nib.load(mov_path).get_fdata())).float().permute(1,0,2)
        
        fixed_mask=torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata()).float().permute(1,0,2)
        moving_mask=torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata()).float().permute(1,0,2)
        
        # fixed_kp=torch.from_numpy(np.genfromtxt(fix_path.replace('images','keypoints').replace('nii.gz','csv'),delimiter=','))
        # moving_kp=torch.from_numpy(np.genfromtxt(mov_path.replace('images','keypoints').replace('nii.gz','csv'),delimiter=','))
        # fixed_kp=(fixed_kp.flip(-1)/torch.tensor(self.shape))*2-1
        # moving_kp=(moving_kp.flip(-1)/torch.tensor(self.shape))*2-1

        if self.masked and not self.downsampled:
            fixed_img=torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata())*fixed_img
            moving_img=torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata())*moving_img
        
        # if self.downsampled:
        #     fixed_img=F.interpolate(fixed_img.view(1,1,self.H,self.W,self.D),size=(self.H//2,self.W//2,self.D//2),mode='trilinear').squeeze()
        #     moving_img=F.interpolate(moving_img.view(1,1,self.H,self.W,self.D), size=(self.H//2,self.W//2,self.D//2), mode='trilinear').squeeze()
        #     if self.masked:
        #         fixed_img*=F.interpolate(torch.from_numpy(nib.load(fix_path.replace('images', 'masks')).get_fdata()).view(1,1,self.H,self.W,self.D),size=(self.H//2,self.W//2,self.D//2),mode='nearest').squeeze()
        #         moving_img*=F.interpolate(torch.from_numpy(nib.load(mov_path.replace('images', 'masks')).get_fdata()).view(1,1,self.H,self.W,self.D),size=(self.H//2,self.W//2,self.D//2),mode='nearest').squeeze()

        if self.transforms is not None:
            fixed_img = self.transforms(fixed_img)
            moving_img = self.transforms(moving_img)
            fixed_mask = self.transforms(fixed_mask)
            moving_mask = self.transforms(moving_mask)
        
        
        return fixed_img.float(), moving_img.float(), fixed_mask, moving_mask