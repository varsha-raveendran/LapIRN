import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F

from nlst import NLST

from Functions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit
from torch.utils.data import DataLoader
from scipy.ndimage import map_coordinates

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LapIRN_diff_fea7.pth',
                    help="Pre-trained Model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
# parser.add_argument("--fixed", type=str,
#                     dest="fixed", default='../Data/image_A.nii',
#                     help="fixed image")
# parser.add_argument("--moving", type=str,
#                     dest="moving", default='../Data/image_B.nii',
#                     help="moving image")
opt = parser.parse_args()

out_path = opt.savepath
# fixed_path = opt.fixed
# moving_path = opt.moving

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path + "/" + "moved_imgs", exist_ok=True)
os.makedirs(out_path + "/" + "disp_field", exist_ok=True)

# if not os.path.isdir(savepath):
#     os.mkdir(savepath)

start_channel = opt.start_channel

def compute_tre(x, y, spacing=(1.5, 1.5, 1.5)):
    return np.linalg.norm((x.numpy() - y.numpy()) * spacing, axis=1)

def test():
    imgshape_4 = (224 / 4, 192 / 4, 224 / 4)
    imgshape_2 = (224 / 2, 192 / 2, 224 / 2)

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).cuda()
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform_unit().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # fixed_img = load_4D(fixed_path)
    # moving_img = load_4D(moving_path)

    # # normalize image to [0, 1]
    # # norm = True
    # # if norm:
    # #     fixed_img = imgnorm(fixed_img)
    # #     moving_img = imgnorm(moving_img)

    # fixed_img = torch.from_numpy(fixed_img).float()
    # moving_img = torch.from_numpy(moving_img).float()
    

    # rescale = tio.RescaleIntensity(percentiles=(0.5, 99.5))
    # preprocess_intensity = tio.Compose([
    #     rescale,
    #     ])

    # moving_img =  preprocess_intensity(moving_img).unsqueeze(dim=0)
    # fixed_img =  preprocess_intensity(fixed_img).unsqueeze(dim=0)
    
    NLST_dataset = NLST("/home/varsha/data/NLST", 'NLST_dataset_train_test_v1.json',
                                downsampled=False, 
                                masked=False,train=False,
                            is_norm=True)

    val_dataloader = DataLoader(NLST_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            val1 = batch["fixed_name"][0][-16:-12]
            val2 = batch["moving_name"][0][-16:-12]
            input_fixed = batch["fixed_img"].to(device)
            
            input_moving = batch["moving_img"].to(device)
            fixed_affine = batch["fixed_affine"][0]
            moving_keypoint = batch["moving_kp"][0]
            fixed_keypoint = batch["fixed_kp"][0]
            # predict
            F_X_Y = model(input_moving.to(device), input_fixed.to(device))
            # F_X_Y = F.interpolate(F_X_Y, size=imgshape, mode='trilinear', align_corners=True)
            # F_X_Y_clone = F_X_Y.clone()
            print(F_X_Y.shape)
            X_Y = transform(input_moving.to(device), F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :] #grid sample
            # F_X_Y_cpu = F_X_Y.flip(1).data.cpu()[0, :, :, :, :]
            
            # # F_X_Y_cpu = F_X_Y_clone.data.cpu()
            # print(F_X_Y_cpu.shape)
            # F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu.permute(1,2,3,0)) #unnorm
            F_X_Y_xyz = torch.zeros(F_X_Y.shape, dtype=F_X_Y.dtype, device=F_X_Y.device)
            _, _, x, y, z = F_X_Y.shape
            F_X_Y_xyz[0, 0] = F_X_Y[0, 2] * (x - 1) / 2
            F_X_Y_xyz[0, 1] = F_X_Y[0, 1] * (y - 1) / 2
            F_X_Y_xyz[0, 2] = F_X_Y[0, 0] * (z - 1) / 2
            
            F_X_Y_xyz_cpu = F_X_Y_xyz.data.cpu().numpy()[0]
                       
            
            fixed_disp_x = map_coordinates(F_X_Y_xyz_cpu[0], fixed_keypoint.numpy().transpose())
            fixed_disp_y = map_coordinates(F_X_Y_xyz_cpu[1], fixed_keypoint.numpy().transpose())
            fixed_disp_z = map_coordinates(F_X_Y_xyz_cpu[2], fixed_keypoint.numpy().transpose())
            lms_fixed_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()
            
            warped_fixed_keypoint = fixed_keypoint + lms_fixed_disp
            
            tre_score = compute_tre(warped_fixed_keypoint, moving_keypoint).mean()
            print(val1, val2, tre_score)
            print(F_X_Y_xyz_cpu.shape)
            # print(fixed_affine)
            moved_path = os.path.join(out_path + '/moved_imgs', f'moved_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
            warped_path = os.path.join(out_path + '/disp_field', f'flow_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
            save_flow(F_X_Y_xyz_cpu, warped_path)
            save_img(X_Y, moved_path,affine=fixed_affine)

    print("Finished")


if __name__ == '__main__':
    imgshape = (224, 192, 224)
    range_flow = 0.4
    test()
