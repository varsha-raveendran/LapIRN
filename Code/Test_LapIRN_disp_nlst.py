import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import map_coordinates
from nlst import NLST
from Functions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D
from miccai2020_model_stage_cpu import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/LapIRN_disp_fea7.pth',
                    help="Pre-trained Model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=14,
                    help="number of start channels")

opt = parser.parse_args()

out_path = opt.savepath

os.makedirs(out_path, exist_ok=True)
os.makedirs(out_path + "/" + "moved_imgs", exist_ok=True)
os.makedirs(out_path + "/" + "disp_field", exist_ok=True)

start_channel = opt.start_channel

def compute_tre(x, y, spacing=(1.5, 1.5, 1.5)):
    return np.linalg.norm((x - y) * spacing, axis=1)


def test():
    orig_imgshape = (224, 192, 224)
    imgshape = (224//2, 192//2, 224//2)
    imgshape_4 = (imgshape[0]//4, imgshape[1]//4, imgshape[2]//4)
    imgshape_2 = (imgshape[0]//2, imgshape[1]//2, imgshape[2]//2)

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1)

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2)

    transform = SpatialTransform_unit()
    device = torch.device('cpu')
    model.load_state_dict(torch.load(opt.modelpath,  map_location=device))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).float()

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    NLST_dataset = NLST("/home/varsha/data/NLST", 'NLST_dataset_train_test_v1.json',
                                downsampled=True
                                , 
                                masked=False,train=False,
                            is_norm=True)
    # overfit_set = torch.utils.data.Subset(NLST_dataset, [2] )

    val_dataloader = DataLoader(NLST_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            print(batch["fixed_name"][0])
            val1 = batch["fixed_name"][0][-16:-12]
            val2 = batch["moving_name"][0][-16:-12]
            input_fixed = batch["fixed_img"]
            
            input_moving = batch["moving_img"]
            
            fixed_affine = batch["fixed_affine"][0]
            print(input_moving.shape)
            F_X_Y = model(input_moving, input_fixed)
            print(F_X_Y.shape)
            # F_X_Y = F.interpolate(F_X_Y, size=orig_imgshape, mode='trilinear', align_corners=True)
            # F_X_Y = F.interpolate(F_X_Y, scale_factor=2, mode='trilinear', align_corners=True) #1. upsample
            # print(F_X_Y.shape)
            
            # X = F.interpolate(input_moving,scale_factor=2, mode='trilinear')
            # print(X.shape)
            # print(grid.shape)
            X_Y = transform(input_moving, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :] #warped img
            
            #2. unnorm and flip
            F_X_Y_xyz = torch.zeros(F_X_Y.shape, dtype=F_X_Y.dtype, device=F_X_Y.device)
            _, _, x, y, z = F_X_Y.shape
            F_X_Y_xyz[0, 0] = F_X_Y[0, 2] * (x - 1) / 2
            F_X_Y_xyz[0, 1] = F_X_Y[0, 1] * (y - 1) / 2
            F_X_Y_xyz[0, 2] = F_X_Y[0, 0] * (z - 1) / 2

            # F_X_Y_clone = F_X_Y.clone()
            # F_X_Y_cpu = F_X_Y_xyz.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_xyz_cpu = F_X_Y_xyz.data.cpu().numpy()[0, :, :, :, :]
            print(F_X_Y_xyz_cpu.shape)   
            moving_keypoint = batch['moving_kp'].squeeze(0).numpy() //2  
            fixed_keypoint = batch['fixed_kp'].squeeze(0).numpy() //2
            print(fixed_keypoint.shape) 
            fixed_disp_x = map_coordinates(F_X_Y_xyz_cpu[0], fixed_keypoint.transpose())
            fixed_disp_y = map_coordinates(F_X_Y_xyz_cpu[1], fixed_keypoint.transpose())
            fixed_disp_z = map_coordinates(F_X_Y_xyz_cpu[2], fixed_keypoint.transpose())
            lms_fixed_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()
            
            warped_fixed_keypoint = fixed_keypoint + lms_fixed_disp
            tre_score = compute_tre(warped_fixed_keypoint, moving_keypoint, 
                                    spacing=(1.5, 1.5, 1.5)).mean()
            print(tre_score)
            # moved_path = os.path.join(out_path + '/moved_imgs', f'moved_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
            # warped_path = os.path.join(out_path + '/disp_field', f'flow_{str(val1).zfill(4)}_{str(val2).zfill(4)}.nii.gz')
            # save_flow(F_X_Y_xyz_cpu.transpose(1,2,3,0), warped_path,affine=fixed_affine)
            # save_img(X_Y, moved_path,affine=fixed_affine)

    print("Finished")


if __name__ == '__main__':
    imgshape = (224, 192, 224)
    range_flow = 0.4
    test()
