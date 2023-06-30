import glob
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

import wandb
from nlst import NLST

from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
    Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3, SpatialTransform_unit, \
    SpatialTransformNearest_unit, smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, MSE

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=30001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=30001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=60001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: suggested range 0 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/PATH/TO/YOUR/DATA',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=2000,
                    help="Number step for freezing the previous level")
parser.add_argument("--logpath", type=str,
                    dest="logpath",
                    default='/PATH/TO/YOUR/DATA',
                    help=" path for stpring logs")
parser.add_argument("--modelpath", type=str,
                    dest="modelpath",
                    default='/PATH/TO/YOUR/DATA',
                    help="path for checkpoints")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "LDR_NLST_NCC_unit_disp_add_reg_1_"
wandb.login()
wandb.init(project="lapirn_nlst", entity="varsha_r", reinit=True)

def compute_tre(x, y, spacing=(1.5, 1.5, 1.5)):
    return np.linalg.norm((x.numpy() - y.numpy()) * spacing, axis=1)

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / num_count


def train_lvl1():
    print("Training lvl1...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                        range_flow=range_flow).to(device)

    loss_similarity = NCC(win=3)
    
    # loss_similarity = MSE()
    loss_Jdet = neg_Jdet_loss
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    
    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = opt.modelpath + "/Stage"
    if not os.path.isdir(opt.modelpath):
        os.mkdir(opt.modelpath)
        
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl1+1))

    NLST_dataset = NLST("/home/varsha/data/NLST", 'NLST_dataset_train_test_v1.json',
                                downsampled=True, 
                                masked=False, is_norm=True)
    
    # overfit_set = torch.utils.data.Subset(NLST_dataset, [2] * 20)

    training_generator = Data.DataLoader(NLST_dataset, batch_size=1,
                                         shuffle=True, num_workers=2)
    print("Length" ,len(training_generator.dataset))
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl1:
        epoch_total_loss = []
        
        for batch_idx, data in enumerate(training_generator):

            X = data['moving_img']
            Y = data['fixed_img']
            X = X.to(device).float()
            Y = Y.to(device).float()
            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()
            epoch_total_loss.append(loss.item())
            wandb.log({"lvl1_step" : step, "lvl1_train_loss": loss.item(), "lvl1_sim_NCC" : loss_multiNCC.item(), "lvl1_Jdet" : loss_Jacobian.item(), "lvl1_regulation_loss" : loss_regulation.item() })

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl1_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(step) + '.npy', lossall)

            step += 1

            if step > iteration_lvl1:
                break
        wandb.log({"lvl1/epoch_loss": np.mean(epoch_total_loss), 'epoch': step + 1})

        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl1.npy', lossall)


def train_lvl2():
    print("Training lvl2...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                          range_flow=range_flow).to(device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
    model_path = sorted(glob.glob(opt.modelpath + "/Stage/" + model_name + "stagelvl1_?????.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = opt.modelpath + "/Stage"
    if not os.path.isdir(opt.modelpath):
        os.mkdir(opt.modelpath)
        
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl2 + 1))

    NLST_dataset = NLST("/home/varsha/data/NLST", 'NLST_dataset_train_test_v1.json',
                                downsampled=True, 
                                masked=False, is_norm=True)
    # overfit_set = torch.utils.data.Subset(NLST_dataset, [2] * 20)

    training_generator = Data.DataLoader(NLST_dataset, batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl2:
        epoch_total_loss = []
        for batch_idx, data in enumerate(training_generator):

            X = data['moving_img']
            Y = data['fixed_img']
            X = X.to(device).float()
            Y = Y.to(device).float()
            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()
            epoch_total_loss.append(loss.item())
            wandb.log({"lvl2_step" : step, "lvl2_train_loss": loss.item(), "lvl2_sim_NCC" : loss_multiNCC.item(), "lvl2_Jdet" : loss_Jacobian.item(), "lvl2_regulation_loss" : loss_regulation.item() })
            
            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(step) + '.npy', lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
        wandb.log({"lvl2/epoch_loss": np.mean(epoch_total_loss), 'epoch': step + 1})

    np.save(model_dir + '/loss' + model_name + 'stagelvl2.npy', lossall)


def train_lvl3():
    print("Training lvl3...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    model_path = sorted(glob.glob(opt.modelpath + "/Stage/" + model_name + "stagelvl2_?????.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    
    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = opt.modelpath + "/Stage"
    if not os.path.isdir(opt.modelpath):
        os.mkdir(opt.modelpath)
        
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl3 + 1))

    NLST_dataset = NLST("/home/varsha/data/NLST", 'NLST_dataset_train_test_v1.json',
                                downsampled=True, 
                                masked=False, is_norm=True)
    # NLST_dataset=NLST(data_path, downsampled=False, masked=True)
    # ts1 = torch.utils.data.Subset(NLST_dataset, [1])
    
    # overfit_set = torch.utils.data.Subset(NLST_dataset, [2] * 20)

    training_generator = Data.DataLoader(NLST_dataset, batch_size=1,
                                         shuffle=True, num_workers=2)
    
    NLST_val_dataset = NLST("/home/varsha/data/NLST", 'NLST_dataset_train_test_v1.json',
                                downsampled=True, 
                                masked=False,train=False, is_norm=True)
    valid_generator = Data.DataLoader(NLST_val_dataset, batch_size=1,
                                shuffle=False, num_workers=2) 
    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda(device).float()
    
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl3:
        epoch_total_loss = []
        for _, data in enumerate(training_generator):
            
            X = data['moving_img'].to(device)
            Y = data['fixed_img'].to(device)
            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            F_X_Y[:, 0, :, :, :] = F_X_Y[:, 0, :, :, :] * (z-1)
            F_X_Y[:, 1, :, :, :] = F_X_Y[:, 1, :, :, :] * (y-1)
            F_X_Y[:, 2, :, :, :] = F_X_Y[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(F_X_Y)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()
            epoch_total_loss.append(loss.item())
            wandb.log({"lvl3_step" : step, "lvl3_train_loss": loss.item(), "lvl3_sim_NCC" : loss_multiNCC.item(), "lvl3_Jdet" : loss_Jacobian.item(), "lvl3_regulation_loss" : loss_regulation.item() })
           
            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Validation
            
            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
            
        dice_total = []
                
        print("\nValidating...")
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_generator):
                X = data['moving_img'].to(device)
                Y = data['fixed_img'].to(device)
                X_label, Y_label =  data['moving_mask'].to(
                    device), data['fixed_mask'].to(device)
                #breakpoint()
                X = F.interpolate(X, size=imgshape, mode='trilinear')
                Y = F.interpolate(Y, size=imgshape, mode='trilinear')
                moving_keypoint = data["moving_kp"][0]
                fixed_keypoint = data["fixed_kp"][0]
                
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)
                #breakpoint()
                #F_X_Y = F.interpolate(F_X_Y, size=imgshape, mode='trilinear', align_corners=True) #why did I comment this out? Added again below now
                F_X_Y = F.interpolate(F_X_Y, size=imgshape, mode='trilinear', align_corners=True)
                X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit).cpu().numpy()[0,
                            0, :, :, :]
                Y_label = Y_label.cpu().numpy()[0, 0, :, :, :]
                

                dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                dice_total.append(dice_score)
                # F_X_Y_xyz = torch.zeros(F_X_Y.shape, dtype=F_X_Y.dtype, device=F_X_Y.device)
                # # breakpoint()
                # _, _, x, y, z = F_X_Y.shape
                # F_X_Y_xyz[0, 0] = F_X_Y[0, 2] * (x - 1) / 2
                # F_X_Y_xyz[0, 1] = F_X_Y[0, 1] * (y - 1) / 2
                # F_X_Y_xyz[0, 2] = F_X_Y[0, 0] * (z - 1) / 2
                
                # F_X_Y_xyz_cpu = F_X_Y_xyz.data.cpu().numpy()[0]                       
                
                
                # fixed_disp_x = map_coordinates(F_X_Y_xyz_cpu[0], fixed_keypoint.numpy().transpose())
                # fixed_disp_y = map_coordinates(F_X_Y_xyz_cpu[1], fixed_keypoint.numpy().transpose())
                # fixed_disp_z = map_coordinates(F_X_Y_xyz_cpu[2], fixed_keypoint.numpy().transpose())
                # lms_fixed_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()
                
                # warped_fixed_keypoint = fixed_keypoint + lms_fixed_disp
                
                # tre_score = compute_tre(warped_fixed_keypoint, moving_keypoint).mean()
                test_data_at = wandb.Artifact("test_samples_" + str(wandb.run.id), type="predictions")            

                table_columns = [ 'moving - source', 'fixed - target', 'warped', 'flow_x', 'flow_y', 'mask_warped', 'mask_fixed', 'dice']
                #'displacement_magn', *list(metrics.keys())
                table_results = wandb.Table(columns = table_columns)
                fixed = Y.to('cpu').detach().numpy()
                fixed = fixed[0,0,:,48,:]
                moving = X.to('cpu').detach().numpy()
                moving = moving[0,0,:,48,:]
                warped = X_Y.to('cpu').detach().numpy()
                warped = warped[0,0,:,48,:]
                
                target_fixed = Y_label
                target_fixed = target_fixed[:,48,:]
                mask_fixed = wandb.Image(fixed, masks={
                            "predictions": {
                                "mask_data": target_fixed
                                
                            }
                            })
                warped_seg = X_Y_label
                warped_seg = X_Y_label[:,48,:]

                mask_warped = wandb.Image(warped, masks={
                            "predictions": {
                                "mask_data": warped_seg
                                
                            }
                            })

                # target_moving = X_label.to('cpu').detach().numpy()
                # target_moving = X_label[0,:,119,:]
                flow_x = F_X_Y[0,0,:,48,:].to('cpu').detach().numpy()
                flow_y = F_X_Y[0,1,:,48,:].to('cpu').detach().numpy()
                
                table_results.add_data(wandb.Image(moving), wandb.Image(fixed),wandb.Image(warped),wandb.Image(flow_x), wandb.Image(flow_y), mask_warped ,mask_fixed, dice_score)
                # Varsha
                test_data_at.add(table_results, "predictions")
                wandb.run.log_artifact(test_data_at) 

            print("Dice mean: ", np.mean(dice_total))
            wandb.log({"dice" : np.mean(dice_total)} )
                
        print("one epoch pass")
        wandb.log({"lvl3/epoch_loss": np.mean(epoch_total_loss), 'epoch': step + 1})

    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


imgshape = (224//2, 192//2, 224//2)
imgshape_4 = (imgshape[0]//4, imgshape[1]//4, imgshape[2]//4)
imgshape_2 = (imgshape[0]//2, imgshape[1]//2, imgshape[2]//2)
# Create and initalize log file
if not os.path.isdir(opt.logpath):
    os.mkdir(opt.logpath)

log_dir = opt.logpath + "/" + model_name + ".txt"

with open(log_dir, "a") as log:
    log.write("Validation Dice log for " + model_name[0:-1] + ":\n")

range_flow = 0.4
train_lvl1()
train_lvl2()
train_lvl3()
