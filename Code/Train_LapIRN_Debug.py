import glob
import os
import sys
from datetime import datetime
from argparse import ArgumentParser
from xmlrpc.client import boolean
import warnings

import numpy as np
import torch
import torch.utils.data as Data
# from torch.utils.tensorboard import SummaryWriter

#import SimpleITK as sitk
import pandas as pd


from Functions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, Dataset_epoch_validation, dice
from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, Miccai2020_LDR_laplacian_unit_add_lvl2, \
    Miccai2020_LDR_laplacian_unit_add_lvl3, SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC
from evaluation import evaluate

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: suggested range 0 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=4,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--validation_check", type=int,
                    dest="validation_check", default=1000,
                    help="frequency of validating model")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=2,
                    help="number of start channels")
parser.add_argument("--model_dir", type=str,
                    dest="model_dir", default='../Model/Stage',
                    help="model director")
parser.add_argument("--log_dir", type=str,
                    dest="log_dir", default="../Logs/train",
                    help="Logs director")
parser.add_argument("--training_pairs_filepath", type=str,
                    dest="training_pairs_filepath",
                    default='C:/Drive_D/Atul/MS/Thesis/3d/Lapirn3d/Code/subjects/train_100_pair_subjects.csv',
                    help="data path for training images")
parser.add_argument("--validation_pairs_datapath", type=str,
                    dest="validation_pairs_datapath",
                    default='C:/Drive_D/Atul/MS/Thesis/3d/Lapirn3d/Code/subjects/val_100_pair_subjects.csv',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=2000,
                    help="Number step for freezing the previous level")
parser.add_argument("--on_server", type=boolean,
                    dest="on_server", default=False,
                    help="Flag to indicate training using server")
parser.add_argument("--datapath", type=str,
                    dest="datapath",

                    default='../Data/OASIS',
                    help="data path for training images")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
validation_check = opt.validation_check
freeze_step = opt.freeze_step
datapath = opt.datapath

# OASIS
names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
training_generator = Data.DataLoader(Dataset_epoch(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
# OASIS (Validation)
imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[-4:]
labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))[-4:]
validation_generator = Data.DataLoader(
                    Dataset_epoch_validation(imgs, labels, norm=True),
                    batch_size=1,
                    shuffle=False, num_workers=2)

# if opt.on_server:
#     datapath = os.path.join(os.sep,'home','student','sharma_atul','3d','neurite-oasis.v1.0')
# else:
#     datapath = os.path.join('c:', os.sep,'Drive_d','Atul','MS','Thesis','dataset','neurite-oasis.v1.0')
# training_pairs_filename = opt.training_pairs_filepath
# validation_pairs_filename = opt.validation_pairs_datapath
# training_pairs = read_pair_list(filename = training_pairs_filename, delim= ',')
# validation_pairs = read_pair_list(filename = validation_pairs_filename, delim= ',')
#
# training_generator = Data.DataLoader(Dataset_epoch(training_pairs, datapath, norm=True), batch_size=1,
#                                         shuffle=False, num_workers=2)
#
# validation_generator = Data.DataLoader(Dataset_epoch_validation(validation_pairs, datapath, norm=True), batch_size=1,
#                                         shuffle=False, num_workers=2)

model_dir = opt.model_dir

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

logs_dir = opt.log_dir
if not os.path.isdir(logs_dir):
    os.mkdir(logs_dir)

model_name = "LapIRN3D_smooth_4_start_4_epochs_1_seg4"


def train_lvl1():
    print("Training lvl1...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow).to(device)

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((4, iteration*epoch_level1 + 1))

    training_counter = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    for epoch in range(epoch_level1):
        for step, [moving_image, fixed_image] in enumerate(training_generator):

            fixed_image = fixed_image.cuda().float()
            moving_image = moving_image.cuda().float()

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            deformation_field, wrapped_moving_image, downsampled_fixed_image_4x, velocity_field, _ = model(moving_image, fixed_image)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(wrapped_moving_image, downsampled_fixed_image_4x)

            deformation_field_norm = transform_unit_flow_to_flow_cuda(deformation_field.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(deformation_field_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = velocity_field.shape
            velocity_field[:, 0, :, :, :] = velocity_field[:, 0, :, :, :] * (z-1)
            velocity_field[:, 1, :, :, :] = velocity_field[:, 1, :, :, :] * (y-1)
            velocity_field[:, 2, :, :, :] = velocity_field[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(velocity_field)

            loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation

            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'epoch "{0}" step "{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    epoch, step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            # plot_graph(training_counter, 1, loss, loss_multiNCC, loss_Jacobian, loss_regulation)

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl1_" + str(epoch) + '_'  + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(epoch) + '_'  + str(step) + '.npy', lossall)

            training_counter += 1
        print("one epoch pass")
    modelname = model_dir + '/' + model_name + "stagelvl1_" + str(epoch_level1*iteration) + '.pth'
    torch.save(model.state_dict(), modelname)
    np.save(model_dir + '/loss_' + model_name + 'stagelvl1.npy', lossall)


def train_lvl2():
    print("Training lvl2...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                          range_flow=range_flow).to(device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
    model_path = sorted(glob.glob(model_dir + '/' + model_name + "stagelvl1_"+  str(epoch_level1*iteration) +".pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((4, iteration*epoch_level2 + 1))

    training_counter = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    for epoch in range(epoch_level2):
        for step, [moving_image, fixed_image] in enumerate(training_generator):

            moving_image = moving_image.to(device).float()
            fixed_image = fixed_image.to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
            deformation_field, wrapped_moving_image, downsampled_fixed_image_2x, velocity_field, _, _ = model(moving_image, fixed_image)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(wrapped_moving_image, downsampled_fixed_image_2x)

            deformation_field_norm = transform_unit_flow_to_flow_cuda(deformation_field.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(deformation_field_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = velocity_field.shape
            velocity_field[:, 0, :, :, :] = velocity_field[:, 0, :, :, :] * (z-1)
            velocity_field[:, 1, :, :, :] = velocity_field[:, 1, :, :, :] * (y-1)
            velocity_field[:, 2, :, :, :] = velocity_field[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(velocity_field)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'epoch "{0}" step "{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    epoch, step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            # plot_graph(training_counter, 2, loss, loss_multiNCC, loss_Jacobian, loss_regulation)

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl2_" + str(epoch) + '_'  + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(epoch) + '_' + str(step) + '.npy', lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            training_counter += 1

        print("one epoch pass")
    modelname = model_dir + '/' + model_name + "stagelvl2_" + str(epoch_level2*iteration) + '.pth'
    torch.save(model.state_dict(), modelname)
    np.save(model_dir + '/loss_' + model_name + 'stagelvl2.npy', lossall)


def train_lvl3():
    print("Training lvl3...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1).to(device)

    # model_path = sorted(glob.glob(model_dir + '/' + model_name + "stagelvl2_"+ str(epoch_level2*iteration) + ".pth"))[-1]
    # model_lvl2.load_state_dict(torch.load(model_path))
    # print("Loading weight for model_lvl2...", model_path)
    #
    # # Freeze model_lvl1 weight
    # for param in model_lvl2.parameters():
    #     param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
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

    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    lossall = np.zeros((4, iteration*epoch_level3 + 1))

    training_counter = 0
    validation_counter = 0

    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.pth"
        print("Loading weight: ", model_path)
        step = 10000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/lossLDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.npy")
        lossall[:, 0:10000] = temp_lossall[:, 0:10000]

    for epoch in range(epoch_level3):
        for step, [moving_image, fixed_image] in enumerate(training_generator):

            moving_image = moving_image.to(device).float()
            fixed_image = fixed_image.to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
            deformation_field, wrapped_moving_image, downsampled_fixed_image_x, velocity_field, _, _, _ = model(moving_image, fixed_image)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(wrapped_moving_image, downsampled_fixed_image_x)

            deformation_field_norm = transform_unit_flow_to_flow_cuda(deformation_field.permute(0,2,3,4,1).clone())

            loss_Jacobian = loss_Jdet(deformation_field_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = velocity_field.shape
            velocity_field[:, 0, :, :, :] = velocity_field[:, 0, :, :, :] * (z-1)
            velocity_field[:, 1, :, :, :] = velocity_field[:, 1, :, :, :] * (y-1)
            velocity_field[:, 2, :, :, :] = velocity_field[:, 2, :, :, :] * (x-1)
            loss_regulation = loss_smooth(velocity_field)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'epoch "{0}" step "{1}" -> training loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                    epoch, step, loss.item(), loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()))
            sys.stdout.flush()

            # plot_graph(training_counter, 3, loss, loss_multiNCC, loss_Jacobian, loss_regulation)

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(epoch) + '_' + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(epoch) + '_' + str(step) + '.npy', lossall)

            # Validation
            if (step % validation_check == 0):
                eval_df = pd.DataFrame()
                model.eval()
                transform_nearest.eval()
                for val_step, [moving_image, fixed_image, moving_image_label, fixed_image_label] in enumerate(validation_generator):

                    with torch.no_grad():
                        moving_image = moving_image.cuda().float()
                        fixed_image = fixed_image.cuda().float()

                        moving_image_label = moving_image_label.cuda().float()
                        fixed_image_label = fixed_image_label.cuda().float()

                        # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
                        deformation_field, wrapped_moving_image, downsampled_fixed_image_x, velocity_field, F_xy_lvl1, F_xy_lvl2, _ = model(moving_image, fixed_image)

                        # 3 level deep supervision NCC
                        val_loss_multiNCC = loss_similarity(wrapped_moving_image, downsampled_fixed_image_x)

                        deformation_field_norm = transform_unit_flow_to_flow_cuda(deformation_field.permute(0,2,3,4,1).clone())

                        val_loss_Jacobian = loss_Jdet(deformation_field_norm, grid)

                        # reg2 - use velocity
                        _, _, x, y, z = velocity_field.shape
                        velocity_field[:, 0, :, :, :] = velocity_field[:, 0, :, :, :] * (z-1)
                        velocity_field[:, 1, :, :, :] = velocity_field[:, 1, :, :, :] * (y-1)
                        velocity_field[:, 2, :, :, :] = velocity_field[:, 2, :, :, :] * (x-1)
                        val_loss_regulation = loss_smooth(velocity_field)

                        val_loss = val_loss_multiNCC + antifold * val_loss_Jacobian + smooth * val_loss_regulation
                        # sys.stdout.write(
                        #     "\r" + 'epoch "{0}" val_step "{1}" -> Validation loss "{2:.4f}" - sim_NCC "{3:4f}" - Jdet "{4:.10f}" -smo "{5:.4f}"'.format(
                        #         epoch, val_step, val_loss.item(), val_loss_multiNCC.item(), val_loss_Jacobian.item(), val_loss_regulation.item()))
                        # sys.stdout.flush()

                        # writer.add_scalar("Validation loss ", val_loss.item(), validation_counter)
                        # writer.add_scalar("Validation sim_NCC ", val_loss_multiNCC.item(), validation_counter)
                        # writer.add_scalar("Validation Jdet ", val_loss_Jacobian.item(), validation_counter)
                        # writer.add_scalar("Validation smo ", val_loss_regulation.item(), validation_counter)

                        wrapped_moving_image_label = transform_nearest(moving_image_label, deformation_field.permute(0, 2, 3, 4, 1), grid_unit).cpu().numpy()[0, 0, :, :, :]

                        # Tranfer the tensors to cpu and convert them to numpy inorder to compute metrics
                        fixed_image = fixed_image.cpu().numpy()[0, 0, :, :, :]
                        wrapped_moving_image = wrapped_moving_image.cpu().numpy()[0, 0, :, :, :]
                        fixed_image_label = fixed_image_label.cpu().numpy()[0, 0, :, :, :]

                        overlap_results_df, surface_distance_results_df = evaluate(fixed_image, wrapped_moving_image,
                                                            sitk.GetImageFromArray(fixed_image_label.transpose(2,1,0)),
                                                            sitk.GetImageFromArray(wrapped_moving_image_label.transpose(2,1,0)))

                        result = pd.concat([overlap_results_df, surface_distance_results_df], axis=1)
                        eval_df = eval_df.append(result, ignore_index=True)
                        validation_counter += 1
                print("Dice: ", np.mean(eval_df['dice']))

                # writer.add_scalar("Validation Dice/step ", np.mean(eval_df['dice']), step)   # dice/validation epoch
                # writer.add_scalar("Validation Mi/step ", np.mean(eval_df['mutual_information']), step)       # Mutual Information/validation epoch
                # writer.add_scalar("Validation volume_similarity/step ", np.mean(eval_df['volume_similarity']), step)       # Volume Similarity/validation epoch

                model.train() # Switch to training

            if step == freeze_step:
                model.unfreeze_modellvl2()

            training_counter += 1

        print("one epoch pass")

    modelname = model_dir + '/' + model_name + "stagelvl3_" + str(epoch_level3*iteration) + '.pth'
    torch.save(model.state_dict(), modelname)
    np.save(model_dir + '/loss_' + model_name + 'stagelvl3.npy', lossall)

def plot_graph(training_counter, level, training_loss, loss_multiNCC, loss_Jacobian, loss_regulation):
    writer.add_scalar("training loss " + str(level), training_loss.item(), training_counter)
    writer.add_scalar("sim_NCC " + str(level), loss_multiNCC.item(), training_counter)
    writer.add_scalar("Jdet " + str(level), loss_Jacobian.item(), training_counter)
    writer.add_scalar("smo " + str(level), loss_regulation.item(), training_counter)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    imgshape = (160, 192, 224)
    imgshape_4 = (160//4, 192//4, 224//4)
    imgshape_2 = (160//2, 192//2, 224//2)

    epoch_level1 = 1
    epoch_level2 = 1
    epoch_level3 = 1

    range_flow = 0.4

    iteration = len(training_generator)

    # writer = SummaryWriter(logs_dir)

    start_t = datetime.now()
    train_lvl1()
    train_lvl2()
    train_lvl3()
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t)
    # writer.flush()
    # writer.close()