import nibabel as nib
import skimage.transform as skTrans
import torch
import torchio as tio

import os
import json

import nibabel as nib
import skimage.transform as skTrans


class MRILiverPairwiseFixed(torch.utils.data.Dataset):
        def __init__(self, root_dir, json_conf='sample.json', masked=False, 
                        downsampled=False, train_transform = None, train=True, is_norm=False):

                self.root_dir = root_dir
                self.image_dir = os.path.join(root_dir,'imagesPreprocessedTr')
                self.keypoint_dir = os.path.join(root_dir,'keypointsTr')
                self.masked = masked
                with open(os.path.join(root_dir,json_conf)) as f:
                        self.dataset_json = json.load(f)

                self.downsampled = downsampled
                self.train = train

                self.is_norm = is_norm

                self.data_pairs = self.dataset_json["training"]
                self.volumes = {}
                
#                 self.__init_dataset(self.image_dir)

                
                #         if self.train :
                #             self.type_data = 'training_paired_images'

                #         else:
                #             self.type_data = 'registration_val'
        
        def __len__(self):

                return len(self.data_pairs)
                #         if self.train:
                #             return self.dataset_json['numPairedTraining']
                #         else:
                #             return len(self.dataset_json['registration_val'])

                #     def get_shape(self):
                #         if self.downsampled:
                #             return [x//2 for x in self.shape]
                #         else:
                #             return self.shape

        def __getitem__(self, idx):
                fix_idx = int(self.data_pairs[idx]['fixed_img'])
                mov_idx = int(self.data_pairs[idx]['moving_img'])

                img_id = self.data_pairs[idx]['img_path']
                volume = nib.load(img_id)
                
                affine = volume.affine
                data = volume.get_fdata()
                W, H, D = data.shape
                
                volume = skTrans.resize(data, (400,400, D), order=1, preserve_range=True)
                # fixed_img = self.volumes[img_id][fix_idx]
                fixed_img = volume[:,:,fix_idx]
                moving_img = volume[:,:,mov_idx]
                
                mask=nib.load(img_id.replace('imagesPreprocessed', 'labelsFixLoc') + '.gz').get_fdata()
                mask = skTrans.resize(mask, (400,400, D), order=0, preserve_range=True)
                
                fixed_mask = mask[:,:,fix_idx]
                moving_mask = mask[:,:,mov_idx]
                
                #         fixed_affine = fixed_img.affine
                #         fixed_img = fixed_img.get_fdata()

                # moving_img = self.volumes[img_id][mov_idx]
                # fixed_img = skTrans.resize(fixed_img, (400,400), order=1, preserve_range=True)
                # moving_img = skTrans.resize(moving_img, (400,400), order=1, preserve_range=True)

                fixed_img=torch.from_numpy(fixed_img).float()
                moving_img=torch.from_numpy(moving_img).float()
                fixed_img = fixed_img.unsqueeze(0)
                moving_img = moving_img.unsqueeze(0)
                
                fixed_mask=torch.from_numpy(fixed_mask).float()
                moving_mask=torch.from_numpy(moving_mask).float()
                fixed_mask = fixed_mask.unsqueeze(0)
                moving_mask = moving_mask.unsqueeze(0)


                #         fixed_mask = fixed_mask.unsqueeze(0)
                #         moving_mask = moving_mask.unsqueeze(0)

                shape = (400, 400)

                zeros = torch.zeros((1, *shape, len(shape)))



                return { "fixed_name" : fix_idx,
                        "moving_name" : mov_idx,
                        "fixed_img" : fixed_img,
                        "moving_img" : moving_img,
                        "fixed_mask" : fixed_mask, 
                        "moving_mask" : moving_mask, 
                        "zero_flow_field" : zeros} 
                #                 
                #                 
                #                 "fixed_affine" : fixed_affine,

        

class MRILiverPairwise(torch.utils.data.Dataset):
        def __init__(self, root_dir, json_conf='sample.json', masked=False, 
                        downsampled=False, train_transform = None, train=True, is_norm=False):

                self.root_dir = root_dir
                self.image_dir = os.path.join(root_dir,'imagesPreprocessedTr')
                self.keypoint_dir = os.path.join(root_dir,'keypointsTr')
                self.masked = masked
                with open(os.path.join(root_dir,json_conf)) as f:
                        self.dataset_json = json.load(f)

                self.downsampled = downsampled
                self.train = train

                self.is_norm = is_norm

                self.data_pairs = self.dataset_json["training"]
                self.volumes = {}
                
                self.__init_dataset(self.image_dir)

                # self.transform = tio.CropOrPad((512,512,129))
                

                #         if self.train :
                #             self.type_data = 'training_paired_images'

                #         else:
                #             self.type_data = 'registration_val'
        
        def __len__(self):

                return len(self.data_pairs)
                #         if self.train:
                #             return self.dataset_json['numPairedTraining']
                #         else:
                #             return len(self.dataset_json['registration_val'])

                #     def get_shape(self):
                #         if self.downsampled:
                #             return [x//2 for x in self.shape]
                #         else:
                #             return self.shape

        def __getitem__(self, idx):
                fix_idx = self.data_pairs[idx]['fixed_img']
                mov_idx = self.data_pairs[idx]['moving_img']

                img_id = self.data_pairs[idx]['img_path']
                
                volume = nib.load(img_id)
                
                affine = volume.affine
                data = volume.get_fdata()
                W, H, D = volume.shape
                
                volume = skTrans.resize(data, (400,400, D), order=1, preserve_range=True)
                
                # fixed_img = self.volumes[img_id][fix_idx]
                
                fixed_img = volume[...,fix_idx]
                moving_img = volume[...,mov_idx]
                #         fixed_affine = fixed_img.affine
                #         fixed_img = fixed_img.get_fdata()

                # moving_img = self.volumes[img_id][mov_idx]
                # fixed_img = skTrans.resize(fixed_img, (400,400), order=1, preserve_range=True)
                # moving_img = skTrans.resize(moving_img, (400,400), order=1, preserve_range=True)

                fixed_img=torch.from_numpy(fixed_img).float()
                moving_img=torch.from_numpy(moving_img).float()
                fixed_img = fixed_img.unsqueeze(0)
                moving_img = moving_img.unsqueeze(0)


                #         fixed_mask = fixed_mask.unsqueeze(0)
                #         moving_mask = moving_mask.unsqueeze(0)

                shape = (400, 400)

                zeros = torch.zeros((1, *shape, len(shape)))



                return { "fixed_name" : fix_idx,
                        "moving_name" : mov_idx,
                        "fixed_img" : fixed_img,
                        "moving_img" : moving_img,
                        # "data" : torch.from_numpy(volume).float().unsqueeze(0), 
                        "zero_flow_field" : zeros} 
                #                 "fixed_mask" : fixed_mask, 
                #                 "moving_mask" : moving_mask,
                #                 
                #                 "fixed_affine" : fixed_affine,
                #                "fixed_kp":fixed_kp,
                #                "moving_kp": moving_kp}
# 
        def __init_dataset(self, image_dir):
                from os import listdir
                from os.path import isfile, join
                volumes = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
                
                for vol in volumes:
                        file_path = os.path.join(image_dir, vol)
                        self.volumes[file_path] = MRILiver(file_path)
                
                



class MRILiver():
        
        def __init__(self, file_path):
                print("setting data", file_path)
                self.file_path = file_path
                volume = nib.load(self.file_path)
                
                self.affine = volume.affine
                self.data = volume.get_fdata()
                
                # add spacing, slice thickness
                
        def __getitem__(self, idx):
                return self.data[...,idx]
                