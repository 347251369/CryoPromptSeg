import json
import os

import scipy.io
import torch
import numpy as np
import albumentations as A

from skimage import io
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def read_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class DataFolder(Dataset):
    def __init__(self,
            cfg,
            partA_image,
            mode
    ):
        self.cfg = cfg
        self.partA_image = partA_image
        self.partB_image = []
        if mode != 'test':
            for path in partA_image:
               self.partB_image.append(path.replace('partA', 'partB'))

        # self.cutoff = cfg.data.denoise.pixel_cutoff

        self.keys = ['image','image2','keypoints','mask']
        self.phase = mode

        self.transform = A.Compose(
          # ToTensorV2:从 (H, W, C) 的 NumPy 格式转换为 PyTorch 张量格式 (C, H, W)
            [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get(mode).transform] + [ToTensorV2()],
            p=1, keypoint_params=A.KeypointParams(format='xy'), 
            additional_targets={'image2':"image"}, is_check_shapes=False
        )

    def __len__(self):
        return len(self.partA_image)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        partA_path = self.partA_image[index]
        # print(partA_path)
        
        if self.phase=='test':
            partB_path=partA_path
            file_name=partA_path.split('/')[-3]
            points_path=self.cfg.data.test_point_path
        else:
            partB_path=self.partB_image[index]
            file_name=partA_path.split('/')[-4]
            points_path=self.cfg.data.train_point_path
            
        # # # 训练
        # if self.phase !='train':
        #     partB_path=partA_path
        #     file_name=partA_path.split('/')[-4]
        #     points_path=self.cfg.data.train_point_path
        # else:
        #     partB_path=self.partB_image[index]
        #     file_name=partA_path.split('/')[-4]
        #     points_path=self.cfg.data.train_point_path
        # print(partB_path)
        anno_json = read_from_json(f'{points_path}/{file_name}/{file_name}.json')
        classes = anno_json.pop('classes')
        data = anno_json
        
        if self.phase !='train':
            csv_path = partA_path.split('/')[-1].replace('.jpg', '.csv')
        else:
            csv_path = partA_path.split('/')[-1].replace('_denoised.jpg', '.csv')
        # print(csv_path)
        values = ([io.imread(partA_path)] + [io.imread(partB_path)]+
                  [np.array(point).reshape(-1, 2) for point in data[csv_path]])

        if self.phase !='train':
            mask_path= partA_path.replace('images', 'masks').replace('.jpg', '_mask.jpg')
        else:
           # mask_path= partA_path.replace('partA', 'masks').replace('.jpg', '_mask.jpg')
            mask_path = partA_path.replace('partA', 'masks').replace('_denoised.jpg', '_mask.jpg')
        # print(mask_path)
        mask = io.imread(mask_path)
        mask = (mask > 0).astype(float)

        values.append(mask)

        sample = dict(zip(self.keys, values))
        res = self.transform(**sample)
        res = list(res.values())

        ori_shape = res[0].shape[1:]
        
        img_partA = np.repeat(res[0], 3, axis=0)
        img_partA = img_partA/255.0
        
        # partB不用3通道，因为在model里面会把partA变成1通道
        # img_partB = np.repeat(res[1], 3, axis=0)
        img_partB = res[1]
        img_partB = img_partB/255.0
        # img = res[0]
        points = torch.Tensor(res[2]) 
        # label
        labels = torch.full((len(points),), 0)

        mask = res[-1]
        
        return img_partA, img_partB , points , labels, mask, torch.as_tensor(ori_shape), partA_path
