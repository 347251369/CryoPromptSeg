
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random
from skimage import io
import pandas as pd
import time
import cv2

def read_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

class TestingDataset(Dataset):
    
    def __init__(self, args, image_path, image_size=1024, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None):

        self.args = args
        self.image_path = image_path
        self.image_size = image_size
        self.mode = mode
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.requires_name = requires_name
        self.point_num = point_num
        
        self.transform = A.Compose(
          # ToTensorV2:从 (H, W, C) 的 NumPy 格式转换为 PyTorch 张量格式 (C, H, W)
            [A.Resize(height=1024, width=1024)] + [ToTensorV2()],
            p=1, keypoint_params=A.KeypointParams(format='xy'), is_check_shapes=False
        )
        
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
      
        image_input = {}
        img_path = self.image_path[index]
        # print('img_path', img_path)
        # img_path /root/autodl-tmp/datasets/test_dataset/10028/denoised/077.jpg
        
        # 读取对应的json文件
        file_name=img_path.split('/')[5]
        points_path=self.prompt_path
        # print('points_path', points_path)  
 
        json_name= img_path.split('/')[-1].replace('.jpg', '.json')
        
        json_path = points_path.replace("*",file_name)+json_name
        # print("json_path",json_path)  
        # json_path /root/autodl-tmp/datasets/test_dataset/10028/points2/077.json
    
        with open(json_path, 'r') as file:
            data = np.array(json.load(file))
        
        all_points=[(x , y) for x, y,_ in data]
        img = io.imread(img_path) 
        
        masks_list = []
        mask_path= img_path.replace('denoised', 'masks').replace('.jpg', '_mask.jpg')
              
        ori_np_mask = io.imread(mask_path)
        ori_np_mask = (ori_np_mask > 0).astype(int)
        
        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {mask_path}"
        h, w = ori_np_mask.shape
        
        
        masks_list.append(ori_np_mask)
        masks_list[0] = cv2.resize(masks_list[0], (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        pro_mask_path = img_path.replace('denoised', 'pro_mask2')
        pro_mask = io.imread(pro_mask_path)
        pro_mask = (pro_mask > 200).astype(np.float32)
        masks_list.append(pro_mask)
        
      
        transformed =  self.transform(image=img, masks=masks_list, keypoints=all_points)
        img = transformed['image']
        # print('img', img.shape)
        masks_list = transformed['masks']
        point_coords_list = transformed['keypoints']
        
        img = np.repeat(img, 3, axis=0)
        img = img/255.0

        point_coords = torch.as_tensor(point_coords_list, dtype=torch.float)
        point_labels = torch.ones(point_coords.shape[:1],dtype=torch.int)
        
        image_input["image"] = img.to(torch.float64)
        image_input["label"] = masks_list[0].to(torch.int64).unsqueeze(0)
        image_input["pro_mask"] = masks_list[1].to(torch.float32).unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        # image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        # print('label_path',image_input["label_path"])
        # label_path /root/autodl-tmp/datasets/test_dataset/10017/masks

        if self.return_ori_mask:
            ori_mask = torch.tensor(ori_np_mask, dtype=torch.float64).unsqueeze(0)
            image_input["ori_label"] = ori_mask
     
        image_name = img_path.split('/')[-1]
        if self.requires_name:
          
          # TODO 返回路径就包括了名字
            image_input["name"] = img_path
            return image_input
        else:
            return image_input


class TrainingDataset(Dataset):
    def __init__(self, args, image_path, point_path, image_size=1024, mode='train', requires_name=True, point_num=1, mask_num=1):

        self.args = args
        self.image_path = image_path
        self.point_path = point_path
        self.image_size = image_size
        self.mode = mode
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        
        self.transform = A.Compose(
          # ToTensorV2:从 (H, W, C) 的 NumPy 格式转换为 PyTorch 张量格式 (C, H, W)
            [A.Resize(height=1024, width=1024), A.HorizontalFlip(p=0.5) , A.VerticalFlip(p=0.5),A.RandomRotate90(p=0.5)] + [ToTensorV2()],
            p=1, keypoint_params=A.KeypointParams(format='xy'), is_check_shapes=False
        )
        
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, index):
        image_input = {}
        
        img_path = self.image_path[index]
        # print('img_path', img_path)

        # 读取对应的json文件
        file_name=img_path.split('/')[5]
        # print(file_name)
        points_path=self.point_path

        anno_json = read_from_json(f'{points_path}/{file_name}/{file_name}.json')
        # print('anno_json', anno_json)
        classes = anno_json.pop('classes')
        data = anno_json
        
        csv_path = img_path.split('/')[-1].replace('.jpg', '.csv')
        
        # all_points= [np.array(point).reshape(-1, 2) for point in data[csv_path]]
        all_points= data[csv_path][0]
        # print(all_points)
        
        img = io.imread(img_path)
        
        masks_list = []
        mask_path= img_path.replace('denoised', 'masks').replace('.jpg', '_mask.jpg')
        # print(mask_path)
        mask = io.imread(mask_path)
        mask = (mask > 0).astype(int)
        masks_list.append(mask)
                
        chosen_indices=random.sample(range(len(all_points)), min(len(all_points), self.point_num))
        
        
        # # mask_curr
        # diameter_path = f'{points_path}/{file_name}/particle_coordinates/'+img_path.split('/')[-1].replace('.jpg', '.csv')
        # # print('diameter_path', diameter_path)
        # df = pd.read_csv(diameter_path)
        # # 获取第三列数据
        # diameter_column = df.iloc[:, 2]
        # # 获取唯一值
        # diameter = diameter_column.unique()
        #  mask_curr = np.zeros_like(mask)
        
      

        prompt_points = []
        for pid in chosen_indices:
            x , y=all_points[pid]
            prompt_points.append((x,y))
            
            
        #     x ,y = int(x), int(y)
        #     # mask_single = np.zeros_like(mask)
        #     mask_single = np.zeros(mask.shape, dtype=np.uint8).copy(order='C')
        #     # 计算圆的半径（diameter / 2）
        #     radius = int(diameter / 2)

        #     # # 获取坐标网格
        #     # yy, xx = np.ogrid[:mask_single.shape[0], :mask_single.shape[1]]
        #     # # 计算每个点到圆心的距离，并生成圆
        #     # distance = (xx - x)**2 + (yy - y)**2
        #     # mask_single[distance <= radius**2] = 255  # 在圆内的点设为 255
        #     # mask_single =(mask_single>0).astype(int) 
            
        #     cv2.circle(mask_single, (x, y), radius, 1, thickness=-1)  # 直接填充圆区域为1
            
        #     mask_curr = np.logical_or(mask_single,mask_curr)
        # # print('mask_curr', mask_curr.shape)
        # mask_curr = np.array(mask_curr,dtype=int)
        
        
        while len(prompt_points)<self.point_num: # repeat prompt to ensure the same size
            prompt_points.append((x,y))
        

        # masks_list.append(mask_curr)
        
        
        pro_mask_path = img_path.replace('denoised', 'pro_mask')
        pro_mask = io.imread(pro_mask_path)
        pro_mask = (pro_mask > 200).astype(np.float32)
        masks_list.append(pro_mask)
        
        # 获取原始 mask[0] 尺寸
        H_orig, W_orig = masks_list[0].shape
        # print('H_orig', H_orig, 'W_orig', W_orig)
        # resize mask0 到 1024x1024
        masks_list[0] = cv2.resize(masks_list[0], (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # 对 all_points 做等比例缩放
        scale_x = 1024 / W_orig
        scale_y = 1024 / H_orig
        scaled_points = [(int(round(x * scale_x)), int(round(y * scale_y))) for (x, y) in prompt_points]

        transformed  =  self.transform(image=img, masks=masks_list, keypoints=scaled_points)

        img = transformed['image']
        # print('img', img.shape)
        masks_list = transformed['masks']
        point_coords_list = transformed['keypoints']
        # print('prompt_points', point_coords_list)
        

        img = np.repeat(img, 3, axis=0)
        img = img/255.0
        
        point_coords = torch.tensor(point_coords_list)
        # print("point_coords",point_coords.shape)
        # point_coords torch.Size([20, 2])
        point_labels = torch.ones(point_coords.shape[:1],dtype=torch.int32)
  
        image_input["image"] = img.to(torch.float64)
        image_input["label"] = masks_list[0].unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["pro_mask"] =  masks_list[1].unsqueeze(0)

        image_name = img_path.split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
          
def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
      # 如果某个键对应的值是列表类型，则直接将其复制到输出字典中。
        if isinstance(v, list):
            out_dict[k] = v
        else:
          # 如果值不是列表类型（通常是一个张量），则对其进行形状调整。具体操作是将张量的第一个维度展平为 -1（即自动计算大小），并保留其余维度的形状。
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict


if __name__ == "__main__":
    train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

