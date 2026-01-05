import os.path

import torch
import glob
import json
import numpy as np
from tqdm import tqdm
from skimage import io
from utils import predict, mkdir
from models.dpa_p2pnet import build_model

from main import parse_args
from mmengine.config import Config
from PIL import Image
from dataset import DataFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from engine import train_one_epoch, evaluate


def main():
    args = parse_args()
    cfg = Config.fromfile(f'config/{args.config}')

    device = torch.device(args.device)

    model = build_model(cfg)
    # ckpt = torch.load(f'checkpoint/{args.resume}/best.pth', map_location='cpu')
    ckpt = torch.load(f'{args.resume}', map_location='cpu')
    pretrained_state_dict = ckpt['model']

    model.load_state_dict(pretrained_state_dict)
    model.eval()
    model.to(device)
    
    val_image_path = list(glob.glob(cfg.data.val_partA_image))
    val_dataset = DataFolder(cfg, val_image_path,'val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=False
    )
    # print(len(val_dataloader))
    # print(cfg.data.val_partA_image)
    
    
    test_dataset_path = list(glob.glob(cfg.data.test_dataset_path))
    test_dataset = DataFolder(cfg, test_dataset_path,'test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=False
    )
    
    print(len(test_dataloader))
    print(cfg.data.test_dataset_path)
    
    metrics, metrics_string = evaluate(
              cfg,
              model,
              test_dataloader,
              device
          )
    print(cfg.data.test_dataset_path)
    with open('table_output.txt', 'w') as file:
        file.write(metrics_string)


    # print(unet_y)
    # arr = unet_y
    # greater_than_1_elements = arr[arr > 1]

    # print(greater_than_1_elements)
    
    # unet_y = (unet_y - unet_y.mean())/unet_y.std()
    # 量化过程会将浮点数范围 [mi, ma] 映射到整数范围 [0, 255]，以便于图像保存。
    # im = Image.fromarray(quantize(unet_y, mi=-3, ma=3))
    # im.save(path, 'png')


    # denoised_image = np.clip(unet_y, 0, 1)  # 如果值在 [0, 1]，先进行限制
    # denoised_image = (denoised_image * 255).astype(np.uint8)  # 将值映射到 [0, 255] 并转换为 uint8 类型

    # # 将图像保存为文件
    # output_path = 'path_to_save/denoised_image.png'  # 修改为你希望保存的路径
    # Image.fromarray(denoised_image).save(output_path)
if __name__ == '__main__':
    main()   


