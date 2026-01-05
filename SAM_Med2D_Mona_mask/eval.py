import sys
import os
import inspect
# 添加当前文件夹到 Python 搜索路径
sys.path.insert(0, os.path.abspath("."))
import segment_anything
from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset,stack_dict_batched
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
import glob


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input
def postprocess_masks(low_res_masks, image_size, original_size):
      ori_h, ori_w = original_size
      masks = F.interpolate(
          low_res_masks,
          (image_size, image_size),
          mode="bilinear",
          align_corners=False,
          )
      
      if ori_h < image_size and ori_w < image_size:
          top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
          left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
          masks = masks[..., top : ori_h + top, left : ori_w + left]
          pad = (top, left)
      else:
          masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
          pad = None 
      return masks, pad
  
def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        low_res_masks = F.interpolate(batched_input.get("pro_mask").float(), size=(args.image_size//4,args.image_size//4))
        
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            # masks=batched_input.get("pro_mask", None),
            masks=low_res_masks,
            # masks=None
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
        
    # print("mask",low_res_masks.shape)
    # mask torch.Size([72, 1, 256, 256])
    low_res_masks = torch.sigmoid(low_res_masks)
    low_res_masks[low_res_masks > 0.5] = int(1)
    low_res_masks[low_res_masks <= 0.5] = int(0)
    
    low_res_masks=torch.max(low_res_masks, dim=0)[0].unsqueeze(0)
    # print("mask",low_res_masks.shape)
    
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions
  
  
def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True
      
      
def evaluate(args , model, test_loader , loggers,epoch, filename ):
  
    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])
            
        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
        
        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        
        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
            
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    print(f"metrics: {test_metrics}")
    loggers.info(f"epoch: {epoch},{filename}_metrics: {test_metrics}")
    
    return test_metrics