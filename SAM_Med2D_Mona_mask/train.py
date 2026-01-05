import sys
import os
import inspect
# 添加当前文件夹到 Python 搜索路径
sys.path.insert(0, os.path.abspath("."))
import segment_anything
from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from DataLoader import TestingDataset,stack_dict_batched
from torch.utils.data import ConcatDataset
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F

from apex import amp
import random
import json
from pathlib import Path
import wandb
import glob
from tensorboardX import SummaryWriter
from utils import *
from eval import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 重新训练前进行修改
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/checkpoint/SAM_Med2D_mona_mask1", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-cryo", help="run model name")
    
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    
    parser.add_argument("--num_workers", type=int, default=16, help="train num_workers")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")

    parser.add_argument("--point_num", type=int, default=50, help="get mask number")
    parser.add_argument("--mask_num", type=int, default=1, help="get mask number")
    
    parser.add_argument("--train_dataset_path_1", type=str, default="/root/autodl-tmp/datasets/train_dataset/*/val/denoised/*.jpg", help="train data path")
    parser.add_argument("--train_point_path_1", type=str, default="/root/Promptpoint/SFS_pro_Unet/datasets/train_dataset", help="train data path")
    
    parser.add_argument("--train_dataset_path_2", type=str, default="/root/Promptpoint/SAM_Med2D_Mona_mask/half_file_paths_all.txt", help="train data path")
    parser.add_argument("--train_point_path_2", type=str, default="/root/Promptpoint/SFS_pro_Unet/datasets/test_dataset", help="train data path")
    
    
    parser.add_argument("--test_dataset_path_1", type=str, default="/root/Promptpoint/SAM_Med2D_Mona_mask/10017.txt", help="train data path")
    parser.add_argument("--test_point_path_1", type=str, default="/root/autodl-tmp/datasets/test_dataset/*/points2/", help="train data path")
    
    parser.add_argument("--test_dataset_path_2", type=str, default="/root/Promptpoint/SAM_Med2D_Mona_mask/10093.txt", help="train data path")
    parser.add_argument("--test_point_path_2", type=str, default="/root/autodl-tmp/datasets/test_dataset/*/points2/", help="train data path")
    
    parser.add_argument("--test_dataset_path", type=str, default="/root/autodl-tmp/datasets/test_dataset/*/", help="test data path")
    parser.add_argument("--test_point_path", type=str, default="/root/Promptpoint/SFS_pro_Unet/datasets/test_dataset", help="test data path")
    
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume") 
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    # parser.add_argument("--sam_checkpoint", type=str, default="/root/Promptpoint/finetune-SAM/sam_vit_b_01ec64.pth", help="sam checkpoint")
    parser.add_argument("--sam_checkpoint", type=str, default="/root/autodl-tmp/checkpoint/SAM_Med2D_mona_mask1/models0/sam_best.pth", help="sam checkpoint")
    
    
    # parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    
    
    # parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    
    
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=False, help="use amp")
    
    
    
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            # 如果键是image或label，将其转换为float类型并移动到目标设备。
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            # 如果值是list或torch.Size类型，保持原样
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
              # 其他类型的数据直接移动到目标设备。
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("pro_mask", None),
                # masks= None,
            )

    else:
        low_res_masks = F.interpolate(batched_input.get("pro_mask").float(), size=(args.image_size//4,args.image_size//4))
        
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=low_res_masks,
            #  masks= None,
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
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
    # TODO 查看返回的内容shape
    # print("mask",low_res_masks.shape)
    # mask torch.Size([2, 1, 256, 256])
    # print(iou_predictions.shape)
    # torch.Size([2, 1])
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch , iter_num, criterion ,writer):

    train_loader = tqdm(train_loader)
    train_losses = []
    # args.metrics  ['iou', 'dice']
    train_iter_metrics = [0] * len(args.metrics)
    
    for batch, batched_input in enumerate(train_loader):
        # batched_input = stack_dict_batched(batched_input)
        # print("batched_input",batched_input['point_coords'].shape)
        # batched_input torch.Size([2, 3, 1024, 1024])
        batched_input = to_device(batched_input, args.device)
        
        for n, value in model.image_encoder.named_parameters():
            # if "Adapter" in n:
            if "mona" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False

        if args.use_amp:
            labels = batched_input["label"].half()
            image_embeddings = model.image_encoder(batched_input["image"].half())

            # B, _, _, _ = image_embeddings.shape
            # image_embeddings_repeat = []
            # for i in range(B):
            #     image_embed = image_embeddings[i]
            #     image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            #     image_embeddings_repeat.append(image_embed)
            # image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            # TODO loss返回的形状，好保存数据
            loss = criterion(masks, labels, iou_predictions)
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=False)

        else:
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])

            # B, _, _, _ = image_embeddings.shape
            # image_embeddings_repeat = []
            # for i in range(B):
            #     image_embed = image_embeddings[i]
            #     image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
            #     image_embeddings_repeat.append(image_embed)
            # image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            # print("image_embeddings",image_embeddings.shape)
            # image_embeddings torch.Size([40, 256, 64, 64])
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False)
            loss = criterion(masks, labels, iou_predictions)
            # print("loss", loss.shape,loss)
            # loss torch.Size([]) tensor(0.8531, device='cuda:0', grad_fn=<AddBackward0>
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if int(batch) % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch}: {SegMetrics(masks, labels, args.metrics)}')

        train_losses.append(loss.item())
        iter_num+=1
        # TODO 查看保存形式

        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        # 更新 tqdm 进度条的后缀信息，以便在训练过程中实时显示当前批次的损失值和 GPU 信息
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        # 保存每一批次的IOU和dice
        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        # 用于累积整个训练过程中每个指标的值。
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
        
        # TODO
        writer.add_scalar('iter/total_loss', loss, iter_num)
        writer.add_scalar('iter/iou', train_batch_metrics[0], iter_num)
        writer.add_scalar('iter/dice', train_batch_metrics[1], iter_num)

    return train_losses, train_iter_metrics , iter_num



def main(args):
    # 使用 Path（来自 pathlib 模块）创建一个目录，路径由 args.dir_checkpoint 指定。
    Path(args.work_dir).mkdir(parents=True,exist_ok = True)
    path_to_json = os.path.join(args.work_dir, "args.json")
    args_dict = vars(args)
    with open(path_to_json, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    
    # 模型加载是否正确 --正确 
    print(segment_anything.__file__)
    # /root/Promptpoint/SAM_Med2D_Mona_mask/segment_anything/__init__.py
    print(inspect.getfile(SamPredictor))
    # /root/Promptpoint/SAM_Med2D_Mona_mask/segment_anything/predictor.py
    # TODO 权重加载
    
    model = sam_model_registry[args.model_type](args).to(args.device) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = FocalDiceloss_IoULoss()

    if args.lr_scheduler:
      # TODO 是否成功
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75], gamma = 0.5)
        print('*******Use MultiStepLR')

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        print("*******Mixed precision with Apex")
    else:
        print('*******Do not use mixed precision')
        
    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # 恢复学习率调度器
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])  
            if 'amp' in checkpoint:
                amp.load_state_dict(checkpoint["amp"])
            # 恢复训练轮数
            start_epoch = checkpoint['epoch']+1
            iter_num = checkpoint['iter_num']
            # best_loss = checkpoint['best_loss']
            best_dice_10017 = checkpoint['best_dice_10017']
            best_dice_10093 = checkpoint['best_dice_10093']
            # average_loss =checkpoint['average_loss']
            print(f"*******load {args.resume}")
            print(f"Resuming training from epoch {start_epoch} and iteration {iter_num}")
    else:
        start_epoch = 0
        # best_loss = 1e10
        best_dice_10017=0
        best_dice_10093=0
        # # 用于记录当前的迭代次数
        iter_num = 0
        
    print("epoch",start_epoch)

    train_image_path_1 = list(glob.glob(args.train_dataset_path_1))
    # txt
    # with open(args.train_dataset_path_1, 'r') as f:
    #     train_image_path_1 = [line.strip() for line in f if line.strip()]
    train_dataset_1 = TrainingDataset(args, train_image_path_1, point_path=args.train_point_path_1,image_size=args.image_size, mode='train', point_num=args.point_num, mask_num=args.mask_num, requires_name = True)
    # txt
    with open(args.train_dataset_path_2, 'r') as f:
        train_image_path_2 = [line.strip() for line in f if line.strip()]
    train_dataset_2 = TrainingDataset(args,train_image_path_2, point_path=args.train_point_path_2,image_size=args.image_size,mode='train',point_num=args.point_num,mask_num=args.mask_num,requires_name=True)
    
    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('*******Train data:', len(train_dataset))   

    # test
    with open(args.test_dataset_path_1, 'r') as f:
        test_image_path_1 = [line.strip() for line in f if line.strip()]
    test_dataset_1 = TestingDataset(args, test_image_path_1, image_size=args.image_size, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=args.test_point_path_1)
    test_loader_1 = DataLoader(test_dataset_1, batch_size = 1, shuffle=False, num_workers=4)
    print('*******Test data1:', len(test_loader_1)) 
    
    with open(args.test_dataset_path_2, 'r') as f:
        test_image_path_2 = [line.strip() for line in f if line.strip()]
    test_dataset_2 = TestingDataset(args, test_image_path_2, image_size=args.image_size, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=args.test_point_path_2)
    test_loader_2 = DataLoader(test_dataset_2, batch_size = 1, shuffle=False, num_workers=4)
    print('*******Test data2:', len(test_loader_2)) 
    
    
    # 在work_dir创建一个logs文件夹,包括一个sam-cryo_20250321-1423.log文件
    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    # TODO
    writer = SummaryWriter(args.work_dir + '/logs')
    
    l = len(train_loader)

    print("Starting training!",start_epoch)
    
    for epoch in range(start_epoch, args.epochs):
        
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models"), exist_ok=True)
        
        train_losses, train_iter_metrics, iter_num = train_one_epoch(args, model, optimizer, train_loader, epoch, iter_num, criterion ,writer)

        if args.lr_scheduler is not None:
            scheduler.step()

        # 一个epoceh的评价iou 和 dice 
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        # 一个epoceh的loss
        average_loss = np.mean(train_losses)
        # 如果使用了学习率调度器（args.lr_scheduler不为None），则从调度器中获取当前的学习率。
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
        loggers.info(f"epoch: {epoch}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")
        
        writer.add_scalar('epoch/lr', lr, epoch)
        writer.add_scalar('epoch/loss', average_loss, epoch)
        writer.add_scalar('epoch/iou', train_iter_metrics[0], epoch)
        writer.add_scalar('epoch/dice', train_iter_metrics[1], epoch)
        
        test_metrics_10017=evaluate(args , model , test_loader_1, loggers , epoch, filename='10017')
        test_metrics_10093=evaluate(args , model , test_loader_2, loggers , epoch, filename='10093')
        
        dice_10017 = float(test_metrics_10017['dice'])
        dice_10093 = float(test_metrics_10093['dice'])
        
        if dice_10017 > best_dice_10017:
            best_dice_10017=dice_10017
            save_path = os.path.join(args.work_dir, "models", f"sam_best_10017.pth")
            state = {
                    'model': model.float().state_dict(), 
                    'epoch': epoch,
                    'iter_num' : iter_num,
                    # 'best_loss':best_loss,
                    "best_dice_10017":best_dice_10017,
                    'test_metrics':test_metrics_10017 
            }
            torch.save(state, save_path)
            
            if args.use_amp:
                model = model.half()
                
        if dice_10093 > best_dice_10093:
            best_dice_10093=dice_10093
            save_path = os.path.join(args.work_dir, "models", f"sam_best_10093.pth")
            state = {
                    'model': model.float().state_dict(), 
                    'epoch': epoch,
                    'iter_num' : iter_num,
                    # 'best_loss':best_loss,
                    "best_dice_10093":best_dice_10093,
                    'test_metrics':test_metrics_10093
                    
            }
            torch.save(state, save_path)
            
            if args.use_amp:
                model = model.half()  
         
          # 保存训练状态
        save_path = os.path.join(args.work_dir, "models", f"sam_latest.pth")
        
        checkpoint = {
            'epoch': epoch,
            'iter_num' : iter_num,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'average_loss':average_loss,
            # 'best_loss': best_loss,
            "best_dice_10017":best_dice_10017,
            "best_dice_10093":best_dice_10093,
            'train_metrics':train_metrics
        }
        if args.lr_scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        if args.use_amp:
            checkpoint["amp"] = amp.state_dict()
        torch.save(checkpoint ,save_path)

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)


