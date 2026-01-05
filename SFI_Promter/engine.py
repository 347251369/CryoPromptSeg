import sys
import math
import itertools
import prettytable as pt

import torch
import torch.nn as nn
from utils import *
from tqdm import tqdm
from eval_map import eval_map
from collections import OrderedDict
import json
from PIL import Image

def train_one_epoch(
        args,
        model,
        train_loader,
        criterion,
        # optimizer_d,
        optimizer_p,
        # scheduler_d,  
        scheduler_p,
        epoch,
        device,
        model_ema=None,
        scaler=None

):
    model.train()
    criteria_d = nn.MSELoss()
    criterion.train()
    
    n = 0
    loss_accum = 0

    log_info = dict()

    # 创建一个 MetricLogger 对象，用于记录和显示训练指标。
    metric_logger = MetricLogger(delimiter="  ")
    # 添加一个平滑值记录器来跟踪学习率。
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # 使用 metric_logger.log_every 方法遍历训练数据加载器，并每隔 args.print_freq 步记录一次日志
    for data_iter_step, (img_partA, img_partB,  masks, points_list, labels_list) in enumerate(
            metric_logger.log_every(train_loader, args.print_freq, header)):
      
        img_partA = img_partA.to(device)
        masks = masks.to(device)
        
        # if data_iter_step % 2 == 0:
        #     optimizer_d.zero_grad()
        #     # img_partB应该是（b,1,h,w)，觉得确认
        #     img_partB = img_partB.to(device)
        #     # print("img_partB:",img_partB.shape)
        #     # img_partB: torch.Size([2, 1, 1024, 1024])
        #     _ , unet_y = model(img_partA)
        #     Loss = criteria_d(unet_y, img_partB)
        #     # print("loss_d:",Loss)          
        #     # print("unet_y:",unet_y.shape)
        #     # unet_y: torch.Size([2, 1, 1024, 1024])
        #     Loss.backward()
        #     optimizer_d.step()#更新参数
            
        #     # 关于topaz打印loss的后续
        #     # 将损失张量转换为标量。
        #     Loss = Loss.item()
        #     # 获取当前批次的大小。
        #     b = img_partA.size(0)
            
        #     n += b
        #     # 计算当前批次损失与累积损失之间的差异。
        #     delta = b*(Loss - loss_accum)
        #     # 更新累积损失。
        #     loss_accum += delta/n
            
        #     log_info["loss_d"]= loss_accum
            
        #     metric_logger.update(loss_d=loss_accum)
            
        targets = {
            'gt_masks': masks,
            'gt_nums': [len(points) for points in points_list],
            'gt_points': [points.view(-1, 2).to(device).float() for points in points_list],
            'gt_labels': [labels.to(device).long() for labels in labels_list],
        }
        
        # 使用 torch.cuda.amp.autocast 上下文管理器来启用自动混合精度
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs, _ = model(img_partA)
            loss_dict = criterion(outputs, targets, epoch)
            losses = sum(loss for loss in loss_dict.values())
            # print("loss_p:",losses)
            # loss_p: tensor(0.2370, device='cuda:0', grad_fn=<AddBackward0>)

        # 使用  reduce_dict 函数对 loss_dict 进行规约（例如在分布式训练中将不同设备上的损失值汇总
        loss_dict_reduced = reduce_dict(loss_dict)
        # 将规约后的损失字典中的所有损失项相加，得到总的损失值
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # print("losses_reduced",losses_reduced)
        # losses_reduced tensor(0.2370, device='cuda:0', grad_fn=<AddBackward0>)
        loss_value = losses_reduced.item()

        # 遍历规约后的损失字典 loss_dict_reduced，将每个损失项累加到 log_info 字典中，用于后续的日志记录或监控。
        for k, v in loss_dict_reduced.items():
            # print(f"{k}: {v.item()}")
            # print(log_info.get(k, 0))
            # loss_reg: 0.06168612465262413
              # 0
              # loss_cls: 0.07033073157072067
              # 0
              # loss_mask: 0.10500810295343399
              # 0
            # log_info 字典中键为 k 的值，如果该键不存在，则返回默认值 0。
            log_info[k] = log_info.get(k, 0) + v.item()

        # 检查 loss_value 是否为有限数值（即不是无穷大或 NaN）。如果不是有限数值，则打印当前损失并终止训练，以防止模型崩溃。
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer_p.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer_p)
            scaler.update()
        else:
            losses.backward()
            if args.clip_grad > 0:  # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer_p.step()

        if model_ema and data_iter_step % args.model_ema_steps == 0:
          # 使用当前模型的参数更新 EMA 模型的参数
            model_ema.update_parameters(model)
            if epoch < args.warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                # 将 EMA 模型的计数器 (`n_averaged`) 重置为0。在预热期内，EMA 模型的参数直接复制自当前模型，
                # 而不进行平滑处理，以确保 EMA 模型能够快速跟上当前模型的变化。
                model_ema.n_averaged.fill_(0)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer_p.param_groups[0]["lr"])
        
        # scheduler_d.step()
        scheduler_p.step()
               
      
    return log_info

@torch.inference_mode()
def evaluate(
        cfg,
        model,
        test_loader,
        device,
        epoch=0,
        calc_map=False,
):
    model.eval()
    # class_names = test_loader.dataset.classes
    # num_classes = len(class_names)
    num_classes = 1

    # 用于存储分类任务的预测结果和真实标签。
    cls_predictions = []
    cls_annotations = []
    # 每个列表包含 num_classes 个元素，分别表示分类任务中的预测数和真实样本数。
    cls_pn, cls_tn = list(torch.zeros(num_classes).to(device) for _ in range(2))
    # 一个长度为 num_classes 的零张量，用于记录分类任务中的其他统计信息（例如召回率分母）。
    cls_rn = torch.zeros(num_classes).to(device)

    # 用于记录检测任务的预测结果和真实标签。
    det_pn, det_tn = list(torch.zeros(1).to(device) for _ in range(2))
    det_rn = torch.zeros(1).to(device)

    iou_scores = []
    
    criteria_d = nn.MSELoss()
    n = 0
    loss_d = 0

    epoch_iterator = tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
                          dynamic_ncols=True, disable=not is_main_process())

    for data_iter_step, (img_partA,img_partB, gt_points, labels, masks, ori_shape , image_path) in enumerate(epoch_iterator):
        assert len(img_partA) == 1, 'batch size must be 1'
        # 确保在分布式训练中，每个测试样本只会被一个进程评估，避免了重复计算，提高了评估效率
        if data_iter_step % get_world_size() != get_rank():  # To avoid duplicate evaluation for some test samples
            continue

        epoch_iterator.set_description(
            "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))

        img_partA = img_partA.to(device)
        img_partB = img_partB.to(device)
        
        pd_points, pd_scores, pd_classes, pd_masks, unet_y = predict(
            model,
            img_partA,
            ori_shape=ori_shape[0].numpy(),
            filtering=cfg.test.filtering,
            nms_thr=cfg.test.nms_thr,
        )
        
        # 这部分只有在predeict_prompts.py中才使用
        save_content = np.concatenate([pd_points, pd_classes[:, None]], axis=-1).tolist()
        # 保存为 JSON 文件

        # val -4
        file_name=image_path[0].split('/')[-3]
        folder_path = f'/root/autodl-tmp/datasets/test_dataset/{file_name}/points2'

        os.makedirs(folder_path, exist_ok=True)
        output_path = f'{folder_path}/{image_path[0].split("/")[-1][:-4]}.json'
        # # 获取目标文件的目录路径
        with open(output_path, 'w') as f:
            json.dump(save_content, f)
            
      #   # 这部分只有在predeict_prompts.py中才使用
      #   denoise_image = unet_y.clone().squeeze().cpu().numpy()
      #   denoise_image = (denoise_image - denoise_image.mean())/denoise_image.std()
      # #  量化过程会将浮点数范围 [mi, ma] 映射到整数范围 [0, 255]，以便于图像保存。
      #   im = Image.fromarray(quantize(denoise_image, mi=-3, ma=3))
      #   folder_path = f'/root/autodl-tmp/datasets/test_dataset/{file_name}/denoised'
      #   os.makedirs(folder_path, exist_ok=True)
      #   denoise_path = f'{folder_path}/{image_path[0].split("/")[-1][:-4]}.jpg'
      #   im.save(denoise_path)
        
         # 这部分只有在predeict_prompts.py中才使用                
        mask = (pd_masks * 255).astype(np.uint8)  # 转换为 0 和 255
        img = Image.fromarray(mask)
        folder_path = f'/root/autodl-tmp/datasets/test_dataset/{file_name}/pro_mask2'
        os.makedirs(folder_path, exist_ok=True)
        pro_mask_path = f'{folder_path}/{image_path[0].split("/")[-1][:-4]}.jpg'
        img.save(pro_mask_path)
        
        # loss_ = criteria_d(unet_y, img_partB).item()
        # b = img_partA.size(0)
        # n += b
        # delta = b*(loss_ - loss_d)
        # loss_d += delta/n
        
        # 计算预测掩码（pd_masks）与真实掩码（masks）之间的交并比（IoU），并将结果存储在 iou_scores 列表中
        if pd_masks is not None:
            masks = masks[0].numpy()
            intersection = (pd_masks * masks).sum()
            union = (pd_masks.sum() + masks.sum() + 1e-7) - intersection
            iou_scores.append(intersection / (union + 1e-7))
            
        # 真实点位及其标签
        gt_points = gt_points[0].reshape(-1, 2).numpy()
        labels = labels[0].numpy()

        cls_annotations.append({'points': gt_points, 'labels': labels})

        # 对每个类别（class）的预测点和真实点进行处理
        cls_pred_sample = []
        for c in range(cfg.data.num_classes):
            # ind 是布尔索引数组，表示哪些预测点属于类别 c
            ind = (pd_classes == c)
            category_pd_points = pd_points[ind]
            category_pd_scores = pd_scores[ind]
            # 真实点位，表示哪些真实点属于类别 c
            category_gt_points = gt_points[labels == c]

            # 使用 np.concatenate 将预测点坐标和分数合并为一个数组
            cls_pred_sample.append(np.concatenate([category_pd_points, category_pd_scores[:, None]], axis=-1))

            pred_num, gd_num = len(category_pd_points), len(category_gt_points)
            cls_pn[c] += pred_num
            cls_tn[c] += gd_num

            if pred_num and gd_num:
              # 调用 get_tp 函数计算正确匹配的数量
                cls_right_nums = get_tp(category_pd_points, category_pd_scores, category_gt_points, thr=cfg.test.match_dis)
                cls_rn[c] += torch.tensor(cls_right_nums, device=cls_rn.device)

        cls_predictions.append(cls_pred_sample)

        # 统计预测点（pd_points）和真实点（gt_points）的数量，并计算正确匹配的数量
        det_pn += len(pd_points)
        det_tn += len(gt_points)

        if len(pd_points) and len(gt_points):
            det_right_nums = get_tp(pd_points, pd_scores, gt_points, thr=cfg.test.match_dis)
            det_rn += torch.tensor(det_right_nums, device=det_rn.device)
    # 分布式的，不会进入
    if get_world_size() > 1:
        dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

        dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        cls_predictions = list(itertools.chain.from_iterable(all_gather(cls_predictions)))
        cls_annotations = list(itertools.chain.from_iterable(all_gather(cls_annotations)))

        iou_scores = np.concatenate(all_gather(iou_scores))
    # eps 是一个极小值，用于防止除零错误
    eps = 1e-7
    # 检测任务的召回率
    det_r = det_rn / (det_tn + eps)
    # 精度
    det_p = det_rn / (det_pn + eps)
    # F1分数
    det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)
    # 然后乘以100以百分比表示。
    det_r = det_r.cpu().numpy() * 100
    det_p = det_p.cpu().numpy() * 100
    det_f1 = det_f1.cpu().numpy() * 100

    # 计算分类任务的召回率、精度和F1分数，并将它们转换为百分比表示。
    cls_r = cls_rn / (cls_tn + eps)
    cls_p = cls_rn / (cls_pn + eps)
    cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

    cls_r = cls_r.cpu().numpy() * 100
    cls_p = cls_p.cpu().numpy() * 100
    cls_f1 = cls_f1.cpu().numpy() * 100
    
    # print(cls_f1)
    # [20.255062]
    
    # 使用 PrettyTable 创建一个表格，添加类名、分类任务的精度、召回率和F1分数。
    table = pt.PrettyTable()
    table.add_column('Class', ["prompter"])
    table.add_column('Precision', cls_p.round(2))
    table.add_column('Recall', cls_r.round(2))
    table.add_column('F1', cls_f1.round(2))

    table.add_row(['---'] * 4)

    det_p, det_r, det_f1 = det_p.round(2)[0], det_r.round(2)[0], det_f1.round(2)[0]
    cls_p, cls_r, cls_f1 = cls_p.mean().round(2), cls_r.mean().round(2), cls_f1.mean().round(2)

    table.add_row(['Det', det_p, det_r, det_f1])
    table.add_row(['Cls', cls_p, cls_r, cls_f1])
    # table.add_row(['loss_d', loss_d,'---', '---'])
    table.add_row(['loss_d', "loss_d",'---', '---'])
    print(table)
    
    # 如果 calc_map 为真，则调用 eval_map 函数计算mAP，并打印结果。
    if calc_map:
        mAP = eval_map(cls_predictions, cls_annotations, cfg.test.match_dis)[0]
        print(f'mAP: {round(mAP * 100, 2)}')

    metrics = {'Det': [det_p, det_r, det_f1], 'Cls': [cls_p, cls_r, cls_f1],
               'IoU': (np.mean(iou_scores) * 100).round(2),'Loss_d':"loss_d"}

    return metrics, table.get_string()
