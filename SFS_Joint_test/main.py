import wandb
import argparse

from utils import *
import glob
from mmengine.config import Config
from dataset import DataFolder
from criterion import build_criterion
from models.dpa_p2pnet import build_model
from engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser('Point prompter')
    parser.add_argument('--config', default='cryoPoint.py', type=str)
    # Wandb（Weights & Biases）是一款专为机器学习和深度学习设计的可视化工具
    parser.add_argument('--run-name', default=None, type=str, help='wandb run name')
    parser.add_argument('--group-name', default=None, type=str, help='wandb group name')

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )

    # * Run Mode
    parser.add_argument('--eval', action='store_true')

    # * Train
    # 随机种子
    parser.add_argument('--seed', default=42, type=int)
    # 预训练权重的位置
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 输出模型的位置
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    
    parser.add_argument("--start-eval", default=0, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    # Logging打印频率
    parser.add_argument("--print-freq", default=5, type=int, help="print frequency")
    parser.add_argument("--use-wandb", action='store_true', help='use wandb for logging')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs.')
    # Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率,前5个epoch
    # 要不要删除，没有用到
    parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs.')
    # 修剪梯度，为了防止梯度爆炸
    parser.add_argument('--clip-grad', type=float, default=0.1,
                        help='Clip gradient norm (default: 0.1)')
    
    # 不用
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    # 控制更新EMA模型的频率，默认值为1。每 model-ema-steps 次迭代更新一次EMA模型。
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=1,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    # 控制EMA模型参数的衰减因子，默认值为0.99
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99)",
    )

    # Mixed precision training parameters混合精度训练可以加速训练过程并减少显存占用。
    # 也没用
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # * Distributed training
    #  Distributed training分布式训练
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    opt = parser.parse_args()

    return opt


def main():
    args = parse_args()
    # 与分布式训练相关的设置,多个GPU进行训练 ，会输出Not using distributed mode
    init_distributed_mode(args)
    # 设置随机种子，使得实验或模型训练的结果可以重现
    set_seed(args)

    cfg = Config.fromfile(f'config/{args.config}')
    if args.output_dir:
        mkdir(f'checkpoint/{args.output_dir}')
        cfg.dump(f'checkpoint/{args.output_dir}/config.py')

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    
    device = torch.device(args.device)
    
    model = build_model(cfg).to(device)
    # model_without_ddp 和 model 是同一个对象的两个名称或引用
    model_without_ddp = model
    
     # 加载权重
    ckpt = torch.load(cfg.sfs.pretrained, map_location='cpu')
    pretrained_state_dict = ckpt['model']
    model.load_state_dict(pretrained_state_dict)

    train_partA_image = list(glob.glob(cfg.data.train_partA_image))
    
    # partA_train_image = []
    # for i in range(len(train_partA_image)):
    #     if 'denoised' in train_partA_image[i]:
    #         partA_train_image.append(train_partA_image[i])
    # print('# training with', len(partA_train_image), 'image pairs')
            
    # train_dataset = DataFolder(cfg, partA_train_image ,'train')
    train_dataset = DataFolder(cfg, train_partA_image ,'test')
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn
    )
    
    val_partA_image = list(glob.glob(cfg.data.val_partA_image))
        
    # partA_val_image = []
    # for i in range(len(val_partA_image)):
    #     if 'denoised' in val_partA_image[i]:
    #         partA_val_image.append(val_partA_image[i])
    # print('# validating on', len(partA_val_image), 'image pairs')
    
    val_dataset = DataFolder(cfg, val_partA_image,'val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=False
    )
   
    # # 待定
    # test_dataset_path = list(glob.glob(cfg.data.test_dataset_path))
    # test_dataset = DataFolder(cfg, test_dataset_path,'test')
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     num_workers=cfg.data.num_workers,
    #     shuffle=False,
    #     drop_last=False
    # )

    if args.eval:
        # checkpoint = torch.load(f'./checkpoint/{args.resume}/best.pth', map_location="cpu")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get('model_ema', ckpt['model']))
        evaluate(
            cfg,
            model,
            test_dataloader,
            device,
            calc_map=True
        )
        return

    model_ema = None
    if args.model_ema:
        print("model_ema is enabled")
        model_ema = ExponentialMovingAverage(model_without_ddp, device=device, decay=args.model_ema_decay)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
        
    # 冻结模型的所有参数
    for params in model_without_ddp.parameters():
        params.requires_grad = False

    # denoise_params = []
    # # for params in model_without_ddp.backbone.unet_encoder.parameters():
    # #     params.requires_grad = True
    # #     denoise_params += [params]
    # for params in model_without_ddp.backbone.unet_decoder.parameters():
    #     params.requires_grad = True
    #     denoise_params += [params]
    # for params in model_without_ddp.backbone.feature_selection.parameters():
    #     params.requires_grad = True
    #     denoise_params += [params]
    
    seg_params = []
    for params in model_without_ddp.backbone.backbone.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.backbone.neck.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.backbone.neck1.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.backbone.feature_selection.leakyunit_u_fpn.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.deform_layer.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.reg_head.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.cls_head.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.conv.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.fuse_blocks.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model_without_ddp.mask_head.parameters():
        params.requires_grad = True
        seg_params += [params]
        

    criterion = build_criterion(cfg, device)
    # 去噪
    # optimizer_d = torch.optim.Adagrad(denoise_params, lr=cfg.optimizer_d.lr)
    # # 学习率调整
    # # actual_lr = cfg.optimizer_p.lr * (cfg.data.batch_size_per_gpu * get_world_size()) / 8  # linear scaling rule
    
    optimizer_p = torch.optim.AdamW(
      # 只选择模型中需要梯度更新的参数。
        filter(lambda p: p.requires_grad, seg_params),
        lr=cfg.optimizer_p.lr,
        weight_decay=cfg.optimizer_p.weight_decay
    )
    Iter_Max = args.epochs * len(train_dataloader)
    # scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=Iter_Max, eta_min=1e-4)
    scheduler_p = optim.lr_scheduler.CosineAnnealingLR(optimizer_p, T_max=Iter_Max, eta_min=5e-5)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.use_wandb and is_main_process():
        wandb.init(
            project='SFS_Joint_test',
            name=args.run_name,
            group=args.group_name,
            config=vars(args),
        )

    # load checkpoint
    max_cls_f1 = 0
    max_cls_recall = 0
    # 检查是否需要恢复训练
    if args.resume:
        # checkpoint = torch.load(f'./checkpoint/{args.resume}/latest.pth', map_location="cpu")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        # optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        optimizer_p.load_state_dict(checkpoint["optimizer_p"])
        # scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        scheduler_p.load_state_dict(checkpoint["scheduler_p"])
        args.start_epoch = checkpoint["epoch"] + 1
        # 如果 checkpoint 中不存在键 "f1"，则返回默认值 0
        
        max_cls_f1 = checkpoint.get("f1", 0)
        max_cls_recall = checkpoint.get("recall", 0)
        
        if model_ema:
            model_ema.module.load_state_dict(checkpoint["model_ema"])
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    
    # print("max_cls_f1:", max_cls_f1)
    # print("max_cls_recall:", max_cls_recall)
    # max_cls_f1: 69.38
    # max_cls_recall: 0
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: 
            train_dataloader.sampler.set_epoch(epoch)
        # 解冻编码器
  
        # if epoch ==args.start_eval:
        #   for params in model.backbone.unet_encoder.parameters():
        #       params.requires_grad = True
        #   for params in model.backbone.backbone.parameters():
        #       params.requires_grad = True
        #   # 收集参数
          
        #   denoise_params = []
        #   seg_params = []

        #   for params in model_without_ddp.backbone.unet_encoder.parameters():
        #       denoise_params.append(params)
        #   for params in model_without_ddp.backbone.unet_decoder.parameters():
        #       denoise_params.append(params)
        #   for params in model_without_ddp.backbone.feature_selection.parameters():
        #       denoise_params.append(params)

        #   for params in model_without_ddp.backbone.backbone.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.backbone.neck.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.backbone.neck1.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.backbone.feature_selection.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.deform_layer.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.reg_head.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.cls_head.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.conv.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.fuse_blocks.parameters():
        #       seg_params.append(params)
        #   for params in model_without_ddp.mask_head.parameters():
        #       seg_params.append(params)

        #   # 获取当前学习率
        #   current_lr_d = optimizer_d.param_groups[0]['lr']
        #   current_lr_p = optimizer_p.param_groups[0]['lr']
          
          # # 重新new优化器
          # optimizer_d = torch.optim.Adagrad(
          #     filter(lambda p: p.requires_grad, denoise_params),
          #     lr=current_lr_d
          # )
          # optimizer_p = torch.optim.AdamW(
          #     filter(lambda p: p.requires_grad, seg_params),
          #     lr=current_lr_p,
          #     weight_decay=cfg.optimizer_p.weight_decay
          # )

          # # 重新new调度器
          # Iter_Max = (args.epochs - epoch) * len(train_dataloader)  # 剩余步数
          # scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
          #     optimizer_d, T_max=Iter_Max, eta_min=1e-4
          # )
          # scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(
          #     optimizer_p, T_max=Iter_Max, eta_min=5e-5
          # )

        log_info = train_one_epoch(
            args,
            model,
            train_dataloader,
            criterion,
            # optimizer_d,
            optimizer_p,
            # scheduler_d,
            scheduler_p,
            epoch,
            device,
            model_ema,
            scaler
        )

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                # "optimizer_d": optimizer_d.state_dict(),
                "optimizer_p": optimizer_p.state_dict(),
                # "scheduler_d": scheduler_d.state_dict(),
                "scheduler_p": scheduler_p.state_dict(),
                "f1": max_cls_f1,
                "recall": max_cls_recall,
                "epoch": epoch,
                "args": args
            }

            if model_ema:
                checkpoint["model_ema"] = model_ema.module.state_dict()

            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()

            save_on_master(
                checkpoint,
                f"checkpoint/{args.output_dir}/latest.pth",
            )
        
        # try:
        if epoch >= args.start_eval:
            metrics, metrics_string = evaluate(
                cfg,
                model_ema or model,
                val_dataloader,
                device,
                epoch,
            )

            log_info.update(dict(zip(["Det Pre", "Det Rec", "Det F1"], metrics['Det'])))
            log_info.update(dict(zip(["Cls Pre", "Cls Rec", "Cls F1"], metrics['Cls'])))
            log_info.update(dict(IoU=metrics['IoU']))
            # log_info.update(dict(Loss_d=metrics['Loss_d']))

            cls_f1 = metrics['Cls'][-1]
            cls_recall = metrics['Cls'][1]
            
            # if max_cls_f1 < cls_f1:
            #     max_cls_f1 = cls_f1
            if max_cls_recall < cls_recall:

                max_cls_recall = cls_recall
                
                checkpoint = {
                    "model": model_without_ddp.state_dict() if not model_ema else model_ema.module.state_dict(),
                    "metrics": metrics_string,
                    "f1": max_cls_f1,
                    "recall": max_cls_recall,
                    "epoch": epoch,
                }
                if args.output_dir:
                    save_on_master(
                        checkpoint,
                        f"checkpoint/{args.output_dir}/best.pth",
                    )
        # except NameError:
        #     # pass
        #     print(f"NameError: A variable is not defined. Please check the variable names.")

        if is_main_process() and args.use_wandb:
            wandb.log(
                log_info,
                step=epoch
            )

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
