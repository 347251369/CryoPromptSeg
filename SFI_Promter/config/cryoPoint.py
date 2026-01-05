prompter = dict(
    # prompter
    pretrained='/root/Promptpoint/prompter/checkpoint/small_test3/best.pth',
    
    backbone=dict(
        model_name='convnext_small',
        pretrained=False,
        num_classes=0,    #num_classes=0：表示不包括最后的分类层，这通常用于特征提取任务，模型输出为最后一层卷积特征。
        global_pool=''    #表示禁用全局池化。模型的输出将是最后一个卷积层的特征图（而非经过池化得到的向量）。
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],   #与convnext_small 4个阶段的输出通道数一致
        out_channels=256,
        num_outs=4,   #输出尺度的数量
        add_extra_convs='on_input',   #在FPN的最后一个输出特征图上添加额外的卷积层。这些卷积层可以进一步提取特征，增强模型的表达能力。
    ),
    Unet_encoder = dict(
        in_channels=1,
        nf=48,
        base_width=11,
        top_width=5,
        pretrained='pretrained/unet_L2_v0.2.2.sav',
    ),
    Unet_decoder = dict(
        out_channels=1,
        nf=48,
        base_width=11,
        top_width=5,
        pretrained='pretrained/unet_L2_v0.2.2.sav',
    ),
    
    feature_selection=dict(
        in_channels=[96, 192, 384, 768],
        inu_channels=[48, 48, 48, 48],
        adaptive_channel=128,
    ),
    dropout=0.1,
    # space=16,
    space=8,
    hidden_dim=256
)

sfs = dict(
    pretrained= '/root/Promptpoint/SFS_Joint/checkpoint/feature_fusion/best.pth'
)
data = dict(    
    # train_partA_image ='/root/autodl-tmp/datasets/sfs_train_dataset/*/train/partA/*.jpg',
    # val_partA_image ='/root/autodl-tmp/datasets/sfs_train_dataset/*/val/partA/*.jpg',
    
    train_partA_image ='/root/autodl-tmp/datasets/train_dataset/*/train/images/*.jpg',
    val_partA_image ='/root/autodl-tmp/datasets/train_dataset/*/val/images/*.jpg',
    train_point_path ='/root/Promptpoint/SFS_pro_Unet/datasets/train_dataset',
    
    # train_dataset_path='/root/autodl-tmp/datasets/train_dataset/*/',
    test_dataset_path ='/root/autodl-tmp/datasets/test_dataset/*/images/*.jpg',
    test_point_path='/root/Promptpoint/SFS_pro_Unet/datasets/test_dataset', 
    
    num_classes=1,
    batch_size_per_gpu=2,
    num_workers=12,
    
    train=dict(transform=[
        # dict(type='RandomCrop',height=1024, width=1024 , p=1),
        dict(type='Resize',height=1024, width=1024),
        # 50% 概率进行水平翻转
        dict(type='HorizontalFlip', p=0.5),
        dict(type='VerticalFlip', p=0.5),
        dict(type='RandomRotate90', p=0.5),
        # 填充，以防不是16的倍数
        dict(type='PadIfNeeded', min_height=None, min_width=None, pad_height_divisor=prompter["space"],
             pad_width_divisor=prompter["space"], position="top_left", p=1),
        # dict(type='Normalize'),
    ]),
    
    val=dict(transform=[
        # dict(type='RandomCrop',height=1024, width=1024 , p=1),
        dict(type='Resize',height=1024, width=1024),
        dict(type='PadIfNeeded', min_height=None, min_width=None, pad_height_divisor=prompter["space"],
             pad_width_divisor=prompter["space"], position="top_left", p=1),
        # dict(type='Normalize'),
    ]),
    
    test=dict(transform=[
        dict(type='Resize',height=1024, width=1024),
        # 50% 概率进行水平翻转
        # dict(type='HorizontalFlip', p=0.5),
        # dict(type='VerticalFlip', p=0.5),
        # dict(type='RandomRotate90', p=0.5),
        
        dict(type='PadIfNeeded', min_height=None, min_width=None, pad_height_divisor=prompter["space"],
             pad_width_divisor=prompter["space"], position="top_left", p=1),
        # dict(type='Normalize'),
    ]),
)

optimizer_d = dict(
    type='adagrad',
    lr=0.001
)
optimizer_p = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=1e-4
)

criterion = dict(
    matcher=dict(type='HungarianMatcher', dis_type='l2', set_cost_point=0.1, set_cost_class=1),
    eos_coef=0.2,
    reg_loss_coef=5e-3,   #0.005
    # reg_loss_coef=0.01,
    cls_loss_coef=1.0,
    mask_loss_coef=1.0
)

test = dict(nms_thr=12, match_dis=12, filtering=False)
