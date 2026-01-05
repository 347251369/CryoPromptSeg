# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import timm
import copy
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.fpn import FPN
from models.u_net import Unet_encoder, Unet_decoder
from models.feature_selection import Basic_fs

class Backbone(nn.Module):
    def __init__(
            self,
            cfg
    ):
        super(Backbone, self).__init__()

        backbone = timm.create_model(
            **cfg.prompter.backbone
        )

        self.backbone = backbone
        
        self.unet_encoder = Unet_encoder(
            **cfg.prompter.Unet_encoder
        )
        
        self.unet_decoder = Unet_decoder(
            **cfg.prompter.Unet_decoder
        )
        
        self.neck = FPN(
            **cfg.prompter.neck
        )

        new_dict = copy.copy(cfg.prompter.neck)
        new_dict['num_outs'] = 1
        self.neck1 = FPN(
            **new_dict
        )
        
        self.feature_selection = Basic_fs(
            **cfg.prompter.feature_selection
        )
        
        
    def forward(self, images):
        # Unet_encoder的forward
        out_u = []
        partA_images = images[:, 0:1, :, :]
        out_u.append(partA_images)
        p1 = self.unet_encoder.enc1(partA_images)
        out_u.append(p1)
        # print(out_u[-1].shape)
        # torch.Size([4, 48, 512, 512])
        
        # convnext的forward
        out_c = []
        out_c.append(self.backbone.stem(images))
        # print(out_c[-1].shape)
        # torch.Size([4, 96, 256, 256])
        
        oc = self.backbone.stages[0](out_c[-1])
        # print(oc.shape)
        # torch.Size([4, 96, 256, 256])
        ou = self.unet_encoder.enc2(out_u[-1])
        # print(ou.shape)
        # torch.Size([4, 48, 256, 256])                    #  
        # convnext为主任务
        a_feat_hat, _, _ = self.feature_selection.leakyunit_u_fpn[0](oc, ou)
        # unet为主任务
        b_feat_hat, _, _ = self.feature_selection.leakyunit_fpn_u[0](ou, oc)
        out_c.append(a_feat_hat)
        out_u.append(b_feat_hat)
        # print(out_u[-1].shape)
        # print(out_c[-1].shape)
        # torch.Size([4, 48, 256, 256])
        # torch.Size([4, 96, 256, 256])

        oc = self.backbone.stages[1](out_c[-1])
        ou = self.unet_encoder.enc3(out_u[-1])                          
        # convnext为主任务
        a_feat_hat, _, _ = self.feature_selection.leakyunit_u_fpn[1](oc, ou)
        # unet为主任务
        b_feat_hat, _, _ = self.feature_selection.leakyunit_fpn_u[1](ou, oc)
        out_c.append(a_feat_hat)
        out_u.append(b_feat_hat)
        # print(out_u[-1].shape)
        # print(out_c[-1].shape)
        # torch.Size([4, 48, 128, 128])
        # torch.Size([4, 192, 128, 128])

        oc = self.backbone.stages[2](out_c[-1])
        ou = self.unet_encoder.enc4(out_u[-1])
        # convnext为主任务
        a_feat_hat, _, _ = self.feature_selection.leakyunit_u_fpn[2](oc, ou)
        # unet为主任务
        b_feat_hat, _, _ = self.feature_selection.leakyunit_fpn_u[2](ou, oc)
        out_c.append(a_feat_hat)
        out_u.append(b_feat_hat)
        # print(out_u[-1].shape)
        # print(out_c[-1].shape)
        # torch.Size([4, 48, 64, 64])
        # torch.Size([4, 384, 64, 64])
        
        oc = self.backbone.stages[3](out_c[-1])
        ou = self.unet_encoder.enc5(out_u[-1])
        # convnext为主任务
        a_feat_hat, _, _ = self.feature_selection.leakyunit_u_fpn[3](oc, ou)
        # unet为主任务
        b_feat_hat, _, _ = self.feature_selection.leakyunit_fpn_u[3](ou, oc)
        out_c.append(a_feat_hat)
        out_u.append(self.unet_encoder.enc6(b_feat_hat))
        
        # x = self.backbone(images)
        
        # # # Unet输入的是一个通道
        # partA_images = images[:, 0:1, :, :] 
        # y = self.unet_encoder(partA_images)
        
        # x, y = self.feature_selection(x, y)
        
        unet_y = self.unet_decoder(out_u)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)
        # list(self.neck(x)) 将4个输出特征图转为列表格式
        # self.neck1(x)[0]  的num_outs=1,通过 [0] 索引，只取列表中的第一个特征图,进行mask
        return list(self.neck(out_c[1:])), self.neck1(out_c[1:])[0] , unet_y


class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        # 使用 np.meshgrid 生成网格坐标，范围是从0到 ceil(w / space) 和 ceil(h / space)。
        # 乘以 space，得到初始锚点坐标。
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space
        # or self.space 确保当余数为0时，取 self.space
        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            pretrained,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256,
            with_mask=False
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.with_mask = with_mask
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1, drop=dropout)

        self.conv = nn.Conv2d(hidden_dim * num_levels, hidden_dim, kernel_size=3, padding=1)

        # 多尺度融合模块：把所有尺度统一通道数后上采样 + 相加
        self.fuse_blocks = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(num_levels)
        ])
        
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=1)
        )
        
        self.init_weights(pretrained)
        
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location='cpu')  # 加载权重文件
            
            state_dict = checkpoint['model']

            # self 代表当前的 Unet_encoder 实例，表示要给它加载权重
            self.load_state_dict(state_dict, strict=False)  # 加载到模型中,
            # 设置 strict=False 以如果 checkpoint 中的权重和 Unet_encoder 的参数不完全匹配
            # （比如 checkpoint 里有额外的层，或者 Unet_encoder 有新的层），不会报错，但不会加载那些不匹配的参数。
    def fuse_features(self, feats):
        # 统一上采样到最高分辨率（feats[0]）
        target_size = feats[0].shape[-2:]
        fused = 0
        for i in range(self.num_levels):
            x = self.fuse_blocks[i](feats[i])  # 1x1 conv 统一通道
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            # 多个 [B, hidden_dim, H, W] 相加，仍是 [B, hidden_dim, H, W]
            fused += x  # Add-fusion
        return fused  # shape: [B, hidden_dim, H, W]
    def forward(self,
                images):
        # extract features
        (feats, feats1, unet_y), proposals = self.backbone(images), self.get_aps(images)
        
        # 计算每个特征图的尺寸 feat_sizes
        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # DPP
        # 将 proposals 转换为适合特征图的网格坐标
        grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0)
        # 从特征图中提取 proposals 的特征
        roi_features = F.grid_sample(feats[0], grid, mode='bilinear', align_corners=True)
        # 计算 deformable offset（偏移量）
        deltas2deform = self.deform_layer(roi_features.permute(0, 2, 3, 1))
        # 更新 proposals
        deformed_proposals = proposals + deltas2deform

        # MSD
        # roi_features 用于存储多尺度特
        # 遍历多尺度特征图，生成对应的网格坐标 grid，并使用 F.grid_sample 从每个特征图中采样 ROI 特征
        roi_features = []
        for i in range(self.num_levels):
            # 根据 deformable proposals 的位置和特征图的步幅（self.strides[i]），生成对应网格坐标 grid。
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))
        # 将所有尺度的 ROI 特征拼接在一起
        roi_features = torch.cat(roi_features, 1)

        # 对拼接后的多尺度特征进行降维
        roi_features = self.conv(roi_features).permute(0, 2, 3, 1)
        # 计算坐标细化偏移量 
        deltas2refine = self.reg_head(roi_features)
        # 计算细化偏移量 deltas2refine，并更新 deformed_proposals 为最终的预测坐标 pred_coords
        pred_coords = deformed_proposals + deltas2refine

        # 生成分类头的输出,对每个 proposal，输出一个向量，表示属于每个类别（包括背景）的 logit
        pred_logits = self.cls_head(roi_features)

        fused_feats = self.fuse_features(feats)  # 融合后的高分辨率特征
        
        output = {
            'pred_coords': pred_coords.flatten(1, 2),
            'pred_logits': pred_logits.flatten(1, 2),
            'pred_masks': F.interpolate(
                self.mask_head(fused_feats), size=images.shape[2:], mode='bilinear', align_corners=True)
                # self.mask_head(feats1), size=images.shape[2:], mode='bilinear', align_corners=True)
        }

        return output, unet_y


def build_model(cfg):
    backbone = Backbone(cfg)

    model = DPAP2PNet(
        backbone,
        pretrained=cfg.prompter.pretrained,
        num_levels=cfg.prompter.neck.num_outs,#特征金字塔的层数
        num_classes=cfg.data.num_classes,
        dropout=cfg.prompter.dropout,
        space=cfg.prompter.space,
        hidden_dim=cfg.prompter.hidden_dim
    )

    return model
