import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import xavier_init
from ops import resize, Upsample
import torch

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, f_m, f_n):
        # query: [B, C, H, W]
        # key:   [B, C, H, W]
        # value: [B, C, H, W]
        # f_m主任务
        b, c, h, w = f_m.shape
        q = self.to_q(f_m)
        k, v = self.to_kv(f_n).chunk(2, dim=1)
        # reshape to (b, head, c//head, h*w)
        # q 本来形状是 [b, inner_dim, h, w] 变 [b, heads, dim_head, h*w]
        q = q.view(b, self.heads, -1, h * w)
        k = k.view(b, self.heads, -1, h * w)
        v = v.view(b, self.heads, -1, h * w)

        # bhid
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.view(b, -1, h, w)
        return self.to_out(out)

class LeakyUnit_adaptive(nn.Module):
    # n_fi是辅助任务的通道数，n_fo是主任务的通道数
    def __init__(self, n_fi, n_fo, co_ch, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')): #dict(type='ReLU')
        super(LeakyUnit_adaptive, self).__init__()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        # self.co_ch = co_ch
        # 卷积模块，用于对输入特征进行1x1卷积操作，将输入通道变到adaptive_channel
        self.fit_m = ConvModule(n_fo, co_ch, kernel_size=1, padding=0,
                                stride=1, bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg
                                )
        # 卷积模块，用于对输入特征进行1x1卷积操作，将输入通道变到adaptive_channel
        self.fit_n = ConvModule(n_fi, co_ch, kernel_size=1, padding=0,
                                stride=1, bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg
                                )
        
        # 新增 Cross Attention 模块
        self.cross_attn = CrossAttention(co_ch)
        
        self.W_r = ConvModule(co_ch + co_ch, co_ch, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )
        self.W = ConvModule(co_ch, co_ch, kernel_size=1, padding=0,
                            stride=1, bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.U = ConvModule(co_ch, co_ch, kernel_size=1, padding=0,
                            stride=1, bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                            )
        self.W_z = ConvModule(co_ch + co_ch, co_ch, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )
        self.sigma = nn.Sigmoid()
        # 变回主任务的通道数n_fo
        self.out = ConvModule(co_ch, n_fo, kernel_size=1, padding=0,
                              stride=1, bias=False,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg
                              )

    def forward(self, f_m, f_n):
        # f_M是主任务的，f_n是辅助任务的
        f_m = self.fit_m(f_m)
        f_n = self.fit_n(f_n)
        # 调整张量 f_n 的尺寸以匹配张量 f_m 的空间维度。
        prev_shape = f_m.shape[2:]
        f_n = F.interpolate(f_n, size=prev_shape)
        
        # 2. Cross Attention 融合
        cross_attn_feat = self.cross_attn(f_m, f_n)
        
        r_mn = self.sigma(self.W_r(torch.cat((f_m, cross_attn_feat), dim=1)))
        f_mn_hat = torch.tanh(self.U(f_m) + self.W(r_mn * cross_attn_feat))
        z_mn = self.sigma(self.W_z(torch.cat((f_m, cross_attn_feat), dim=1)))
        f_m_out = z_mn * f_m + (1 - z_mn) * f_mn_hat
        
        f_m_out = self.out(f_m_out)
        return f_m_out, r_mn, z_mn
    # 遍历模型中的所有模块，并对其中的二维卷积层 (nn.Conv2d) 使用 Xavier 均匀分布进行权重初始化。
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UpSampleFeatureFusionModel(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 bilinear=True,
                 align_corners=False
                 ):
        super(UpSampleFeatureFusionModel, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners)  # 'nearest'/bilinear
            # self.conv1 = FusionLayer(in_channels,in_channels)
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv1 = FusionLayer(in_channels, in_channels)
            self.conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, higher_features, lower_features):
        higher_features = self.up(higher_features)
        # input is CHW
        diffY = torch.tensor([lower_features.size()[2] - higher_features.size()[2]])
        diffX = torch.tensor([lower_features.size()[3] - higher_features.size()[3]])
        higher_features = F.pad(higher_features, [diffX // 2, diffX - diffX // 2,
                                                  diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        fused_features = torch.cat([lower_features, higher_features], dim=1)
        # x = self.conv1(x)
        return self.conv(fused_features)

class Basic_fs(nn.Module):

    def __init__(self,
                 in_channels,
                 inu_channels,
                 adaptive_channel=256,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(Basic_fs, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.inu_channels = inu_channels
        self.leakyunit_u_fpn = nn.ModuleList()
        self.leakyunit_fpn_u = nn.ModuleList()
        self.adaptive_channel = adaptive_channel
        # self.LeakyUnit_adaptive=LeakyUnit_adaptive
        '''
        only the last 3 layers of FPN_seg and the first 3 layers of UNet_gan implement soft selection
        '''
        for i in range(0, len(in_channels)):
            #  self.leakyunit_u_fpn以fpn为主任务
            self.leakyunit_u_fpn.append(LeakyUnit_adaptive(self.inu_channels[i],
                                                           self.in_channels[i], self.adaptive_channel))
            #  self.leakyunit_fpn_u以unet为主任务
            self.leakyunit_fpn_u.append(LeakyUnit_adaptive(self.in_channels[i],
                                                           self.inu_channels[i], self.adaptive_channel))
        
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    # 初始化模型权重
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs_a, inputs_b):

        out_a = []
        out_b = []

        out_b.append(inputs_b[0])
        out_b.append(inputs_b[1])
        
        for i in range(0, len(self.in_channels)):
            # fpn为主任务
            a_feat_hat, _, _ = self.leakyunit_u_fpn[i](inputs_a[i], inputs_b[i+2])
            # unet为主任务
            b_feat_hat, _, _ = self.leakyunit_fpn_u[i](inputs_b[i+2], inputs_a[i])

            out_a.append(a_feat_hat)
            out_b.append(b_feat_hat)
        
        return tuple(out_a), tuple(out_b)



