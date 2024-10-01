import torch
from torch import nn
from backbone.Shunted.SSA import *
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
import torch.fft as fft
import torch.nn.functional as F
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

channel_lay1 = 64
channel_lay2 = 128
channel_lay3 = 256
channel_lay4 = 512
class DWC(nn.Module):
    """
        深度可分离卷积 = 深度卷积 + 逐点卷积调整通道 ，gruops=in_channels就可以实现逐通道卷积
    """

    def __init__(self, in_channel, out_channel):
        super(DWC, self).__init__()
        # 深度卷积
        self.conv_group = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        # 逐点卷积调整通道
        self.conv_point = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1)
        # BN
        self.bn = nn.BatchNorm2d(out_channel)
        # activate
        self.act = nn.PReLU()

    def forward(self, inputs):
        x = self.conv_group(inputs)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.act(x)
        return x

def DOWN1(in_, out_):
    return nn.Sequential(
        nn.Conv2d(in_, out_, 3, 1, 1),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    )

class Mlp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, in_features, 1, 1, 0)
        self.dwconv = nn.Conv2d(in_features, in_features, 1, 1, 0)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(in_features, in_features, 1, 1, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SS2D(nn.Module):
    def __init__(self,d_model,d_state=16,d_conv=3,expand=2,dt_rank="auto",dt_min=0.001,dt_max=0.1,dt_init="random",
            dt_scale=1.0,dt_init_floor=1e-4,dropout=0.,conv_bias=True,bias=False,device=None,dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model               # d_state="auto", # 20240109
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner,out_channels=self.d_inner,groups=self.d_inner,bias=conv_bias,kernel_size=d_conv,
            padding=(d_conv - 1) // 2,**factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(xs, dts,As, Bs, Cs, Ds, z=None,delta_bias=dt_projs_bias,delta_softplus=True,return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out

class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,stride=1,padding=1,groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,out_channels=out_ch, kernel_size=1,stride=1, padding=0,groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class SPR(nn.Module):
    def __init__(self,inch1,attn_drop_rate: float = 0,d_state: int = 16,**kwargs,):
        super(SPR, self).__init__()
        self.mamba1 = SS2D(d_model=inch1, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.convh1 = DWC(inch1,inch1)
        self.convx1 = DWC(inch1, inch1)
        self.convl1 = DWC(inch1,inch1)
        self.conv = DWC(inch1,inch1)
        self.sig = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.convf = DWC(inch1,inch1)
        self.SSAA = SA(inch1)

    def forward(self, H,L):
        H1 = self.convh1(H) + H
        L1 = self.convl1(self.avg(L) * H1)
        X = H1 + L1

        H3 = self.convx1(self.sig(self.mamba1(X)) * X + X)
        C1 = self.SSAA(X)

        x = self.conv(H3 + C1) + X

        return x

class SA(nn.Module):
    def __init__(self, inp_channel):
        super().__init__()
        self.max1 = nn.AdaptiveMaxPool2d(1)
        self.k = DEPTHWISECONV(inp_channel, inp_channel)
        self.v = DEPTHWISECONV(inp_channel, inp_channel)
        self.mlp1 = Mlp(1)
        self.mlp2 = Mlp(1)
        self.sig = nn.Sigmoid()
        self.dwconv = DEPTHWISECONV(inp_channel, inp_channel)
        self.CBR = DWC(inp_channel, inp_channel)

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, C//2,dim=1)                                                       #1,256,10,10
        max_values = torch.where(x1 > x2, x1, x2)
        cosine_similarity1 = F.cosine_similarity(x1, max_values, dim=1).unsqueeze(1)           # 1, 1, 10, 10
        cosine_similarity2 = F.cosine_similarity(x2, max_values, dim=1).unsqueeze(1)

        C1 = self.sig(self.mlp1(cosine_similarity1))           # 1,1,10,10
        C2 = self.sig(self.mlp2(cosine_similarity2))
        w1 = C1 / (C1 + C2)
        w2 = C2 / (C1 + C2)
        x = torch.cat((w1 * x1, w2 * x2),dim=1)
        x = self.CBR(x) + x
        # k = self.k(x)
        # v = self.v(x)
        # x = self.sig(x * k) * v + x

        return x

class BetaPredictor(nn.Module):
    def __init__(self, in_channels):
        super(BetaPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels//4, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        beta = self.sigmoid(self.conv3(x))
        return beta

class MRFLayer(nn.Module):
    def __init__(self, inch1, num_iter=10):
        super(MRFLayer, self).__init__()
        self.num_iter = num_iter
        self.beta_predictor = BetaPredictor(inch1)

    def forward(self, x):
        x_denoised = x.clone()

        for _ in range(self.num_iter):
            # 预测当前的 beta 值
            beta = self.beta_predictor(x).expand_as(x)

            # 上下左右四个方向的移动
            x_up = torch.roll(x_denoised, shifts=-1, dims=2)
            x_down = torch.roll(x_denoised, shifts=1, dims=2)
            x_left = torch.roll(x_denoised, shifts=-1, dims=3)
            x_right = torch.roll(x_denoised, shifts=1, dims=3)

            # 计算邻居的平均值
            neighbors_mean = (x_up + x_down + x_left + x_right) / 4

            # 更新去噪图像
            x_denoised = neighbors_mean * beta + x * (1 - beta)

        return x_denoised

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up4 = DOWN1(channel_lay4, channel_lay3)
        self.up3 = DOWN1(channel_lay3, channel_lay2)
        self.up2 = DOWN1(channel_lay2, channel_lay1)
        self.mrf1 = MRFLayer(channel_lay3)
        self.mrf2 = MRFLayer(channel_lay2)

    def forward(self,f4,f3,f2,f1):
        x1 = self.up4(f4)
        a1 = x1 + f3
        x2 = self.up3(a1)
        a2 = f2 + x2
        y2 = self.mrf2(a2) + a2
        x3 = self.up2(y2) + f1
        return x3,x2,x1

class shunttiny(nn.Module):
    def __init__(self):
        super(shunttiny, self).__init__()
        self.rgb_net = shunted_t()
        self.d_net = shunted_t()

        self.spr4 = SPR(channel_lay4)
        self.spr3 = SPR(channel_lay3)
        self.spr2 = SPR(channel_lay2)
        self.spr1 = SPR(channel_lay1)

        self.up4 = DOWN1(channel_lay4, channel_lay3)
        self.up3 = DOWN1(channel_lay3, channel_lay2)
        self.up2 = DOWN1(channel_lay2, channel_lay1)
        self.decoder = Decoder()

        #################          监督     ##################
        self.s1 = nn.Sequential(
            nn.Conv2d(channel_lay1, 1, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.jdy2 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.jdy3 = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )
        self.jdx1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.jdx2 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.jdx3 = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )
        self.jdx4 = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True),
        )
        ##############################################################################

    def forward(self, rgb, d):
        d = torch.cat((d, d, d), dim=1)
        rgb_list = self.rgb_net(rgb)
        depth_list = self.d_net(d)

        r1 = rgb_list[0]
        r2 = rgb_list[1]
        r3 = rgb_list[2]
        r4 = rgb_list[3]

        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]

        f4 = self.spr4(r4, d4)
        f3 = self.spr3(r3, d3)
        f2 = self.spr2(r2, d2)
        f1 = self.spr1(r1, d1)
        y1, y2, y3 = self.decoder(f4, f3, f2, f1)
        outs = self.s1(y1)

        out1 = self.jdx1(f1)
        out2 = self.jdx2(f2)
        out3 = self.jdx3(f3)
        out4 = self.jdx4(f4)

        return  outs,out1,out2,out3,out4

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.rgb_net.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.rgb_net.load_state_dict(model_dict_r)

        model_dict_d = self.d_net.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_d.update(state_dict_d)
        self.d_net.load_state_dict(model_dict_d)
        ############################################################
        print('self.rgb_uniforr loading', 'self.depth_unifor loading')

if __name__ == '__main__':
    rgb = torch.randn([1, 3, 320, 320]).cuda()                                   # batch_size=1，通道3，图片尺寸320*320
    depth = torch.randn([1, 1, 320, 320]).cuda()
    model = shunttiny().cuda()
    a = model(rgb, depth)
    print(a[0].shape)
    from FLOP import *
    CalParams(model, rgb, depth)



