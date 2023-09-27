
from timm.loss import SoftTargetCrossEntropy
from timm.models.layers import drop

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from typing import List
from timm.models.layers import DropPath


_cur_active: torch.Tensor = None  # B1ff

def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    '''
    获取活动的ex或ii
    :param H: 模型的高度
    :param W: 模型的宽度
    :param returning_active_ex: 是否返回活动ex
    :return: 活动ex或ii
    '''
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi

# def _get_active_ex_or_ii(H, W, returning_active_ex=True):
#     h_repeat = H // _cur_active.shape[-2]
#     w_repeat = W // _cur_active.shape[-1]
#
#     active_ex = _cur_active.expand(-1, -1, h_repeat, -1, w_repeat).reshape(*_cur_active.shape[:2], H, W)
#     if returning_active_ex:
#         return active_ex
#     else:
#         return active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):

    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3],
                              returning_active_ex=True)  # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    '''
    :param x: (B, C, H, W)
    :return: (B, C, H, W)
    '''
    # 获取x的形状
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    # 将x的维度变换为(B, C, H, W)
    bhwc = x.permute(0, 2, 3, 1)
    # 获取x中非masked位置的特征，并将其转换为一个矩阵，维度为(B, C, H, W)
    nc = bhwc[ii]  # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)  # use BN1d to normalize this flatten feature `nc`

    # 将特征转换为(B, C, H, W)
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward  # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward  # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim == 4:  # BHWC or BCHW
            if self.data_format == "channels_last":  # BHWC
                if self.sparse:
                    # 获取激活的ex或ii
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], returning_active_ex=False)
                    # 获取激活的ex
                    nc = x[ii]
                    # 调用父类的forward函数
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    # 初始化x
                    x = torch.zeros_like(x)
                    x[ii] = nc
                    return x
                else:
                    # 调用父类的forward函数
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:  # channels_first, BCHW
                if self.sparse:
                    # 获取激活的ex或ii
                    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                    # 将x的第二维排列到第一维
                    bhwc = x.permute(0, 2, 3, 1)
                    # 获取激活的ex
                    nc = bhwc[ii]
                    # 调用父类的forward函数
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)

                    # 初始化x
                    x = torch.zeros_like(bhwc)
                    x[ii] = nc
                    return x.permute(0, 3, 1, 2)
                else:
                    # 将x的第一维排列到第二维
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    # 调用父类的forward函数
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:  # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[
               :-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'

class SparseConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=True, ks=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks // 2, groups=dim)  # depthwise conv
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sparse = sparse

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)  # GELU(0) == (0), so there is no need to mask x (no need to `x *= _get_active_ex_or_ii`)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)

        x = input + self.drop_path(x)
        return x

    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.3,
                 layer_scale_init_value=1e-6, sparse=True,):
        super().__init__()

        self.dims: List[int] = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            SparseConvNeXtLayerNorm(dims[0], eps=1e-6, data_format="channels_first", sparse=sparse)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                SparseConvNeXtLayerNorm(dims[i], eps=1e-6, data_format="channels_first", sparse=sparse),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[SparseConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, sparse=sparse) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.depths = depths

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def get_downsample_ratio(self) -> int:
        return 32

    def get_feature_map_channels(self) -> List[int]:
        return self.dims

    def forward(self, x):

        ls = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            ls.append(x)
        return ls

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}, layer_scale_init_value={self.layer_scale_init_value:g}'

@register_model
def convnext_tiny(**kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def convnext_small(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

    return model


@register_model
def convnext_base(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

    return model


@register_model
def convnext_large(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

    return model


@register_model
def convnext_xlarge(**kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)

    return model

@register_model
def build_convnext(model_name=None, **kwargs):

    if model_name == 'convnext_tiny':
        return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    elif model_name == 'convnext_small':
        return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.2, **kwargs)
    elif model_name == 'convnext_base':
        return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], drop_path_rate=0.3, **kwargs)
    elif model_name == 'convnext_large':
        return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], drop_path_rate=0.4, **kwargs)
    elif model_name == 'convnext_xlarge':
        return ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    else:
        raise NotImplementedError


class SparseEncoder(nn.Module):
    def __init__(self, cnn, input_size, sbn=False, verbose=False):
        super(SparseEncoder, self).__init__()
        self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn, verbose=verbose, sbn=sbn)
        self.input_size, self.downsample_raito, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()

    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias = m.bias is not None
            oup = SparseConv2d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation,
                                   return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode,
                                   count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m: nn.BatchNorm2d
            oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps,
                                                                        momentum=m.momentum, affine=m.affine,
                                                                        track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
            m: nn.LayerNorm
            oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError

        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup

    def forward(self, x):
        return self.sp_cnn(x)

# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


pretrain_default_model_kwargs = {

    'convnext_small': dict(sparse=True, drop_path_rate=0.2),
    'convnext_base': dict(sparse=True, drop_path_rate=0.3),
    'convnext_large': dict(sparse=True, drop_path_rate=0.4),
}
# for kw in pretrain_default_model_kwargs.values():
#     kw['pretrained'] = False
#     kw['num_classes'] = 0
#     kw['global_pool'] = ''


def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False):

    kwargs = pretrain_default_model_kwargs[name]
    if drop_path_rate != 0:
        kwargs['drop_path_rate'] = drop_path_rate
    print(f'[build_sparse_encoder] model kwargs={kwargs}')

    cnn = build_convnext(name)

    return SparseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)


