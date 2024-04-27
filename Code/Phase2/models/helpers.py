import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc
from models.weight_init import trunc_normal_


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    
    
class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --------------------------------------------------------
# Backbones for GMFlowNet
# --------------------------------------------------------
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x



class NeighborWindowAttention(nn.Module):
    """ Patch-based OverLapping multi-head self-Attention (POLA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window (or patch).
        num_heads (int): Number of attention heads.
        neig_win_num (int): Number of neighbor windows. Default: 1
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, neig_win_num=1, 
                    qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., use_proj=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.use_proj = use_proj

        # define a parameter table of relative position bias
        self.n_win = 2*neig_win_num + 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((self.n_win + 1) * window_size[0] - 1) * ((self.n_win + 1) * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww

        coords_h_neig = torch.arange(self.n_win * self.window_size[0])
        coords_w_neig = torch.arange(self.n_win * self.window_size[1])
        coords_neig = torch.stack(torch.meshgrid([coords_h_neig, coords_w_neig]))  # 2, Wh, Ww

        coords_flat = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords_neig_flat = torch.flatten(coords_neig, 1)  # 2, (n_win*Wh)*(n_win*Ww)
        relative_coords = coords_flat[:, :, None] - coords_neig_flat[:, None, :]  # 2, Wh*Ww, (n_win*Wh)*(n_win*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww,(n_win*Wh)*(n_win*Ww), 2
        relative_coords[:, :, 0] += self.n_win * self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.n_win * self.window_size[1] - 1
        relative_coords[:, :, 0] *= (self.n_win + 1) * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """ Forward function.

        Args:
            q: input queries with shape of (num_windows*B, N, C)
            k: input keys with shape of (num_windows*B, N, C)
            v: input values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N_q, C = q.shape
        N_kv = k.shape[1]
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        dim_per_head = C // self.num_heads
        q = self.Wq(q).reshape(B_, N_q, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        k = self.Wk(k).reshape(B_, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)
        v = self.Wv(v).reshape(B_, N_kv, self.num_heads, dim_per_head).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.n_win*self.window_size[0] * self.n_win*self.window_size[1], -1)  # Wh*Ww,(n_win*Wh)*(n_win*Ww),nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, (n_win*Wh)*(n_win*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N_q, N_kv) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N_q, N_kv)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_q, C)
        if self.use_proj:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x



class BasicConvEncoder(nn.Module):
    """docstring for BasicConvEncoder"""
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicConvEncoder, self).__init__()
        self.norm_fn = norm_fn

        half_out_dim = max(output_dim // 2, 64)

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(half_out_dim)
            self.norm3 = nn.BatchNorm2d(output_dim)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm2 = nn.InstanceNorm2d(half_out_dim)
            self.norm3 = nn.InstanceNorm2d(output_dim)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, half_out_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(half_out_dim, output_dim, kernel_size=3, stride=2, padding=1)

        # # output convolution; this can solve mixed memory warning, not know why
        # self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = F.relu(self.norm2(self.conv2(x)), inplace=True)
        x = F.relu(self.norm3(self.conv3(x)), inplace=True)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x




class POLATransBlock(nn.Module):
    """ Transformer block with POLA.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window/patch size.
        neig_win_num (int): Number of overlapped windows
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, neig_win_num=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.neig_win_num = neig_win_num
        self.mlp_ratio = mlp_ratio

        self.n_win = 2 * neig_win_num + 1

        self.norm1 = norm_layer(dim)

        self.attn = NeighborWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, neig_win_num=neig_win_num,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, H, W, attn_mask=None):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # print('LocalTransBlock x.shape: ', x.shape)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # partition windows
        x_win = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_win = x_win.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # pad and unfold
        pad_size = self.neig_win_num * self.window_size
        key_val = F.pad(x, (0, 0, pad_size, pad_size, pad_size, pad_size)) # B, H'+2*1*win, W'+2*1*win, C
        key_val = F.unfold(key_val.permute(0, 3, 1, 2), self.n_win*self.window_size, stride=self.window_size)
        key_val = key_val.permute(0,2,1).reshape(-1, C, (self.n_win*self.window_size)**2).permute(0,2,1) # (B*num_win, (3*3)*win_size*win_size, C)

        # Local attention feature
        attn_windows = self.attn(x_win, key_val, key_val, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    

class POLAUpdate(nn.Module):
    """ POLA update for GMFlowNet.
        A PyTorch impl of : `Global Matching with Overlapping Attention for Optical Flow Estimation`  -
          https://arxiv.org/abs/2203.11335

    Args:
        embed_dim (int): Number of linear projection output channels. Default: 256.
        depths (int): Number of POLA blocks.
        num_heads (int): Number of attention head in each POLA block.
        window_size (int): Window/patch size. Default: 7.
        neig_win_num: Number of overlapped Windows/patches
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 embed_dim=256,
                 depth=6,
                 num_head=8,
                 window_size=7,
                 neig_win_num=1,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_feature = embed_dim
        self.num_head = num_head
        self.win_size = window_size
        self.neig_win_num = neig_win_num

        self.use_checkpoint = use_checkpoint

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            POLATransBlock(
                dim=self.num_feature,
                num_heads=self.num_head,
                window_size=self.win_size,
                neig_win_num=self.neig_win_num,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(self.num_feature)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B L C

        # calculate attention mask for ConvAlikeLocalTransBlock
        img_mask = torch.zeros((1, H, W, 1), device=x.device)  # 1 H W 1

        pad_r = (self.win_size - W % self.win_size) % self.win_size
        pad_b = (self.win_size - H % self.win_size) % self.win_size
        pad_extra = self.neig_win_num * self.win_size
        img_mask = F.pad(img_mask, (0, 0, pad_extra, pad_r + pad_extra,
                                    pad_extra, pad_b + pad_extra),
                         mode='constant', value=float(-100.0))

        # unfold
        n_win = 2 * self.neig_win_num + 1
        mask_windows = F.unfold(img_mask.permute(0, 3, 1, 2), n_win * self.win_size, stride=self.win_size)
        mask_windows = mask_windows.permute(0, 2, 1).reshape(-1, (
                    n_win * self.win_size) ** 2)  # (num_win, (3*3)*win_size*win_size)
        attn_mask = mask_windows.unsqueeze(1).repeat(1, self.win_size * self.win_size, 1)

        # update features
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W, attn_mask)
            else:
                x = blk(x, H, W, attn_mask)

        x = self.norm(x)
        out = x.view(-1, H, W, self.num_feature).permute(0, 3, 1, 2).contiguous()

        if is_list:
            out = torch.split(out, [batch_dim, batch_dim], dim=0)

        return out
