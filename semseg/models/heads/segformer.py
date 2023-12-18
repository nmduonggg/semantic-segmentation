from git import Sequence
import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg
    
class RCSegFormerHead(nn.Module):
    """Modified version for row-col output"""
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred_row = nn.Conv2d(embed_dim, 2, 1)
        self.linear_pred_col = nn.Conv2d(embed_dim, 2, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        row_seg = self.linear_pred_row(self.dropout(seg))
        col_seg = self.linear_pred_col(self.dropout(seg))
        return (row_seg, col_seg)
    
class RCSegFormerHeadV2(nn.Module):
    """Modified version for row-col output"""
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred_row = nn.Sequential(
            nn.MaxPool2d((3, 7), (1, 1), (1, 3)),
            nn.Conv2d(embed_dim, embed_dim//2, (3, 7), (1, 1), (1, 3)), nn.ReLU(True),
            nn.Conv2d(embed_dim//2, embed_dim//4, (3, 7), (1, 1), (1, 3)), nn.ReLU(True),
            nn.Conv2d(embed_dim//4, embed_dim//8, (3, 7), (1, 1), (1, 3)), nn.ReLU(True),
            nn.Conv2d(embed_dim//8, 2, 1, 1, 0))
        self.linear_pred_col = nn.Sequential(
            nn.MaxPool2d((7, 3), (1, 1), (3, 1)),
            nn.Conv2d(embed_dim, embed_dim//2, (7, 3), (1, 1), (3, 1)), nn.ReLU(True),
            nn.Conv2d(embed_dim//2, embed_dim//4, (7, 3), (1, 1), (3, 1)), nn.ReLU(True),
            nn.Conv2d(embed_dim//4, embed_dim//8, (7, 3), (1, 1), (3, 1)), nn.ReLU(True),
            nn.Conv2d(embed_dim//8, 2, 1, 1, 0))
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        row_seg = self.linear_pred_row(self.dropout(seg))
        col_seg = self.linear_pred_col(self.dropout(seg))
        return (row_seg, col_seg)


class RCSegFormerHeadV3(nn.Module):
    """Modified version for row-col output"""
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 2):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*4, embed_dim)
        self.linear_pred_row = nn.Sequential(
            nn.MaxPool2d((3, 7), (1, 1), (1, 3)),
            nn.Conv2d(embed_dim, embed_dim//2, (3, 7), (1, 1), (1, 3)), nn.ReLU(True),
            nn.Conv2d(embed_dim//2, embed_dim//4, (3, 7), (1, 1), (1, 3)), nn.ReLU(True),
            nn.Conv2d(embed_dim//4, embed_dim//8, (3, 7), (1, 1), (1, 3)), nn.ReLU(True),
            nn.Conv2d(embed_dim//8, 2, 1, 1, 0))
        self.linear_pred_col = nn.Sequential(
            nn.MaxPool2d((7, 3), (1, 1), (3, 1)),
            nn.Conv2d(embed_dim, embed_dim//2, (7, 3), (1, 1), (3, 1)), nn.ReLU(True),
            nn.Conv2d(embed_dim//2, embed_dim//4, (7, 3), (1, 1), (3, 1)), nn.ReLU(True),
            nn.Conv2d(embed_dim//4, embed_dim//8, (7, 3), (1, 1), (3, 1)), nn.ReLU(True),
            nn.Conv2d(embed_dim//8, 2, 1, 1, 0))
        self.row_cuer = RCSegFormerCueV3(embed_dim, mode='row')
        self.col_cuer = RCSegFormerCueV3(embed_dim, mode='col')
        self.dropout = nn.Dropout2d(0.1)
        

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        
        row_cue = self.row_cuer(seg)
        col_cue = self.col_cuer(seg)
        row_seg = self.linear_pred_row(self.dropout(seg))
        col_seg = self.linear_pred_col(self.dropout(seg))
        col_seg = col_cue*0.02 + col_seg*0.98
        row_seg = row_cue*0.02 + row_seg*0.98
        
        return (row_seg, col_seg)

class RCSegFormerCueV3(nn.Module):
    """Row Column Prediction Cue"""
    def __init__(self, in_channels, mode):
        super().__init__()
        assert(mode in ['row', 'col']), f"mode {mode} not in ['row', 'col']"
        if mode=='row':
            k, s, p = (3, 7), (1, 1), (1, 3)
        else:
            k, s, p = (7, 3), (1, 1), (3, 1)
            
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, k, s, p), nn.ReLU(True),
            nn.Conv2d(in_channels//2, in_channels//4, k, s, p), nn.ReLU(True),
            nn.Conv2d(in_channels//4, 2, k, s, p)
        )
        print(f"[INFO] Create {mode.capitalize()} cuer network")
        
    def forward(self, x: Tensor):
        _, _, h, w = x.size()
        x = self.conv(x)
        mean_dim = int(2 if self.mode=='col' else 3)
        x = torch.mean(x, dim=mean_dim, keepdim=True)
        if self.mode=='col':
            x = x.repeat(1, 1, h, 1)
        else:
            x = x.repeat(1, 1, 1, w)
        return x
    
