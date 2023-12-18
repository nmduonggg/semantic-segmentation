import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import RCSegFormerHead, RCSegFormerHeadV2, RCSegFormerHeadV3


class RCSegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B1', num_classes: int = 2) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = RCSegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        y = self.backbone(x)
        yr, yc = self.decode_head(y)   # 4x reduction in image size
        yr = F.interpolate(yr, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        yc = F.interpolate(yc, size=x.shape[2:], mode='bilinear', align_corners=False) 
        return yr, yc
    
class RCSegFormerV2(BaseModel):
    def __init__(self, backbone: str = 'MiT-B1', num_classes: int = 2) -> None:
        super().__init__(backbone, num_classes)
        print("[INFO] Initialize RCSegFormer-V2...")
        self.decode_head = RCSegFormerHeadV2(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        y = self.backbone(x)
        yr, yc = self.decode_head(y)   # 4x reduction in image size
        yr = F.interpolate(yr, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        yc = F.interpolate(yc, size=x.shape[2:], mode='bilinear', align_corners=False) 
        return yr, yc
    
class RCSegFormerV3(BaseModel):
    '''V3 of RCSegFormer, increase prediction head size'''
    def __init__(self, backbone: str = 'MiT-B1', num_classes: int = 2) -> None:
        super().__init__(backbone, num_classes)
        print("[INFO] Initialize RCSegFormer-V3...")
        self.decode_head = RCSegFormerHeadV3(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        y = self.backbone(x)
        yr, yc = self.decode_head(y)   # 4x reduction in image size
        yr = F.interpolate(yr, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        yc = F.interpolate(yc, size=x.shape[2:], mode='bilinear', align_corners=False) 
        return yr, yc

if __name__ == '__main__':
    model = RCSegFormerV2('MiT-B1')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    yr, yc = model(x)
    print(yr.shape, yc.shape)