from .upernet import UPerHead
from .segformer import SegFormerHead, RCSegFormerHead, \
    RCSegFormerHeadV2, RCSegFormerHeadV3
from .sfnet import SFHead
from .fpn import FPNHead
from .fapn import FaPNHead
from .fcn import FCNHead
from .condnet import CondHead
from .lawin import LawinHead

__all__ = ['UPerHead', 'SegFormerHead', 'SFHead', 'FPNHead', 'FaPNHead', 'FCNHead', 'CondHead', 'LawinHead', 'RCSegFormerHead', 'RCSegFormerHeadV2', 'RCSegFormerHeadV3']