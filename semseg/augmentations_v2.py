import albumentations as A
from albumentations.augmentations.geometric.resize import LongestMaxSize
import cv2
from numpy import interp

def get_train_augmentation(max_size: int, seg_fill: int = 0):
    return A.Compose([
        # A.ColorJitter(brightness=0.0, contrast=0.5, saturation=0.5, hue=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.3, 1.), p=0.5),
        # A.RandomAutoContrast(p=0.2),
        #  A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.GaussianBlur((3, 3), p=0.5),
        A.ToGray(p=0.2),
        # A.Rotate(limit=10, p=1, border_mode=cv2.BORDER_CONSTANT),
        LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_val_augmentation(max_size: int):
    return A.Compose([
        LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
def get_test_augmentation(max_size: int):
    return A.Compose([
        LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT),
    ])