import torch 
import os
import tqdm
import json
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class SplitDataset(Dataset):
    CLASSES = ['background', 'row', 'col']
    PALETTE = torch.tensor([[255, 255 , 255], [0, 255, 255], [255, 255, 0]])

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        
        self.CLASSES = self.CLASSES
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.label_dicts = self.get_label_dicts(root, split)
        if not self.label_dicts: raise Exception(f"No images found in {root}")
        print(f"Found {len(self.label_dicts)} {split} images.")
        
    def get_label_dicts(self, root, split):
        label_dicts = []
        labels_path = os.path.join(root, 'gts', split)
        img_dir = os.path.join(root, 'images ')
        
        if os.path.isfile(labels_path):
            with open(labels_path, 'r') as file:
                for line in tqdm.tqdm(file, total=350000):
                    # dict_keys(['img_fn', 'img_split', 'row_label', 'column_label', 'rels', 'h_origin', 'w_origin'])
                    out = json.loads(line)
                    out['img_path'] = os.path.join(img_dir, out['img_fn'])
                    label_dicts.append(out)
        elif os.path.isdir(labels_path):
            for json_file in tqdm.tqdm(os.listdir(labels_path)):
                if os.path.isdir(os.path.join(labels_path, json_file)): 
                    continue
                
                with open(os.path.join(labels_path, json_file), 'r') as j:
                    out = json.loads(j.read())
                    out['img_path'] = os.path.join(img_dir, out['img_fn'])
                    label_dicts.append(out)

        return label_dicts

    def __len__(self) -> int:
        return len(self.label_dicts)

    def __getitem__(self, idx: int):
        assert(idx < len(self.label_dicts))
        label_dict = self.label_dicts[idx]
        # Image
        img_path = label_dict['img_path']
        image = cv2.imread(img_path).astype('float32')  
        
        if image.ndim==2:
            image = image[np.newaxis]
            
        row_line = np.array(label_dict['row_label']).reshape(-1, 1)
        col_line = np.array(label_dict['column_label']).reshape(1, -1)
        row_mask = row_line * np.ones((image.shape[0], image.shape[1]))
        col_mask = col_line * np.ones((image.shape[0], image.shape[1]))
    
        if self.transform:
            tfs = self.transform(image=image, masks=[row_mask, col_mask])
            image = tfs['image']
            row_mask, col_mask = tfs['masks']
        # image np to tensor
        # plt.imsave('./_img_test.jpg', image.astype(np.uint8))
        image = image / 255.
        image = torch.from_numpy(image.transpose((2,0,1))).type(torch.FloatTensor)
        row_mask = torch.from_numpy(row_mask).type(torch.LongTensor)
        col_mask = torch.from_numpy(col_mask).type(torch.LongTensor)    
            
        return image, (row_mask, col_mask)

if __name__ == '__main__':
    import albumentations as A
    from albumentations.augmentations.geometric.resize import LongestMaxSize
    def get_train_augmentation(max_size: int, seg_fill: int = 0):
        return A.Compose([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.3, 1.), p=1.0),
            #  A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.GaussianBlur((3, 3), p=1.0),
            A.ToGray(p=1.0),
            # A.Rotate(limit=10, p=1, border_mode=cv2.BORDER_CONSTANT),
            LongestMaxSize(max_size, interpolation=cv2.INTER_NEAREST),
            # A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            A.PadIfNeeded(min_height=max_size, min_width=max_size, border_mode=cv2.BORDER_CONSTANT)
        ])
        
    tfs = get_train_augmentation(1024)
    dts = SplitDataset(root='/home/nguyenduong/Data/Real_Data/BCTC/BCTC', split='val', transform=tfs)
    outs = './test_labels'
    if not os.path.exists(outs):
        os.makedirs(outs)
    for i in tqdm.tqdm(range(50), total=50):
        img, lbls = dts[i]
        np_img = img.numpy().squeeze().transpose(1,2,0) * 255.
        row_m = lbls[0].numpy().squeeze()
        col_m = lbls[1].numpy().squeeze()
        plt.imsave(os.path.join(outs, f'./{i}_img_test.jpg'), np_img.astype(np.uint8))
        plt.imsave(os.path.join(outs, f'./{i}_row_mask.jpg'), row_m)
        plt.imsave(os.path.join(outs, f'./{i}_col_mask.jpg'), col_m)
        