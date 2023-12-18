import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semseg.augmentations_v2 import get_val_augmentation, get_test_augmentation
from numpy import int64, ndarray
import numpy as np
import cv2

from rich.console import Console
console = Console()


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE'][0]
        self.tf_pipeline = get_val_augmentation(self.size)

    def preprocess(self, image: ndarray) -> ndarray:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        tfs = self.tf_pipeline(image=image)
        image = tfs['image']
        
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        orig_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = orig_map.softmax(dim=1).argmax(dim=1).cpu()
        heat_map = (orig_map.softmax(dim=1).cpu()*255).type(torch.int64).squeeze().permute(1, 2, 0)
        heat_map = torch.cat([torch.zeros_like(heat_map)[:, :, :2], heat_map[:, :, 1:]], dim=-1)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)
            heat_image = (orig_img.permute(1, 2, 0) * 0.4) + (heat_map * 0.6)

            heat_image = heat_image.numpy()

        image = draw_text(seg_image, seg_map, self.labels)
        return image, heat_map.numpy()

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool):
        image = cv2.imread(img_fname).astype('float32')
        img = self.preprocess(image)
        orig_img = get_test_augmentation(self.size)(image=image)['image']
        orig_img = torch.from_numpy(orig_img.transpose(2,0,1)).type(torch.FloatTensor)
        tensor_img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).type(torch.FloatTensor).to(self.device)
        row_map, col_map = self.model_forward(tensor_img)
        row_map , row_heat= self.postprocess(orig_img, row_map, overlay)
        col_map, _ = self.postprocess(orig_img, col_map, overlay)
        return row_map, col_map, row_heat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/split_segformer_v3.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)
    
    semseg = SemSeg(cfg)

    with console.status("[bright_green]Processing..."):
        if test_file.is_file():
            console.rule(f'[green]{test_file}')
            row_map, col_map, row_heat = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
            row_map.save(save_dir / f"{str(test_file.stem)}_row.png")
            col_map.save(save_dir / f"{str(test_file.stem)}_col.png")
            cv2.imwrite(f"{save_dir}/{str(test_file.stem)}_row_heat.png", row_heat)
        else:
            files = test_file.glob('*.*')
            for i, file in enumerate(files):
                console.rule(f'[green]{file}')
                row_map, col_map, row_heat = semseg.predict(str(file), cfg['TEST']['OVERLAY'])
                row_map.save(save_dir / f"{i}_row.png")
                col_map.save(save_dir / f"{i}_col.png")
                cv2.imwrite(f"{save_dir}/{i}_row_heat.png", row_heat)

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")