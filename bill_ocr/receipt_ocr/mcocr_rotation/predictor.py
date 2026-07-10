"""MobileNetV3 line-crop classifier: 0° vs 180°."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from .mobilenetv3 import mobilenetv3


class _NewPad:
    def __init__(self, t_size: tuple[int, int] = (64, 192), fill: tuple[int, int, int] = (255, 255, 255)):
        from torchvision.transforms.functional import pad

        self.t_size = t_size
        self.fill = fill
        self._pad = pad

    def __call__(self, img: Image.Image) -> Image.Image:
        target_h, target_w = self.t_size
        w, h = img.size
        im_scale = h / max(w, 1)
        target_scale = target_h / target_w
        if im_scale < target_scale:
            new_w = int(round(target_h / im_scale))
            out_im = img.resize((new_w, target_h))
        else:
            new_w = h / target_scale
            pad_x = int(round((new_w - w) / 2))
            out_im = self._pad(img, (pad_x, 0, pad_x, 0), self.fill, "constant")
            out_im = out_im.resize((self.t_size[1], self.t_size[0]))
        return out_im


class PageRotationModel:
    def __init__(self, weight_path: str | Path, device: torch.device | None = None):
        self.class_list = ["0", "180"]
        self.device = device or torch.device("cpu")
        self.im_h, self.im_w = 64, 192
        net = mobilenetv3(n_class=2, dropout=0.2, input_size=64)
        net.load_state_dict(torch.load(str(weight_path), map_location="cpu", weights_only=False))
        self.net = net.to(self.device).eval()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                _NewPad(t_size=(64, 192), fill=(255, 255, 255)),
                transforms.Resize((64, 192), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def classify_crop(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.transform(pil).view(1, 3, self.im_h, self.im_w).to(self.device)
        with torch.no_grad():
            out = self.net(Variable(x))
            probs = nn.Softmax(dim=1)(out).cpu().numpy()[0]
        idx = int(probs.argmax())
        return self.class_list[idx], float(probs[idx])


def load_rotation_model(weight_path: str | Path, device: str | None = None) -> PageRotationModel:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return PageRotationModel(weight_path, dev)
