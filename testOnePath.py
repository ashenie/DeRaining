import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm

from models.unet import UNet

# ===== metrics =====
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

import lpips
import pyiqa


# ======================
# config
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "PathOne.pth"

rain_dir = "test/rain"
clean_dir = "test/clean"

save_dir = "results_batch"

os.makedirs(save_dir, exist_ok=True)

image_size = 128
save_images = True


# ======================
# model
# ======================

# 单步版本:
# rain(3) + edge(3) + label(1) + dir_feat(12)
# = 19 channels

model = UNet(
    in_channels=22,
    out_channels=3
).to(device)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()


# ======================
# metrics
# ======================
lpips_fn = lpips.LPIPS(net='alex').to(device)

niqe_model = pyiqa.create_metric(
    'niqe',
    device=device
)


# ======================
# transform
# ======================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def load_image(path):

    img = Image.open(path).convert("RGB")

    return transform(img).unsqueeze(0)


def to_01(x):

    return (x + 1) / 2


def tensor_to_np(x):

    return (
        x.detach()
        .cpu()
        .numpy()
        .transpose(0,2,3,1)
    )


# ======================
# sobel
# ======================
def sobel(x):

    kernel_x = torch.tensor(
        [[1,0,-1],
         [2,0,-2],
         [1,0,-1]],
        dtype=torch.float32,
        device=x.device
    ).view(1,1,3,3)

    kernel_y = torch.tensor(
        [[1,2,1],
         [0,0,0],
         [-1,-2,-1]],
        dtype=torch.float32,
        device=x.device
    ).view(1,1,3,3)

    B, C, H, W = x.shape

    grad_x = F.conv2d(
        x,
        kernel_x.repeat(C,1,1,1),
        padding=1,
        groups=C
    )

    grad_y = F.conv2d(
        x,
        kernel_y.repeat(C,1,1,1),
        padding=1,
        groups=C
    )

    return torch.sqrt(
        grad_x**2 + grad_y**2 + 1e-6
    )


# ======================
# direction feature
# ======================
def rain_direction_feature(x):

    device = x.device

    kernels = [

        # horizontal
        torch.tensor(
            [[0,0,0],
             [1,0,-1],
             [0,0,0]],
            dtype=torch.float32,
            device=device
        ),

        # 45°
        torch.tensor(
            [[0,1,0],
             [0,0,-1],
             [0,0,0]],
            dtype=torch.float32,
            device=device
        ),

        # vertical
        torch.tensor(
            [[0,1,0],
             [0,0,0],
             [0,-1,0]],
            dtype=torch.float32,
            device=device
        ),

        # 135°
        torch.tensor(
            [[0,0,0],
             [0,0,1],
             [0,-1,0]],
            dtype=torch.float32,
            device=device
        ),
    ]

    B, C, H, W = x.shape

    feats = []

    for k in kernels:

        k = k.view(1,1,3,3).repeat(C,1,1,1)

        f = F.conv2d(
            x,
            k,
            padding=1,
            groups=C
        )

        feats.append(f)

    return torch.cat(feats, dim=1)


# ======================
# single-step inference
# ======================
def infer_single_step(model, rain, label):

    B, C, H, W = rain.shape

    label_map = (
        label.view(-1,1,1,1)
        .float()
        .expand(-1,1,H,W)
    )

    with torch.no_grad():

        edge = sobel(rain)

        dir_feat = rain_direction_feature(rain)

        # ======================
        # 输入:
        # rain      -> 3
        # edge      -> 3
        # label     -> 1
        # dir_feat  -> 12
        # total     -> 19
        # ======================

        model_input = torch.cat([
            rain,
            rain,
            edge,
            label_map,
            dir_feat
        ], dim=1)

        pred_res, rain_mask = model(model_input)

        alpha = 0.15

        clean_pred = (
            rain
            - alpha * rain_mask * pred_res
        ).clamp(-1, 1)

    return clean_pred


# ======================
# test
# ======================
rain_files = sorted(
    os.listdir(rain_dir)
)

psnr_total = 0
ssim_total = 0
lpips_total = 0
niqe_total = 0

count = 0


for name in tqdm(rain_files):

    rain_path = os.path.join(
        rain_dir,
        name
    )

    clean_path = os.path.join(
        clean_dir,
        name
    )

    if not os.path.exists(clean_path):
        continue

    rain = load_image(rain_path).to(device)

    clean = load_image(clean_path).to(device)

    # label
    label = torch.tensor(
        [0],
        device=device
    )

    # ======================
    # inference
    # ======================
    pred = infer_single_step(
        model,
        rain,
        label
    )

    # ======================
    # convert
    # ======================
    pred_01 = to_01(pred)

    clean_01 = to_01(clean)

    pred_np = tensor_to_np(pred_01)[0]

    clean_np = tensor_to_np(clean_01)[0]

    # ======================
    # PSNR
    # ======================
    psnr = peak_signal_noise_ratio(
        clean_np,
        pred_np,
        data_range=1.0
    )

    # ======================
    # SSIM
    # ======================
    ssim_val = structural_similarity(
        clean_np,
        pred_np,
        channel_axis=2,
        data_range=1.0
    )

    # ======================
    # LPIPS
    # ======================
    lp = lpips_fn(
        pred,
        clean
    ).item()

    # ======================
    # NIQE
    # ======================
    niqe_score = niqe_model(
        pred_01
    ).item()

    # ======================
    # accumulate
    # ======================
    psnr_total += psnr

    ssim_total += ssim_val

    lpips_total += lp

    niqe_total += niqe_score

    count += 1

    # ======================
    # save image
    # ======================
    if save_images:

        vis = torch.cat([
            rain,
            pred,
            clean
        ], dim=0)

        save_image(
            to_01(vis),
            os.path.join(save_dir, name),
            nrow=3
        )


# ======================
# final result
# ======================
print("\n===== Test Result =====")

print(f"PSNR : {psnr_total/count:.2f}")

print(f"SSIM : {ssim_total/count:.4f}")

print(f"LPIPS: {lpips_total/count:.4f}")

print(f"NIQE : {niqe_total/count:.4f}")