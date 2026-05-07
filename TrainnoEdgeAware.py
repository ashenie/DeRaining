import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from pytorch_msssim import ssim

from utils.dataset import RainMixDataset
from models.unet import UNet
from ema import EMA


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 4
lr = 3e-5
epochs = 200
image_size = 128


# ======================
# dataset
# ======================
dataset = RainMixDataset(
    rain100h_path="dataset/Rain100H",
    rain100l_path="dataset/Rain100L",
    size=image_size,
    ratio=0.5
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)


# =====================================================
# 不使用 edge-aware
#
# 输入:
# x         -> 3
# rain      -> 3
# label     -> 1
# dir_feat  -> 12
#
# total     -> 19
# =====================================================

model = UNet(
    in_channels=19,
    out_channels=3
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr
)

ema = EMA(model, decay=0.999)

mse = nn.MSELoss()
l1 = nn.L1Loss()


# ======================
# rain loss
# ======================
def rain_loss_final(pred, rain, clean, mask):

    rain_gt = rain - clean

    rain_pred = rain - pred

    diff = torch.abs(
        rain_pred - rain_gt
    )

    brightness = torch.mean(
        rain_gt,
        dim=1,
        keepdim=True
    )

    weight = (
        1 + 4 * brightness.clamp(min=0)
    )

    focus = mask.detach()

    loss_main = torch.mean(
        diff * weight * focus
    )

    return loss_main


# ======================
# util
# ======================
def to_01(x):

    return (x + 1) / 2


# ======================
# VGG
# ======================
vgg = models.vgg16(
    pretrained=True
).features[:16].to(device).eval()


for p in vgg.parameters():
    p.requires_grad = False


def to_vgg(x):

    x = (x + 1) / 2

    mean = torch.tensor(
        [0.485,0.456,0.406],
        device=x.device
    ).view(1,3,1,1)

    std = torch.tensor(
        [0.229,0.224,0.225],
        device=x.device
    ).view(1,3,1,1)

    return (x - mean) / std


def perceptual_loss(x, y):

    x = to_vgg(x)

    y = to_vgg(y)

    return F.l1_loss(
        vgg(x),
        vgg(y)
    )


# ======================
# FFT loss
# ======================
def fft_loss(pred, target):

    pred_f = torch.fft.fft2(pred)

    target_f = torch.fft.fft2(target)

    pred_mag = torch.abs(pred_f)

    target_mag = torch.abs(target_f)

    return F.l1_loss(
        pred_mag,
        target_mag
    )


# ======================
# direction feature
# ======================
def rain_direction_feature(x):

    device = x.device

    kernels = [

        torch.tensor(
            [[0,0,0],
             [1,0,-1],
             [0,0,0]],
            dtype=torch.float32,
            device=device
        ),

        torch.tensor(
            [[0,1,0],
             [0,0,-1],
             [0,0,0]],
            dtype=torch.float32,
            device=device
        ),

        torch.tensor(
            [[0,1,0],
             [0,0,0],
             [0,-1,0]],
            dtype=torch.float32,
            device=device
        ),

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

        k = (
            k.view(1,1,3,3)
            .repeat(C,1,1,1)
        )

        f = F.conv2d(
            x,
            k,
            padding=1,
            groups=C
        )

        feats.append(f)

    return torch.cat(feats, dim=1)


# =====================================================
# training
# =====================================================
for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch in dataloader:

        rain = batch["rain"].to(device)

        clean = batch["clean"].to(device)

        label = batch["label"].to(device)

        B, C, H, W = rain.shape

        label_map = (
            label.view(-1,1,1,1)
            .float()
            .expand(-1,1,H,W)
        )

        x = rain.clone()

        # ======================
        # 双阶段恢复
        # ======================
        alpha_list = [0.15, 0.07]

        for alpha in alpha_list:

            # 不再使用 edge
            dir_feat = rain_direction_feature(x)

            # ==================================
            # 19 channels
            # ==================================

            model_input = torch.cat([
                x,
                rain,
                label_map,
                dir_feat
            ], dim=1)

            pred_res, rain_mask = model(
                model_input
            )

            x = (
                x
                - alpha * rain_mask * pred_res
            ).clamp(-1, 1)

        clean_pred = x


        # ======================
        # losses
        # ======================

        loss_mse = F.mse_loss(
            clean_pred,
            clean
        )

        loss_ssim = 1 - ssim(
            to_01(clean_pred),
            to_01(clean),
            data_range=1.0,
            size_average=True
        )

        loss_rain = rain_loss_final(
            clean_pred,
            rain,
            clean,
            rain_mask
        )

        loss_fft = fft_loss(
            clean_pred,
            clean
        )

        # ======================
        # final loss
        # ======================
        loss = (
            0.6 * loss_mse
            + 0.25 * loss_rain
            + 0.1 * loss_ssim
            + 0.05 * loss_fft
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        ema.update(model)

        total_loss += loss.item()

    print(
        f"Epoch {epoch} "
        f"Loss: {total_loss/len(dataloader):.4f}"
    )

    if epoch % 5 == 0:

        torch.save(
            ema.ema_model.state_dict(),
            f"ema_{epoch}.pth"
        )