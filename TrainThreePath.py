import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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

dataset = RainMixDataset(
    rain100h_path="dataset/Rain100H",
    rain100l_path="dataset/Rain100L",
    size=image_size,
    ratio=0.5   # H/L均衡
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

model = UNet(in_channels=22, out_channels=3).to(device)
# 4通道：3(RGB) + 1(label)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

ema = EMA(model, decay=0.999)
mse = nn.MSELoss()
l1 = nn.L1Loss()



def sobel(x):
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                            dtype=torch.float32, device=x.device).view(1,1,3,3)
    kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],
                            dtype=torch.float32, device=x.device).view(1,1,3,3)

    B, C, H, W = x.shape

    grad_x = F.conv2d(x, kernel_x.repeat(C,1,1,1), padding=1, groups=C)
    grad_y = F.conv2d(x, kernel_y.repeat(C,1,1,1), padding=1, groups=C)

    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

def rain_loss_final(pred, rain, clean, mask):
    rain_gt = rain - clean
    rain_pred = rain - pred

    # ===== 基础误差 =====
    diff = torch.abs(rain_pred - rain_gt)

    # ===== 🔥 亮度增强（加强版）=====
    brightness = torch.mean(rain_gt, dim=1, keepdim=True)
    weight = 1 + 4 * brightness.clamp(min=0)   # 🔥 原来是2 → 提高到4

    # ===== 🔥 mask 聚焦 =====
    focus = mask.detach()  # 防止mask被影响

    # ===== 主 loss（重点）=====
    loss_main = torch.mean(diff * weight * focus)

    # 梯度（方向）
    grad_gt = sobel(rain_gt)
    grad_pred = sobel(rain_pred)

    loss_grad = torch.mean(torch.abs(grad_pred - grad_gt))

    return loss_main + 0.2 * loss_grad

def to_01(x):
    return (x + 1) / 2

vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()

def to_vgg(x):
    x = (x + 1) / 2  # [-1,1] → [0,1]

    mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)

    return (x - mean) / std

def perceptual_loss(x, y):
    x = to_vgg(x)
    y = to_vgg(y)
    return F.l1_loss(vgg(x), vgg(y))

def edge_protect_loss(pred, target):
    edge_p = sobel(pred)
    edge_t = sobel(target)
    return F.l1_loss(edge_p, edge_t)

def fft_loss(pred, target):
    pred_f = torch.fft.fft2(pred)
    target_f = torch.fft.fft2(target)

    pred_mag = torch.abs(pred_f)
    target_mag = torch.abs(target_f)

    return F.l1_loss(pred_mag, target_mag)

def rain_direction_feature(x):
    device = x.device

    kernels = [
        torch.tensor([[0,0,0],[1,0,-1],[0,0,0]], dtype=torch.float32, device=device),
        torch.tensor([[0,1,0],[0,0,-1],[0,0,0]], dtype=torch.float32, device=device),
        torch.tensor([[0,1,0],[0,0,0],[0,-1,0]], dtype=torch.float32, device=device),
        torch.tensor([[0,0,0],[0,0,1],[0,-1,0]], dtype=torch.float32, device=device),
    ]

    B, C, H, W = x.shape
    feats = []

    for k in kernels:
        k = k.view(1,1,3,3).repeat(C,1,1,1)
        f = F.conv2d(x, k, padding=1, groups=C)
        feats.append(f)

    return torch.cat(feats, dim=1)  # 4C 通道

for p in vgg.parameters():
    p.requires_grad = False

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for batch in dataloader:

        rain = batch["rain"].to(device)
        clean = batch["clean"].to(device)
        label = batch["label"].to(device)

        B, C, H, W = rain.shape
        label_map = label.view(-1,1,1,1).float().expand(-1,1,H,W)

        x = rain.clone()
        
        alpha_list = [0.15, 0.07, 0.03]  # 🔥 三阶段
        # ===== iterative =====
        for alpha in alpha_list:
            edge = sobel(x)
            dir_feat = rain_direction_feature(x)
            model_input = torch.cat([x, rain, edge, label_map, dir_feat], dim=1)

            pred_res, rain_mask = model(model_input)
            
            x = (x- alpha * rain_mask * pred_res).clamp(-1, 1)
        
        clean_pred = x

        # ===== loss =====
        loss_mse = F.mse_loss(clean_pred, clean)

        #loss_per = perceptual_loss(clean_pred, clean)

        loss_ssim = 1 - ssim(
            to_01(clean_pred),
            to_01(clean),
            data_range=1.0,
            size_average=True
        )

        loss_rain = rain_loss_final(clean_pred, rain, clean, rain_mask)

        loss_edge = edge_protect_loss(clean_pred, clean)

        loss_fft=fft_loss(clean_pred, clean)

        loss = (
            0.5 * loss_mse
            + 0.25 * loss_rain
            + 0.1 * loss_edge
            + 0.1 * loss_ssim
            + 0.05 * loss_fft
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        
        ema.update(model)
        total_loss += loss.item()


    print(f"Epoch {epoch} Loss: {total_loss/len(dataloader):.4f}")

    if epoch % 5 == 0:
        torch.save(ema.ema_model.state_dict(), f"ema_{epoch}.pth")