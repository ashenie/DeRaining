import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import cv2
from MPRNet import MPRNet
import utils

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import pyiqa
from skimage import img_as_ubyte

# ======================
# config（统一你的实验）
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rain_dir = "test/rain"
clean_dir = "test/clean"
save_dir = "results_batch"

model_path = "checkpoint/model_deraining.pth"

save_images = True
os.makedirs(save_dir, exist_ok=True)

# ======================
# model
# ======================
model = MPRNet()
checkpoint = torch.load(model_path, map_location=device)

state_dict = checkpoint["state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v

model.load_state_dict(new_state_dict)
model.eval()

# ======================
# metrics
# ======================
lpips_fn = lpips.LPIPS(net='alex').to(device)
niqe_model = pyiqa.create_metric('niqe', device=device)

def normalize(img):
    """
    PReNet 原始代码使用:
    [0,255] -> [0,1]
    """
    return img / 255.0

# ======================
# utils
# ======================
def load_img(path):
    
    import numpy as np

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    img = np.float32(img) / 255.0

    return img

def to_tensor(img):
    return torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)

def to_np(x):
    return x.detach().cpu().numpy().transpose(0,2,3,1)

def save_img(path, img):
    """
    img: numpy array (H, W, C), range [0,1]
    """
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


# ======================
# file list（统一方式）
# ======================
rain_files = sorted(os.listdir(rain_dir))

psnr_total = 0
ssim_total = 0
lpips_total = 0
niqe_total = 0
count = 0

# ======================
# inference
# ======================

files = sorted(os.listdir(rain_dir))

for name in tqdm(files):

    rain_path = os.path.join(rain_dir, name)
    clean_path = os.path.join(clean_dir, name)

    if not os.path.exists(clean_path):
        continue

    # ======================
    # 读取 (0~1)
    # ======================
    rain = load_img(rain_path).astype(np.float32)
    clean = load_img(clean_path).astype(np.float32)

    # 尺寸对齐
    if rain.shape != clean.shape:
        clean = utils.resize_np(clean, rain.shape)

    # ======================
    # 转 tensor [0,1]
    # ======================
    rain_t = torch.from_numpy(rain).permute(2,0,1).unsqueeze(0).to(device)
    clean_t = torch.from_numpy(clean).permute(2,0,1).unsqueeze(0).to(device)

    # ======================
    # 推理
    # ======================
    with torch.no_grad():
        restored = model(rain_t)

    # 👉 MPRNet 是多阶段输出
    if isinstance(restored, (list, tuple)):
        restored = restored[-1]

    # 👉 clamp 到 [0,1]
    restored = torch.clamp(restored, 0, 1)



    # Debug（建议保留）
    # print("min/max:", restored.min().item(), restored.max().item())

    # ======================
    # numpy
    # ======================
    pred_np = restored[0].detach().cpu().numpy().transpose(1,2,0)
    clean_np = clean_t[0].detach().cpu().numpy().transpose(1,2,0)

    pred_np = np.clip(pred_np, 0, 1)
    clean_np = np.clip(clean_np, 0, 1)

    # ======================
    # PSNR
    # ======================
    psnr = peak_signal_noise_ratio(clean_np, pred_np, data_range=1.0)

    # ======================
    # SSIM
    # ======================
    ssim = structural_similarity(clean_np, pred_np, channel_axis=2, data_range=1.0)

    # ======================
    # LPIPS（必须 [-1,1]）
    # ======================
    lp = lpips_fn(
        (restored * 2 - 1).detach(),
        (clean_t * 2 - 1).detach()
    ).item()

    # ======================
    # NIQE
    # ======================
    niqe_val = niqe_model(restored.detach()).item()

    # ======================
    # 累计
    # ======================
    psnr_total += psnr
    ssim_total += ssim
    lpips_total += lp
    niqe_total += niqe_val
    count += 1

    # ======================
    # 保存
    # ======================
    save_path = os.path.join(save_dir, name)
    save_img(save_path, pred_np)

# ======================
# 平均结果
# ======================
print("\n===== MPRNet Test Result =====")
print(f"PSNR : {psnr_total/count:.2f}")
print(f"SSIM : {ssim_total/count:.4f}")
print(f"LPIPS: {lpips_total/count:.4f}")
print(f"NIQE : {niqe_total/count:.4f}")
