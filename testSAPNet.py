import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.autograd import Variable

from network import SAPNet
from utilsSAPNet import *

# ===== metrics =====
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import pyiqa

# ======================
# config
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

test_rain_dir = "test/rain"
test_clean_dir = "test/clean"

save_dir = "results_batch"
os.makedirs(save_dir, exist_ok=True)

image_size = 128   # 🔥统一尺寸

# ======================
# model
# ======================
print("Loading model...")

model = SAPNet(
    recurrent_iter=6,
    use_dilation=True
).to(device)

model = nn.DataParallel(model)

ckpt_path = "checkpoint/SAPNet.pth"   # 改成你的路径
model.load_state_dict(torch.load(ckpt_path, map_location=device))

model.eval()

# ======================
# metrics
# ======================
lpips_fn = lpips.LPIPS(net='alex').to(device)
niqe_model = pyiqa.create_metric('niqe', device=device)

# ======================
# tools
# ======================
def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 🔥统一尺寸（关键）
    img = cv2.resize(img, (image_size, image_size))

    img = np.float32(img) / 255.0
    return img

def to_tensor(img):
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

def save_img(path, img):
    img = (img * 255).clip(0,255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

# ======================
# test loop
# ======================
rain_files = sorted(os.listdir(test_rain_dir))

psnr_total = 0
ssim_total = 0
lpips_total = 0
niqe_total = 0
count = 0

time_test = 0

for name in tqdm(rain_files):

    rain_path = os.path.join(test_rain_dir, name)
    clean_path = os.path.join(test_clean_dir, name)

    if not os.path.exists(clean_path):
        continue

    # ======================
    # load
    # ======================
    rain = load_img(rain_path)
    clean = load_img(clean_path)

    rain_t = to_tensor(rain).to(device)
    clean_t = to_tensor(clean).to(device)

    # ======================
    # forward
    # ======================
    with torch.no_grad():

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        out, _ = model(rain_t)
        out = torch.clamp(out, 0, 1)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

        time_test += (end - start)

    # ======================
    # tensor -> numpy
    # ======================
    pred = out[0].cpu().numpy().transpose(1,2,0)
    gt = clean

    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)

    # ======================
    # PSNR
    # ======================
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)

    # ======================
    # SSIM
    # ======================
    ssim = structural_similarity(
        gt,
        pred,
        channel_axis=2,
        data_range=1.0
    )

    # ======================
    # LPIPS（需要 [-1,1]）
    # ======================
    pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).to(device)

    lp = lpips_fn(pred_t*2-1, gt_t*2-1).item()

    # ======================
    # NIQE
    # ======================
    niqe = niqe_model(pred_t).item()

    # ======================
    # accumulate
    # ======================
    psnr_total += psnr
    ssim_total += ssim
    lpips_total += lp
    niqe_total += niqe
    count += 1

    # ======================
    # save
    # ======================
    save_path = os.path.join(save_dir, name)
    save_img(save_path, pred)

# ======================
# result
# ======================
print("\n===== SAPNet Test Result =====")
print(f"PSNR : {psnr_total/count:.2f}")
print(f"SSIM : {ssim_total/count:.4f}")
print(f"LPIPS: {lpips_total/count:.4f}")
print(f"NIQE : {niqe_total/count:.4f}")
print("Avg time:", time_test/count)