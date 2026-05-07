import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
from tqdm import tqdm

# ===== metrics =====
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import pyiqa

# ===== model =====
from runpy import run_path

# ======================
# 配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

input_dir = "test/rain"
gt_dir = "test/clean"
result_dir = "results_ddn"
model_path = "checkpoint/deraining_DDN.pth"

os.makedirs(result_dir, exist_ok=True)

# ======================
# 加载模型
# ======================
parameters = {
    'in_channels': 3,
    'window_size': 8,
    'use_bias': True,
    'reduction': 4,
    'out_channels': 3
}

print("Loading model...")

load_arch = run_path(
    os.path.join('SADT_arch.py')
)

model = load_arch['SADT'](**parameters)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['params'])

model = model.to(device)
model.eval()

print("Model loaded!")

# ======================
# metrics
# ======================
lpips_fn = lpips.LPIPS(net='alex').to(device)
niqe_model = pyiqa.create_metric('niqe', device=device)

# ======================
# 工具函数
# ======================
def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def save_img(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def img_to_tensor(img):
    return torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0)

def tensor_to_np(x):
    return x.detach().cpu().numpy().transpose(0,2,3,1)

# ======================
# 文件列表（简化版）
# ======================
rain_files = sorted(os.listdir(input_dir))

assert len(rain_files) > 0, "No test images found!"

# ======================
# 测试
# ======================
avg_psnr = 0
avg_ssim = 0
avg_lpips = 0
avg_niqe = 0

count = 0

img_multiple_of = 64

with torch.no_grad():
    for name in tqdm(rain_files):

        inp_path = os.path.join(input_dir, name)
        gt_path = os.path.join(gt_dir, name)

        if not os.path.exists(gt_path):
            continue

        # ===== load =====
        img = load_img(inp_path)
        gt = load_img(gt_path)

        # ===== 尺寸对齐（关键）=====
        if img.shape != gt.shape:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]))

        input_tensor = img_to_tensor(img).to(device)

        # ===== padding =====
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H = (h + img_multiple_of) // img_multiple_of * img_multiple_of
        W = (w + img_multiple_of) // img_multiple_of * img_multiple_of

        pad_h = H - h
        pad_w = W - w

        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), 'reflect')

        # ===== inference =====
        restored = model(input_tensor)[0]
        restored = torch.clamp(restored, 0, 1)

        # ===== remove pad =====
        restored = restored[:, :, :h, :w]

        # ===== numpy =====
        restored_np = tensor_to_np(restored)[0]
        gt_np = gt.astype(np.float32) / 255.

        restored_np = np.clip(restored_np, 0, 1)

        # ===== PSNR =====
        psnr = peak_signal_noise_ratio(gt_np, restored_np, data_range=1.0)

        # ===== SSIM =====
        ssim = structural_similarity(
            gt_np,
            restored_np,
            channel_axis=2,
            data_range=1.0
        )

        # ===== tensor =====
        restored_tensor = torch.from_numpy(restored_np).permute(2,0,1).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(gt_np).permute(2,0,1).unsqueeze(0).to(device)

        # ===== LPIPS =====
        lp = lpips_fn(
            restored_tensor * 2 - 1,
            gt_tensor * 2 - 1
        ).item()

        # ===== NIQE =====
        niqe_score = niqe_model(restored_tensor).item()

        # ===== 累加 =====
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lp
        avg_niqe += niqe_score

        count += 1

        # ===== 保存结果 =====
        save_path = os.path.join(result_dir, name)
        save_img(save_path, (restored_np * 255).astype(np.uint8))

# ======================
# 输出
# ======================
avg_psnr /= count
avg_ssim /= count
avg_lpips /= count
avg_niqe /= count

print("\n===== DDN Test Result =====")
print(f"PSNR : {avg_psnr:.2f}")
print(f"SSIM : {avg_ssim:.4f}")
print(f"LPIPS: {avg_lpips:.4f}")
print(f"NIQE : {avg_niqe:.4f}")