import os
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch.autograd import Variable

# ======================
# PReNet
# ======================
from networks import PReNet

# ======================
# metrics
# ======================
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

import lpips
import pyiqa

# ======================
# 配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "checkpoint/PReNet6.pth"

rain_dir = "test/rain"
clean_dir = "test/clean"

save_dir = "results_batch"
os.makedirs(save_dir, exist_ok=True)

save_images = True

recurrent_iter = 6

# ======================
# 加载模型
# ======================
print("Loading PReNet6...")

model = PReNet(
    recurrent_iter,
    use_GPU=(device == "cuda")
)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model = model.to(device)
model.eval()

print("Model loaded!")

# ======================
# LPIPS / NIQE
# ======================
lpips_fn = lpips.LPIPS(net='alex').to(device)

niqe_model = pyiqa.create_metric(
    'niqe',
    device=device
)

# ======================
# 工具
# ======================

def normalize(img):
    """
    PReNet 原始代码使用:
    [0,255] -> [0,1]
    """
    return img / 255.0


def tensor_to_np(x):

    return (
        x.detach()
        .cpu()
        .numpy()
        .transpose(0, 2, 3, 1)
    )


def load_image_cv(path):

    img = cv2.imread(path)

    # BGR -> RGB
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    img = cv2.resize(img, (128, 128))
    img = normalize(
        np.float32(img)
    )

    img = np.expand_dims(
        img.transpose(2, 0, 1),
        0
    )

    img = torch.Tensor(img)

    return img


def save_tensor_image(tensor, path):

    img = tensor.detach().cpu().numpy()[0]

    img = img.transpose(1, 2, 0)

    img = np.clip(img, 0, 1)

    img = np.uint8(img * 255)

    # RGB -> BGR
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    cv2.imwrite(path, img)


# ======================
# 推理
# ======================

def infer(model, rain):

    with torch.no_grad():

        out, _ = model(rain)

        out = torch.clamp(
            out,
            0.,
            1.
        )

    return out


# ======================
# 测试
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

    # ======================
    # load image
    # ======================

    rain = load_image_cv(
        rain_path
    ).to(device)

    clean = load_image_cv(
        clean_path
    ).to(device)

    # ======================
    # inference
    # ======================

    pred = infer(
        model,
        rain
    )

    # ======================
    # numpy
    # ======================

    pred_np = tensor_to_np(pred)[0]
    clean_np = tensor_to_np(clean)[0]

    pred_np = np.clip(
        pred_np,
        0,
        1
    )

    clean_np = np.clip(
        clean_np,
        0,
        1
    )

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
        pred * 2 - 1,
        clean * 2 - 1
    ).item()

    # ======================
    # NIQE
    # ======================

    niqe_score = niqe_model(
        pred
    ).item()

    # ======================
    # 累加
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

        save_tensor_image(
            pred,
            os.path.join(
                save_dir,
                name
            )
        )

# ======================
# 输出
# ======================

print("\n===== PReNet6 Test Result =====")

print(
    f"PSNR : {psnr_total / count:.2f}"
)

print(
    f"SSIM : {ssim_total / count:.4f}"
)

print(
    f"LPIPS: {lpips_total / count:.4f}"
)

print(
    f"NIQE : {niqe_total / count:.4f}"
)