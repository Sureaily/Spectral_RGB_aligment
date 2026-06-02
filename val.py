import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import tifffile
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
from skimage.util import img_as_ubyte

from tutorial_dataset import MyDataset
from cldm.model import create_model


# ==================== 评估指标（与训练脚本一致，基于 [0,1] 区间）====================
class Loss_RMSE(nn.Module):
    def forward(self, outputs, label):
        error = outputs - label
        return torch.sqrt(torch.mean(error.pow(2)))

class Loss_MRAE_custom(nn.Module):
    def forward(self, outputs, label):
        mask = label == 0
        label_safe = label.clone()
        if mask.any():
            label_safe[mask] = 1e-5
        return torch.mean(torch.abs(outputs - label) / label_safe)

class Loss_PSNR(nn.Module):
    def forward(self, im_true_01, im_fake_01, data_range=255):
        N, C, H, W = im_true_01.shape
        true = im_true_01.clamp(0, 1).mul(data_range).reshape(N, C * H * W)
        fake = im_fake_01.clamp(0, 1).mul(data_range).reshape(N, C * H * W)
        mse = nn.MSELoss(reduction='none')(true, fake).sum(dim=1) / (C * H * W)
        psnr = 10 * torch.log10(data_range ** 2 / mse)
        return psnr.mean()

class Loss_SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def _gaussian_window(self, C, device):
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device) - self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g /= g.sum()
        g = g.outer(g).reshape(1, 1, self.window_size, self.window_size)
        return g.expand(C, 1, self.window_size, self.window_size)

    def forward(self, im_true_01, im_fake_01):
        C = im_true_01.shape[1]
        window = self._gaussian_window(C, im_true_01.device)
        mu1 = F.conv2d(im_true_01, window, groups=C, padding=self.window_size // 2)
        mu2 = F.conv2d(im_fake_01, window, groups=C, padding=self.window_size // 2)
        mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(im_true_01 * im_true_01, window, groups=C, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(im_fake_01 * im_fake_01, window, groups=C, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(im_true_01 * im_fake_01, window, groups=C, padding=self.window_size // 2) - mu12
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


# ==================== 相机退化模拟函数（与参考脚本一致，操作 [0,255] uint8）====================
def add_gaussian_noise_uint8(img_uint8, noise_level):
    """添加高斯噪声，使得 PSNR 落在预设范围"""
    psnr_target = {"slight": 38.0, "medium": 30.0, "heavy": 20.0}[noise_level]
    max_val = 255.0
    mse = (max_val ** 2) / (10 ** (psnr_target / 10))
    sigma = np.sqrt(mse)
    noise = np.random.normal(0, sigma, img_uint8.shape).astype(np.float32)
    noisy = img_uint8.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def apply_color_cast_uint8(img_uint8, delta_e_range):
    """施加色偏，使平均 CIEDE2000 ΔE 落在给定范围"""
    ref_lab = rgb2lab(img_uint8)
    low, high = 0.0, 30.0
    best_d = 0.0
    for _ in range(30):
        d = (low + high) / 2.0
        temp_lab = ref_lab.copy()
        temp_lab[:, :, 1] += d
        temp_lab[:, :, 2] += d
        temp_lab[:, :, 1] = np.clip(temp_lab[:, :, 1], -128, 127)
        temp_lab[:, :, 2] = np.clip(temp_lab[:, :, 2], -128, 127)
        delta = deltaE_ciede2000(ref_lab, temp_lab)
        avg_delta = np.mean(delta)
        if delta_e_range[0] <= avg_delta <= delta_e_range[1]:
            best_d = d
            break
        elif avg_delta < delta_e_range[0]:
            low = d
        else:
            high = d
        best_d = d

    final_lab = ref_lab.copy()
    final_lab[:, :, 1] += best_d
    final_lab[:, :, 2] += best_d
    final_lab[:, :, 1] = np.clip(final_lab[:, :, 1], -128, 127)
    final_lab[:, :, 2] = np.clip(final_lab[:, :, 2], -128, 127)
    cast_uint8 = img_as_ubyte(lab2rgb(final_lab))
    return cast_uint8


def get_color_cast_range(level):
    ranges = {"slight": (1.5, 3.0), "medium": (5.0, 8.0), "heavy": (12.0, 18.0)}
    return ranges.get(level, None)


# ==================== 辅助函数：对多光谱条件图像施加退化 ====================
def degrade_hint(hint_tensor, noise_level, color_cast_level):
    """
    hint_tensor: 形状 (B, C, H, W)，值域 [-1, 1] (与模型输入一致)
    返回: 退化后的同形状 tensor
    """
    if noise_level == 'none' and color_cast_level == 'none':
        return hint_tensor

    B, C, H, W = hint_tensor.shape
    degraded = []
    for b in range(B):
        # 转成 (H, W, C) numpy，值域 [-1,1]
        img = hint_tensor[b].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        # 映射到 [0,255] uint8
        img_uint8 = np.clip((img + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

        # 提取前三个通道（RGB对应波段，假设是 G,R,RE）用于色偏
        rgb_uint8 = img_uint8[:, :, :3]  # (H, W, 3)
        nir_uint8 = img_uint8[:, :, 3:4] if C == 4 else None  # 第四通道保持不变

        # 应用高斯噪声（对所有三个可见光通道）
        if noise_level != 'none':
            rgb_uint8 = add_gaussian_noise_uint8(rgb_uint8, noise_level)
            if nir_uint8 is not None:
                nir_uint8 = add_gaussian_noise_uint8(nir_uint8, noise_level)

        # 应用色偏（仅对可见光三通道）
        if color_cast_level != 'none':
            delta_range = get_color_cast_range(color_cast_level)
            rgb_uint8 = apply_color_cast_uint8(rgb_uint8, delta_range)
            # 近红外不施加色偏

        # 合并通道
        if nir_uint8 is not None:
            img_uint8_degraded = np.concatenate([rgb_uint8, nir_uint8], axis=-1)
        else:
            img_uint8_degraded = rgb_uint8

        # 转回 [-1,1] float32
        img_degraded = img_uint8_degraded.astype(np.float32) / 127.5 - 1.0
        degraded_tensor = torch.from_numpy(img_degraded).permute(2, 0, 1)
        degraded.append(degraded_tensor)

    return torch.stack(degraded, dim=0).to(hint_tensor.device, dtype=hint_tensor.dtype)


# ==================== 时间戳提取 ====================
def extract_timestamp(source_filename):
    basename = os.path.basename(source_filename)
    parts = basename.split('_')
    if len(parts) >= 3:
        timestamp = '_'.join(parts[:3])
    else:
        timestamp = basename
    return timestamp


# ==================== 主程序 ====================
if __name__ == '__main__':
    # -------------------- 参数解析 --------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', type=str, default='none',
                        choices=['none', 'slight', 'medium', 'heavy'],
                        help='Apply Gaussian noise degradation to hint image')
    parser.add_argument('--color_cast_level', type=str, default='none',
                        choices=['none', 'slight', 'medium', 'heavy'],
                        help='Apply color cast degradation to hint image')
    parser.add_argument('--dataset', type=str, default='train',
                        help='Which dataset to evaluate: train (default) or test')
    opt = parser.parse_args()

    # -------------------- 路径配置 --------------------
    config_path = './models/cldm_v15.yaml'
    controlnet_ckpt = '/mnt/data/Sureaily/ControlNet-main/lightning_logs/version_29/checkpoints/epoch=59-step=14099.ckpt'
    vae_weights_path = '/mnt/data/Sureaily/ControlNet-main/vae_4ch_epoch10.pth'

    # 数据集选择
    if opt.dataset == 'test':
        dataset = MyDataset('/mnt/data/Sureaily/ControlNet_datasets/prompt_test.json')
    else:
        dataset = MyDataset()  # 默认训练集

    batch_size = 8
    num_workers = 8
    output_dir = "val_outputs"

    target_bands_dir = os.path.join(output_dir, "target_bands")
    generated_bands_dir = os.path.join(output_dir, "generated_bands")
    os.makedirs(target_bands_dir, exist_ok=True)
    os.makedirs(generated_bands_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- 加载模型 ----------
    model = create_model(config_path).cpu()
    ckpt = torch.load(controlnet_ckpt, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    filtered_state = {k: v for k, v in state.items()
                      if not k.startswith('first_stage_model.') and not k.startswith('cond_stage_model.')}
    model.load_state_dict(filtered_state, strict=False)

    vae_ckpt = torch.load(vae_weights_path, map_location='cpu')
    if 'state_dict' in vae_ckpt:
        vae_state = {k.replace('first_stage_model.', ''): v
                     for k, v in vae_ckpt['state_dict'].items()
                     if k.startswith('first_stage_model.')}
    else:
        vae_state = vae_ckpt
    model.first_stage_model.load_state_dict(vae_state, strict=True)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    bands = ['G', 'R', 'RE', 'NIR']
    rmse_total = [0.0] * 4
    mrae_total = [0.0] * 4
    psnr_total = [0.0] * 4
    ssim_total = [0.0] * 4
    count = 0

    loss_rmse = Loss_RMSE()
    loss_mrae = Loss_MRAE_custom()
    loss_psnr = Loss_PSNR()
    loss_ssim = Loss_SSIM(data_range=1.0)

    print(f"Degradation settings: noise={opt.noise_level}, color_cast={opt.color_cast_level}\n")
    print("Evaluating ControlNet with training-aligned protocol...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            # ---------- 相机退化模拟（施加在 hint 上）----------
            if opt.noise_level != 'none' or opt.color_cast_level != 'none':
                # hint 原始形状 (B, H, W, C)，转成 (B, C, H, W) 便于处理
                hint = batch['hint'].permute(0, 3, 1, 2)  # 暂时用通道在前
                degraded_hint = degrade_hint(hint, opt.noise_level, opt.color_cast_level)
                # 转回 (B, H, W, C) 放回 batch（因为 log_images 可能需要这种排列？）
                batch['hint'] = degraded_hint.permute(0, 2, 3, 1)
            else:
                # 保持原始 hint 不变
                pass

            # 将数据移到 GPU
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 生成
            images = model.log_images(batch_gpu, split="val")
            pred = images['reconstruction']
            target = batch_gpu['jpg']

            # 数据后处理
            if target.ndim == 4 and target.shape[-1] == 4:
                target = target.permute(0, 3, 1, 2)
            if pred.ndim == 4 and pred.shape[-1] == 4:
                pred = pred.permute(0, 3, 1, 2)

            if pred.shape[1] == 3 and target.shape[1] == 4:
                target = target[:, :3, :, :]
            elif pred.shape[1] == 4 and target.shape[1] == 3:
                pred = pred[:, :3, :, :]

            if pred.shape[2:] != target.shape[2:]:
                pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

            pred_01 = (pred + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0

            # 指标计算
            for c in range(4):
                true_c = target_01[:, c:c+1, :, :]
                fake_c = pred_01[:, c:c+1, :, :]

                rmse_total[c] += loss_rmse(fake_c, true_c).item() * target.size(0)
                mrae_total[c] += loss_mrae(fake_c, true_c).item() * target.size(0)
                psnr_total[c] += loss_psnr(true_c, fake_c, data_range=255).item() * target.size(0)
                ssim_total[c] += loss_ssim(true_c, fake_c).item() * target.size(0)

            # 图像保存
            for i in range(target.size(0)):
                abs_idx = batch_idx * batch_size + i
                item = dataset.data[abs_idx]
                ts = extract_timestamp(item['source'])

                true_img = target[i].cpu().numpy().transpose(1, 2, 0)
                tifffile.imwrite(os.path.join(output_dir, f"target_{ts}.tif"),
                                 ((true_img + 1) / 2 * 255).astype(np.uint8))
                rgb_true = ((true_img[:, :, :3] + 1) / 2 * 255).astype(np.uint8)
                Image.fromarray(rgb_true).save(os.path.join(output_dir, f"target_rgb_{ts}.png"))

                fake_img = pred[i].cpu().numpy().transpose(1, 2, 0)
                tifffile.imwrite(os.path.join(output_dir, f"generated_{ts}.tif"),
                                 ((fake_img + 1) / 2 * 255).astype(np.uint8))
                rgb_fake = ((fake_img[:, :, :3] + 1) / 2 * 255).astype(np.uint8)
                Image.fromarray(rgb_fake).save(os.path.join(output_dir, f"generated_rgb_{ts}.png"))

                for c, band_name in enumerate(bands):
                    band_true = true_img[:, :, c]
                    band_true_uint8 = np.clip((band_true + 1) / 2 * 255, 0, 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(target_bands_dir, f"target_{band_name}_{ts}.tif"), band_true_uint8)

                    band_fake = fake_img[:, :, c]
                    band_fake_uint8 = np.clip((band_fake + 1) / 2 * 255, 0, 255).astype(np.uint8)
                    tifffile.imwrite(os.path.join(generated_bands_dir, f"generated_{band_name}_{ts}.tif"), band_fake_uint8)

            count += target.size(0)

    # 输出结果
    print(f"\nEvaluated on {count} images (dataset: {'train' if opt.dataset=='train' else 'test'})")
    print(f"Degradation applied: noise={opt.noise_level}, color_cast={opt.color_cast_level}")
    print("Band\tRMSE\t\tMRAE\t\tPSNR(dB)\tSSIM")
    print("-----------------------------------------------------------")
    for i, band in enumerate(bands):
        print(f"{band}\t{rmse_total[i]/count:.6f}\t{mrae_total[i]/count:.6f}\t"
              f"{psnr_total[i]/count:.2f}\t\t{ssim_total[i]/count:.4f}")
    print("-----------------------------------------------------------")
    avg_rmse = np.mean(rmse_total) / count
    avg_mrae = np.mean(mrae_total) / count
    avg_psnr = np.mean(psnr_total) / count
    avg_ssim = np.mean(ssim_total) / count
    print(f"Avg\t{avg_rmse:.6f}\t{avg_mrae:.6f}\t{avg_psnr:.2f}\t\t{avg_ssim:.4f}")