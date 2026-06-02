import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import tifffile

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    # ----- 从 batch 提取时间戳（DJI_20251107113933_0130 形式）-----
    def _extract_timestamp(self, source_path):
        """从完整路径或文件名中提取 DJI 时间戳"""
        base = os.path.basename(source_path)
        parts = base.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:3])
        else:
            return base

    def _get_sample_timestamps(self, batch):
        """返回一个列表，每个元素是该 batch 中每张图的时间戳字符串"""
        if 'source' not in batch:
            return ['unknown'] * len(batch['jpg'])
        return [self._extract_timestamp(p) for p in batch['source']]

    # ----- 自适应拉伸到 [0,255] -----
    def _stretch_to_uint8(self, image):
        """image: numpy array (H,W) 或 (H,W,C)，返回 uint8 数组，单通道会被保留为 2D。"""
        img = image.copy()
        if img.ndim == 2:
            # 单通道直接拉伸
            vmin, vmax = img.min(), img.max()
            if vmax - vmin > 1e-8:
                img = (img - vmin) / (vmax - vmin)
            else:
                img = img - vmin
            return np.clip(img * 255, 0, 255).astype(np.uint8)
        else:
            # 多通道，逐通道处理
            out = np.zeros_like(img, dtype=np.uint8)
            for c in range(img.shape[-1]):
                ch = img[:, :, c]
                vmin, vmax = ch.min(), ch.max()
                if vmax - vmin > 1e-8:
                    ch = (ch - vmin) / (vmax - vmin)
                else:
                    ch = ch - vmin
                out[:, :, c] = np.clip(ch * 255, 0, 255).astype(np.uint8)
            return out

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx,
                  timestamps=None):
        root = os.path.join(save_dir, "image_log", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            # 假设 images[k] 是已堆叠的网格（或单张），但原代码中 make_grid 前会有 batch 维度
            # 我们直接使用 images[k] 作为原始 batch 图像（已限制数量）
            # 注意：这里 images[k] 是 B x C x H x W 或 B x H x W 的张量列表（原代码已 detach 并 clamp）
            # 但由于在 log_img 中已经做了 make_grid，所以这里已经是网格图，不再是逐样本的。
            # 为了能够按样本命名时间戳，我们需要在 log_img 中直接保存每个样本，而不是先 make_grid。
            # 我们需要调整 log_img 的逻辑，不再使用 make_grid，而是逐样本保存。
            pass

    # 重新设计 log_img，逐样本保存而不使用 make_grid
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            # 提取时间戳
            timestamps = self._get_sample_timestamps(batch)  # 列表，长度 = batch size

            # 处理每种图像 (reconstruction, samples, conditioning 等)
            for k, tensor_list in images.items():
                N = min(tensor_list.shape[0], self.max_images)
                for i in range(N):
                    # 获取单个样本，形状 (C, H, W) 或 (H, W)
                    img_tensor = tensor_list[i].detach().cpu()
                    if self.clamp:
                        img_tensor = torch.clamp(img_tensor, -1., 1.)

                    # 转换为 numpy
                    if img_tensor.ndim == 3:
                        # (C, H, W) -> (H, W, C)
                        img_np = img_tensor.permute(1, 2, 0).numpy()
                    else:
                        img_np = img_tensor.numpy()  # (H, W)

                    # 若需要 rescale （从 [-1,1] 到 [0,1]）
                    if self.rescale:
                        img_np = (img_np + 1.0) / 2.0

                    # 拉伸到 uint8
                    img_uint8 = self._stretch_to_uint8(img_np)

                    # 获取时间戳
                    ts = timestamps[i] if i < len(timestamps) else f"{global_step:06d}_{batch_idx:06d}_{i:02d}"

                    # 构建文件名：{timestamp}_{k}.tif
                    root = os.path.join(pl_module.logger.save_dir, "image_log", split)
                    os.makedirs(root, exist_ok=True)

                    if img_uint8.ndim == 2:
                        filename = f"{ts}_{k}.tif"
                        tifffile.imwrite(os.path.join(root, filename), img_uint8)
                    elif img_uint8.shape[-1] == 4:
                        bands = ['G', 'R', 'RE', 'NIR']
                        for idx, band in enumerate(bands):
                            ch_data = img_uint8[:, :, idx]
                            filename = f"{ts}_{k}_{band}.tif"
                            tifffile.imwrite(os.path.join(root, filename), ch_data)
                    elif img_uint8.shape[-1] == 3:
                        filename = f"{ts}_{k}.tif"
                        tifffile.imwrite(os.path.join(root, filename), img_uint8)
                    else:
                        filename = f"{ts}_{k}.tif"
                        tifffile.imwrite(os.path.join(root, filename), img_uint8)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")