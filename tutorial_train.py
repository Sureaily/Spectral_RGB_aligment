# """Controlnet模型训练（验证集 RMSE/MRAE/PSNR/SSIM + 余弦退火学习率）"""
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from cldm.logger import ImageLogger
# from cldm.model import create_model
# import resource
#
# resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
#
#
# # ===================== 指标计算类 =====================
# class Loss_RMSE(nn.Module):
#     def forward(self, outputs, label):
#         error = outputs - label
#         return torch.sqrt(torch.mean(error.pow(2)))
#
# class Loss_MRAE_custom(nn.Module):
#     def forward(self, outputs, label):
#         mask = label == 0
#         label_safe = label.clone()
#         if mask.any():
#             label_safe[mask] = 1e-5
#         return torch.mean(torch.abs(outputs - label) / label_safe)
#
# class Loss_PSNR(nn.Module):
#     def forward(self, im_true_01, im_fake_01, data_range=255):
#         N, C, H, W = im_true_01.shape
#         true = im_true_01.clamp(0, 1).mul(data_range).reshape(N, C * H * W)
#         fake = im_fake_01.clamp(0, 1).mul(data_range).reshape(N, C * H * W)
#         mse = nn.MSELoss(reduction='none')(true, fake).sum(dim=1) / (C * H * W)
#         psnr = 10 * torch.log10(data_range ** 2 / mse)
#         return psnr.mean()
#
# class Loss_SSIM(nn.Module):
#     def __init__(self, window_size=11, sigma=1.5, data_range=1.0):
#         super().__init__()
#         self.window_size = window_size
#         self.sigma = sigma
#         self.data_range = data_range
#
#     def _gaussian_window(self, C, device):
#         coords = torch.arange(self.window_size, dtype=torch.float32, device=device) - self.window_size // 2
#         g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
#         g /= g.sum()
#         g = g.outer(g).reshape(1, 1, self.window_size, self.window_size)
#         return g.expand(C, 1, self.window_size, self.window_size)
#
#     def forward(self, im_true_01, im_fake_01):
#         C = im_true_01.shape[1]
#         window = self._gaussian_window(C, im_true_01.device)
#
#         mu1 = F.conv2d(im_true_01, window, groups=C, padding=self.window_size // 2)
#         mu2 = F.conv2d(im_fake_01, window, groups=C, padding=self.window_size // 2)
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu12 = mu1 * mu2
#         sigma1_sq = F.conv2d(im_true_01 * im_true_01, window, groups=C, padding=self.window_size // 2) - mu1_sq
#         sigma2_sq = F.conv2d(im_fake_01 * im_fake_01, window, groups=C, padding=self.window_size // 2) - mu2_sq
#         sigma12 = F.conv2d(im_true_01 * im_fake_01, window, groups=C, padding=self.window_size // 2) - mu12
#
#         C1 = (0.01 * self.data_range) ** 2
#         C2 = (0.03 * self.data_range) ** 2
#         ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
#                    ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#         return ssim_map.mean()
#
#
# if __name__ == '__main__':
#     # ================= 配置 =================
#     resume_path = '/mnt/data/Sureaily/ControlNet-main/lightning_logs/version_45/checkpoints/epoch=500-step=4508.ckpt'
#     vae_weights_path = '/mnt/data/Sureaily/ControlNet-main/vae_4ch_epoch410.pth'
#     batch_size = 1
#     logger_freq = 600
#     learning_rate = 1e-5
#     sd_locked = True
#     devices = 2
#     max_epochs = 500
#
#     # ================= 创建模型 =================
#     model = create_model('./models/cldm_v15.yaml').cpu()
#
#     checkpoint = torch.load(resume_path, map_location='cpu')
#     if 'state_dict' in checkpoint:
#         checkpoint = checkpoint['state_dict']
#     model_dict = model.state_dict()
#     filtered = {}
#     for k, v in checkpoint.items():
#         if k.startswith('cond_stage_model.') or k.startswith('first_stage_model.'):
#             continue
#         if k in model_dict and v.shape == model_dict[k].shape:
#             filtered[k] = v
#     model.load_state_dict(filtered, strict=False)
#
#     vae_state = torch.load(vae_weights_path, map_location='cpu')
#     model.first_stage_model.load_state_dict(vae_state, strict=True)
#
#     model.sd_locked = sd_locked
#     for p in model.first_stage_model.parameters():
#         p.requires_grad = False
#
#     model.learning_rate = learning_rate
#
#     # ================= 数据集 =================
#     train_dataset = MyDataset('/mnt/data/Sureaily/ControlNet_datasets/prompt.json')
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               num_workers=4, pin_memory=True)
#
#     val_dataset = MyDataset('/mnt/data/Sureaily/ControlNet_datasets/prompt_test.json')
#     print(f"[INFO] Validation dataset size: {len(val_dataset)}")
#     if len(val_dataset) == 0:
#         raise RuntimeError("Validation dataset is empty! Check prompt_test.json and image paths.")
#
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
#                             num_workers=2, pin_memory=True)
#
#     # 快速测试能否取出一批数据
#     try:
#         test_batch = next(iter(val_loader))
#         print(f"[INFO] Validation batch loaded successfully. Keys: {test_batch.keys()}, "
#               f"target shape: {test_batch['jpg'].shape}")
#     except Exception as e:
#         print(f"[ERROR] Failed to load a validation batch: {e}")
#         raise
#
#     # ================= 动态注入验证方法 =================
#     def val_dataloader(self):
#         return val_loader
#     model.val_dataloader = val_dataloader.__get__(model, type(model))
#
#     loss_rmse = Loss_RMSE()
#     loss_mrae = Loss_MRAE_custom()
#     loss_psnr = Loss_PSNR()
#     loss_ssim = Loss_SSIM()
#
#     def validation_step(self, batch, batch_idx):
#         # 强制打印（确保步骤被执行）
#         if batch_idx == 0:
#             print(">>> [VALIDATION STEP] called")
#
#         with torch.no_grad():
#             images = self.log_images(batch, split="val")
#         pred = images['reconstruction']
#         target = batch['jpg']
#
#         # 形状调试打印（仅在首个 batch）
#         if batch_idx == 0:
#             print(f"   pred shape: {pred.shape}, target shape: {target.shape}")
#
#         # 统一到 (B, C, H, W)
#         if target.ndim == 4 and target.shape[-1] == 4:
#             target = target.permute(0, 3, 1, 2)
#         if pred.ndim == 4 and pred.shape[-1] == 4:
#             pred = pred.permute(0, 3, 1, 2)
#
#         # 通道适配（若 pred 3 通道，target 4 通道，仅取前3比较）
#         if pred.shape[1] == 3 and target.shape[1] == 4:
#             target = target[:, :3, :, :]
#             print(f"   [ADAPT] Using only first 3 channels of target. New target shape: {target.shape}")
#         elif pred.shape[1] == 4 and target.shape[1] == 3:
#             pred = pred[:, :3, :, :]
#             print(f"   [ADAPT] Using only first 3 channels of pred. New pred shape: {pred.shape}")
#
#         # 空间尺寸适配
#         if pred.shape[2:] != target.shape[2:]:
#             pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)
#             print(f"   [RESIZE] pred resized to {pred.shape}")
#
#         # 归一化到 [0,1]
#         pred_01 = (pred + 1.0) / 2.0
#         target_01 = (target + 1.0) / 2.0
#
#         # 计算指标
#         rmse_val = loss_rmse(pred_01, target_01)
#         mrae_val = loss_mrae(pred_01, target_01)
#         psnr_val = loss_psnr(target_01, pred_01)
#         ssim_val = loss_ssim(target_01, pred_01)
#
#         self.log('val/rmse', rmse_val, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
#         self.log('val/mrae', mrae_val, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
#         self.log('val/psnr', psnr_val, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
#         self.log('val/ssim', ssim_val, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
#
#         if batch_idx == 0:
#             print(f"   batch 0 metrics: RMSE={rmse_val.item():.4f}, MRAE={mrae_val.item():.4f}, "
#                   f"PSNR={psnr_val.item():.2f}, SSIM={ssim_val.item():.4f}")
#
#         return {'pred_01': pred_01, 'target_01': target_01}
#     model.validation_step = validation_step.__get__(model, type(model))
#
#     def validation_epoch_end(self, outputs):
#         metrics = self.trainer.callback_metrics
#         rmse = metrics.get('val/rmse')
#         mrae = metrics.get('val/mrae')
#         psnr = metrics.get('val/psnr')
#         ssim = metrics.get('val/ssim')
#         # 强制打印，不论是否 global_zero
#         if rmse is not None:
#             print(f"========== [EPOCH {self.current_epoch}] Validation Summary ==========")
#             print(f"RMSE: {rmse:.6f} | MRAE: {mrae:.6f} | PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f}")
#     model.validation_epoch_end = validation_epoch_end.__get__(model, type(model))
#
#     # ================= 余弦退火学习率 =================
#     def configure_optimizers(self):
#         params = list(self.control_model.parameters())
#         if not self.sd_locked:
#             params += list(self.model.diffusion_model.parameters())
#         opt = torch.optim.AdamW(params, lr=self.learning_rate)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
#         return {
#             'optimizer': opt,
#             'lr_scheduler': {
#                 'scheduler': scheduler,
#                 'interval': 'epoch',
#                 'frequency': 1,
#             }
#         }
#     model.configure_optimizers = configure_optimizers.__get__(model, type(model))
#
#     # ================= Trainer =================
#     logger = ImageLogger(batch_frequency=logger_freq)
#     trainer = pl.Trainer(
#         accelerator='gpu',
#         devices=devices,
#         strategy='ddp',
#         precision=16,
#         callbacks=[logger],
#         accumulate_grad_batches=4,
#         gradient_clip_val=1.0,
#         max_epochs=max_epochs,
#         check_val_every_n_epoch=1,
#         num_sanity_val_steps=1,   # 确保至少尝试运行一次验证
#     )
#
#     torch.cuda.empty_cache()
#     trainer.fit(model, train_loader)


"""Controlnet模型训练"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 根据需要取消注释并指定 GPU
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

# 所有顶层代码只保留导入和函数/类定义，不要执行任何多进程相关操作

if __name__ == '__main__':
    # ================= 配置 =================
    resume_path = '/mnt/data/Sureaily/ControlNet-main/lightning_logs/version_45/checkpoints/epoch=500-step=4508.ckpt'
    vae_weights_path = '/mnt/data/Sureaily/ControlNet-main/vae_4ch_epoch10.pth'
    batch_size = 1
    logger_freq = 600
    learning_rate = 1e-5
    sd_locked = True
    devices = 3          # 建议先用单卡，若要多卡改为实际数量并确保显存足够
    max_epochs = 500
    # ================= 创建模型 =================
    model = create_model('./models/cldm_v15.yaml').cpu()

    # 加载训练过的 ControlNet 检查点（跳过 VAE 和文本编码器）
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_dict = model.state_dict()
    filtered = {}
    for k, v in checkpoint.items():
        if k.startswith('cond_stage_model.') or k.startswith('first_stage_model.'):
            continue
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)

    # 加载单独训练的 VAE
    vae_state = torch.load(vae_weights_path, map_location='cpu')
    model.first_stage_model.load_state_dict(vae_state, strict=True)

    # ================= 冻结 / 解冻 =================
    model.sd_locked = sd_locked
    # 显式冻结 VAE，确保不参与训练
    for p in model.first_stage_model.parameters():
        p.requires_grad = False

    # 打印可训练参数（应只有 control_model 的参数）
    print("Trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    # ================= 设置训练参数 =================
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    # ================= 数据 =================
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ================= Trainer =================
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=3,
        strategy='ddp',
        precision=16,
        callbacks=[logger],
        accumulate_grad_batches=4,
        max_epochs=max_epochs,
        gradient_clip_val=1.0,
    )

    torch.cuda.empty_cache()
    trainer.fit(model, dataloader)