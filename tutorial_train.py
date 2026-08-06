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
    max_epochs = 300
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
