"""只训练VAE网络（加入学习率余弦退火衰减）"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
if torch.cuda.is_available():
    torch.cuda.set_device(0)
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.model import create_model
from torch.optim.lr_scheduler import CosineAnnealingLR   # 新增导入

# ========================= 配置 =========================
resume_path = '/mnt/data/Sureaily/ControlNet-main/vae_4ch_epoch410.pth'
batch_size = 1
learning_rate = 1e-7
total_epochs = 500
save_every = 10

# ========================= 初始化模型 =========================
model = create_model('./models/cldm_v15.yaml').cpu()

print(f"Loading VAE weights from {resume_path} ...")
vae_weights = torch.load(resume_path, map_location='cpu')
model.first_stage_model.load_state_dict(vae_weights, strict=True)
print("Loaded VAE weights successfully.")

# ========================= 冻结 / 解冻 =========================
for p in model.first_stage_model.parameters():
    p.requires_grad = True

model.model.diffusion_model.requires_grad_(False)
model.control_model.requires_grad_(False)

model.cuda()
optimizer = torch.optim.AdamW(model.first_stage_model.parameters(), lr=learning_rate)

# ========================= 学习率调度器（余弦退火） =========================
scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-10)

# ========================= 数据 =========================
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

# ========================= 训练循环 =========================
print("Start training VAE...")
for epoch in range(1, total_epochs + 1):
    total_loss = 0.0
    for batch in dataloader:
        x = batch['jpg'].cuda()                     # (B, H, W, 4)
        x = x.permute(0, 3, 1, 2).contiguous()     # (B, 4, H, W)

        # 1. 编码，得到分布参数
        posterior = model.first_stage_model.encode(x)
        # 2. 手动重参数化采样
        mean = posterior.mean
        logvar = posterior.logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps
        # 3. 解码
        x_recon = model.first_stage_model.decode(z)

        loss = torch.nn.functional.mse_loss(x_recon, x)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.first_stage_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(dataset)
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d}/{total_epochs} | MSE: {avg_loss:.6f} | LR: {current_lr:.2e}")

    # 在每个epoch结束时更新学习率
    scheduler.step()

    if epoch % save_every == 0:
        save_path = f"vae_4ch_epoch{epoch}.pth"
        torch.save(model.first_stage_model.state_dict(), save_path)
        print(f"   Saved VAE to {save_path}")

print("Training finished.")