import numpy as np
from tutorial_dataset import MyDataset

dataset = MyDataset()
print(f"数据集总长度: {len(dataset)}")

item = dataset[89]
jpg = item['jpg']     # target 四通道，形状 (512,512,4)
txt = item['txt']
hint = item['hint']   # source 三通道，形状 (512,512,3)

print(f"文本提示: {txt}")
print(f"target 形状: {jpg.shape}  (应为 (512,512,4))")
print(f"hint 形状:   {hint.shape}  (应为 (512,512,3))")

# ---------- 检查四通道各波段数值范围 ----------
bands_name = ['G', 'R', 'RE', 'NIR']  # 预设顺序
for i, band in enumerate(bands_name):
    ch = jpg[:, :, i]
    print(f"波段 {band} (通道{i}):  min={ch.min():.3f}, max={ch.max():.3f}, mean={ch.mean():.3f}")

# ---------- 保存四个单波段灰度图（便于肉眼确认） ----------
import tifffile
for i, band in enumerate(bands_name):
    # 由于 target 归一化到 [-1,1]，需反向映射到 [0,255] 再转为 uint8
    ch_uint8 = np.clip((jpg[:, :, i] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    tifffile.imwrite(f"debug_target_band_{band}.tif", ch_uint8)
    print(f"已保存波段 {band} 至 debug_target_band_{band}.tif")

# 同时保存 hint 的 RGB 图（方便对比）
import cv2
hint_uint8 = np.clip(hint * 255, 0, 255).astype(np.uint8)
cv2.imwrite("debug_hint.jpg", cv2.cvtColor(hint_uint8, cv2.COLOR_RGB2BGR))
print("已保存 hint 预览至 debug_hint.jpg")