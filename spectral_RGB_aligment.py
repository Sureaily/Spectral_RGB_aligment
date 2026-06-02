import torch
import numpy as np
from PIL import Image
import cv2
import os
import json
import tifffile
import argparse
from collections import defaultdict
from scipy.spatial import KDTree
from scipy import ndimage
from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
from lightglue.utils import rbd
import warnings
warnings.filterwarnings('ignore')

# ==================== 使用方法 ====================
"""
用法示例：
python script.py \
    --jsonl_path /path/to/prompt.json \
    --source_root /path/to/rgb_images \
    --target_root /path/to/spectral_images \
    --output_dir /path/to/output \
    --min_inliers 2000 \
    --grid_size 20 \
    --sigma_factor 0.1 \
    --max_keypoints 7000 \
    --max_void_ratio 0.1 \
    --min_fill_ratio 0.6 \
    --features aliked disk superpoint sift

说明：
- 成功配准的图像会裁剪并保存至 output_dir
- 无法配准（内点不足/分布不均/公共区域为空等）的 RGB 图像会保存至 output_dir/failed_images/ 目录下，
  文件名格式为 {原图名}_failed.jpg，并在图像上绘制已检测到的内点（红色圆点，带白色外轮廓）。
"""
# =================================================

# -------------------- 参数解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RGB与光谱图像配准裁剪工具")
    parser.add_argument('--jsonl_path', type=str,
                        default='/root/autodl-tmp/pytorch-CycleGAN-and-pix2pix-master/datasets/controlnet-dataset/prompt.json',
                        help='JSONL文件路径，包含source-target映射')
    parser.add_argument('--source_root', type=str,
                        default='/root/autodl-tmp/pytorch-CycleGAN-and-pix2pix-master/datasets/controlnet-dataset/',
                        help='RGB图像根目录')
    parser.add_argument('--target_root', type=str,
                        default='/root/autodl-tmp/pytorch-CycleGAN-and-pix2pix-master/datasets/controlnet-dataset/',
                        help='光谱图像根目录')
    parser.add_argument('--output_dir', type=str,
                        default='/root/autodl-tmp/LightGlue-main/Dataset',
                        help='裁剪结果输出目录')
    parser.add_argument('--min_inliers', type=int, default=1000,
                        help='最小内点数量，低于此值跳过该组')
    parser.add_argument('--grid_size', type=int, default=20,
                        help='局部单应性网格划分数量')
    parser.add_argument('--sigma_factor', type=float, default=1.5,
                        help='高斯权重的sigma因子（相对于图像对角线）')
    parser.add_argument('--max_keypoints', type=int, default=10000,
                        help='每个特征提取器最大关键点数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备 (cuda/cpu)')
    # 均匀度判别参数（填充率）
    parser.add_argument('--max_void_ratio', type=float, default=0.1,
                        help='最大允许的无内点连通域面积占比（面积/总网格数），超过则考虑拒绝')
    parser.add_argument('--min_fill_ratio', type=float, default=0.6,
                        help='最小允许的填充率（面积/外接矩形面积），高于此值且面积占比超限则拒绝（紧凑区域）')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='跳过已存在输出文件的组')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                        help='不跳过已存在输出，强制重新处理')
    parser.add_argument('--features', type=str, nargs='+',
                        default=['aliked', 'disk', 'superpoint', 'sift'],
                        help='使用的特征类型列表')
    parser.add_argument('--reverse', action='store_true',
                        help='是否倒序处理组（用于双服务器并行）')
    return parser.parse_args()

args = parse_args()

# 设置设备
DEVICE = torch.device(args.device)
JSONL_PATH = args.jsonl_path
SOURCE_ROOT = args.source_root
TARGET_ROOT = args.target_root
OUTPUT_DIR = args.output_dir
MIN_INLIERS = args.min_inliers
GRID_SIZE = args.grid_size
SIGMA_FACTOR = args.sigma_factor
MAX_KEYPOINTS = args.max_keypoints
MAX_VOID_RATIO = args.max_void_ratio
MIN_FILL_RATIO = args.min_fill_ratio
SKIP_EXISTING = args.skip_existing

# -------------------- 初始化特征提取器和匹配器 --------------------
AVAILABLE_FEATURES = {
    'aliked': (ALIKED, {'max_num_keypoints': MAX_KEYPOINTS}),
    'disk': (DISK, {'max_num_keypoints': MAX_KEYPOINTS}),
    'superpoint': (SuperPoint, {'max_num_keypoints': MAX_KEYPOINTS, 'threshold': 0.005}),
    'sift': (SIFT, {'max_num_keypoints': MAX_KEYPOINTS, 'backend': 'opencv'}),
}

extractors = {}
matchers = {}

for feat in args.features:
    if feat not in AVAILABLE_FEATURES:
        print(f"警告：特征 {feat} 不可用，跳过")
        continue
    cls, kwargs = AVAILABLE_FEATURES[feat]
    extractors[feat] = cls(**kwargs).eval().to(DEVICE)
    matchers[feat] = LightGlue(features=feat,  depth_confidence=-1, width_confidence=-1).eval().to(DEVICE)

if not extractors:
    raise RuntimeError("未初始化任何特征提取器，请检查 --features 参数")

print(f"已初始化特征器: {list(extractors.keys())}")
print(f"使用设备: {DEVICE}")

# -------------------- 辅助函数：绘制特征点 --------------------
def draw_keypoints(img, keypoints, radius=5, color=(0, 0, 255), thickness=-1, outline=True):
    """
    在图像上绘制特征点（带白色外轮廓）
    参数:
        img: numpy数组 (H,W,3) BGR顺序
        keypoints: 点列表 [(x,y), ...]
        radius: 圆点半径
        color: 内部颜色 (B,G,R)，默认红色
        thickness: -1表示填充
        outline: 是否添加白色外轮廓
    """
    img_copy = img.copy()
    for (x, y) in keypoints:
        pt = (int(x), int(y))
        if outline:
            cv2.circle(img_copy, pt, radius + 1, (255, 255, 255), thickness + 1)
        cv2.circle(img_copy, pt, radius, color, thickness)
    return img_copy

# -------------------- 图像加载函数 --------------------
def load_rgb_image(path, resize=None):
    img = Image.open(path).convert('RGB')
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    img = np.array(img) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float()

def load_tiff_robust(path, resize=None, return_raw=False):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    raw = arr.astype(np.uint16)          # 保留原始 uint16
    # 归一化到 [0,1]（用于显示/匹配）
    if arr.dtype == np.uint16:
        arr_norm = arr / 65535.0
    elif arr.dtype == np.uint8:
        arr_norm = arr / 255.0
    else:
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    if resize:
        # 对 uint8 可视化图进行缩放，原始 raw 按同样比例缩放（用最近邻或双线性）
        raw = cv2.resize(raw, resize, interpolation=cv2.INTER_NEAREST)
        arr_norm = cv2.resize((arr_norm * 255).astype(np.uint8), resize) / 255.0
    else:
        arr_norm = arr_norm * 255.0
    # 返回归一化后的 tensor（用于匹配）和原始 raw 数据
    arr_rgb = np.stack([arr_norm, arr_norm, arr_norm], axis=-1)
    tensor_img = torch.from_numpy(arr_rgb).permute(2, 0, 1).float()
    return tensor_img, raw

# -------------------- 计算匹配内点（多特征融合+RANSAC） --------------------
def compute_inliers(ms_img, rgb_img):
    """返回内点匹配对 (pts_src, pts_dst) 和内点数量"""
    all_pts_src = []
    all_pts_dst = []
    for name in extractors:
        extractor = extractors[name]
        matcher = matchers[name]
        feats_dst = extractor.extract(rgb_img)
        feats_src = extractor.extract(ms_img)
        if len(feats_dst['keypoints']) == 0 or len(feats_src['keypoints']) == 0:
            continue
        matches = matcher({'image0': feats_dst, 'image1': feats_src})
        feats_dst, feats_src, matches = [rbd(x) for x in [feats_dst, feats_src, matches]]
        matches = matches['matches']
        pts_dst = feats_dst['keypoints'][matches[..., 0]].cpu().numpy()
        pts_src = feats_src['keypoints'][matches[..., 1]].cpu().numpy()
        if len(pts_dst) > 0:
            all_pts_dst.append(pts_dst)
            all_pts_src.append(pts_src)
    if not all_pts_dst:
        return None, None, 0
    merged_pts_dst = np.vstack(all_pts_dst)
    merged_pts_src = np.vstack(all_pts_src)
    if len(merged_pts_dst) < 4:
        return None, None, 0
    # 使用RANSAC得到内点
    H, mask = cv2.findHomography(merged_pts_src, merged_pts_dst, cv2.RANSAC, 3)
    if H is None or mask is None:
        return None, None, 0
    mask = mask.ravel().astype(bool)
    pts_dst_inliers = merged_pts_dst[mask]
    pts_src_inliers = merged_pts_src[mask]
    inliers = np.sum(mask)
    return pts_src_inliers, pts_dst_inliers, inliers

# -------------------- 局部单应性变换相关函数 --------------------
def weighted_homography(pts_src, pts_dst, weights):
    """加权直接线性变换估计单应性矩阵 H (3x3)"""
    assert len(pts_src) == len(pts_dst) == len(weights)
    n = len(pts_src)
    A = []
    for i in range(n):
        x, y = pts_src[i]
        u, v = pts_dst[i]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A) * np.repeat(weights, 2)[:, np.newaxis]
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H / H[2, 2]

def local_homography_warp(img_src, pts_src, pts_dst, img_shape_dst, grid_size=8, sigma=None):
    """
    使用局部单应性变换将源图像扭曲到目标图像空间。
    每个网格的邻域半径基于网格对角线自适应计算，内点不足时自动扩大范围。
    
    参数:
        img_src: 源图像 (H_src, W_src, 3) numpy数组，值域[0,255]
        pts_src: 源图像上的匹配内点 (N,2)
        pts_dst: 目标图像上的匹配内点 (N,2)
        img_shape_dst: 目标图像尺寸 (H_dst, W_dst)
        grid_size: 网格划分数，默认8
        sigma: 半径比例因子，若为None则使用全局SIGMA_FACTOR。最终半径 = diag * (1 + sigma)
    返回:
        warped: 扭曲后的图像 (H_dst, W_dst, 3)
        mask: 有效像素掩码 (H_dst, W_dst)
    """
    h_dst, w_dst = img_shape_dst
    h_src, w_src = img_src.shape[:2]

    # 确定sigma因子
    if sigma is None:
        sigma = SIGMA_FACTOR

    # 预计算全局单应矩阵（用于回退）
    H_global = None
    if len(pts_src) >= 4:
        H_global, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3)

    # 生成网格边界
    grid_y = np.linspace(0, h_dst, grid_size + 1).astype(np.int32)
    grid_x = np.linspace(0, w_dst, grid_size + 1).astype(np.int32)

    # 存储每个网格单元的单应性矩阵
    H_cells = []
    # 构建KDTree加速邻域搜索
    tree = KDTree(pts_dst)

    # 最大尝试次数和半径扩大倍数
    max_attempts = 3
    expansion_factor = 2.0

    # 遍历每个网格单元
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = grid_y[i], grid_y[i+1]
            x_start, x_end = grid_x[j], grid_x[j+1]
            # 网格中心点
            cy = (y_start + y_end) // 2
            cx = (x_start + x_end) // 2
            center = np.array([cx, cy], dtype=np.float32)

            # 计算网格的对角线长度
            cell_w = x_end - x_start
            cell_h = y_end - y_start
            diag = np.sqrt(cell_w**2 + cell_h**2)
            base_radius = diag * (1 + sigma)   # 初始半径

            # 动态扩大半径直到内点数 >= 4 或达到最大尝试次数
            indices = []
            radius = base_radius
            for attempt in range(max_attempts):
                indices = tree.query_ball_point(center, r=radius)
                if len(indices) >= 4:
                    break
                radius *= expansion_factor
            else:
                # 尝试多次后仍不足4个内点，使用全局单应（如果可用）
                if H_global is not None:
                    H_cells.append(H_global)
                else:
                    H_cells.append(None)
                continue

            # 获取邻域内的点
            pts_dst_local = pts_dst[indices]
            pts_src_local = pts_src[indices]
            # 计算高斯权重，标准差 = radius / 3 （使得距离为radius时权重衰减到约0.011）
            dists = np.linalg.norm(pts_dst_local - center, axis=1)
            sigma_gauss = radius / 3.0
            weights = np.exp(- (dists**2) / (2 * sigma_gauss**2))
            # 加权估计单应矩阵
            H = weighted_homography(pts_src_local, pts_dst_local, weights)
            H_cells.append(H)

    # 创建映射矩阵和掩码
    map_x = np.zeros((h_dst, w_dst), dtype=np.float32)
    map_y = np.zeros((h_dst, w_dst), dtype=np.float32)
    mask = np.zeros((h_dst, w_dst), dtype=np.uint8)

    # 逐网格应用单应性逆映射
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = grid_y[i], grid_y[i+1]
            x_start, x_end = grid_x[j], grid_x[j+1]
            cell_idx = i * grid_size + j
            H = H_cells[cell_idx]
            if H is None:
                continue

            # 生成网格内所有像素坐标
            yv, xv = np.meshgrid(np.arange(y_start, y_end), np.arange(x_start, x_end), indexing='ij')
            pts_dst_grid = np.stack([xv.ravel(), yv.ravel()], axis=-1).astype(np.float32)
            # 逆映射到源图像坐标
            H_inv = np.linalg.inv(H)
            ones = np.ones((len(pts_dst_grid), 1), dtype=np.float32)
            pts_hom = np.hstack([pts_dst_grid, ones])
            pts_src_hom = (H_inv @ pts_hom.T).T
            pts_src_grid = pts_src_hom[:, :2] / pts_src_hom[:, 2:3]

            # 有效性检查
            valid_x = (pts_src_grid[:, 0] >= 0) & (pts_src_grid[:, 0] < w_src - 1)
            valid_y = (pts_src_grid[:, 1] >= 0) & (pts_src_grid[:, 1] < h_src - 1)
            valid = valid_x & valid_y

            # 填充映射表和掩码
            map_x[y_start:y_end, x_start:x_end] = pts_src_grid[:, 0].reshape(y_end-y_start, x_end-x_start)
            map_y[y_start:y_end, x_start:x_end] = pts_src_grid[:, 1].reshape(y_end-y_start, x_end-x_start)
            mask[y_start:y_end, x_start:x_end] = valid.reshape(y_end-y_start, x_end-x_start).astype(np.uint8)

    # 图像扭曲
    warped = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, mask

# -------------------- 最大内接矩形函数 --------------------
def largest_inner_rectangle(mask):
    h, w = mask.shape
    height = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                height[i, j] = height[i-1, j] + 1 if i > 0 else 1
            else:
                height[i, j] = 0
    max_area = 0
    best_rect = (0, 0, 0, 0)
    for i in range(h):
        stack = []
        for j in range(w):
            cur_height = height[i, j]
            if not stack or cur_height > stack[-1][1]:
                stack.append((j, cur_height))
            else:
                last_j = j
                while stack and stack[-1][1] >= cur_height:
                    last_j, hh = stack.pop()
                    area = hh * (j - last_j)
                    if area > max_area:
                        max_area = area
                        best_rect = (last_j, i - hh + 1, j - last_j, hh)
                stack.append((last_j, cur_height))
        while stack:
            last_j, hh = stack.pop()
            area = hh * (w - last_j)
            if area > max_area:
                max_area = area
                best_rect = (last_j, i - hh + 1, w - last_j, hh)
    return best_rect

# -------------------- 均匀度判别函数（使用填充率） --------------------
def check_large_compact_void(points, image_shape, grid_size, max_void_ratio, min_fill_ratio):
    """
    检测是否存在大面积且紧凑的无内点连通区域。
    仅考虑面积最大的无内点连通域，如果其面积占比 > max_void_ratio 且
    填充率（面积/外接矩形面积） > min_fill_ratio，则返回不均匀（False）。
    返回 (is_uniform, area_ratio, fill_ratio)
    """
    h, w = image_shape
    grid_h = h / grid_size
    grid_w = w / grid_size

    # 标记无内点网格（1表示无点，0表示有点）
    grid_map = np.ones((grid_size, grid_size), dtype=np.uint8)
    for (x, y) in points:
        col = min(int(x // grid_w), grid_size - 1)
        row = min(int(y // grid_h), grid_size - 1)
        grid_map[row, col] = 0

    # 八连通标记
    labeled, num_features = ndimage.label(grid_map, structure=np.ones((3,3)))
    if num_features == 0:
        return True, 0, 0

    # 找出面积最大的连通域
    max_area = 0
    max_mask = None
    for label_id in range(1, num_features+1):
        mask = (labeled == label_id)
        area = np.sum(mask)
        if area > max_area:
            max_area = area
            max_mask = mask

    if max_mask is None:
        return True, 0, 0

    total_cells = grid_size * grid_size
    area_ratio = max_area / total_cells

    # 如果面积占比未超过阈值，直接返回均匀
    if area_ratio <= max_void_ratio:
        return True, area_ratio, 0

    # 计算外接矩形
    rows, cols = np.where(max_mask)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    bounding_area = width * height
    if bounding_area == 0:
        return True, area_ratio, 0

    fill_ratio = max_area / bounding_area  # 填充率
    if fill_ratio > min_fill_ratio:
        return False, area_ratio, fill_ratio
    else:
        return True, area_ratio, fill_ratio

# -------------------- 主流程 --------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    failed_dir = os.path.join(OUTPUT_DIR, 'failed_images')
    os.makedirs(failed_dir, exist_ok=True)

    # 读取JSONL文件并按source分组
    print("读取映射文件...")
    groups = defaultdict(list)
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            src = item['source']
            tgt = item['target']
            groups[src].append(tgt)

    # 转换为列表并可选倒序
    items = list(groups.items())
    if args.reverse:
        items.reverse()
        print("已启用倒序处理")
    total = len(items)
    print(f"共找到 {total} 组RGB图像")

    success = 0
    skip_no_match = 0
    skip_others = 0
    skip_existing = 0

    BAND_NAMES = ['G', 'R', 'NIR', 'RE']

    for idx, (src_rel, tgt_list) in enumerate(items, 1):
        print(f"\n===== 处理第 {idx}/{total} 组: {src_rel} =====")

        rgb_path = os.path.join(SOURCE_ROOT, src_rel)
        if not os.path.exists(rgb_path):
            print(f"  RGB图像不存在: {rgb_path}，跳过")
            skip_others += 1
            continue

        if len(tgt_list) != 4:
            print(f"  光谱图像数量为 {len(tgt_list)}，预期4，跳过")
            skip_others += 1
            continue

        # 基础文件名（与输出一致）
        base_name = os.path.splitext(os.path.basename(src_rel))[0]
        if base_name.endswith('_D'):
            base_name = base_name[:-2]

        # 检查输出文件是否已存在
        if SKIP_EXISTING:
            rgb_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_rgb_cropped.jpg")
            all_exist = os.path.exists(rgb_out_path)
            for i, tgt_rel in enumerate(tgt_list):
                tgt_name = os.path.basename(tgt_rel)
                if '_MS_' in tgt_name:
                    band = tgt_name.split('_MS_')[-1].split('.')[0]
                else:
                    band = BAND_NAMES[i]
                ms_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_MS_{band}_cropped.tif")
                if not os.path.exists(ms_out_path):
                    all_exist = False
                    break
            if all_exist:
                print(f"  输出文件已存在，跳过")
                skip_existing += 1
                continue

        # 加载RGB图像
        try:
            rgb_img = load_rgb_image(rgb_path).to(DEVICE)
            rgb_np = (rgb_img.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            h_rgb, w_rgb = rgb_np.shape[:2]
        except Exception as e:
            print(f"  加载RGB图像失败: {e}，跳过")
            skip_others += 1
            continue

        aligned_images = []
        masks = []
        all_success = True
        success_pts_dst_list = []
        last_pts_dst = None          # 保存最近一次计算的内点（即使不满足条件）

        for tgt_rel in tgt_list:
            tgt_path = os.path.join(TARGET_ROOT, tgt_rel)
            if not os.path.exists(tgt_path):
                print(f"  光谱图像不存在: {tgt_path}，跳过整组")
                all_success = False
                break

            try:
                ms_img = load_tiff_robust(tgt_path).to(DEVICE)
                ms_np = (ms_img.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
            except Exception as e:
                print(f"  加载光谱图像失败: {e}，跳过整组")
                all_success = False
                break

            # 计算匹配内点
            pts_src, pts_dst, inliers = compute_inliers(ms_img, rgb_img)
            last_pts_dst = pts_dst   # 记录本次计算结果（可能为 None）

            if pts_src is None or inliers < MIN_INLIERS:
                print(f"  匹配内点不足 ({inliers} < {MIN_INLIERS})，跳过整组")
                all_success = False
                break

            # 均匀度判别
            is_uniform, area_ratio, fill_ratio = check_large_compact_void(
                pts_dst, (h_rgb, w_rgb), grid_size=GRID_SIZE,
                max_void_ratio=MAX_VOID_RATIO, min_fill_ratio=MIN_FILL_RATIO
            )
            if not is_uniform:
                print(f"  内点分布不均匀：最大无内点区域面积占比 {area_ratio:.3f} > {MAX_VOID_RATIO}，填充率 {fill_ratio:.3f} > {MIN_FILL_RATIO}，跳过整组")
                all_success = False
                break

            # 记录成功的内点
            success_pts_dst_list.append(pts_dst)

            # 局部单应性变换
            aligned, mask = local_homography_warp(
                ms_np, pts_src, pts_dst, (h_rgb, w_rgb),
                grid_size=GRID_SIZE, sigma=None
            )
            aligned_images.append(aligned)
            masks.append(mask)

        # 如果因匹配失败而跳过，保存失败图像（优先绘制成功的内点，否则绘制最后一次计算的内点）
        if not all_success:
            # 确定要绘制的点集
            if success_pts_dst_list:
                all_pts_dst = np.vstack(success_pts_dst_list)
            elif last_pts_dst is not None:
                all_pts_dst = last_pts_dst
            else:
                all_pts_dst = np.empty((0,2))

            # 绘制内点并保存（注意：rgb_np 是 BGR 顺序，draw_keypoints 中 color 也是 BGR）
            img_with_pts = draw_keypoints(rgb_np, all_pts_dst, radius=8, color=(255, 0, 0), thickness=-1, outline=True)
            failed_path = os.path.join(failed_dir, f"{base_name}_failed.jpg")
            Image.fromarray(img_with_pts).save(failed_path, quality=95)
            print(f"  保存失败图像（含内点）至: {failed_path}")
            skip_no_match += 1
            continue

        # 计算公共区域
        rgb_mask = np.ones((h_rgb, w_rgb), dtype=np.uint8)
        all_masks = [rgb_mask] + masks
        intersection_mask = all_masks[0].copy()
        for m in all_masks[1:]:
            intersection_mask = cv2.bitwise_and(intersection_mask, m)

        if np.sum(intersection_mask) == 0:
            print("  公共区域为空，跳过")
            # 公共区域为空也保存失败图像（绘制所有成功配准的内点）
            all_pts_dst = np.vstack(success_pts_dst_list) if success_pts_dst_list else np.empty((0,2))
            img_with_pts = draw_keypoints(rgb_np, all_pts_dst, radius=5, color=(0,0,255), thickness=-1, outline=True)
            failed_path = os.path.join(failed_dir, f"{base_name}_failed.jpg")
            Image.fromarray(img_with_pts).save(failed_path, quality=95)
            print(f"  保存失败图像（含内点）至: {failed_path}")
            skip_others += 1
            continue

        x, y, w, h = largest_inner_rectangle(intersection_mask)
        if w == 0 or h == 0:
            print("  最大内接矩形面积为0，跳过")
            # 矩形面积为0也保存失败图像
            all_pts_dst = np.vstack(success_pts_dst_list) if success_pts_dst_list else np.empty((0,2))
            img_with_pts = draw_keypoints(rgb_np, all_pts_dst, radius=5, color=(0,0,255), thickness=-1, outline=True)
            failed_path = os.path.join(failed_dir, f"{base_name}_failed.jpg")
            Image.fromarray(img_with_pts).save(failed_path, quality=95)
            print(f"  保存失败图像（含内点）至: {failed_path}")
            skip_others += 1
            continue

        x_max = x + w - 1
        y_max = y + h - 1
        print(f"  裁剪矩形: [{x}, {y}] -> [{x_max}, {y_max}], 尺寸 {h}x{w}")

        # 保存RGB
        rgb_cropped = rgb_np[y:y_max+1, x:x_max+1]
        rgb_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_rgb_cropped.jpg")
        Image.fromarray(rgb_cropped).save(rgb_out_path, format='JPEG', quality=95)
        print(f"  保存RGB裁剪: {rgb_out_path}")

        # 保存光谱
        for i, aligned in enumerate(aligned_images):
            cropped = aligned[y:y_max+1, x:x_max+1]
            tgt_name = os.path.basename(tgt_list[i])
            if '_MS_' in tgt_name:
                band = tgt_name.split('_MS_')[-1].split('.')[0]
            else:
                band = BAND_NAMES[i]
            ms_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_MS_{band}_cropped.tif")
            tifffile.imwrite(ms_out_path, cropped)
            print(f"  保存光谱裁剪: {ms_out_path}")

        success += 1

    # 统计输出
    print("\n" + "="*50)
    print(f"处理完成！总计 {total} 组")
    print(f"成功裁剪（本次处理）: {success} 组")
    print(f"跳过已存在输出: {skip_existing} 组")
    print(f"因匹配点不足/分布不均跳过: {skip_no_match} 组")
    print(f"其他原因跳过: {skip_others} 组")
    print("="*50)

if __name__ == "__main__":
    main()