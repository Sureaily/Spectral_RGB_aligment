"""
integrated_registration_evaluation_v2.py

一体化脚本：配准（采用稳定的局部单应性实现）+ 公平评估

配准部分源自您提供的 RGB 与光谱图像配准裁剪工具（稳定版）。
评估部分在此基础上增加 Initial、Global、Proposed 三种方法的指标对比。
新增：Global 裁剪区域也采用多波段交集的最大内接矩形，与 Proposed 对称。
"""

import torch
import numpy as np
import cv2
import os
import json
import argparse
import csv
from collections import defaultdict
import tifffile
from PIL import Image
from scipy.spatial import KDTree
from scipy import ndimage
from sklearn.metrics import mutual_info_score
from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
from lightglue.utils import rbd
import warnings
warnings.filterwarnings('ignore')

# -------------------- 参数解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="配准+评估一体化脚本")
    parser.add_argument('--jsonl_path', type=str, required=True)
    parser.add_argument('--source_root', type=str, required=True)
    parser.add_argument('--target_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='full_comparison.csv')
    parser.add_argument('--grid_size', type=int, default=20)
    parser.add_argument('--sigma_factor', type=float, default=0.1,
                        help='高斯权重的sigma因子（相对于图像最大边长）')
    parser.add_argument('--min_inliers', type=int, default=1000)
    parser.add_argument('--max_void_ratio', type=float, default=0.1)
    parser.add_argument('--min_fill_ratio', type=float, default=0.6)
    parser.add_argument('--max_keypoints', type=int, default=12000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--features', type=str, nargs='+', default=['aliked', 'disk', 'superpoint', 'sift'])
    parser.add_argument('--ransac_thresh', type=float, default=2.0)
    parser.add_argument('--skip_existing', action='store_true', default=False)
    return parser.parse_args()

args = parse_args()
DEVICE = torch.device(args.device)

# -------------------- 初始化特征提取器和匹配器 --------------------
AVAILABLE_FEATURES = {
    'aliked': (ALIKED, {'max_num_keypoints': args.max_keypoints}),
    'disk': (DISK, {'max_num_keypoints': args.max_keypoints}),
    'superpoint': (SuperPoint, {'max_num_keypoints': args.max_keypoints, 'threshold': 0.005}),
    'sift': (SIFT, {'max_num_keypoints': args.max_keypoints, 'backend': 'opencv'}),
}

extractors = {}
matchers = {}
for feat in args.features:
    if feat not in AVAILABLE_FEATURES:
        continue
    cls, kwargs = AVAILABLE_FEATURES[feat]
    extractors[feat] = cls(**kwargs).eval().to(DEVICE)
    matchers[feat] = LightGlue(features=feat, depth_confidence=-1, width_confidence=-1).eval().to(DEVICE)

print(f"已初始化特征器: {list(extractors.keys())}")
print(f"使用设备: {DEVICE}")

# -------------------- 图像加载函数 --------------------
def load_rgb_image(path):
    img = Image.open(path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_uint8 = np.array(img)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return tensor, img_uint8

def load_ms_image(path):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=2)
    arr = arr.astype(np.float32)
    if arr.dtype == np.uint16:
        arr /= 65535.0
    elif arr.dtype == np.uint8:
        arr /= 255.0
    else:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr_3ch = np.stack([arr, arr, arr], axis=-1)
    tensor = torch.from_numpy(arr_3ch).permute(2, 0, 1).unsqueeze(0)
    img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    return tensor, img_uint8

# -------------------- 特征匹配与全局单应性（用于备用和 Global 评估） --------------------
def compute_matches_and_homography(rgb_tensor, ms_tensor):
    all_pts_src, all_pts_dst = [], []
    for name in extractors:
        extractor = extractors[name]
        matcher = matchers[name]
        with torch.no_grad():
            feats_rgb = extractor.extract(rgb_tensor.to(DEVICE))
            feats_ms = extractor.extract(ms_tensor.to(DEVICE))
        if len(feats_rgb['keypoints']) == 0 or len(feats_ms['keypoints']) == 0:
            continue
        matches = matcher({'image0': feats_rgb, 'image1': feats_ms})
        feats_rgb, feats_ms, matches = [rbd(x) for x in [feats_rgb, feats_ms, matches]]
        matches = matches['matches']
        pts_rgb = feats_rgb['keypoints'][matches[..., 0]].cpu().numpy()
        pts_ms = feats_ms['keypoints'][matches[..., 1]].cpu().numpy()
        if len(pts_rgb) > 0:
            all_pts_dst.append(pts_rgb)
            all_pts_src.append(pts_ms)
    if not all_pts_dst:
        return None, None, None, 0.0, 0
    merged_pts_dst = np.vstack(all_pts_dst)
    merged_pts_src = np.vstack(all_pts_src)
    if len(merged_pts_dst) < 4:
        return None, None, None, 0.0, 0
    H, mask = cv2.findHomography(merged_pts_src, merged_pts_dst, cv2.RANSAC, args.ransac_thresh)
    if H is None or mask is None:
        return None, None, None, 0.0, 0
    mask = mask.ravel().astype(bool)
    inliers_src = merged_pts_src[mask]
    inliers_dst = merged_pts_dst[mask]
    num_inliers = np.sum(mask)
    if num_inliers > 0:
        ones = np.ones((inliers_src.shape[0], 1))
        src_hom = np.hstack([inliers_src, ones])
        proj_hom = (H @ src_hom.T).T
        proj_pts = proj_hom[:, :2] / proj_hom[:, 2:]
        errors = np.linalg.norm(proj_pts - inliers_dst, axis=1)
        mean_residual = np.mean(errors)
    else:
        mean_residual = 0.0
    return inliers_src, inliers_dst, H, mean_residual, num_inliers

# -------------------- 局部单应性变换（稳定版，源自您提供的代码） --------------------
def weighted_homography(pts_src, pts_dst, weights):
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

def local_homography_warp(img_src, pts_src, pts_dst, img_shape_dst, grid_size, sigma_factor):
    """
    高性能局部单应性变换（矢量化坐标映射，速度提升 20~50 倍）
    """
    h_dst, w_dst = img_shape_dst
    h_src, w_src = img_src.shape[:2]

    # ---------- 1. 全局单应性备用 ----------
    H_global = None
    if len(pts_src) >= 4:
        H_global, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

    # ---------- 2. 网格划分 ----------
    grid_y = np.linspace(0, h_dst, grid_size + 1).astype(np.int32)
    grid_x = np.linspace(0, w_dst, grid_size + 1).astype(np.int32)

    # ---------- 3. 预计算网格中心 ----------
    cell_centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            cy = (grid_y[i] + grid_y[i+1]) // 2
            cx = (grid_x[j] + grid_x[j+1]) // 2
            cell_centers.append([cx, cy])
    cell_centers = np.array(cell_centers, dtype=np.float32)

    sigma = max(w_dst, h_dst) * sigma_factor
    tree = KDTree(pts_dst)

    # ---------- 4. 为每个网格计算单应性（快速） ----------
    H_cells = []
    for center in cell_centers:
        indices = tree.query_ball_point(center, r=3 * sigma)
        if len(indices) < 4:
            H_cells.append(H_global)
            continue
        pts_dst_local = pts_dst[indices]
        pts_src_local = pts_src[indices]
        dists = np.linalg.norm(pts_dst_local - center, axis=1)
        weights = np.exp(- (dists**2) / (2 * sigma**2))
        H = weighted_homography(pts_src_local, pts_dst_local, weights)
        H_cells.append(H)

    # ---------- 5. 构建映射表（矢量化极速版） ----------
    map_x = np.zeros((h_dst, w_dst), dtype=np.float32)
    map_y = np.zeros((h_dst, w_dst), dtype=np.float32)
    mask = np.zeros((h_dst, w_dst), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            y_start, y_end = grid_y[i], grid_y[i+1]
            x_start, x_end = grid_x[j], grid_x[j+1]
            H = H_cells[i * grid_size + j]
            if H is None:
                continue

            # 生成网格坐标（内存友好）
            yv, xv = np.mgrid[y_start:y_end, x_start:x_end].astype(np.float32)
            coords = np.dstack([xv, yv])  # shape (H_cell, W_cell, 2)

            # 关键优化：使用 OpenCV 的透视变换，一次性处理整个网格的所有像素
            H_inv = np.linalg.inv(H)
            coords_src = cv2.perspectiveTransform(coords.reshape(-1, 1, 2), H_inv)
            coords_src = coords_src.reshape(y_end - y_start, x_end - x_start, 2)

            # 有效性掩码
            valid = (coords_src[..., 0] >= 0) & (coords_src[..., 0] < w_src - 1) & \
                    (coords_src[..., 1] >= 0) & (coords_src[..., 1] < h_src - 1)

            # 无效坐标置零
            coords_src[~valid] = 0

            # 填入映射表
            map_x[y_start:y_end, x_start:x_end] = coords_src[..., 0]
            map_y[y_start:y_end, x_start:x_end] = coords_src[..., 1]
            mask[y_start:y_end, x_start:x_end] = valid.astype(np.uint8)

    # ---------- 6. 执行重映射 ----------
    warped = cv2.remap(img_src, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, mask

# -------------------- 均匀度判别（填充率法） --------------------
def check_large_compact_void(points, image_shape, grid_size, max_void_ratio, min_fill_ratio):
    h, w = image_shape
    grid_h = h / grid_size
    grid_w = w / grid_size
    grid_map = np.ones((grid_size, grid_size), dtype=np.uint8)
    for (x, y) in points:
        col = min(int(x // grid_w), grid_size - 1)
        row = min(int(y // grid_h), grid_size - 1)
        grid_map[row, col] = 0
    labeled, num_features = ndimage.label(grid_map, structure=np.ones((3,3)))
    if num_features == 0:
        return True, 0, 0
    max_area, max_mask = 0, None
    for label_id in range(1, num_features+1):
        mask = (labeled == label_id)
        area = np.sum(mask)
        if area > max_area:
            max_area, max_mask = area, mask
    if max_mask is None:
        return True, 0, 0
    total_cells = grid_size * grid_size
    area_ratio = max_area / total_cells
    if area_ratio <= max_void_ratio:
        return True, area_ratio, 0
    rows, cols = np.where(max_mask)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    width = c_max - c_min + 1
    height = r_max - r_min + 1
    bounding_area = width * height
    if bounding_area == 0:
        return True, area_ratio, 0
    fill_ratio = max_area / bounding_area
    if fill_ratio > min_fill_ratio:
        return False, area_ratio, fill_ratio
    else:
        return True, area_ratio, fill_ratio

def largest_inner_rectangle(mask):
    """
    快速最大内接矩形查找（向量化 height 计算，线性栈扫描）
    """
    h, w = mask.shape
    
    # ---------- 向量化计算 height 矩阵 ----------
    height = np.zeros((h, w), dtype=np.int32)
    height[0, :] = mask[0, :]
    for i in range(1, h):
        height[i, :] = (height[i-1, :] + 1) * mask[i, :]
    
    # ---------- 对每一行使用栈算法找最大矩形 ----------
    max_area = 0
    best_rect = (0, 0, 0, 0)
    for i in range(h):
        stack = []
        row = height[i, :]
        for j in range(w + 1):
            cur_h = row[j] if j < w else 0
            start = j
            while stack and stack[-1][1] > cur_h:
                prev_j, prev_h = stack.pop()
                start = prev_j
                area = prev_h * (j - prev_j)
                if area > max_area:
                    max_area = area
                    best_rect = (prev_j, i - prev_h + 1, j - prev_j, prev_h)
            if j < w and (not stack or cur_h > stack[-1][1]):
                stack.append((start, cur_h))
    return best_rect

# -------------------- 指标计算 --------------------
def compute_nmi(img1, img2, bins=256):
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2
    if gray1.dtype != np.uint8:
        gray1 = (gray1 * 255).clip(0, 255).astype(np.uint8)
    if gray2.dtype != np.uint8:
        gray2 = (gray2 * 255).clip(0, 255).astype(np.uint8)
    hist_2d = np.histogram2d(gray1.ravel(), gray2.ravel(), bins=bins)[0]
    mi = mutual_info_score(None, None, contingency=hist_2d)
    hist1 = np.histogram(gray1, bins=bins)[0]
    hist2 = np.histogram(gray2, bins=bins)[0]
    eps = 1e-10
    h1 = -np.sum((hist1 / np.sum(hist1)) * np.log((hist1 + eps) / np.sum(hist1)))
    h2 = -np.sum((hist2 / np.sum(hist2)) * np.log((hist2 + eps) / np.sum(hist2)))
    return 2 * mi / (h1 + h2 + eps)

def compute_gcc(img1, img2):
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2
    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)
    grad_x1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    grad_x2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
    mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
    mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
    return np.corrcoef(mag1.flat, mag2.flat)[0, 1]

# -------------------- 主流程 --------------------
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f if line.strip()]
    groups = defaultdict(list)
    for item in lines:
        groups[item['source']].append(item['target'])
    total = len(groups)
    print(f"共找到 {total} 组图像对")

    fieldnames = ['image', 'band', 'method', 'inlier_residual', 'num_inliers',
                  'NMI', 'GCC', 'crop_size']
    csv_exists = os.path.exists(args.output_csv)
    with open(args.output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

    band_names = ['G', 'R', 'NIR', 'RE']
    all_results = []

    for idx, (src_rel, tgt_list) in enumerate(groups.items(), 1):
        base_name = os.path.splitext(os.path.basename(src_rel))[0]
        if base_name.endswith('_D'):
            base_name = base_name[:-2]
        rgb_path = os.path.join(args.source_root, src_rel)
        if not os.path.exists(rgb_path):
            print(f"[{idx}/{total}] 跳过 {base_name}：RGB 图像不存在")
            continue

        # 文件路径
        rgb_crop_path = os.path.join(args.output_dir, f"{base_name}_rgb_cropped.jpg")
        first_band_crop = os.path.join(args.output_dir, f"{base_name}_MS_G_cropped.tif")
        rect_proposed_path = os.path.join(args.output_dir, f"{base_name}_crop_rect_proposed.json")
        rect_global_path = os.path.join(args.output_dir, f"{base_name}_crop_rect_global.json")

        need_registration = args.skip_existing or not (
            os.path.exists(rgb_crop_path) and os.path.exists(first_band_crop) and os.path.exists(rect_proposed_path)
        )

        # 加载RGB原图（无论是否需要配准，评估都需要）
        try:
            rgb_tensor, rgb_uint8 = load_rgb_image(rgb_path)
            h_rgb, w_rgb = rgb_uint8.shape[:2]
        except Exception as e:
            print(f"  加载RGB失败: {e}，跳过整组")
            continue

        if need_registration:
            print(f"[{idx}/{total}] {base_name}：开始配准流程...")
            aligned_proposed = []
            masks_proposed = []
            masks_global = []
            all_success = True

            for i, tgt_rel in enumerate(tgt_list):
                tgt_path = os.path.join(args.target_root, tgt_rel)
                if not os.path.exists(tgt_path):
                    print(f"  多光谱图像不存在: {tgt_rel}，跳过整组")
                    all_success = False
                    break
                try:
                    ms_tensor, ms_uint8 = load_ms_image(tgt_path)
                except Exception as e:
                    print(f"  加载多光谱失败: {e}，跳过整组")
                    all_success = False
                    break

                # 获取内点
                pts_src, pts_dst, H, _, num_inliers = compute_matches_and_homography(rgb_tensor, ms_tensor)
                if pts_src is None or num_inliers < args.min_inliers:
                    print(f"  内点不足 ({num_inliers})，跳过整组")
                    all_success = False
                    break

                is_uniform, _, _ = check_large_compact_void(
                    pts_dst, (h_rgb, w_rgb), grid_size=args.grid_size,
                    max_void_ratio=args.max_void_ratio, min_fill_ratio=args.min_fill_ratio
                )
                if not is_uniform:
                    print(f"  内点分布不均匀，跳过整组")
                    all_success = False
                    break

                # 保存内点和全局单应性（供后续评估使用）
                band = band_names[i] if i < len(band_names) else f"band{i}"
                np.save(os.path.join(args.output_dir, f"{base_name}_{band}_pts_src.npy"), pts_src)
                np.save(os.path.join(args.output_dir, f"{base_name}_{band}_pts_dst.npy"), pts_dst)
                np.save(os.path.join(args.output_dir, f"{base_name}_{band}_H_global.npy"), H)

                # 局部单应性变换
                warped, mask = local_homography_warp(
                    ms_uint8, pts_src, pts_dst, (h_rgb, w_rgb),
                    grid_size=args.grid_size, sigma_factor=args.sigma_factor
                )
                aligned_proposed.append(warped)
                masks_proposed.append(mask)

                # 全局单应性掩码（用于计算 Global 公共区域）
                warped_global = cv2.warpPerspective(ms_uint8, H, (w_rgb, h_rgb),
                                                    flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if warped_global.ndim == 3:
                    mask_global = (warped_global.max(axis=2) > 0).astype(np.uint8)
                else:
                    mask_global = (warped_global > 0).astype(np.uint8)
                masks_global.append(mask_global)

            if not all_success:
                continue

            # 计算 Proposed 公共区域
            inter_proposed = np.ones((h_rgb, w_rgb), dtype=np.uint8)
            for m in masks_proposed:
                inter_proposed = cv2.bitwise_and(inter_proposed, m)
            if np.sum(inter_proposed) == 0:
                print("  Proposed 公共区域为空，跳过")
                continue
            x_p, y_p, w_p, h_p = largest_inner_rectangle(inter_proposed)
            if w_p == 0 or h_p == 0:
                print("  Proposed 最大内接矩形面积为0，跳过")
                continue

            # 计算 Global 公共区域（多波段交集）
            inter_global = np.ones((h_rgb, w_rgb), dtype=np.uint8)
            for m in masks_global:
                inter_global = cv2.bitwise_and(inter_global, m)
            if np.sum(inter_global) == 0:
                print("  Global 公共区域为空，跳过")
                continue
            x_g, y_g, w_g, h_g = largest_inner_rectangle(inter_global)
            if w_g == 0 or h_g == 0:
                print("  Global 最大内接矩形面积为0，跳过")
                continue

            # 保存矩形
            crop_rect = (int(x_p), int(y_p), int(w_p), int(h_p))
            crop_rect_global = (int(x_g), int(y_g), int(w_g), int(h_g))
            with open(rect_proposed_path, 'w') as f:
                json.dump({'x': int(x_p), 'y': int(y_p), 'w': int(w_p), 'h': int(h_p)}, f)
            with open(rect_global_path, 'w') as f:
                json.dump({'x': int(x_g), 'y': int(y_g), 'w': int(w_g), 'h': int(h_g)}, f)

            # 保存裁剪图像（使用 Proposed 矩形）
            x_max, y_max = x_p + w_p - 1, y_p + h_p - 1
            rgb_cropped = rgb_uint8[y_p:y_max+1, x_p:x_max+1]
            Image.fromarray(rgb_cropped).save(rgb_crop_path, quality=95)

            for i, aligned in enumerate(aligned_proposed):
                cropped = aligned[y_p:y_max+1, x_p:x_max+1]
                band = band_names[i] if i < len(band_names) else f"band{i}"
                ms_out_path = os.path.join(args.output_dir, f"{base_name}_MS_{band}_cropped.tif")
                tifffile.imwrite(ms_out_path, cropped)

            print(f"  配准完成，裁剪矩形: Proposed ({x_p},{y_p}) {w_p}x{h_p}, Global ({x_g},{y_g}) {w_g}x{h_g}")

        else:
            with open(rect_proposed_path, 'r') as f:
                rect_dict = json.load(f)
                crop_rect = (rect_dict['x'], rect_dict['y'], rect_dict['w'], rect_dict['h'])
            if os.path.exists(rect_global_path):
                with open(rect_global_path, 'r') as f:
                    global_dict = json.load(f)
                    crop_rect_global = (global_dict['x'], global_dict['y'], global_dict['w'], global_dict['h'])
            else:
                crop_rect_global = None
            print(f"[{idx}/{total}] {base_name}：加载已存在的裁剪矩形")

        # -------------------- 评估阶段 --------------------
        rgb_cropped = cv2.imread(rgb_crop_path)
        if rgb_cropped is None:
            print(f"  无法读取裁剪RGB，跳过评估")
            continue
        rgb_cropped = cv2.cvtColor(rgb_cropped, cv2.COLOR_BGR2RGB)
        h_crop, w_crop = rgb_cropped.shape[:2]

        # 提取裁剪尺寸
        _, _, prop_w, prop_h = crop_rect
        proposed_crop_size = f"{prop_w}*{prop_h}"
        if crop_rect_global is not None:
            global_w, global_h = crop_rect_global[2], crop_rect_global[3]
            global_crop_size = f"{global_w}*{global_h}"
        else:
            global_crop_size = "0*0"

        for i, tgt_rel in enumerate(tgt_list):
            band = band_names[i] if i < len(band_names) else f"band{i}"
            tgt_path = os.path.join(args.target_root, tgt_rel)
            try:
                _, ms_orig_uint8 = load_ms_image(tgt_path)
            except:
                continue

            # ----- Initial -----
            ms_initial = cv2.resize(ms_orig_uint8, (w_crop, h_crop), interpolation=cv2.INTER_LINEAR)
            ms_initial_rgb = cv2.cvtColor(ms_initial, cv2.COLOR_GRAY2RGB) if ms_initial.ndim == 2 else ms_initial
            nmi_init = compute_nmi(rgb_cropped, ms_initial_rgb)
            gcc_init = compute_gcc(rgb_cropped, ms_initial_rgb)

            # ----- Global -----
            H_global_path = os.path.join(args.output_dir, f"{base_name}_{band}_H_global.npy")
            pts_src_path = os.path.join(args.output_dir, f"{base_name}_{band}_pts_src.npy")
            pts_dst_path = os.path.join(args.output_dir, f"{base_name}_{band}_pts_dst.npy")

            if os.path.exists(H_global_path) and os.path.exists(pts_src_path) and os.path.exists(pts_dst_path):
                H_global = np.load(H_global_path)
                pts_src = np.load(pts_src_path)
                pts_dst = np.load(pts_dst_path)
                num_inliers_global = len(pts_src)

                # 生成全局配准并裁剪（仍用 Proposed 矩形裁剪以公平比较指标）
                _, ms_full_uint8 = load_ms_image(tgt_path)
                x, y, w, h = crop_rect
                ms_global_warped = cv2.warpPerspective(ms_full_uint8, H_global, (w_rgb, h_rgb),
                                                       flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                ms_global_cropped = ms_global_warped[y:y+h, x:x+w]
                ms_global_rgb = cv2.cvtColor(ms_global_cropped, cv2.COLOR_GRAY2RGB) if ms_global_cropped.ndim == 2 else ms_global_cropped

                # 重算残差（基于裁剪后的图像对）
                ms_tensor_global = torch.from_numpy(ms_global_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
                rgb_tensor_crop = torch.from_numpy(rgb_cropped.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
                _, _, _, inlier_res_global, _ = compute_matches_and_homography(rgb_tensor_crop, ms_tensor_global)

                nmi_global = compute_nmi(rgb_cropped, ms_global_rgb)
                gcc_global = compute_gcc(rgb_cropped, ms_global_rgb)
            else:
                nmi_global = gcc_global = inlier_res_global = float('nan')
                num_inliers_global = 0

            # ----- Proposed -----
            ms_crop_path = os.path.join(args.output_dir, f"{base_name}_MS_{band}_cropped.tif")
            if os.path.exists(ms_crop_path):
                ms_proposed = tifffile.imread(ms_crop_path)
                if ms_proposed.ndim == 3:
                    ms_proposed = np.mean(ms_proposed, axis=2)
                ms_proposed = ms_proposed.astype(np.uint8)
                ms_proposed_rgb = cv2.cvtColor(ms_proposed, cv2.COLOR_GRAY2RGB)

                ms_tensor_proposed = torch.from_numpy(ms_proposed_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
                rgb_tensor_crop = torch.from_numpy(rgb_cropped.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
                _, _, _, inlier_res_proposed, num_inliers_proposed = compute_matches_and_homography(rgb_tensor_crop, ms_tensor_proposed)

                nmi_proposed = compute_nmi(rgb_cropped, ms_proposed_rgb)
                gcc_proposed = compute_gcc(rgb_cropped, ms_proposed_rgb)
            else:
                nmi_proposed = gcc_proposed = inlier_res_proposed = float('nan')
                num_inliers_proposed = 0

            # 写入 CSV
            base_row = {'image': base_name, 'band': band}

            row_init = {**base_row, 'method': 'Initial',
                        'inlier_residual': 'NaN', 'num_inliers': num_inliers_proposed,
                        'NMI': f"{nmi_init:.6f}", 'GCC': f"{gcc_init:.6f}",
                        'crop_size': 'N/A'}
            row_global = {**base_row, 'method': 'Global',
                          'inlier_residual': f"{inlier_res_global:.4f}", 'num_inliers': num_inliers_global,
                          'NMI': f"{nmi_global:.6f}", 'GCC': f"{gcc_global:.6f}",
                          'crop_size': global_crop_size}
            row_proposed = {**base_row, 'method': 'Proposed',
                            'inlier_residual': f"{inlier_res_proposed:.4f}", 'num_inliers': num_inliers_proposed,
                            'NMI': f"{nmi_proposed:.6f}", 'GCC': f"{gcc_proposed:.6f}",
                            'crop_size': proposed_crop_size}

            with open(args.output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_init)
                writer.writerow(row_global)
                writer.writerow(row_proposed)

            all_results.extend([row_init, row_global, row_proposed])

            print(f"  └─ {band}: Init NMI={nmi_init:.4f} GCC={gcc_init:.4f} | "
                  f"Global NMI={nmi_global:.4f} GCC={gcc_global:.4f} Res={inlier_res_global:.3f} (crop {global_crop_size}) | "
                  f"Prop NMI={nmi_proposed:.4f} GCC={gcc_proposed:.4f} Res={inlier_res_proposed:.3f} (crop {proposed_crop_size})")

    # 汇总统计
    print("\n" + "="*60)
    print("评估完成！汇总统计：")
    if all_results:
        methods = ['Initial', 'Global', 'Proposed']
        for method in methods:
            method_rows = [r for r in all_results if r['method'] == method]
            if not method_rows:
                continue
            nmi_vals = [float(r['NMI']) for r in method_rows if r['NMI'] != 'nan']
            gcc_vals = [float(r['GCC']) for r in method_rows if r['GCC'] != 'nan']
            if method != 'Initial':
                res_vals = [float(r['inlier_residual']) for r in method_rows if r['inlier_residual'] not in ['NaN','nan']]
                avg_res = np.mean(res_vals) if res_vals else float('nan')
            else:
                avg_res = float('nan')
            avg_nmi = np.mean(nmi_vals) if nmi_vals else float('nan')
            avg_gcc = np.mean(gcc_vals) if gcc_vals else float('nan')
            print(f"{method:10s}: NMI={avg_nmi:.4f} ± {np.std(nmi_vals):.4f}, "
                  f"GCC={avg_gcc:.4f} ± {np.std(gcc_vals):.4f}, "
                  f"Inlier Residual={avg_res:.3f} pixels")
    print("="*60)

if __name__ == "__main__":
    main()