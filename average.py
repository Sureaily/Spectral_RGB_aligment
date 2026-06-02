"""
analyze_results.py
从配准评估输出的 CSV 文件计算统计平均值。
"""

import csv
import numpy as np
from collections import defaultdict

def safe_float(val):
    """安全转换为浮点数，无效值返回 nan"""
    if val in ['NaN', 'nan', 'N/A', '']:
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan

def safe_int(val):
    """安全转换为整数，无效值返回 nan"""
    if val in ['NaN', 'nan', 'N/A', '']:
        return np.nan
    try:
        return int(val)
    except ValueError:
        return np.nan

def main():
    csv_path = '/root/autodl-tmp/LightGlue-main/alignment_full_comparison.csv'  # 修改为实际路径
    methods = ['Initial', 'Global', 'Proposed']
    bands = ['G', 'R', 'NIR', 'RE']

    # 存储所有数据
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    print("=" * 70)
    print("CSV 统计分析结果")
    print("=" * 70)

    # ---------- 1. 按方法统计 num_inliers 和 crop_size ----------
    print("\n【按方法统计】")
    print(f"{'Method':<10} {'Avg Inliers':<15} {'Avg Crop Width':<15} {'Avg Crop Height':<15}")
    print("-" * 55)

    for method in methods:
        method_rows = [r for r in data if r['method'] == method]

        # 内点数（Initial 不计）
        inlier_vals = []
        if method != 'Initial':
            for r in method_rows:
                val = safe_int(r['num_inliers'])
                if not np.isnan(val):
                    inlier_vals.append(val)
        avg_inliers = np.mean(inlier_vals) if inlier_vals else np.nan

        # 裁剪尺寸（解析 crop_size 字段）
        crop_widths = []
        crop_heights = []
        for r in method_rows:
            crop_str = r.get('crop_size', '')
            if crop_str and crop_str != 'N/A' and '*' in crop_str:
                try:
                    w_str, h_str = crop_str.split('*')
                    crop_widths.append(int(w_str))
                    crop_heights.append(int(h_str))
                except ValueError:
                    continue
        avg_w = np.mean(crop_widths) if crop_widths else np.nan
        avg_h = np.mean(crop_heights) if crop_heights else np.nan

        inlier_str = f"{avg_inliers:.1f}" if not np.isnan(avg_inliers) else "N/A"
        width_str = f"{avg_w:.1f}" if not np.isnan(avg_w) else "N/A"
        height_str = f"{avg_h:.1f}" if not np.isnan(avg_h) else "N/A"
        print(f"{method:<10} {inlier_str:<15} {width_str:<15} {height_str:<15}")

    # ---------- 2. 分波段统计 inlier_residual, NMI, GCC ----------
    print("\n【分波段统计 (inlier_residual, NMI, GCC)】")
    for band in bands:
        print(f"\n--- 波段: {band} ---")
        print(f"{'Method':<10} {'Inlier Residual (pix)':<25} {'NMI':<15} {'GCC':<15}")
        print("-" * 65)

        for method in methods:
            band_method_rows = [r for r in data
                                if r['band'] == band and r['method'] == method]
            if not band_method_rows:
                continue

            # inlier_residual
            res_vals = []
            if method != 'Initial':
                for r in band_method_rows:
                    val = safe_float(r['inlier_residual'])
                    if not np.isnan(val):
                        res_vals.append(val)
            avg_res = np.mean(res_vals) if res_vals else np.nan
            res_str = f"{avg_res:.4f}" if not np.isnan(avg_res) else "N/A"

            # NMI
            nmi_vals = []
            for r in band_method_rows:
                val = safe_float(r['NMI'])
                if not np.isnan(val):
                    nmi_vals.append(val)
            avg_nmi = np.mean(nmi_vals) if nmi_vals else np.nan
            nmi_str = f"{avg_nmi:.4f}" if not np.isnan(avg_nmi) else "N/A"

            # GCC
            gcc_vals = []
            for r in band_method_rows:
                val = safe_float(r['GCC'])
                if not np.isnan(val):
                    gcc_vals.append(val)
            avg_gcc = np.mean(gcc_vals) if gcc_vals else np.nan
            gcc_str = f"{avg_gcc:.4f}" if not np.isnan(avg_gcc) else "N/A"

            print(f"{method:<10} {res_str:<25} {nmi_str:<15} {gcc_str:<15}")

    print("\n" + "=" * 70)
    print("统计完成。")
    print("=" * 70)

if __name__ == "__main__":
    main()