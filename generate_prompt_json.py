#!/usr/bin/env python3
"""
根据新的命名规则生成 prompt.json：
  - source: DJI_{timestamp}_{index}.jpg
  - target: DJI_{timestamp}_{index}.tif
  - prompt: "DJI Mavic 3M"
 python generate_prompt_json.py --source_dir /mnt/data/Sureaily/ControlNet_datasets/source
 --target_dir /mnt/data/Sureaily/ControlNet_datasets/target
 --output /mnt/data/Sureaily/ControlNet_datasets/prompt.json
输出 JSON Lines 格式，每行一对。
"""

import re
import json
import argparse
from pathlib import Path


def extract_base_id(filename: str) -> str:
    """
    从文件名中提取基础 ID: DJI_20251107113232_0001
    要求格式: DJI_{timestamp}_{index}.{ext}
    """
    match = re.match(r"(DJI_\d+_\d+)\.\w+$", filename, re.IGNORECASE)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description="生成 ControlNet prompt.json (单对单)")
    parser.add_argument("--source_dir", default="source", help="源图像目录 (默认: source)")
    parser.add_argument("--target_dir", default="target", help="目标图像目录 (默认: target)")
    parser.add_argument("--output", default="prompt_test.json", help="输出 JSON 文件 (默认: prompt.json)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    output_path = Path(args.output)

    if not source_dir.is_dir():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"目标目录不存在: {target_dir}")

    # 构建 source 索引: base_id -> 文件名
    source_map = {}
    for file in source_dir.iterdir():
        if file.is_file() and file.suffix.lower() in ('.jpg', '.jpeg'):
            base_id = extract_base_id(file.name)
            if base_id:
                if base_id in source_map:
                    print(f"警告: 重复的 source base_id {base_id}，使用最后找到的: {file.name}")
                source_map[base_id] = file.name
            else:
                print(f"跳过不符合命名规则的 source 文件: {file.name}")

    print(f"找到 {len(source_map)} 个 source 文件")

    # 遍历 target 文件，匹配并生成记录
    records = []
    skipped_no_source = 0
    skipped_invalid_name = 0

    for file in target_dir.iterdir():
        if not file.is_file() or file.suffix.lower() not in ('.tif', '.tiff'):
            continue

        base_id = extract_base_id(file.name)
        if not base_id:
            print(f"跳过无法识别命名规则的 target 文件: {file.name}")
            skipped_invalid_name += 1
            continue

        if base_id not in source_map:
            print(f"警告: target {file.name} 找不到对应的 source (base_id: {base_id})")
            skipped_no_source += 1
            continue

        source_filename = source_map[base_id]
        record = {
            "source": f"source/{source_filename}",
            "target": f"target/{file.name}",
            "prompt": "DJI Mavic 3M"
        }
        records.append(record)

    # 写入 JSON Lines 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"成功生成 {len(records)} 条记录，写入 {output_path}")
    if skipped_no_source:
        print(f"有 {skipped_no_source} 个 target 文件缺少对应 source")
    if skipped_invalid_name:
        print(f"有 {skipped_invalid_name} 个 target 文件无法提取有效 base_id")


if __name__ == "__main__":
    main()