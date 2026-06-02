# import json
# import cv2
# import numpy as np
#
# from torch.utils.data import Dataset
#
#
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         with open('/mnt/data/Sureaily/ControlNet_datasets/prompt.json', 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         item = self.data[idx]
#
#         source_filename = item['source']
#         target_filename = item['target']
#         prompt = item['prompt']
#
#         # 读取 source（3通道，BGR -> RGB）
#         source = cv2.imread('/mnt/data/Sureaily/ControlNet_datasets/' + source_filename)
#         source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#
#         # 读取 target（4通道，保留原样）
#         target = cv2.imread('/mnt/data/Sureaily/ControlNet_datasets/' + target_filename, cv2.IMREAD_UNCHANGED)
#         # 注意：此时 target 的通道顺序就是 [G, R, RE, NIR]（无需重排）
#
#         # 统一 resize
#         source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)
#         target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_LINEAR)
#
#         # 归一化
#         source = source.astype(np.float32) / 255.0  # [0, 1]
#         target = (target.astype(np.float32) / 127.5) - 1.0  # [-1, 1]  (假设原始为 uint8)
#
#         return dict(jpg=target, txt=prompt, hint=source)

import json
import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, json_path='/mnt/data/Sureaily/ControlNet_datasets/prompt_test.json'):
        self.data = []
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/mnt/data/Sureaily/ControlNet_datasets/' + source_filename)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.imread('/mnt/data/Sureaily/ControlNet_datasets/' + target_filename, cv2.IMREAD_UNCHANGED)

        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_LINEAR)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(
            jpg=target,
            txt=prompt,
            hint=source,
            source=source_filename,  # ★ 新增：原始文件路径（或仅文件名）
            target_path=target_filename  # 若需要也可加上
        )