import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ======== 文件路径 ========
compressed_path = Path("/Users/runqingyang/Downloads/blend_from_sequence_compressed.csv")
original_path   = Path("/Users/runqingyang/Downloads/blenshaperuthtalk3.csv")

# ======== jawOpen 列号推断函数（如果没有 header 就用第 18 列 index=17） ========
def find_jawopen_column(path):
    with open(path, "r") as f:
        first_line = f.readline().strip()

    # 判断是否是纯数字 → 无表头
    if not any(c.isalpha() for c in first_line):
        print(f"{path.name} 没有 header，使用默认 jawOpen index=17")
        return 17

    # 有 header 的情况
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for idx, col in enumerate(header):
            if col.lower() == "jawopen":
                print(f"{path.name} 找到 jawOpen 列 index={idx}")
                return idx

    raise ValueError(f"{path.name} 有 header，但是没有找到 jawOpen 列")

# jawOpen 列号
col_compressed = find_jawopen_column(compressed_path)
col_original   = find_jawopen_column(original_path)

# ======== 加载 jawOpen ========
def load_jawopen(path, col):
    data = np.loadtxt(path, delimiter=",")
    if col >= data.shape[1]:
        raise ValueError(f"{path.name}: jawOpen 列 {col} 超出范围")
    return data[:, col]

jaw_compressed = load_jawopen(compressed_path, col_compressed)
jaw_original   = load_jawopen(original_path, col_original)

# ======== 绘图 ========
plt.figure(figsize=(12, 5))

plt.plot(jaw_original, label="Original jawOpen (blenshaperuthtalk3.csv)", alpha=0.8)
plt.plot(jaw_compressed, label="Compressed jawOpen (~51 frames)", marker='o')

plt.ylim(0, 1)  # 你说 jawOpen 范围应该是 0–1
plt.xlabel("Frame Index")
plt.ylabel("jawOpen Value")
plt.title("jawOpen Comparison: Original vs Compressed")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()