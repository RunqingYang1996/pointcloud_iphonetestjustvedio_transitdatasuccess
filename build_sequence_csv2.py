import numpy as np
import csv
from pathlib import Path
import math

# ======== 路径设置 ========
viseme_path = Path("/Users/runqingyang/Downloads/blenshaperuthtalk3_visemes.csv")
output_full_path = Path("/Users/runqingyang/Downloads/blend_from_sequence_full.csv")
output_compressed_path = Path("/Users/runqingyang/Downloads/blend_from_sequence_compressed.csv")

# ======== 从 viseme CSV 读取 label → 68 维向量 ========
viseme_vectors = {}

with open(viseme_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # label, frame_index, v1..v68
    for row in reader:
        if not row:
            continue
        label = row[0]
        vec = np.array(row[2:], dtype=float)
        if label not in viseme_vectors:
            viseme_vectors[label] = vec

print("找到的 labels:", viseme_vectors.keys())

# 我们需要的 labels:
required_labels = {"k", "a", "r", "i", "u", "sil"}
missing = required_labels - set(viseme_vectors.keys())
if missing:
    raise ValueError(f"viseme CSV 缺少这些 label: {missing}")

# ======== sequence 定义 ========
sequence = [
    (187, "S"),
    (237, "i"),
    (312, "p"),
    (375, "u"),
    (450, "S"),
    (500, "t"),
    (562, "@"),
    (712, "s"),
    (750, "p"),
    (837, "O"),
    (937, "t"),
    ##########
    (1037, "i"),
    (1125, "E"),
    (1150, "t"),
    (1250, "o"),
    (1375, "k"),
    (1462, "u"),
    (1562, "S"),
    (1625, "t"),
    (1762, "a"),
    (1837, "t"),
    (1875, "t"),
    ##########
    (1937, "u"),
    (1962, "T"),
    (2037, "@"),
    (2137, "s"),
    (2275, "o"),
    (2387, "f"),
    (2462, "@"),
    (2637, "sil"),
    (2787, "@"),
    (2887, "S"),
    (2950, "u"),
    ##########
    (3062, "r"),
    (3112, "i"),
    (3125, "k"),
    (3287, "E"),
    (3312, "f"),
    (3387, "r"),
    (3450, "i"),
    (3500, "u"),
    (3575, "E"),
    (3587, "t"),
    (3687, "T"),
    ##########
    (3737, "a"),
    (3750, "t"),
    (3825, "T"),
    (3887, "@"),
    (4000, "t"),
    (4037, "O"),
    (4187, "s"),
    (4250, "S"),
    (4287, "u"),
    (4312, "t"),
    (4437, "f"),
    ##########
    (4562, "e"),
    (4625, "t"),
    (4762, "s"),
    (4875, "u"),
    (4975, "t"),
    (5150, "sil"),
    #(9150, "sil"),
]
start_label = "sil"

# ======== 索引检查 ========
for (i1, _), (i2, _) in zip(sequence, sequence[1:]):
    if i2 <= i1:
        raise ValueError(f"sequence index 不是严格递增: {i1}->{i2}")

# ======== 总帧数 ========
last_idx = sequence[-1][0]
total_frames = last_idx + 1  # 0..837 共 838 行
dim = 68
print(f"生成完整序列: total={total_frames} 行, 每行 {dim} 维")

# ======== 输出矩阵 ========
out = np.zeros((total_frames, dim), dtype=np.float64)

# ======== 工具函数 ========
def fill_constant(start, end, vec):
    if end < start:
        return
    out[start:end+1, :] = vec

def fill_interp(start, end, vec_from, vec_to):
    if end < start:
        return
    if start == end:
        out[start] = vec_to
        return
    n = end - start
    for f in range(start, end+1):
        t = (f - start) / n
        out[f] = (1-t)*vec_from + t*vec_to
def fill_interp2(start, end, vec_from, vec_to, sharpness=12.0):
    """
    用 Sigmoid 进行平滑插值：
    s(t) = 1 / (1 + exp(-(t-0.5)*sharpness))
    sharpness 值越大，变化越陡峭。
    """
    if end < start:
        return
    if start == end:
        out[start] = vec_to
        return

    n = end - start
    for f in range(start, end + 1):
        t = (f - start) / n  # 0->1
        s = 1.0 / (1.0 + np.exp(-(t - 0.5) * sharpness))
        out[f] = (1 - s) * vec_from + s * vec_to
# ======== 计算每段中点 ========
midpoints = []
for (cur_idx, cur_label), (next_idx, _) in zip(sequence, sequence[1:]):
    m = int(round((cur_idx + next_idx) / 2))
    midpoints.append((m, cur_label))

print("\n=== midpoints ===")
for m,l in midpoints:
    print(m, "->", l)

# ======== 开始构造完整序列 ========

# (1) 0 ~ first_idx-1 = sil
first_idx, first_label = sequence[0]
sil_vec = viseme_vectors[start_label]
fill_constant(0, first_idx-1, sil_vec)

# (2) first_idx ~ m0  = sil → first_label
m0, lab0 = midpoints[0]
vec_to = viseme_vectors[lab0]
fill_interp(first_idx, m0, sil_vec, vec_to)

# (3) 中点之间逐段处理
for j in range(len(midpoints)-1):
    m_j, lab_j = midpoints[j]
    m_next, lab_next = midpoints[j+1]
    vec_from = viseme_vectors[lab_j]
    vec_to = viseme_vectors[lab_next]
    fill_interp(m_j+1, m_next, vec_from, vec_to)

# (4) 最后一段: m_last+1 ~ last_idx = last_label_j → final sequence label
m_last, lab_last = midpoints[-1]
last_idx, last_label = sequence[-1]
vec_from = viseme_vectors[lab_last]
vec_to = viseme_vectors[last_label]
fill_interp(m_last+1, last_idx, vec_from, vec_to)

# ======== 保存完整 CSV ========
np.savetxt(output_full_path, out, delimiter=",", fmt="%.6f")
print(f"\n✅ 已保存完整序列到: {output_full_path}")

# =====================================================================
# =====================   压 缩 操 作 （新增）  ========================
# =====================================================================

# 按比例压缩：838/100*6 = 50.28 → round → 50 或 51
target_frames = int(round(total_frames / 100 * 6))

print(f"\n压缩比例: {total_frames} → {target_frames} 行")

# 生成 boundaries（等分 total_frames）
boundaries = np.linspace(0, total_frames, num=target_frames+1, endpoint=True, dtype=int)

compressed = np.zeros((target_frames, dim), dtype=float)

for i in range(target_frames):
    start = boundaries[i]
    end = boundaries[i+1]  # [start, end)
    if end <= start:
        compressed[i] = out[min(start, total_frames-1)]
    else:
        compressed[i] = out[start:end, :].mean(axis=0)

# ======== 保存压缩后的 CSV ========
np.savetxt(output_compressed_path, compressed, delimiter=",", fmt="%.6f")

print(f"✅ 已保存压缩序列到: {output_compressed_path}")
print(f"   最终行数: {compressed.shape[0]}")