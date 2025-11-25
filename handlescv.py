import numpy as np
import csv
from pathlib import Path

# ======== 路径设置 ========
input_path = Path("/Users/runqingyang/Downloads/blenshaperuthtalk3.csv")

# 输出文件
interp_output_path = input_path.with_name("blenshaperuthtalk3_1000fps.csv")
viseme_output_path = input_path.with_name("blenshaperuthtalk3_visemes.csv")

# ======== 参数：原始 & 目标帧率 ========
orig_fps = 60.0
target_fps = 1000.0

# ======== 读取原始 CSV ========
data = np.loadtxt(input_path, delimiter=",")   # 若有表头：加 skiprows=1

if data.shape[1] < 68:
    raise ValueError(f"CSV 只有 {data.shape[1]} 列，少于 68 列。")

# 只取前 61 维做插值
orig_61 = data[:, :61]
num_orig_frames = orig_61.shape[0]

print(f"原始帧数: {num_orig_frames}, 每帧 61 维。")

# ======== 构造时间轴并线性插值到 1000fps ========
t_orig = np.arange(num_orig_frames) / orig_fps
t_end = t_orig[-1]

num_new_frames = int(round(t_end * target_fps)) + 1
t_new = np.arange(num_new_frames) / target_fps

print(f"插值后帧数: {num_new_frames} (约 {t_end:.3f} 秒 @ {target_fps} fps)")

# 逐维插值
new_61 = np.zeros((num_new_frames, 61), dtype=np.float64)
for j in range(61):
    new_61[:, j] = np.interp(t_new, t_orig, orig_61[:, j])

# ======== 合成 68 维 ========
zeros_7 = np.zeros((num_new_frames, 7), dtype=np.float64)
new_68 = np.hstack([new_61, zeros_7])

# ======== 保存插值后的最终 CSV ========
np.savetxt(interp_output_path, new_68, delimiter=",", fmt="%.6f")
print(f"已保存 1000fps CSV 到: {interp_output_path}")

# ======== 自动取整：区间中点 => label ========

pairs = [
    ((187+237)/2, "S"),
    ((237+312)/2, "i"),
    ((312+375)/2, "p"),
    ((375+450)/2, "u"),
    ((500+562)/2, "t"),
    #((562+652)/2, "@"),
    #((1125+1150)/2, "@"),
    ((2582+2582)/2, "@"),
    ((712+750)/2, "s"),
    ((837+937)/2, "O"),
    ((1125+1150)/2, "E"),
    ((1250+1375)/2, "o"),
    ((1375+1487)/2, "k"),
    ((1775+1875)/2, "a"),
    ((2037+2087)/2, "T"),
    ((2387+2462)/2, "f"),
    ((5149+5149)/2, "sil"),
    ((3387+3450)/2, "r"),
    ((4562+4625)/2, "e"),
]

# 四舍五入成最终整数行号（1-based）
row_label_map_1based = { int(round(pos)): label for pos, label in pairs }

print("\n最终生成的 row_label_map_1based：")
for k,v in row_label_map_1based.items():
    print(k, "→", v)

# ======== 标注帧 ========
frame_labels = [""] * num_new_frames

for row_1based, label in row_label_map_1based.items():
    idx = int(row_1based) - 1  # convert to 0-based
    if 0 <= idx < num_new_frames:
        frame_labels[idx] = label
    else:
        print(f"⚠️ 警告: 行 {row_1based} 超出范围（总帧数 {num_new_frames}），跳过。")

# ======== lips 顺序 ========
lips_order = [
    "p", "t", "S", "T", "f", "k",
    "i", "r", "s", "u", "@", "a",
    "e", "E", "o", "O", "sil"
]

# ======== 写入第二个 CSV：包含 label + frame_index + 68 个值 ========
with open(viseme_output_path, "w", newline="") as f:
    writer = csv.writer(f)

    # header
    header = ["label", "frame_index"] + [f"v{i+1}" for i in range(68)]
    writer.writerow(header)

    for label in lips_order:
        for idx, lab in enumerate(frame_labels):
            if lab == label:
                # 写 label、1-based 行号、该行 68 个值
                row = [label, idx + 1] + new_68[idx].tolist()
                writer.writerow(row)

print(f"\n已保存 viseme 68维值 CSV 到: {viseme_output_path}")