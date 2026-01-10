import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心数据生成
# ==========================================
N = 8
num_pulses = 18
np.random.seed(42) # 固定种子

# 初始化两个全零矩阵
z1 = np.zeros((N, N))

# --- 关键：先随机确定“位置” (坐标) ---
# 这些坐标在两张图中必须完全一样
idx_x = np.random.randint(0, N, num_pulses)
idx_y = np.random.randint(0, N, num_pulses)

# --- 为图 A 赋值 ---
# 生成一组 15.0 ~ 20.0 的高数值 (显示为红/深橙色)
amps_A = np.random.uniform(15.0, 20.0, num_pulses)
z1[idx_x, idx_y] = amps_A

# 添加底噪 (不影响激活位置的视觉判断)
noise = np.random.normal(0, 0.5, z1.shape)
z1_final = z1 + noise

# ==========================================
# 2. 绘制图 A
# ==========================================
fig, ax = plt.subplots(figsize=(6, 6))

# 制作白底掩码 (隐藏噪声)
threshold = 5.0
z1_masked = np.ma.masked_where(z1_final < threshold, z1_final)

# 绘图
ax.imshow(z1_masked.T, cmap='jet', origin='lower', extent=[0, N, 0, N], vmin=0, vmax=20)

# 网格与样式设置
ax.set_xticks(np.arange(0, N + 1, 1))
ax.set_yticks(np.arange(0, N + 1, 1))
ax.grid(which='major', color='lightgray', linestyle='-', linewidth=1)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

#plt.title("Figure A: Original State")
plt.tight_layout()
plt.show()

# ==========================================
# 3. 生成图 B 数据 (基于图 A 修改)
# ==========================================
z2 = z1.copy() # 先复制，保证左半边和位置完全一样

# --- 关键逻辑：修改右半部分的“数值” ---
# 我们遍历之前的那些脉冲坐标
for i in range(num_pulses):
    current_x = idx_x[i]
    current_y = idx_y[i]
    
    # 如果脉冲位于右半边 (x >= 4)
    # 注意：因为绘图时用了 .T (转置)，这里的 x 对应屏幕上的横轴
    if current_x >= N // 2:
        # 修改数值：变成 5.0 ~ 8.0 (显示为蓝/青色)
        # 这预示着数值变小，或者符号发生了翻转
        new_val = np.random.uniform(5.0, 8.0)
        z2[current_x, current_y] = new_val

# 添加同样的噪声分布，保持背景质感一致
z2_final = z2 + noise

# ==========================================
# 4. 绘制图 B
# ==========================================
fig, ax = plt.subplots(figsize=(6, 6))

# 制作白底掩码
z2_masked = np.ma.masked_where(z2_final < threshold, z2_final)

# 绘图 (vmin/vmax 必须与图 A 保持一致，这样颜色变化才有意义)
ax.imshow(z2_masked.T, cmap='jet', origin='lower', extent=[0, N, 0, N], vmin=0, vmax=20)

# 网格与样式设置 (同上)
ax.set_xticks(np.arange(0, N + 1, 1))
ax.set_yticks(np.arange(0, N + 1, 1))
ax.grid(which='major', color='lightgray', linestyle='-', linewidth=1)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# (可选) 加一条红线辅助看分界
# ax.axvline(x=N//2, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
