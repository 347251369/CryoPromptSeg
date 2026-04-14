import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import sys
import io

# 修复中文输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# <--- [修改] 全局字体配置：改为学术常用的 Serif 字体 (Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun'] # SimSun 用于兼容可能的中文，纯英文可只留 Times New Roman
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12  # <--- [修改] 设置基础字体大小

# 原始数据
data = np.array([
    [10028, -9.44,  224, 5.50,  0.819],
    [10081, -20.33, 154, 9.66,  0.847],
    [10345, -16.83, 149, 3.97,  0.727],
    [10532, -16.52, 174, 18.32, 0.729],
    [11056, -33.89, 164, 18.37, 0.748], 
    [10093, -21.71, 172, 14.08, 0.601],
    [10017, -14.46, 108, 36.75, 0.727]
])

# 提取特征
features = data[:, 1:4]
f1_scores = data[:, 4]

# 数据预处理
processed_features = features.copy()
processed_features[:, 0] = -features[:, 0]  # SNR 取反
processed_features[:, 1] = -features[:, 1]  # Diameter 取反

# 归一化
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(processed_features)

# 计算难度指数
difficulty = np.sqrt(np.sum(normalized_data**2, axis=1))

# ========== 优化球体大小映射 ==========
diff_norm = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min())
min_size = 100
max_size = 2000
sizes = min_size + diff_norm * (max_size - min_size)

# ========== 创建学术风格图表 ==========
fig = plt.figure(figsize=(10, 8)) # <--- [修改] 显式指定 dpi，保证保存质量
ax = fig.add_subplot(111, projection='3d')

# 背景设置
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 网格线
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')

# 配色方案
scatter = ax.scatter(
    normalized_data[:, 0],
    normalized_data[:, 1],
    normalized_data[:, 2],
    s=sizes,
    c=f1_scores,
    cmap='RdYlGn',
    vmin=0.5,
    vmax=0.9,
    alpha=0.9,
    edgecolors='black',
    linewidth=1,
    depthshade=True
)

#标签放在球体右边，注意不要阻挡
for i, row in enumerate(data):
    # 1. 计算基于球体大小的偏移量
    # sizes[i] 是面积，半径 r ~ sqrt(sizes[i])
    # 我们主要向 X 轴正方向（右侧）偏移
    radius_factor = np.sqrt(sizes[i]) * 0.0025 
    
    # 2. 定义偏移量
    x_offset = 0.03 + radius_factor  # 向右偏移基础值 + 半径补偿
    z_offset = 0.01                  # 轻微向上偏移，避免与球体底部视觉重叠
    
    ax.text(
        normalized_data[i, 0] + x_offset, # X轴向右移
        normalized_data[i, 1],            # Y轴不变
        normalized_data[i, 2] + z_offset, # Z轴轻微上移
        f'{int(row[0])}',
        fontsize=11,
        ha='left',      # 【关键】左对齐，让文字从偏移点向右生长，避开球体
        va='center',    # 垂直居中
        color='#333333',
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1) # 【可选】添加半透明白色背景，进一步防止线条穿过文字
    )

# <--- [修改] 坐标轴标签设置：增大字号，增加间距，统一字体
ax.set_xlabel('SNR', fontsize=12, labelpad=10, fontname='Times New Roman')
ax.set_ylabel('Diameter', fontsize=12, labelpad=10, fontname='Times New Roman')
ax.set_zlabel('Density', fontsize=12, labelpad=6, fontname='Times New Roman')

# <--- [修改] 标题设置
ax.set_title('Task Difficulty vs. F1 Performance', fontsize=16, pad=25, fontname='Times New Roman', fontweight='bold')

# <--- [修改] 颜色条美化
cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1, aspect=15)
cbar.set_label('F1-Score', fontsize=12, labelpad=10,fontname='Times New Roman')
# <--- [修改] 颜色条刻度：方向向内，加粗
cbar.ax.tick_params(direction='out', length=6, width=1.5, labelsize=12)

# ========== 优化视角和坐标轴范围 ==========
# 修改坐标轴范围，留出负空间让原点不在角落
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)

# 调整视角（关键修改）
ax.view_init(elev=25, azim=60)  # 改为 125 度，原点会显示在中心偏右

# 设置等比例坐标轴
ax.set_box_aspect([1, 1, 1])

# 原点标记（更明显）
ax.scatter(0, 0, 0, c='darkred', s=50, marker='o', 
           alpha=0.8, edgecolors='black', linewidth=1,
           label='Origin (0,0,0)', zorder=10)

# 添加从原点到各点的虚线（可选，增强视觉效果）
for i in range(len(normalized_data)):
    ax.plot(
        [0, normalized_data[i, 0]],
        [0, normalized_data[i, 1]],
        [0, normalized_data[i, 2]],
        linestyle='--',
        linewidth=0.5,
        color='gray',
        alpha=0.3
    )

# 坐标轴面板颜色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')

# <--- [修改] 坐标轴刻度美化：关键！方向向内 (direction='in')，加粗，增大字号
ax.tick_params(axis='x', which='major', direction='in', length=6, width=1.5, labelsize=12, pad=5)
ax.tick_params(axis='y', which='major', direction='in', length=6, width=1.5, labelsize=12, pad=5)
ax.tick_params(axis='z', which='major', direction='in', length=6, width=1.5, labelsize=12, pad=5)

# <--- [修改] 图例美化
ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black', fancybox=False)

plt.tight_layout()

# 保存高清图
plt.savefig('task_difficulty_f1.png', dpi=300, bbox_inches='tight')
plt.savefig('task_difficulty_f1.pdf', bbox_inches='tight')

plt.show()

# 打印统计
print("\n" + "=" * 80)
print("Task Difficulty and F1 Performance Statistics")
print("=" * 80)
print(f"{'Rank':<6}│{'SN':<10}│{'Difficulty':<12}│{'F1 Score':<10}│{'Size':<8}│{'Rating':<10}")
print("-" * 80)

ranked_indices = np.argsort(-difficulty)
for rank, idx in enumerate(ranked_indices, 1):
    if f1_scores[idx] >= 0.8:
        rating = "Excellent"
    elif f1_scores[idx] >= 0.7:
        rating = "Good"
    elif f1_scores[idx] >= 0.6:
        rating = "Fair"
    else:
        rating = "Poor"
    print(f"{rank:<6}│{int(data[idx, 0]):<10}│{difficulty[idx]:<12.4f}│{f1_scores[idx]:<10.3f}│{sizes[idx]:<8.0f}│{rating:<10}")

# 【新增】打印具体距离大小（未排序原始顺序）
print("\nDetailed Difficulty Distances (Original Order):")
print("-" * 40)
for i in range(len(data)):
    sn = int(data[i, 0])
    diff_val = difficulty[i]
    print(f"SN {sn}: Difficulty Distance = {diff_val:.4f}")

# 【新增】或者打印归一化后的坐标距离原点的欧氏距离验证
print("\nVerification (Euclidean Distance from Origin):")
print("-" * 40)
for i in range(len(normalized_data)):
    sn = int(data[i, 0])
    # 计算欧氏距离: sqrt(x^2 + y^2 + z^2)
    dist = np.linalg.norm(normalized_data[i])
    print(f"SN {sn}: Calculated Dist = {dist:.4f} | Stored Difficulty = {difficulty[i]:.4f}")

print("=" * 80)
print(f"F1: mean={np.mean(f1_scores):.3f}, max={np.max(f1_scores):.3f}, min={np.min(f1_scores):.3f}")
print(f"Difficulty: mean={np.mean(difficulty):.3f}, max={np.max(difficulty):.3f}, min={np.min(difficulty):.3f}")
print(f"Sphere Size: min={sizes.min():.0f}, max={sizes.max():.0f}")
print("=" * 80 + "\n")