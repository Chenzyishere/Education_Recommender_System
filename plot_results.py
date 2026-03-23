import matplotlib.pyplot as plt

# 1. 填入你刚才运行出的真实 Loss 数据
epochs = list(range(1, 11))
loss_values = [
    0.6466, 0.6211, 0.6072, 0.5963, 0.5867,
    0.5764, 0.5638, 0.5553, 0.5452, 0.5382
]

# 2. 设置绘图风格
plt.figure(figsize=(9, 6)) # 设置画布大小
plt.plot(epochs, loss_values,
         color='#1f77b4',      # 经典学术蓝
         marker='o',           # 圆点标记
         linestyle='-',        # 实线
         linewidth=2,          # 线宽
         markersize=6,         # 点的大小
         label='Training Loss')

# 3. 装饰图表（论文加分项）
plt.title('SAKT Model Training Loss Convergence', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.xticks(epochs)             # 确保横坐标显示 1-10 的整数
plt.grid(True, linestyle='--', alpha=0.7) # 添加虚线网格，方便观察数值
plt.legend(loc='upper right', fontsize=11)

# 4. 在最后一个点标注数值（体现精确性）
plt.annotate(f'{loss_values[-1]:.4f}',
             xy=(10, loss_values[-1]),
             xytext=(9.5, loss_values[-1] + 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

# 5. 保存并显示
plt.tight_layout() # 自动调整布局，防止标签被遮挡
plt.savefig('data/training_loss_curve.png', dpi=300) # 保存 300dpi 高清图，直接打印不模糊
print("✅ 论文配图已生成：data/training_loss_curve.png")
plt.show()