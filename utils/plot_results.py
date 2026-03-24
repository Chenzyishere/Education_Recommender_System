import matplotlib.pyplot as plt
import numpy as np


def plot_paper_figure():
    # 使用你刚才跑出的最新真实数据
    models = ['PureCF', 'DKT', 'SAKT', 'KG-SAKT']
    auc_scores = [0.5874, 0.8458, 0.8198, 0.8122]
    logic_scores = [42.5, 82.22, 77.49, 98.83]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # 绘制 AUC 柱状图 (左轴)
    bars = ax1.bar(x, auc_scores, width, color='#3498db', alpha=0.7, label='Best AUC', edgecolor='black', linewidth=1)
    ax1.set_ylabel('Best AUC (Higher is better)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)

    # 在柱状图上方标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{height:.4f}', ha='center', va='bottom')

    # 绘制 Logic Score 折线图 (右轴)
    ax2 = ax1.twinx()
    ax2.plot(x, logic_scores, color='#e74c3c', marker='D', markersize=10, linewidth=3, label='Logic Score (%)')
    ax2.set_ylabel('Logic Score (%) (Logical Consistency)', color='#e74c3c', fontsize=12, fontweight='bold')
    ax2.set_ylim(30, 110)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # 标注折线图数值
    for i, score in enumerate(logic_scores):
        ax2.text(i, score + 3, f'{score}%', color='#e74c3c', ha='center', fontweight='bold')

    plt.title('Comparison of Models on Accuracy and Logical Consistency', fontsize=14, pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig('experimental_results_comparison.png')
    print("🎨 论文级对比图已生成：experimental_results_comparison.png")


if __name__ == "__main__":
    plot_paper_figure()