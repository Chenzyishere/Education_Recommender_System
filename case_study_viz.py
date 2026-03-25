import matplotlib.pyplot as plt
import networkx as nx


def plot_case_study():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 定义知识点坐标 (模拟教材逻辑顺序)
    nodes = {
        "分数基础": (1, 1),
        "小数运算": (2, 2),
        "百分数转换": (3, 3),
        "复合利率应用": (4, 4)
    }

    # 2. 模拟三位学生的推荐路径
    # [起始点, DKT推荐, KG-SAKT推荐]
    cases = [
        {"name": "学生 A (薄弱型)", "start": "分数基础", "dkt": "百分数转换", "kg": "分数基础"},
        {"name": "学生 B (进阶型)", "start": "小数运算", "dkt": "复合利率应用", "kg": "百分数转换"},
        {"name": "学生 C (拔尖型)", "start": "百分数转换", "kg": "复合利率应用", "dkt": "复合利率应用"}
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    for i, case in enumerate(cases):
        ax = axes[i]
        G = nx.DiGraph()

        # 添加背景节点
        for node, pos in nodes.items():
            G.add_node(node, pos=pos)

        pos = nx.get_node_attributes(G, 'pos')

        # 绘制知识点背景（灰色）
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#ecf0f1', node_size=2000, edgecolors='#bdc3c7')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

        # 绘制 DKT 推荐路径 (红色虚线 - 代表跳跃或不合理)
        if case['dkt'] != case['start']:
            ax.annotate("", xy=pos[case['dkt']], xytext=pos[case['start']],
                        arrowprops=dict(arrowstyle="->", color="#e74c3c", ls="--", lw=2, connectionstyle="arc3,rad=0.3",
                                        label="DKT"))

        # 绘制 KG-SAKT 推荐路径 (绿色实线 - 代表逻辑合理)
        # 如果是强化练习，画一个自环或加粗
        if case['kg'] == case['start']:
            ax.scatter(*pos[case['start']], s=2500, facecolors='none', edgecolors='#2ecc71', linewidths=3)
        else:
            ax.annotate("", xy=pos[case['kg']], xytext=pos[case['start']],
                        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=3, label="KG-SAKT (Ours)"))

        ax.set_title(f"{case['name']}\n路径对比", fontsize=12, fontweight='bold')
        ax.axis('off')

    # 添加全局图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', lw=2, ls='--', label='DKT (存在逻辑跳跃)'),
        Line2D([0], [0], color='#2ecc71', lw=3, label='KG-SAKT (遵循先修逻辑)'),
        Line2D([0], [0], marker='o', color='w', label='知识点', markerfacecolor='#ecf0f1', markersize=10,
               markeredgecolor='#bdc3c7')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig('case_study_paths.png')
    print("✅ 案例分析路径图已生成：case_study_paths.png")


if __name__ == "__main__":
    plot_case_study()