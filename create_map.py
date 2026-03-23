import pandas as pd

# 构造你的知识地图数据
data = {
    'p_id': [11, 47, 70, 311, 277],
    'name': ['整数四则运算', '分数的定义', '简单方程求解', '线性函数与图像', '几何图形属性'],
    'pre_id': [0,11,11,70,47],
    'description': ['基础数学技能', '需要整数基础', '需要运算基础', '需要方程基础', '结合分数与比例']
}

df = pd.DataFrame(data)

# 保存到 data 文件夹
# 如果提示缺少 openpyxl，记得执行: pip install openpyxl
try:
    df.to_excel('data/knowledge_map.xlsx', index=False)
    print("成功生成 data/knowledge_map.xlsx！现在去运行 recommend.py 吧。")
except Exception as e:
    print(f"生成失败，请检查 data 文件夹是否存在。错误信息: {e}")