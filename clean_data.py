import pandas as pd

# 1. 读取原始文件
# 2009版数据通常很大，且编码多为 ISO-8859-1
raw_file = 'data/skill_builder_data.csv'
print("正在读取 ASSISTments 2009 数据集...")

try:
    # 只读取核心三列：学生ID、知识点ID、是否正确
    # 注意：2009版中 correct 这一列有时叫 'correct'，有时需要根据 'order_id' 排序
    df = pd.read_csv(raw_file, encoding='ISO-8859-1', low_memory=False)

    # 2. 筛选核心列 (确保列名准确)
    # 如果下载的文件列名有细微差别，请检查 CSV 第一行
    keep_columns = ['user_id', 'skill_id', 'correct']
    df = df[keep_columns]

    # 3. 彻底清洗
    # 去掉任何一列有缺失值的行
    df = df.dropna()

    # 过滤掉 correct 里面不是 0 或 1 的异常值 (如果有)
    df = df[df['correct'].isin([0, 1])]

    # 转换类型为整数，减小内存占用
    df['user_id'] = df['user_id'].astype(int)
    df['skill_id'] = df['skill_id'].astype(int)
    df['correct'] = df['correct'].astype(int)

    # 4. 关键：按 user_id 排序，保证时间序列的逻辑正确
    df = df.sort_values(by=['user_id'])

    # 5. 保存为项目通用的数据格式
    df.to_csv('data/raw_data.csv', index=False)

    print("-" * 30)
    print(f"清洗成功！")
    print(f"总记录数: {len(df)} 条")
    print(f"独立学生数: {df['user_id'].nunique()} 人")
    print(f"知识点总数: {df['skill_id'].nunique()} 个")
    print("-" * 30)
    print("现在可以去运行 train.py 进行正式训练了。")

except Exception as e:
    print(f"读取失败，请检查文件名或列名。错误信息: {e}")