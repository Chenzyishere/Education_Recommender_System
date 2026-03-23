import pandas as pd
df = pd.read_csv('data/raw_data.csv')
print(df['skill_id'].value_counts().head(10)) # 查看出现次数最多的10个知识点ID