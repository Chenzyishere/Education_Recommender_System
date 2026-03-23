import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # 确保文件开头 import 了 numpy
from torch.utils.data import DataLoader, TensorDataset
from data_preprocess import prepare_sequences  # 导入你之前的预处理函数
from model import SAKTModel  # 导入你的模型结构



# 1. 超参数设置（实验配置）
BATCH_SIZE = 32  # 每次喂给模型多少个学生的数据
MAX_SEQ = 50  # 序列长度
EPOCHS = 10  # 训练多少轮
LEARNING_RATE = 0.001  # 学习率，决定模型“学习”的速度


def train():
    # 2. 准备数据
    print("正在加载并预处理数据...")
    sequences, n_skills = prepare_sequences('data/raw_data.csv', max_seq=MAX_SEQ)

    # 将 list 转换为 PyTorch 能识别的 Tensor 格式
    # s_data: 知识点序列, a_data: 答题结果序列
    # 先转成 numpy 数组，再转成 Tensor，这样速度更快，也不会出警告
    s_data = torch.LongTensor(np.array([s for s, a in sequences]))
    a_data = torch.FloatTensor(np.array([a for s, a in sequences]))

    dataset = TensorDataset(s_data, a_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    model = SAKTModel(n_skills=n_skills, max_seq=MAX_SEQ).to(device)
    criterion = nn.BCELoss()  # 二分类交叉熵损失，适合预测 0 或 1
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    print("开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_s, batch_a in loader:
            batch_s, batch_a = batch_s.to(device), batch_a.to(device)

            # 前向传播：模型预测结果
            optimizer.zero_grad()
            output = model(batch_s)

            # 计算损失（只计算有效题目，排除补齐的 -1）
            mask = batch_a != -1
            loss = criterion(output[mask], batch_a[mask])

            # 反向传播：优化模型参数
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(loader):.4f}")

    # 5. 保存模型权重（这非常重要，以后做推荐直接调用这个文件）
    torch.save(model.state_dict(), 'data/sakt_model.pth')
    print("训练完成！模型已保存至 data/sakt_model.pth")


if __name__ == "__main__":
    train()