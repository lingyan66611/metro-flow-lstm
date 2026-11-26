# -*- coding: utf-8 -*-
"""双层 LSTM：64 → 32，Dropout 0.1，早停 patience 15（对应实现方案第 7 章）。

结构：
    输入 (batch, 12, 8) → LSTM(64) → Dropout → LSTM(32) → 取最后时刻 → Linear(2)
输出：
    下一格的 进站、出站（归一化尺度）
"""
import time

import numpy as np
import torch
import torch.nn as nn

import config as C


class MetroLSTM(nn.Module):
    def __init__(self, n_feature, hidden1=None, hidden2=None, dropout=None):
        super().__init__()
        hidden1 = hidden1 or C.HIDDEN1
        hidden2 = hidden2 or C.HIDDEN2
        dropout = C.DROPOUT if dropout is None else dropout
        self.lstm1 = nn.LSTM(n_feature, hidden1, batch_first=True)    # 第一层：抓短期波动
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)      # 第二层：再抽象一层时序特征
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden2, 2)                             # 输出进站、出站两个数

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])          # 只取最后一格的状态做预测
        return self.head(out)


def train_model(data, epochs=None, patience=None, verbose=True):
    """训练并返回（最好的权重、训练曲线、最优验证损失）。"""
    epochs = epochs or C.EPOCHS
    patience = patience or C.PATIENCE
    torch.manual_seed(C.SEED)                    # 固定随机种子，结果可复现
    np.random.seed(C.SEED)

    X_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"][:, 0, :])      # horizon=1，取第 0 个
    X_val = torch.from_numpy(data["X_val"])
    y_val = torch.from_numpy(data["y_val"][:, 0, :])

    model = MetroLSTM(X_train.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    loss_fn = nn.MSELoss()

    最优损失 = float("inf")
    最优权重 = None
    等待 = 0
    曲线 = []

    for epoch in range(1, epochs + 1):
        model.train()
        乱序 = torch.randperm(len(X_train))                  # 每轮打乱训练样本
        累计 = 0.0
        for i in range(0, len(X_train), C.BATCH_SIZE):
            批 = 乱序[i:i + C.BATCH_SIZE]
            opt.zero_grad()
            预测 = model(X_train[批])
            损失 = loss_fn(预测, y_train[批])
            损失.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪，防止 LSTM 爆梯度
            opt.step()
            累计 += 损失.item() * len(批)
        训练损失 = 累计 / len(X_train)

        model.eval()
        with torch.no_grad():
            验证损失 = loss_fn(model(X_val), y_val).item()   # 用它决定是否早停

        曲线.append({"epoch": epoch, "train": round(训练损失, 6), "val": round(验证损失, 6)})
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print("  epoch %3d  训练 %.5f  验证 %.5f" % (epoch, 训练损失, 验证损失))

        if 验证损失 < 最优损失 - 1e-5:
            最优损失 = 验证损失
            最优权重 = {k: v.clone() for k, v in model.state_dict().items()}   # 存最好的一版
            等待 = 0
        else:
            等待 += 1
            if 等待 >= patience:
                if verbose:
                    print("  早停：验证损失连续 %d 轮没降，停在第 %d 轮" % (patience, epoch))
                break

    model.load_state_dict(最优权重)              # 回到验证最好的那轮
    return model, 曲线, 最优损失


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32))).numpy()
