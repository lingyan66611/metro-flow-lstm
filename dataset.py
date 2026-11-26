# -*- coding: utf-8 -*-
"""构造样本：12 个时间片 → 下一格的进站/出站。

关键点（对应实现方案第 6 章）：
1. 滑窗严格按时间顺序，不跨天打乱；
2. 归一化只用训练段统计量，避免把测试期的极值泄漏进模型；
3. 70/15/15 按时间切，前 70% 训练、接着 15% 验证、最后 15% 测试。

用法（其余脚本 import）：
    from dataset import prepare
    data = prepare(station=15)
"""
import numpy as np

import config as C
import features


def _windows(矩阵, lookback, horizon=1):
    """把 (T, F) 切成 (N, lookback, F) 的输入和 (N, horizon, 2) 的目标。"""
    一维 = 矩阵[:, :2]                                   # 只对进站、出站两个量做预测
    N = len(矩阵) - lookback - horizon + 1
    X = np.empty((N, lookback, 矩阵.shape[1]), dtype=np.float32)
    y = np.empty((N, horizon, 2), dtype=np.float32)
    for i in range(N):
        X[i] = 矩阵[i:i + lookback]                       # 过去 12 格的全部特征
        y[i] = 一维[i + lookback:i + lookback + horizon]   # 紧接其后的那一格
    return X, y


def prepare(station=None, lookback=None, verbose=True):
    station = station if station is not None else C.STATION_ID
    lookback = lookback or C.LOOKBACK

    矩阵, 索引, 路径 = features.build(station)           # (T, 8)
    N = len(矩阵) - lookback
    训练段 = int(N * C.TRAIN_RATIO)
    验证段 = int(N * C.VAL_RATIO)

    拟合行数 = 训练段 + lookback                          # 训练样本覆盖到的序列行数
    最小值 = 矩阵[:拟合行数, :2].min(axis=0)              # 只用训练段拟合 Min-Max
    最大值 = 矩阵[:拟合行数, :2].max(axis=0)
    跨度 = np.where(最大值 - 最小值 == 0, 1.0, 最大值 - 最小值)

    缩放后 = 矩阵.copy()
    缩放后[:, :2] = (矩阵[:, :2] - 最小值) / 跨度          # 只缩放两列客流，时间特征保持不变

    X, y = _windows(缩放后, lookback)
    分段 = {
        "train": slice(0, 训练段),
        "val": slice(训练段, 训练段 + 验证段),
        "test": slice(训练段 + 验证段, None),
    }
    data = {
        "station": station,
        "X_train": X[分段["train"]], "y_train": y[分段["train"]],
        "X_val": X[分段["val"]], "y_val": y[分段["val"]],
        "X_test": X[分段["test"]], "y_test": y[分段["test"]],
        "scaler": {"min": 最小值.tolist(), "range": 跨度.tolist()},
        "test_index": 索引.iloc[lookback + 训练段 + 验证段:].reset_index(drop=True),
        "train_index": 索引.iloc[lookback:lookback + 训练段].reset_index(drop=True),
        "feature_path": str(路径),
    }
    if verbose:
        print("站 %d：总序列 %d 格，滑窗后 %d 样本，训练/验证/测试 = %d / %d / %d"
              % (station, len(矩阵), N, 训练段, 验证段, N - 训练段 - 验证段))
    return data


def inverse(值, scaler):
    """把归一化后的预测值还原成真实客流。"""
    最小 = np.array(scaler["min"], dtype=np.float32)
    跨度 = np.array(scaler["range"], dtype=np.float32)
    return 值 * 跨度 + 最小


if __name__ == "__main__":
    prepare()
