# -*- coding: utf-8 -*-
"""四个评估指标：MAE、RMSE、MAPE、R²（论文第 5 章口径）。"""
import numpy as np


def mae(真实, 预测):
    return float(np.mean(np.abs(真实 - 预测)))


def rmse(真实, 预测):
    return float(np.sqrt(np.mean((真实 - 预测) ** 2)))


def mape(真实, 预测, 阈值=1.0):
    """真实值太小的点会让百分比爆炸，只统计 |真实| ≥ 阈值 的样本。"""
    掩码 = np.abs(真实) >= 阈值
    if not np.any(掩码):
        return float("nan")
    return float(np.mean(np.abs((真实[掩码] - 预测[掩码]) / 真实[掩码])) * 100)


def r2(真实, 预测):
    总平方 = np.sum((真实 - 真实.mean()) ** 2)
    if 总平方 == 0:
        return float("nan")
    return float(1 - np.sum((真实 - 预测) ** 2) / 总平方)


def 全部(真实, 预测):
    """两个目标（进站、出站）分别算，再给平均。"""
    out = {}
    for i, 名称 in enumerate(["inflow", "outflow"]):
        out[名称] = {
            "MAE": round(mae(真实[:, i], 预测[:, i]), 4),
            "RMSE": round(rmse(真实[:, i], 预测[:, i]), 4),
            "MAPE": round(mape(真实[:, i], 预测[:, i]), 4),
            "R2": round(r2(真实[:, i], 预测[:, i]), 4),
        }
    out["avg"] = {k: round(float(np.mean([out["inflow"][k], out["outflow"][k]])), 4)
                  for k in ["MAE", "RMSE", "MAPE", "R2"]}
    return out
