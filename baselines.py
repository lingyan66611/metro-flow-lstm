# -*- coding: utf-8 -*-
"""三个对照模型：ARIMA(2,1,2)、SVR(RBF, C=100)、BP(64-32)。

三者与 LSTM 用同一套滑窗、同一套 70/15/15 划分、同一套指标（对应实现方案第 7 章）。
ARIMA 只用历史客流本身，SVR 与 BP 用与 LSTM 完全相同的 12×8 输入。

用法：
    python baselines.py --station 15
产物：
    results/baseline_metrics.json
"""
import argparse
import json
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

import config as C
import dataset
import features
import metrics


def 跑_arima(序列, 训练结束, 测试开始, 测试长度):
    """在训练段拟合一次，然后一步一更新地滚动预测（不重复拟合，速度快）。"""
    res = ARIMA(序列[:训练结束], order=(2, 1, 2)).fit()
    for i in range(测试开始 - 训练结束):
        res = res.append(np.array([序列[训练结束 + i]]), refit=False)   # 用已知真实值更新状态
    预测 = []
    for i in range(测试长度):
        预测.append(float(res.forecast(1)[0]))
        if i + 1 < 测试长度:
            res = res.append(np.array([序列[测试开始 + i]]), refit=False)
    return np.array(预测).reshape(-1, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", type=int, default=C.STATION_ID)
    args = ap.parse_args(argv)

    data = dataset.prepare(args.station)
    真实 = dataset.inverse(data["y_test"][:, 0, :], data["scaler"])

    训练窗口 = len(data["X_train"])
    验证窗口 = len(data["X_val"])
    测试窗口 = len(data["X_test"])
    训练结束 = 训练窗口 + C.LOOKBACK                 # 训练段覆盖到的序列行号
    测试开始 = 训练结束 + 验证窗口

    矩阵, _, _ = features.build(args.station)
    结果 = {}

    # ---------- ARIMA ----------
    print("ARIMA(2,1,2) 拟合中…")
    arima_pred = np.hstack([
        跑_arima(矩阵[:, 0], 训练结束, 测试开始, 测试窗口),   # 进站
        跑_arima(矩阵[:, 1], 训练结束, 测试开始, 测试窗口),   # 出站
    ])
    结果["ARIMA"] = metrics.全部(真实, arima_pred)

    # ---------- SVR ----------
    print("SVR(RBF, C=100) 拟合中…")
    X_train = data["X_train"].reshape(训练窗口, -1)          # 12×8 拉平成 96 维
    X_val = data["X_val"].reshape(验证窗口, -1)
    X_test = data["X_test"].reshape(测试窗口, -1)
    X_fit = np.vstack([X_train, X_val])                      # SVR 没有早停，验证段并进训练
    y_fit = np.vstack([data["y_train"][:, 0, :], data["y_val"][:, 0, :]])
    svr_pred = np.hstack([
        SVR(kernel="rbf", C=100).fit(X_fit, y_fit[:, i]).predict(X_test).reshape(-1, 1)
        for i in range(2)
    ])
    结果["SVR"] = metrics.全部(真实, dataset.inverse(svr_pred, data["scaler"]))

    # ---------- BP ----------
    print("BP(64-32) 拟合中…")
    bp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                      max_iter=2000, early_stopping=True, random_state=C.SEED)
    bp.fit(X_fit, y_fit)
    bp_pred = bp.predict(X_test).reshape(-1, 2)
    结果["BP"] = metrics.全部(真实, dataset.inverse(bp_pred, data["scaler"]))

    # ---------- 汇总，方便和 LSTM 一起看 ----------
    if C.RESULT_LSTM.exists():
        结果["LSTM"] = json.loads(C.RESULT_LSTM.read_text(encoding="utf-8"))
        结果["LSTM"] = {k: 结果["LSTM"][k] for k in ["inflow", "outflow", "avg"]}

    结果["station"] = args.station
    C.RESULT_BASE.write_text(json.dumps(结果, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 测试段对比（站 %d，平均两目标） ===" % args.station)
    print("%-8s %10s %10s %10s %8s" % ("模型", "MAE", "RMSE", "MAPE(%)", "R2"))
    for 名称 in ["LSTM", "ARIMA", "SVR", "BP"]:
        if 名称 in 结果:
            m = 结果[名称]["avg"]
            print("%-8s %10.2f %10.2f %10.2f %8.3f"
                  % (名称, m["MAE"], m["RMSE"], m["MAPE"], m["R2"]))
    print("结果：results/baseline_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
