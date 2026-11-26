# -*- coding: utf-8 -*-
"""训练主脚本：特征 → 滑窗 → 双层 LSTM → 指标与图表。

用法：
    python train.py                 # 默认站 15
    python train.py --station 30 --epochs 60
产物：
    results/lstm_metrics.json       四个指标
    results/lstm_predictions.csv    测试段逐格真实值 vs 预测值
    results/lstm_station15.pt       最优权重
    figures/lstm_loss.pdf、.svg      训练/验证损失曲线
    figures/lstm_test.pdf、.svg      测试段预测对比
    figures/lstm_error.pdf、.svg     残差分布
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")                        # 不弹窗，直接存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config as C
import dataset
import metrics
import model_lstm


def 存图(fig, 名称):
    """只存矢量图：PDF 用于论文排版（LaTeX 可直接插入），SVG 可直接插进 Word / PPT。

    不输出 PNG：位图在论文里缩放会糊，且期刊/学校通常要求 300 dpi 以上或直接要矢量。
    """
    for 格式 in C.FIGURE_FORMATS:
        路径 = C.FIGURE_DIR / ("%s.%s" % (名称, 格式))
        fig.savefig(路径, format=格式, bbox_inches="tight")   # 收紧空白边，避免插图时留大片白
    plt.close(fig)


def 画图(station, 曲线, 测试真实, 测试预测, 索引):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]   # 让中文标题正常显示
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42            # 把文字写成 TrueType 而不是 Type3，PDF 可正常检索/编辑

    fig, ax = plt.subplots(figsize=(6, 4))
    轮次 = [c["epoch"] for c in 曲线]
    ax.plot(轮次, [c["train"] for c in 曲线], label="训练")
    ax.plot(轮次, [c["val"] for c in 曲线], label="验证")
    ax.set_xlabel("轮次"); ax.set_ylabel("MSE（归一化）")
    ax.set_title("站 %d 训练曲线" % station)
    ax.legend(); fig.tight_layout()
    存图(fig, "lstm_loss")

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(测试真实))
    ax.plot(x, 测试真实[:, 0], label="真实进站", color="#333333", lw=0.9)
    ax.plot(x, 测试预测[:, 0], label="预测进站", color="#d62728", lw=0.9, alpha=0.85)
    ax.plot(x, 测试真实[:, 1], label="真实出站", color="#999999", lw=0.9)
    ax.plot(x, 测试预测[:, 1], label="预测出站", color="#1f77b4", lw=0.9, alpha=0.85)
    每天起点 = 索引.index[索引["bucket"] == "06:00"].tolist()   # 只在每天第一格标日期，避免标签重复
    if 每天起点:
        ax.set_xticks(每天起点)
        ax.set_xticklabels([str(索引.iloc[p]["date"])[5:10] for p in 每天起点])
    ax.set_xlabel("日期（测试段按天分段）"); ax.set_ylabel("5 min 客流（人次）")
    ax.set_title("站 %d 测试段预测对比" % station)
    ax.legend(ncol=4, fontsize=8); fig.tight_layout()
    存图(fig, "lstm_test")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(测试预测[:, 0] - 测试真实[:, 0], bins=40, color="#999999")
    ax.set_xlabel("残差（预测 - 真实）"); ax.set_ylabel("格数")
    ax.set_title("站 %d 进站残差分布" % station)
    fig.tight_layout()
    存图(fig, "lstm_error")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", type=int, default=C.STATION_ID)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    args = ap.parse_args(argv)

    data = dataset.prepare(args.station)
    print("开始训练：LSTM %d-%d，Dropout %.1f，batch %d，最多 %d 轮"
          % (C.HIDDEN1, C.HIDDEN2, C.DROPOUT, C.BATCH_SIZE, args.epochs))

    model, 曲线, 最优损失 = model_lstm.train_model(
        data, epochs=args.epochs, patience=args.patience)

    预测缩放 = model_lstm.predict(model, data["X_test"])
    预测 = dataset.inverse(预测缩放, data["scaler"])                 # 还原到人次
    真实 = dataset.inverse(data["y_test"][:, 0, :], data["scaler"])

    结果 = metrics.全部(真实, 预测)
    结果.update({
        "station": args.station,
        "samples": {"train": int(len(data["X_train"])),
                    "val": int(len(data["X_val"])),
                    "test": int(len(data["X_test"]))},
        "best_val_loss": round(float(最优损失), 6),
        "epochs_run": len(曲线),
        "scaler": data["scaler"],
    })
    C.RESULT_LSTM.write_text(json.dumps(结果, ensure_ascii=False, indent=2), encoding="utf-8")

    索引 = data["test_index"]
    pd.DataFrame({
        "date": 索引["date"].astype(str), "bucket": 索引["bucket"],
        "true_in": 真实[:, 0].round(1), "pred_in": 预测[:, 0].round(1),
        "true_out": 真实[:, 1].round(1), "pred_out": 预测[:, 1].round(1),
    }).to_csv(C.RESULT_PRED, index=False, encoding="utf-8")

    torch.save(model.state_dict(), C.RESULT_DIR / ("lstm_station%d.pt" % args.station))
    画图(args.station, 曲线, 真实, 预测, 索引)

    print("\n=== 测试段指标（站 %d） ===" % args.station)
    for 目标 in ["inflow", "outflow", "avg"]:
        m = 结果[目标]
        print("%-8s MAE %8.2f  RMSE %8.2f  MAPE %6.2f%%  R2 %6.3f"
              % (目标, m["MAE"], m["RMSE"], m["MAPE"], m["R2"]))
    print("结果：results/lstm_metrics.json、lstm_predictions.csv，图在 figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
