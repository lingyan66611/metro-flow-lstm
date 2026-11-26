# -*- coding: utf-8 -*-
"""特征工程：把单站清账单变成“每个时间片一行”的特征表。

特征（对应实现方案第 6 章）：
    客流本身：inflow、outflow
    时间位置：slot（当天第几格，0—208）、sin/cos（把一天编码成环形，避免首尾断裂）
    日历信息：星期几、是否周末、是否节假日

用法：
    python features.py               # 默认站 15（火车东站）
    python features.py --station 30
"""
import argparse
import sys

import numpy as np
import pandas as pd

import config as C

特征列 = ["inflow", "outflow", "slot", "sin_slot", "cos_slot",
          "dow", "is_weekend", "is_holiday"]


def build(station):
    """产出 (T, 8) 的特征矩阵和对应的日期、时刻索引。"""
    路径 = C.DATA_DIR / ("clean_station%s.csv" % station)
    if not 路径.exists():                                   # 没跑单站清洗就用全量清账单里切
        df = pd.read_csv(C.CLEAN_TABLE, encoding="utf-8")
        df = df[df["station_id"] == station]
    else:
        df = pd.read_csv(路径, encoding="utf-8")

    df = df.sort_values(["date", "bucket"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["slot"] = df.groupby("date").cumcount()              # 当天第几格：0—208
    assert df["slot"].max() == C.SLOT_PERIOD - 1, "每天格数不是 209，口径不一致"

    df["sin_slot"] = np.sin(2 * np.pi * df["slot"] / C.SLOT_PERIOD)   # 环形编码，23:20 与 06:00 相邻
    df["cos_slot"] = np.cos(2 * np.pi * df["slot"] / C.SLOT_PERIOD)
    df["dow"] = df["date"].dt.dayofweek                     # 0=周一 … 6=周日
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(C.HOLIDAYS).astype(int)

    out = df[特征列].astype(float)
    索引 = df[["date", "bucket"]]

    路径_out = C.DATA_DIR / ("features_station%s.csv" % station)
    pd.concat([索引, out], axis=1).to_csv(路径_out, index=False, encoding="utf-8")
    return out.values, 索引, 路径_out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", type=int, default=C.STATION_ID)
    args = ap.parse_args(argv)

    矩阵, 索引, 路径 = build(args.station)
    print("特征表：%s" % 路径)
    print("形状 %s，时间范围 %s — %s，每天 %d 格"
          % (矩阵.shape, 索引["date"].min().date(), 索引["date"].max().date(), C.SLOT_PERIOD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
