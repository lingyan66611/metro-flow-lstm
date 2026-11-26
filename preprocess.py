# -*- coding: utf-8 -*-
"""清洗：把 data/archive_5min.csv 变成可直接建模的整洁序列。

做三件事（对应实现方案第 4 章）：
1. 识别缺失日：某站某天 209 格进站+出站全是 0，说明这天该站没有记录，用同期均值补；
2. 分时段异常截断：同一站同一时刻，跨天用 IQR 判异常，超过 Q3 + IQR_K×IQR 的截到上界；
3. 写清账单 data/clean_5min.csv 和报告 results/clean_report.json。

用法：
    python preprocess.py                 # 全量 80 站
    python preprocess.py --station 15    # 只处理一个站，快一些
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C


def load_archive():
    """读 5 min 长表；日期与桶拼成真实时间戳，便于排序和算特征。"""
    df = pd.read_csv(C.ARCHIVE_TABLE, encoding="utf-8")
    df["ts"] = pd.to_datetime(df["date"] + " " + df["bucket"], format="%Y-%m-%d %H:%M")   # 该格的起始时刻
    df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)
    return df


def fill_missing_days(df):
    """整天全零 = 缺失，用该站同格在其它天的均值补上（没有记录不等于没客流）。"""
    total = df["inflow"] + df["outflow"]
    zero_day = total.groupby([df["station_id"], df["date"]]).sum()
    zero_days = zero_day[zero_day == 0]
    df["is_filled"] = False
    if zero_days.empty:
        return df, []

    filled_list = []
    for (station, date) in zero_days.index:
        mask_other = (df["station_id"] == station) & (df["date"] != date)     # 该站其它天
        ref = (df[mask_other].groupby("bucket")[["inflow", "outflow"]].mean()
               .round().astype(int))                                          # 同格同期均值
        mask_cur = (df["station_id"] == station) & (df["date"] == date)
        cur = df.loc[mask_cur].set_index("bucket")
        df.loc[mask_cur, "inflow"] = cur.index.map(ref["inflow"]).astype(int)
        df.loc[mask_cur, "outflow"] = cur.index.map(ref["outflow"]).astype(int)
        df.loc[mask_cur, "is_filled"] = True
        filled_list.append({"station": int(station), "date": date,
                            "cells": int(mask_cur.sum())})
    return df, filled_list


def clip_outliers(df):
    """分时段 IQR：同一站同一时刻在 25 天里的分布，超过上界的截断。"""
    stats = (df.groupby(["station_id", "bucket"])["inflow"]
             .quantile([0.25, 0.75]).unstack())          # Q1、Q3
    q1 = stats[0.25].rename("q1")
    q3 = stats[0.75].rename("q3")
    q1_out = (df.groupby(["station_id", "bucket"])["outflow"]
              .quantile([0.25, 0.75]).unstack())
    bounds = pd.DataFrame({"in_hi": np.floor(q3 + C.IQR_K * (q3 - q1)),
                           "out_hi": np.floor(q1_out[0.75]
                                              + C.IQR_K * (q1_out[0.75] - q1_out[0.25]))})
    df = df.join(bounds, on=["station_id", "bucket"])

    over_in = df["inflow"] > df["in_hi"]
    over_out = df["outflow"] > df["out_hi"]
    df["is_clipped"] = over_in | over_out
    df.loc[over_in, "inflow"] = df.loc[over_in, "in_hi"].astype(int)     # 截到上界，不删点
    df.loc[over_out, "outflow"] = df.loc[over_out, "out_hi"].astype(int)
    return df.drop(columns=["in_hi", "out_hi"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", type=int, default=None, help="只处理指定站号")
    args = ap.parse_args(argv)

    df = load_archive()
    if args.station is not None:
        df = df[df["station_id"] == args.station].copy()      # 只留一个站
        df = df.drop(columns=["in_hi", "out_hi"], errors="ignore")

    df, filled = fill_missing_days(df)
    df = clip_outliers(df)

    out = df[["date", "station_id", "bucket", "inflow", "outflow",
              "raw_count", "is_filled", "is_clipped"]].sort_values(["date", "station_id", "bucket"])
    out_path = C.CLEAN_TABLE if args.station is None else C.DATA_DIR / ("clean_station%s.csv" % args.station)
    out.to_csv(out_path, index=False, encoding="utf-8")

    zero_rate = float((df["inflow"] + df["outflow"] == 0).mean())     # 零值格占比，判断稀疏程度
    report = {
        "rows": int(len(out)),
        "stations": int(out["station_id"].nunique()),
        "days": int(out["date"].nunique()),
        "zero_cell_rate": round(zero_rate, 4),
        "filled_days": filled,
        "filled_cells": int(df["is_filled"].sum()),
        "clipped_cells": int(df["is_clipped"].sum()),
        "inflow_total": int(df["inflow"].sum()),
        "outflow_total": int(df["outflow"].sum()),
    }
    (C.RESULT_DIR / "clean_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("清洗完成：%s（%d 行，%d 站，%d 天）" % (out_path, report["rows"], report["stations"], report["days"]))
    print("零值格 %.2f%%，补齐缺失 %d 格（%d 个整日），异常截断 %d 格"
          % (zero_rate * 100, report["filled_cells"], len(filled), report["clipped_cells"]))
    print("报告：results/clean_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
