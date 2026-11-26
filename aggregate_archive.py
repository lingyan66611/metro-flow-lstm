# -*- coding: utf-8 -*-
"""原始刷卡记录 → 5 min 进出站客流表。

数据源：archive/Metro_train/record_YYYY-MM-DD.csv（每行一位乘客刷卡，约 235 万条/天）
输出：data/days/<date>.csv（按天缓存）、data/archive_5min.csv（合并后的长表）

三条处理规则（对账后确定，不要随意改）：
1. 只统计运营时段 [06:00:00, 23:25:00)，时段外刷卡单独计数后丢弃；
2. status=1 记为进站，status=0 记为出站；
3. 按 5 分钟向下取整分桶，默认不去重（与 HZDT 完全一致）。

用法：
    python aggregate_archive.py              # 处理全部 25 天
    python aggregate_archive.py --days 2     # 只处理前两天，快速验证
    python aggregate_archive.py --force      # 忽略日缓存重新算
"""
import argparse
import csv
import sys
from collections import defaultdict

import config as C


def bucket_of(time_str):
    """'HH:MM:SS' → 该时刻所属 5 min 格的起始标签。

    例：14:37:20 归到 '14:35'；14:42:59 归到 '14:40'。
    """
    minute = int(time_str[3:5])
    return "%s:%02d" % (time_str[0:2], minute // C.FREQ_MIN * C.FREQ_MIN)


def aggregate_day(path, date_str):
    """把一天的刷卡记录压成 {(站点, 桶): [进站数, 出站数, 原始条数]}。"""
    cnt = defaultdict(lambda: [0, 0, 0])
    lines = defaultdict(set)                      # 记录每个站属于哪条线路
    seen = set() if C.DEDUPE else None            # 只有开启去重时才占用内存

    total_rows = outside = dup = 0
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header[:7] != ["time", "lineID", "stationID", "deviceID", "status", "userID", "payType"]:
            raise ValueError("字段与预期不符：%s" % header)      # 表头变了就立刻报错，避免静默算错
        for row in reader:
            total_rows += 1
            t = row[0][11:19]                              # "2019-01-01 06:00:05" 取时分秒部分
            if not (C.OP_START <= t < C.OP_END):           # 运营时段外（夜间工时、设备测试）
                outside += 1
                continue
            station = int(row[2])
            status = row[4]
            if C.DEDUPE:
                key = (row[0], row[2], row[3], row[5], status)   # 时间+站+设备+用户+状态 视为同一条
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)
            cell = (station, bucket_of(t))
            rec = cnt[cell]
            if status == "1":
                rec[0] += 1                                    # 进站 +1
            elif status == "0":
                rec[1] += 1                                    # 出站 +1
            rec[2] += 1                                        # 该格原始记录数，用来说明数据规模
            lines[station].add(row[1])

    stats = {
        "date": date_str,
        "rows": total_rows,
        "outside_window": outside,
        "duplicates_removed": dup,
        "cells": len(cnt),
        "total_in": sum(v[0] for v in cnt.values()),
        "total_out": sum(v[1] for v in cnt.values()),
    }
    return cnt, lines, stats


def write_day(cnt, date_str, out_path):
    """按完整网格写出（80 站 × 209 格）。

    没有刷卡记录的格子显式写 0，而不是省略这一行：
    否则下游滑窗会以为这天少了几片，导致时间轴错位。
    """
    buckets = C.day_buckets()
    zero = [0, 0, 0]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "station_id", "bucket", "inflow", "outflow", "raw_count"])
        for station in C.STATIONS:                     # 站号 0—80，跳过 54
            for b in buckets:                          # 06:00—23:20
                rec = cnt.get((station, b), zero)
                w.writerow([date_str, station, b, rec[0], rec[1], rec[2]])


def merge_days(days=None):
    """把按天缓存拼成一张总表，供对账和后续建模直接读取。"""
    out_path = C.ARCHIVE_TABLE
    rows = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["date", "station_id", "bucket", "inflow", "outflow", "raw_count"])
        for date in (days or C.TRAIN_DAYS):
            day_file = C.DAY_DIR / ("%s.csv" % date)
            if not day_file.exists():
                print("  [跳过] 缺少 %s" % day_file.name)
                continue
            with open(day_file, "r", encoding="utf-8", newline="") as fin:
                for i, line in enumerate(fin):
                    if i == 0:
                        continue                       # 跳过表头
                    fout.write(line)                   # 原样拼接，不做二次转换
                    rows += 1
    print("合并完成：%s（%d 行）" % (out_path, rows))
    return rows


def load_station_lines():
    """把上次写好的站点—线路对照读回来（日缓存里没有线路信息，不能拿日期覆盖它）。"""
    mapping = {}
    if not C.STATION_LINE_TABLE.exists():
        return mapping
    with open(C.STATION_LINE_TABLE, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[1]:
                continue
            ids = row[1].split("|")
            if all(len(x) <= 2 and x.isalnum() for x in ids):   # 线路号是 A/B/C 这类短代号
                mapping[int(row[0])] = set(ids)
    return mapping


def refill_station_lines(needed):
    """日缓存缺线路信息时，回原始记录补扫（只用到 stationID、lineID 两列）。"""
    print("[补算] 日缓存不含线路号，回原始记录补扫…")
    mapping = {}
    for date in C.TRAIN_DAYS:
        src = C.TRAIN_RECORD_DIR / ("record_%s.csv" % date)
        if not src.exists():
            continue
        with open(src, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                mapping.setdefault(int(row[2]), set()).add(row[1])
        if needed <= set(mapping):                     # 需要的站都补到了就提前停
            break
    return mapping


def write_station_lines(lines):
    with open(C.STATION_LINE_TABLE, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station_id", "line_ids"])
        for station in sorted(lines):
            w.writerow([station, "|".join(sorted(lines[station]))])   # 一个站可能被多条线共用


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="只处理前 N 天")
    ap.add_argument("--force", action="store_true", help="忽略日缓存，重新聚合")
    args = ap.parse_args(argv)

    days = C.TRAIN_DAYS[: args.days] if args.days else C.TRAIN_DAYS
    lines = defaultdict(set)
    seen_stations = set()                          # 记录到底有哪些站出现过（含缓存分支）
    known = load_station_lines()
    print("口径：%d min，%s—%s，去重=%s，共 %d 天"
          % (C.FREQ_MIN, C.OP_START[:5], C.OP_END[:5], C.DEDUPE, len(days)))

    for date in days:
        src = C.TRAIN_RECORD_DIR / ("record_%s.csv" % date)
        day_file = C.DAY_DIR / ("%s.csv" % date)
        if day_file.exists() and not args.force:
            print("[缓存] %s" % day_file.name)                 # 已算过，直接用
            with open(day_file, "r", encoding="utf-8", newline="") as fh:
                for i, row in enumerate(csv.reader(fh)):
                    if i and row:
                        seen_stations.add(int(row[1]))         # 只登记站号，线路栏在缓存里没有
            continue
        if not src.exists():
            print("[缺失] %s" % src)
            continue
        cnt, day_lines, stats = aggregate_day(src, date)
        write_day(cnt, date, day_file)
        for station, ids in day_lines.items():
            lines[station] |= ids
            seen_stations.add(station)
        print("[完成] %s 原始 %d 条，时段外 %d 条，去重 %d 条，%d 格，进站 %d 出站 %d"
              % (date, stats["rows"], stats["outside_window"],
                 stats["duplicates_removed"], stats["cells"],
                 stats["total_in"], stats["total_out"]))

    for station, ids in known.items():             # 已有对照先并进来
        lines[station] |= ids
    if seen_stations - set(lines):                 # 还有站不知道线路，回原始记录补
        lines.update(refill_station_lines(seen_stations))

    write_station_lines(lines)
    merge_days(days)
    print("站点数：%d，缺失站：%s" % (len(seen_stations), C.MISSING_STATION))
    return 0


if __name__ == "__main__":
    sys.exit(main())
