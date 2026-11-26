# -*- coding: utf-8 -*-
"""逐格对账：自己聚合的表 vs 论文用的 HZDT.xlsx。

输出：
    results/align_report.json    汇总指标
    results/align_mismatch.csv   有差异的格子（本次为空，只有表头）

判定方式：
    以 (日期, 站点, 5min桶) 为键，逐格比较进站、出站两个数。
    HZDT 独有的 23:25 末格单独统计，不当作差异——它是分桶边界造成的伪格。
"""
import csv
import json
import sys

import config as C
from hzdt_reader import iter_hzdt, summarize


def load_archive(path=None):
    """读 data/archive_5min.csv，返回 {(日期, 站点, 桶): (进站, 出站, 原始条数)}。"""
    path = path or C.ARCHIVE_TABLE
    table = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader)                                     # 跳过表头
        for date, station, bucket, inflow, outflow, raw in reader:
            table[(date, int(station), bucket)] = (int(inflow), int(outflow), int(raw))
    return table


def compare():
    arch = load_archive()
    hzdt = {}
    for date, station, bucket, inflow, outflow in iter_hzdt():
        hzdt[(date, station, bucket)] = (inflow, outflow)

    mismatch = []
    only_arch = only_hz = only_hz_nonzero = 0
    match = 0
    sum_abs_in = sum_abs_out = 0
    tail_cells = tail_nonzero = 0

    for key in set(arch) | set(hzdt):
        a = arch.get(key)
        b = hzdt.get(key)
        if b is not None and key[2] == C.TAIL_BUCKET:     # 末格先挑出来，不计入对账分母
            tail_cells += 1
            if b[0] or b[1]:
                tail_nonzero += 1
            continue
        if a is None:
            only_hz += 1
            if b[0] or b[1]:
                only_hz_nonzero += 1                       # 非零才算真差异
            continue
        if b is None:
            only_arch += 1                                 # 自己的表少了格子
            continue
        if (a[0], a[1]) == b:
            match += 1                                     # 两个数都一样
        else:
            d_in = a[0] - b[0]
            d_out = a[1] - b[1]
            sum_abs_in += abs(d_in)
            sum_abs_out += abs(d_out)
            mismatch.append([key[0], key[1], key[2], a[0], b[0], d_in, a[1], b[1], d_out])

    mismatch.sort(key=lambda r: (r[0], r[1], r[2]))
    with open(C.RESULT_DIR / "align_mismatch.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "station_id", "bucket", "arch_in", "hzdt_in", "diff_in",
                    "arch_out", "hzdt_out", "diff_out"])
        for row in mismatch:
            w.writerow(row)

    valid = match + len(mismatch) + only_arch
    report = {
        "archive_cells": len(arch),
        "hzdt_cells": len(hzdt),
        "valid_cells": valid,
        "matched_cells": match,
        "mismatch_cells": len(mismatch),
        "match_rate": round(100.0 * match / valid, 4) if valid else 0.0,
        "only_in_archive": only_arch,
        "only_in_hzdt": only_hz,
        "only_in_hzdt_nonzero": only_hz_nonzero,
        "sum_abs_diff_inflow": sum_abs_in,
        "sum_abs_diff_outflow": sum_abs_out,
        "hzdt_tail_cells": tail_cells,
        "hzdt_tail_nonzero": tail_nonzero,
        "hzdt_summary": summarize(),
    }
    (C.RESULT_DIR / "align_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, mismatch


def main():
    report, mismatch = compare()
    print("聚合表 %d 格，HZDT %d 格，可比 %d 格"
          % (report["archive_cells"], report["hzdt_cells"], report["valid_cells"]))
    print("完全一致 %d 格（%.2f%%），数值不一致 %d 格，仅聚合有 %d 格"
          % (report["matched_cells"], report["match_rate"],
             report["mismatch_cells"], report["only_in_archive"]))
    print("差异绝对值：进站 %d，出站 %d"
          % (report["sum_abs_diff_inflow"], report["sum_abs_diff_outflow"]))
    print("HZDT 末格 23:25 共 %d 个，其中非零 %d 个（只含 23:25:00 这一秒，属伪格）"
          % (report["hzdt_tail_cells"], report["hzdt_tail_nonzero"]))
    if mismatch:
        print("差异前 10 行：")
        for row in mismatch[:10]:
            print("  ", row)
    print("报告：results/align_report.json，清单：results/align_mismatch.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
