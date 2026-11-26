# -*- coding: utf-8 -*-
"""对账一键执行：聚合原始记录 → 合并 → 与 HZDT 逐格比对。

用法：
    python run_align.py            # 全量 25 天，首次约 7 分钟，之后读缓存
    python run_align.py --days 1   # 单日快速验证
    python run_align.py --force    # 重新聚合
"""
import argparse
import sys

import aggregate_archive as agg
import compare_hzdt as cmp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="只处理前 N 天")
    ap.add_argument("--force", action="store_true", help="重新聚合已有日缓存")
    args = ap.parse_args(argv)

    agg_args = []
    if args.days:
        agg_args += ["--days", str(args.days)]
    if args.force:
        agg_args.append("--force")
    agg.main(agg_args)                       # 第一步：原始 csv → 5 min 表

    report, _ = cmp.compare()                # 第二步：两张表逐格比较

    print("\n=== 对齐结论 ===")
    print("一致率 %.2f%%（%d / %d 格）"
          % (report["match_rate"], report["matched_cells"], report["valid_cells"]))
    if report["mismatch_cells"] == 0 and report["only_in_archive"] == 0:
        print("聚合表与 HZDT 有效格完全一致，口径可以冻结。")
        print("HZDT 末格 23:25 共 %d 个（非零 %d 个），为边界伪格，建模时剔除。"
              % (report["hzdt_tail_cells"], report["hzdt_tail_nonzero"]))
    else:
        print("存在差异，请打开 results/align_mismatch.csv 逐条核对后再冻结口径。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
