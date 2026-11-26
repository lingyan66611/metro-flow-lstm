# -*- coding: utf-8 -*-
"""读取 HZDT.xlsx，只用 Python 标准库（zipfile + ElementTree）。

为什么不用 pandas / openpyxl：
只为对一次账装一个库不划算；自己解 xlsx 还能避开不同 Excel 引擎对日期序列号的解释差异。

用法：
    from hzdt_reader import iter_hzdt
    for date, station, bucket, inflow, outflow in iter_hzdt():
        ...
"""
import datetime as dt
import zipfile
import xml.etree.ElementTree as ET

import config as C

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
EXCEL_EPOCH = dt.date(1899, 12, 30)     # Excel 日期序列号起点，已扣掉 1900 年闰年 bug 的 1 天


def _shared_strings(zf):
    """xlsx 把重复出现的文字存在 sharedStrings.xml，单元格里只放编号。"""
    try:
        content = zf.read("xl/sharedStrings.xml").decode("utf-8")
    except KeyError:
        return []
    out = []
    for item in ET.fromstring(content):
        pieces = [(t.text or "") for t in item.iter(NS + "t")]
        out.append("".join(pieces))         # 富文本会被拆成多个 t，需要拼回去
    return out


def _serial_to_bucket(serial):
    """Excel 日期序列号 → 5 min 桶标签 'HH:MM'。

    小数部分有浮点误差（06:05 可能存成 06:04:59.99），
    所以先四舍五入到最近的分钟，再向下取整到 5 分钟。
    """
    minutes = int(round((float(serial) - int(float(serial))) * 1440)) % 1440   # 一天的分钟数，跨 0 点会绕回
    minutes = minutes // C.FREQ_MIN * C.FREQ_MIN
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def _cell_value(cell, shared):
    t = cell.get("t")                       # t="s" 表示值在共享字符串表里
    v = cell.find(NS + "v")
    if v is None or v.text is None:
        return None
    if t == "s":
        return shared[int(v.text)]
    if t == "inlineStr":
        return v.text
    return v.text                           # 数字、日期都按原始文本返回，后面再转


def iter_hzdt(path=None, sheet="xl/worksheets/sheet1.xml"):
    """逐行产出 (日期, 站点, 桶, 进站, 出站)，内存只保留当前一行。"""
    path = path or C.HZDT_XLSX
    with zipfile.ZipFile(path) as z:
        shared = _shared_strings(z)
        with z.open(sheet) as fh:
            for event, node in ET.iterparse(fh, events=("end",)):
                if node.tag != NS + "row":
                    continue
                cells = {}
                for c in node:
                    if c.tag != NS + "c":
                        continue
                    ref = c.get("r") or ""              # 例如 "D2"
                    col = "".join(ch for ch in ref if ch.isalpha())
                    cells[col] = _cell_value(c, shared)
                node.clear()                            # 清掉已解析节点，否则 42 万行会吃光内存
                if cells.get("A") in (None, "date"):
                    continue                            # 表头行
                date = cells["A"]
                if len(date) == 5 and date.isdigit():   # 极少数情况下日期被存成序列号
                    date = (EXCEL_EPOCH + dt.timedelta(days=int(date))).isoformat()
                station = int(float(cells["C"]))        # C 列：站点编号
                bucket = _serial_to_bucket(cells["D"])  # D 列：时间（日期+时刻的序列号）
                inflow = int(float(cells["G"]))         # G 列：进站客流
                outflow = int(float(cells["H"]))        # H 列：出站客流
                yield date, station, bucket, inflow, outflow


def summarize():
    """快速检查：日期数、站点数、每天格数、末格是否异常。"""
    dates, stations = set(), set()
    cells_per_day = {}
    tail_cells = tail_nonzero = 0
    for date, station, bucket, inflow, outflow in iter_hzdt():
        dates.add(date)
        stations.add(station)
        cells_per_day[date] = cells_per_day.get(date, 0) + 1
        if bucket == "23:25":
            tail_cells += 1
            if inflow or outflow:
                tail_nonzero += 1                   # 末格本应近乎为空，非零说明是伪格
    return {
        "dates": len(dates),
        "date_range": [min(dates), max(dates)] if dates else [],
        "stations": len(stations),
        "station_min": min(stations) if stations else None,
        "station_max": max(stations) if stations else None,
        "cells_per_day": sorted(set(cells_per_day.values())),
        "tail_2325_cells": tail_cells,
        "tail_2325_nonzero": tail_nonzero,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summarize(), ensure_ascii=False, indent=2))
