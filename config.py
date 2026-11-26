# -*- coding: utf-8 -*-
"""口径配置：整个项目唯一的参数与路径来源。

改口径只改这一个文件，其他脚本一律 import config，不各自写死。
"""
from pathlib import Path

# ==================== 路径 ====================
ROOT = Path(__file__).resolve().parent          # 本文件所在目录，即“初版”文件夹
CODE_DIR = ROOT.parent                          # 上一级：代码
PROJ = CODE_DIR.parent                          # 项目根：地铁人流量预测（archive 在这里）

ARCHIVE_DIR = PROJ / "archive"                  # Kaggle 下载的原始刷卡数据
TRAIN_RECORD_DIR = ARCHIVE_DIR / "Metro_train"                 # 25 个训练日，每天一个 csv
TESTA_RECORD = ARCHIVE_DIR / "Metro_testA" / "testA_record_2019-01-28.csv"  # A 组：给 01-28，预测 01-29
TESTB_RECORD = ARCHIVE_DIR / "Metro_testB" / "testB_record_2019-01-26.csv"  # B 组：给 01-26，预测 01-27
HZDT_XLSX = PROJ / "2024毕业设计论文" / "HZDT.xlsx"               # 论文用的 5 min 汇总表，用来对账

DATA_DIR = ROOT / "data"                        # 所有中间数据
DAY_DIR = DATA_DIR / "days"                     # 按天缓存，重跑时不必再解析原始大文件
RESULT_DIR = ROOT / "results"                   # 对账报告与差异清单
FIGURE_DIR = ROOT / "figures"                   # 后续作图统一放这里
FIGURE_FORMATS = ("pdf", "svg")                 # 只输出矢量图：PDF 给 LaTeX，SVG 给 Word/PPT
for _d in (DATA_DIR, DAY_DIR, RESULT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ARCHIVE_TABLE = DATA_DIR / "archive_5min.csv"        # 聚合后的 5 min 长表（80 站 × 209 格 × 天数）
STATION_LINE_TABLE = DATA_DIR / "station_lines.csv"  # 站点属于哪条线路（A/B/C）

# ==================== 研究口径（冻结，改动须同步论文） ====================
FREQ_MIN = 5               # 时间粒度：5 分钟，与“预测未来 5 分钟客流”一致
OP_START = "06:00:00"      # 运营开始时刻（含）
OP_END = "23:25:00"        # 运营结束时刻（不含），与 HZDT 的边界一致
SLICES_PER_DAY = 209       # 每天 06:00—23:20 共 209 格
TAIL_BUCKET = "23:25"      # HZDT 末格：只装“23:25:00 这一秒”的刷卡，属边界伪格，建模要剔除
DEDUPE = False             # 是否去重；对账确认 HZDT 未去重，所以保持 False

STATION_TOTAL = 81         # 提交模板覆盖 0—80 号站
MISSING_STATION = 54       # 原始记录里从头到尾没有出现过的站
STATIONS = [s for s in range(STATION_TOTAL) if s != MISSING_STATION]   # 实际参与建模的 80 个站
TRAIN_DAYS = ["2019-01-%02d" % d for d in range(1, 26)]                 # 01-01 至 01-25


def day_buckets():
    """生成一天的 5 min 桶标签：06:00、06:05 … 23:20，共 209 个。"""
    import datetime as _dt
    start = _dt.datetime.strptime(OP_START, "%H:%M:%S")
    end = _dt.datetime.strptime(OP_END, "%H:%M:%S")
    out = []
    while start < end:
        out.append(start.strftime("%H:%M"))     # 只记到分钟，作为该格的起始时刻
        start += _dt.timedelta(minutes=FREQ_MIN)
    return out


# ==================== 竞赛口径（只用于提交，不进论文主结果） ====================
COMPETITION_FREQ_MIN = 10                # 官方提交是 10 min 一格
COMPETITION_STATIONS = list(range(81))   # 81 个站全都要填

# ==================== 建模口径（后续预处理与训练读取） ====================
LOOKBACK = 12              # 输入 12 个时间片，即 1 小时
HORIZON = 1                # 预测未来 1 个时间片
TRAIN_RATIO = 0.70         # 训练集占 70%
VAL_RATIO = 0.15           # 验证集占 15%，剩下 15% 作测试
SEED = 42                  # 随机种子，保证结果可复现
METRICS = ["MAE", "RMSE", "MAPE", "R2"]   # 论文第 5 章用的四个指标

HOLIDAYS = ["2019-01-01"]  # 元旦；数据期内只有这一天可确认是节假日

# ==================== 清洗与特征（对应实现方案第 4、6 章） ====================
CLEAN_TABLE = DATA_DIR / "clean_5min.csv"        # 清洗后的长表
IQR_K = 3.0                # 分时段 IQR 异常判定倍数：超过 Q3 + 3×IQR 就截断
FEATURE_PATH = DATA_DIR / "features_station%s.csv"   # 单站特征表，%s 换成站号
SLOT_PERIOD = 209          # 一天 209 格，用于 sin/cos 编码

# ==================== 模型与训练（对应实现方案第 7、8 章） ====================
STATION_ID = 15            # 主实验站：15 号火车东站（全峰型典型站）
HIDDEN1 = 64               # 第一层 LSTM 单元数
HIDDEN2 = 32               # 第二层 LSTM 单元数
DROPOUT = 0.1              # Dropout 比例（论文初稿写 0.2，此处以立项报告 0.1 为准）
EPOCHS = 120               # 最大训练轮数
PATIENCE = 15              # 验证集连续 15 轮不降就早停（立项报告口径）
BATCH_SIZE = 64            # 批大小
LR = 0.001                 # 学习率
TARGET_COLS = ["inflow", "outflow"]   # 同时预测进站与出站两个量

RESULT_LSTM = RESULT_DIR / "lstm_metrics.json"
RESULT_PRED = RESULT_DIR / "lstm_predictions.csv"
RESULT_BASE = RESULT_DIR / "baseline_metrics.json"

