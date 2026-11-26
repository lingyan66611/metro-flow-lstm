# 第二步：清洗 → 特征 → 滑窗 → LSTM（含三个对照模型）

前置：`run_align.py` 已跑通，`data/archive_5min.csv` 就绪。整条链路按论文口径（5 min、12→1、70/15/15）实现。

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 口径、路径、模型超参的唯一来源 |
| `preprocess.py` | 缺失日补齐、分时段 IQR 异常截断，输出 `data/clean_5min.csv` |
| `features.py` | 单站特征表：客流、时段 sin/cos、星期、周末、节假日 |
| `dataset.py` | 12→1 滑窗、按时间 70/15/15 切分、只用训练段做 Min-Max |
| `model_lstm.py` | 双层 LSTM 64-32 + Dropout 0.1 + 早停 15 |
| `train.py` | 训练、评估四指标、存权重与图 |
| `metrics.py` | MAE / RMSE / MAPE / R² |
| `baselines.py` | ARIMA(2,1,2)、SVR(RBF, C=100)、BP(64-32) 对照 |

## 运行顺序

```powershell
cd "E:\learn\科研训练\地铁人流量预测\代码\初版"
python preprocess.py          # 约 20 秒，418 000 行 → clean_5min.csv
python features.py            # 默认站 15（火车东站）
python train.py               # LSTM，约 1 分钟
python baselines.py           # ARIMA / SVR / BP，约 1 分钟
```

换站：`python features.py --station 30`，`python train.py --station 30`。

## 当前结果（站 15，测试段 783 格，平均两个目标）

| 模型 | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| LSTM(64-32) | 68.53 | 88.86 | 23.55% | 0.780 |
| ARIMA(2,1,2) | 64.90 | 85.14 | 25.20% | 0.807 |
| SVR(RBF,C=100) | 72.08 | 91.67 | 53.92% | 0.771 |
| BP(64-32) | 73.09 | 95.45 | 40.90% | 0.751 |

单看平均，ARIMA 略好于 LSTM。原因不在模型本身：本数据只有 25 天、5 213 个样本，而进站序列的日周期极强（每 209 格重复一次），单步预测下 ARIMA 的差分+自回归几乎能直接套住昨天的同一时刻；LSTM 参数量大、样本少，验证损失早停在 76 轮已经接近过拟合边界。写进论文时应如实报告，并把它作为"下一步可改进"的入口，而不是隐瞒。

三个可以直接做的改进（按性价比排序）：

1. **把滞后项显式喂给 LSTM**：加入 lag-209（昨日同刻）、lag-1、滚动均值（3 格 / 12 格），让网络不必自己从 12 步里反推日周期 —— 这是当前损失最大的信息缺口。
2. **改多步直接预测**：一次输出未来 6 格（30 min），比递归单步更贴近论文"未来 5 分钟滚动"的应用场景，也更容易体现 LSTM 相对 ARIMA 的优势。
3. **调参 + 正则**：`HIDDEN1/2` 降到 32-16、`DROPOUT` 提到 0.2、`LR` 降到 5e-4，配合更长的输入窗（`LOOKBACK=24` 或 36）。样本少时应先减容量。

## 产物

- `data/clean_5min.csv`：清洗后长表（418 000 行，80 站 × 209 格 × 25 天）
- `data/features_station15.csv`：5 225 行 × 8 列
- `results/clean_report.json`：零值格占比 0.16%，异常截断 3 801 格，无整日缺失
- `results/lstm_metrics.json`、`results/lstm_predictions.csv`、`results/lstm_station15.pt`
- `results/baseline_metrics.json`
- `figures/lstm_loss.pdf`、`.svg`：训练/验证损失曲线
- `figures/lstm_test.pdf`、`.svg`：测试段预测对比
- `figures/lstm_error.pdf`、`.svg`：进站残差分布

图只输出矢量格式：LaTeX 排版直接插 `.pdf`，Word / PPT 插 `.svg`（可无损缩放，不会发糊）。格式列表在 `config.FIGURE_FORMATS`，想加 EPS 加一个后缀即可。

## 口径提醒

- 目标同时含进站、出站两列；论文若只报总量，把两列相加后重算指标即可。
- MAPE 只在真实值 ≥ 1 的格上计算，否则低谷时段会把百分比放大到失真。
- 归一化统计量只来自训练段（`dataset.prepare` 里 `拟合行数 = 训练段 + LOOKBACK`），测试段极值不会泄漏。
- 54 号站无记录，`config.STATIONS` 已排除；换站时不要传 54。
