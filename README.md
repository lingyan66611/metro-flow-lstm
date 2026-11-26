# 基于 LSTM 的轨道交通站点客流预测

## 项目简介
校级科研训练项目。以杭州地铁 15 号火车东站为对象，用双层 LSTM（64-32）预测 5 分钟粒度进出站客流，并与 ARIMA、SVR、BP 三条基线对比。

## 复现说明
本项目复现叶影 2024 届本科毕业设计中的 LSTM 客流预测方法。
数据口径（5 min）、滑窗方式（12→1）、数据集划分（70/15/15）与评价指标（MAE、RMSE、MAPE、R²）对齐原文第 5 章。
- 复现范围：5 min 聚合与预处理、双层 LSTM、指标计算
- 与原文差异：Dropout 取 0.1（依立项报告），缺失补齐用线性插值加同期均值
- 本项目增量：compare_hzdt.py 逐格对账自建聚合表与原文 HZDT.xlsx；补 ARIMA、SVR、BP 三条基线；如实报告 ARIMA 单步略优及原因
- 原文 PDF 与 HZDT.xlsx 属校内资料，未入库

## 数据
Kaggle HZDT，2019 年 1 月杭州地铁 AFC。
80 站 × 209 格 × 25 天 = 41.8 万个 5 分钟客流格；15 号站 5,225 时间片，滑窗后 5,213 样本。
原始数据与中间表体积大，未入库，可用 preprocess.py 重建。

## 运行
pip install -r requirements.txt
python preprocess.py
python train.py
python baselines.py
改 config.py 的 STATION_ID 可换站点。

## 结果
MAE 68.53 / RMSE 88.86 / R² 0.780。
说明：25 天样本、日周期极强，单步预测下 ARIMA 略优，报告如实保留此结论。

## 目录结构
config.py 参数 | dataset.py 滑窗 | features.py 特征 | model_lstm.py 模型 | metrics.py 指标 | compare_hzdt.py 口径对账
