# Forecasting Codex Demo

区域/天需求预测项目（每次输出未来连续 7 天）。

## 1. 环境准备

建议 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据准备

原始数据默认路径：
- `data/starbucks_customer_ordering_patterns.csv`

如果该文件已存在，可直接执行后续步骤。

## 3. 运行流程（主线）

### 3.1 构建区域日级面板

```bash
python3 main/run_data_pipeline.py
```

输出：`data/processed/region_day_panel.csv`

### 3.2 构建特征

```bash
python3 main/run_feature_pipeline.py
```

输出：`data/processed/region_day_features.csv`

### 3.3 训练基线模型（Naive + RF + XGBoost）

```bash
python3 main/train.py
```

输出：
- 指标：`artifacts/metrics/model_comparison.csv`
- 预测明细：`artifacts/metrics/eval_predictions.csv`
- 模型：`artifacts/models/`

### 3.4 画诊断图

```bash
python3 main/plot_model_diagnostics.py
python3 main/plot_top20_store_diagnostics.py
```

输出目录：`figures/model_diagnostics/`

## 4. 调参（可选）

### 4.1 RF + XGBoost 贝叶斯调参

```bash
python3 main/tune_models_bayes.py
```

### 4.2 XGBoost 专项优化

```bash
python3 main/tune_xgboost_focus.py
```

## 5. TimesBERT 技术积累（PyTorch）

### 5.1 训练入口

```bash
python3 main/run_timesbert_region.py
```

输出（成功时）：
- 模型：`artifacts/models/timesbert_region.pt`
- 指标：`artifacts/metrics/timesbert_metrics.json`
- 评估明细：`artifacts/metrics/timesbert_eval_predictions.csv`
- 未来 7 天预测：`data/predictions/region_daily_forecast_next_7_days_timesbert.csv`

### 5.2 模型结构图

```bash
python3 main/plot_timesbert_architecture.py
```

输出：`figures/timesbert_model_structure.png`

## 6. 常见问题

`python3 main/run_timesbert_region.py` 报 `No module named 'torch'`：
- 说明当前环境未安装 PyTorch。
- 重新执行：`pip install -r requirements.txt`。

## 7. 项目结构

- `main/`：任务入口脚本
- `feature_pipeline/`：数据与特征流水线
- `algorithm/`：模型结构定义（含 TimesBERT）
- `utils/`：基础工具（路径管理）
- `docs/`：项目文档
- `artifacts/`：模型与指标产物
- `figures/`：图表产物
