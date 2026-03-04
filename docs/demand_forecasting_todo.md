# 区域/天需求预测项目 TODO（未来7天滚动预测）

## 0. 项目目标与范围确认
- [x] 明确预测目标：`region + date -> demand(订单量)`
- [x] 明确预测窗口：每次执行输出未来连续 7 天
- [x] 业务使用场景：区域级排班、备货、投放策略
- [x] 固化输入输出契约：
  - 输入：`data/starbucks_customer_ordering_patterns.csv`
  - 输出：`data/predictions/region_daily_forecast_next_7_days_<run_date>.csv`

交付物：
- [ ] `docs/project_scope.md`

---

## 1. 数据理解与质量评估
- [x] 数据字典整理：字段含义、类型、缺失率、异常值规则
- [x] 统计时间范围、区域覆盖、日级样本分布
- [x] 检查关键字段质量（`order_date` 可解析率、`order_id` 重复率、`region` 合法性）
- [x] 诊断低活跃区域占比
- [x] 评估节假日/周末效应

交付物：
- [ ] `docs/data_profile.md`
- [x] `docs/data_quality_report.md`

---

## 2. 可复现数据处理层
- [x] 明细聚合到区域/天粒度
- [x] 补齐区域全日期面板
- [x] 缺失需求填0
- [x] 数据处理脚本化、稳定排序、可复现
- [x] 产出中间层：`data/processed/region_day_panel.csv`

交付物：
- [x] `main/data_pipeline.py`
- [x] `data/processed/region_day_panel.csv`

---

## 3. 特征工程
- [x] 日历特征：`dow/day/month/weekofyear/is_weekend`
- [x] 时序特征：`lag_1/7/14/28`, `rolling_mean_7/28`
- [x] 区域统计特征：`region_avg_demand`, `region_std_demand`
- [x] 增强特征：
  - [x] `order_channel` 区域占比
  - [x] `drink_category` 区域占比
  - [x] 节假日与区域交互特征

交付物：
- [x] `feature_pipeline/features.py`
- [x] `main/feature_pipeline.py`
- [x] `main/run_feature_pipeline.py`
- [x] `docs/feature_spec.md`
- [x] `data/processed/region_day_features.csv`

---

## 4. 建模基线
- [x] 基线模型 A：Naive（lag_7）
- [x] 基线模型 B：RandomForest
- [x] 候选模型 C：XGBoost
- [x] 统一训练入口与参数接口
- [x] 固化训练产物目录：
  - [x] `artifacts/models/`
  - [x] `artifacts/metrics/`

当前模型对比结果（区域/天，验证窗口：`2025-12-03` 到 `2025-12-30`，`n_eval=140`）：

| model | MAE | RMSE | MAPE(%) |
|:--|--:|--:|--:|
| random_forest | 3.9422 | 4.9103 | 15.4045 |
| xgboost | 4.1622 | 5.3023 | 16.3627 |
| naive_lag7 | 5.3357 | 7.0554 | 21.0260 |

来源文件：`artifacts/metrics/model_comparison.csv`

XGBoost 专项优化结论（贝叶斯优化）：
- 已完成 80 trials 专项优化，XGB 的 RMSE 从 `5.3023` 提升到 `4.8854`；
- 仍略低于 RF tuned（`4.8517`）；
- 200 trials 结果：RMSE `4.8926`（较 80 trials 略差）；
- 原因分析见：`docs/xgboost_optimization_analysis.md`。

当前生产候选结论（区域/天）：
- 主模型：`random_forest_tuned`（当前离线指标最优且稳定）
- 备选模型：`xgboost_tuned_focus`（已显著提升，但仍略逊于 RF tuned）
- 对照基线：`naive_lag7`

交付物：
- [x] `main/train.py`
- [x] `main/tune_models_bayes.py`
- [x] `artifacts/metrics/model_comparison.csv`
- [x] `artifacts/metrics/model_comparison_tuned.csv`

---

## 5. 时间序列评估设计
- [ ] 划分策略文档（按时间切分 + 防泄漏说明）
- [ ] 滚动回测（>=3折）
- [ ] 指标体系：MAE / RMSE / MAPE / 分位数误差
- [x] 误差诊断图：
  - [x] `figures/model_diagnostics/daily_mean_actual_vs_pred.png`
  - [x] `figures/model_diagnostics/top_entities_summary.csv`
  - [x] `figures/model_diagnostics/top*_actual_vs_pred.png`（区域级）

交付物：
- [ ] `docs/evaluation_protocol.md`
- [ ] `docs/backtesting_report.md`

---

## 6. 7天递推预测输出
- [ ] 统一预测入口（区域级，未来7天）
- [ ] 输出格式与元信息标准化（`run_date/model_version/data_cutoff_date`）
- [ ] 文件落盘：`data/predictions/region_daily_forecast_next_7_days_<run_date>.csv`

交付物：
- [ ] `main/predict.py`
- [ ] `data/predictions/*.csv`

---

## 7. 深度模型技术积累（TimesBERT，PyTorch）
- [x] 完成 TimesBERT 结构定义：`algorithm/timesbert.py`
- [x] 完成区域级训练入口：`main/timesbert_region.py` + `main/run_timesbert_region.py`
- [x] 完成模型结构图：`figures/timesbert_model_structure.png`
- [ ] 在当前环境跑通训练并产出 MAE/RMSE/MAPE（阻塞：环境缺少 `torch`，且当前网络受限无法安装）

阶段结论：
- 已完成工程级代码骨架与数据适配（`region_day_panel.csv`）。
- 当前可做代码评审与结构迭代，但暂无法给出 TimesBERT 的离线效果对比结果。
