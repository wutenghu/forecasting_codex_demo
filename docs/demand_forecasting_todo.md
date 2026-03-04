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

---

## 8. Future Further Improve（基于当前结论）

### 8.1 数据层面（最高优先级）
- [ ] 扩展时间跨度到 `>= 24` 个月（当前 730 天虽可用，但对节假日与年度周期鲁棒性仍有限）。
- [ ] 引入外生数据：天气、降雨、温度、区域经济活动、营销日历、价格与折扣。
- [ ] 提升可追溯性：增加数据版本号与 `data_cutoff_date`，形成训练/预测可追踪血缘。
- [ ] 建立滚动数据质量监控：异常波动、突增突降、字段漂移自动告警。

理由：当前 RF/XGB 的差距已很小，下一步提升空间更依赖“可预测信号增量”，不是单纯换模型。

### 8.2 特征层面（高优先级）
- [ ] 引入更长滞后与季节窗口：`lag_35/42/56`、`rolling_mean_56`、`rolling_std_28/56`。
- [ ] 构造区域偏差校准特征：按区域维护残差均值/分位统计，用于后处理修正（针对 West/Southeast 低估、Southwest/Northeast 高估）。
- [ ] 增加“节假日前后效应”与“发薪日/月底效应”特征。
- [ ] 将渠道/品类占比从“同日聚合”改造为“可预测版本”（lag share、近7日均值 share），避免未来信息不可得。

理由：当前主要问题从“全局趋势”转向“区域偏差校准 + 稳定性提升”。

### 8.3 模型层面（中优先级）
- [ ] 先做评估协议升级：rolling-origin `>= 3` 折，按窗口汇总 MAE/RMSE/MAPE/wMAPE。
- [ ] 在 RF 主线下做两段式策略：`RF 主模型 + 区域后校准器`（轻量线性或分位回归校正）。
- [ ] XGBoost 继续作为 challenger，固定 80-trial 左右预算，避免过度调参带来的验证过拟合。
- [ ] 深度模型（TimesBERT/TFT）仅作为技术积累分支：在补齐 `torch` 环境后跑离线对比，不进入主线，直到收益阈值达标（RMSE 至少提升 5% 且跨窗稳定）。

理由：现阶段 `random_forest_tuned` 已是最稳最优，模型侧应以“稳健增益”而非“复杂度增加”为导向。
