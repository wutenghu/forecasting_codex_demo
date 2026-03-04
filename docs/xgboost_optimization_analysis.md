# XGBoost 专项优化分析（区域/天）

## 1. 优化目标
- 目标：在相同评估口径下让 XGBoost 超过 RandomForest（重点看 RMSE）。
- 数据：`data/processed/region_day_features.csv`
- 评估窗口：`2025-12-03` 到 `2025-12-30`（`n_eval=140`）

## 2. 对比结果

### 2.1 初始基线（`main/train.py`）
- random_forest：RMSE `4.9103`
- xgboost：RMSE `5.3023`
- naive_lag7：RMSE `7.0554`

### 2.2 通用贝叶斯调参后（`main/tune_models_bayes.py`）
- random_forest_tuned：RMSE `4.8517`
- xgboost_tuned：RMSE `4.9021`
- naive_lag7：RMSE `7.0554`

### 2.3 XGBoost 专项优化后（`main/tune_xgboost_focus.py`, 80 trials）
- xgboost_tuned_focus：RMSE `4.8854`，MAE `3.9788`，MAPE `15.6915`

### 2.4 XGBoost 专项优化（200 trials）
- xgboost_tuned_focus：RMSE `4.8926`，MAE `3.9859`，MAPE `15.7181`

结论：
- XGBoost 已明显优于其初始版本（`5.3023 -> 4.8854`）。
- 但仍未超过调优后的 RF（`4.8854 > 4.8517`，差值约 `0.0337` RMSE）。
- 200 trials 的最优调参分数虽更低，但最终 eval 反而变差，说明出现调参验证集过拟合。
- 当前建议：生产优先 `random_forest_tuned`，保留 `xgboost_tuned_focus` 作为挑战者模型持续跟踪。

## 3. 收敛检查
- Optuna TPE（80 trials）最优 tuning-valid RMSE：`5.3474`
- 最优 trial 出现在第 `71` 次。
- 最后10次 trial 的 `best_so_far` 不再下降，已接近平台期。

说明：我们又尝试了 200 trials，tuning-valid 从 `5.3474` 下降到 `5.3441`，但最终 eval RMSE 从 `4.8854` 上升到 `4.8926`，因此采用 80 trials 结果作为更稳健版本。

## 4. 为什么 XGBoost 仍略逊于 RF
- 样本规模较小：区域级只有 5 个区域，训练样本量（有效）不大，XGBoost 的参数空间更大，更容易在小样本时不稳定。
- 目标分布稀疏+离散：需求为计数型且波动较大，RF 对这种“少量强特征 + 非线性分段”通常更稳健。
- 特征维度与信噪比：当前可用的“可预测特征”（去掉同日泄漏特征后）相对有限，XGBoost 的优势空间被压缩。
- 调参过拟合风险：扩大试验次数后，tuning-valid 继续变好但 eval 变差，说明已进入验证集过拟合区间。

## 5. 已落地产物
- XGBoost 专项调参脚本：`main/tune_xgboost_focus.py`
- 结果文件：
  - `artifacts/metrics/xgb_focus_metrics.json`
  - `artifacts/metrics/xgb_optuna_trials_focus.csv`
  - `artifacts/metrics/eval_predictions_xgb_focus.csv`
  - `artifacts/models/xgboost_tuned_focus.joblib`

## 6. 图表诊断补充结论（区域维度）
- 结合 `figures/model_diagnostics/top*_actual_vs_pred.png` 与 `top_entities_summary.csv`：
  - `West`、`Southeast` 存在一定低估（pred_sum 低于 actual_sum）；
  - `Southwest`、`Northeast` 存在一定高估；
  - `Midwest` 聚合层面偏差相对更小。
- 这说明当前模型的主要问题已从“全局趋势失配”转向“区域偏差校准”，后续可优先做区域校准项或分区域后处理。
