# Starbucks Customer Ordering Patterns 项目背景

## 1. 项目来源
- 数据集来源：Kaggle  
  https://www.kaggle.com/datasets/likithagedipudi/starbucks-customer-ordering-patterns
- 本地数据位置：
  - `data/starbucks_customer_ordering_patterns.csv`
  - `data/starbucks-customer-ordering-patterns.zip`（原始压缩包）

## 2. 业务背景
本项目围绕咖啡订单行为展开，当前主线是区域/天需求预测；目标是通过历史订单数据分析消费者在不同时段、不同渠道、不同区域与不同人群特征下的下单规律。  
典型业务问题包括：
- 区域维度如何预测未来7天需求并优化排班与备货；
- 哪类区域在节假日/周末更敏感；
- 哪些订单结构因素会影响区域级需求波动；
- 如何做区域级运营策略（促销、投放、供应协同）。

## 3. 数据内容概览
核心字段包含：
- 订单与时间：`order_id`, `order_date`, `order_time`, `day_of_week`
- 渠道与空间属性：`order_channel`, `store_id`, `store_location_type`, `region`（当前预测主键使用 `region`）
- 用户特征：`customer_age_group`, `customer_gender`, `is_rewards_member`
- 订单结构：`cart_size`, `num_customizations`, `drink_category`, `has_food_item`, `order_ahead`
- 结果指标：`total_spend`, `fulfillment_time_min`, `customer_satisfaction`

## 4. 本项目目标（建议）
- 需求预测：按区域/天预测订单量，并一次输出未来连续7天；
- 用户洞察：识别高价值客群和高复购行为模式；
- 履约优化：分析影响 `fulfillment_time_min` 的关键因子；
- 体验优化：建模或分析 `customer_satisfaction` 的影响因素；
- 运营决策：形成可执行的区域运营策略与指标看板。

## 5. 可交付成果（建议）
- 一份可复现的数据清洗与特征工程流程；
- 一套基础分析报告（EDA + 区域需求关键结论）；
- 一个区域级预测模型基线（可持续迭代）；
- 一份面向业务的结论文档（策略建议 + 风险提示）。

## 6. 当前进展快照（2026-03-05）
- 当前最优离线模型：`random_forest_tuned`（区域/天验证 RMSE `4.8517`）。
- XGBoost 状态：专项贝叶斯优化后 RMSE `4.8854`，较初始有显著提升，但仍略逊于 RF tuned。
- 诊断图已落地：`figures/model_diagnostics/`（含日均对比、误差分布、Top 区域真实 vs 预测）。
- 深度模型技术积累：已完成 TimesBERT 的 PyTorch 工程实现与结构图，但因环境缺少 `torch`，尚未完成离线指标对比。
