# Feature Spec（region-day demand forecasting）

## 1. 数据输入与输出
- 输入1：`data/processed/region_day_panel.csv`
- 输入2：`data/starbucks_customer_ordering_patterns.csv`（用于聚合渠道/品类上下文）
- 输出：`data/processed/region_day_features.csv`
- 主键：`region + order_date`
- 粒度：区域/天（region-day）

## 2. 目标字段
- `demand`：区域在当天的订单量（由订单明细聚合得到）

## 3. 特征分组与定义

### 3.1 日历特征
- `dow`：星期（0=Mon, 6=Sun）
- `day`：月内日（1-31）
- `month`：月份（1-12）
- `weekofyear`：ISO周序号
- `is_weekend`：是否周末（Sat/Sun=1）

### 3.2 渠道占比特征（order_channel）
按 `region + order_date` 聚合渠道占比，缺失填0：
- `channel_share_drive_thru`
- `channel_share_in_store_cashier`
- `channel_share_kiosk`
- `channel_share_mobile_app`

### 3.3 饮品占比特征（drink_category）
按 `region + order_date` 聚合品类占比，缺失填0：
- `drink_share_brewed_coffee`
- `drink_share_espresso`
- `drink_share_frappuccino`
- `drink_share_other`
- `drink_share_refresher`
- `drink_share_tea`

### 3.4 节假日与区域特征
- `is_us_federal_holiday`
- `is_region_midwest`, `is_region_northeast`, `is_region_southeast`, `is_region_southwest`, `is_region_west`
- `is_holiday_region_midwest`, `is_holiday_region_northeast`, `is_holiday_region_southeast`, `is_holiday_region_southwest`, `is_holiday_region_west`

### 3.5 时序特征
- 滞后：`lag_1`, `lag_7`, `lag_14`, `lag_28`
- 滚动：`rolling_mean_7`, `rolling_mean_28`（先 `shift(1)` 再 rolling）

### 3.6 区域统计特征
- `region_avg_demand`
- `region_std_demand`
- 编码：`region_code`

## 4. 防泄漏与可复现规则
- 时序特征只使用历史信息（`shift(1)` 保证不看当天目标）。
- 训练默认不使用 `channel_share_*` / `drink_share_*`（这些是当天聚合特征，未来预测时不可直接获得）。
- 全流程使用稳定排序（`mergesort`），并在配置保留 `random_seed`。

## 5. 缺失值策略
- 占比特征在无订单日填0。
- 滞后/滚动在序列开头存在自然缺失，由训练阶段统一处理（常见是筛掉缺失行）。

## 6. 当前输出字段清单（37列）
- 标识与目标：`region`, `order_date`, `demand`
- 日历：`dow`, `day`, `month`, `weekofyear`, `is_weekend`
- 渠道占比：`channel_share_*`（4列）
- 品类占比：`drink_share_*`（6列）
- 节假日与区域：`is_us_federal_holiday`, `is_region_*`（5列）, `is_holiday_region_*`（5列）
- 时序：`lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_mean_28`
- 区域统计：`region_avg_demand`, `region_std_demand`
- 编码：`region_code`
