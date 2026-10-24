# -*- coding: utf-8 -*-
"""Olist 延迟配送预测 — 第 2 步：标签构造 + 预测时点字段白名单（防泄漏）
标签口径（与项目计划一致）：样本 = delivered 且两时间字段完整；is_late = 实际送达时间 > 预计送达时间。
注意：本脚本同时输出"日期粒度"口径对照，暴露 1292 笔"当天送达"被精确时刻比较判为延迟的口径隐患。
"""
import pandas as pd
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. 标签构造 ----------
orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
orders['delivered_dt'] = pd.to_datetime(orders['order_delivered_customer_date'], errors='coerce')
orders['estimated_dt'] = pd.to_datetime(orders['order_estimated_delivery_date'], errors='coerce')

# 样本范围：delivered 且两时间字段完整
sample = orders[orders['order_status'] == 'delivered'].copy()
sample = sample[sample[['delivered_dt', 'estimated_dt']].notna().all(axis=1)].copy()

# 口径A：精确时刻比较（计划口径，延迟率 8.11%）
sample['is_late'] = (sample['delivered_dt'] > sample['estimated_dt']).astype(int)

# 口径B：日期粒度比较（仅作对照）
sample['is_late_date'] = (sample['delivered_dt'].dt.normalize() > sample['estimated_dt'].dt.normalize()).astype(int)

n = len(sample)
n_late = int(sample['is_late'].sum())
n_late_date = int(sample['is_late_date'].sum())
same_day_misjudged = int((sample['is_late'] == 1) & (sample['is_late_date'] == 0).astype(bool).sum()) if False else \
    int(((sample['delivered_dt'].dt.normalize() == sample['estimated_dt'].dt.normalize()) & (sample['is_late'] == 1)).sum())

print('=== 标签构造结果 ===')
print(f'delivered 订单: {len(orders[orders["order_status"]=="delivered"])}')
print(f'两时间字段完整样本: {n}')
print(f'口径A(精确时刻>): 延迟 {n_late}  | 延迟率 {n_late/n*100:.4f}%   <-- 计划口径 8.11%')
print(f'口径B(日期>):      延迟 {n_late_date}  | 延迟率 {n_late_date/n*100:.4f}%')
print(f'口径隐患: 当天送达但被口径A判延迟 = {same_day_misjudged} 笔')

# 保存标签（口径A）
label_out = os.path.join(DATA_DIR, 'label_is_late.csv')
sample[['order_id', 'delivered_dt', 'estimated_dt', 'is_late']].to_csv(label_out, index=False)
print(f'标签已保存: {label_out}')

# ---------- 2. 预测时点字段白名单 ----------
# 分类: id=标识符, feature=下单时可得(可用), label=标签基准(禁作特征),
#       post_hoc=事后字段(禁用), review=需审查(可能下单后更新)
WHITELIST = {
    'olist_orders_dataset.csv': {
        'order_id': 'id',
        'customer_id': 'id',
        'order_status': 'post_hoc',                    # 最终状态，下单时=created
        'order_purchase_timestamp': 'feature',         # 下单时刻 → 时间特征(星期/月/时)
        'order_approved_at': 'post_hoc',               # 付款批准，下单后
        'order_delivered_carrier_date': 'post_hoc',    # 交付承运人，下单后
        'order_delivered_customer_date': 'label',      # 标签基准(实际送达)
        'order_estimated_delivery_date': 'label',      # 标签基准(预计送达)，作特征会泄漏
    },
    'olist_customers_dataset.csv': {
        'customer_id': 'id',
        'customer_unique_id': 'feature',               # 可作客户级特征，但历史频次须按时间窗口防泄漏
        'customer_zip_code_prefix': 'feature',
        'customer_city': 'feature',
        'customer_state': 'feature',
    },
    'olist_geolocation_dataset.csv': {
        'geolocation_zip_code_prefix': 'feature',
        'geolocation_lat': 'feature',
        'geolocation_lng': 'feature',
        'geolocation_city': 'feature',
        'geolocation_state': 'feature',
    },
    'olist_order_items_dataset.csv': {
        'order_id': 'id',
        'order_item_id': 'feature',                    # 商品序号(下单时可得)
        'product_id': 'id',
        'seller_id': 'id',
        'shipping_limit_date': 'review',               # 发货截止日，需确认下单后是否更新
        'price': 'feature',
        'freight_value': 'feature',
    },
    'olist_order_payments_dataset.csv': {
        'order_id': 'id',
        'payment_sequential': 'review',                # 多笔支付序号，付款时产生
        'payment_type': 'feature',                     # 支付方式(下单时选择)
        'payment_installments': 'feature',             # 分期数(下单时选择)
        'payment_value': 'review',                     # 实际付款金额，≈下单应付金额
    },
    'olist_order_reviews_dataset.csv': {
        'review_id': 'post_hoc',
        'order_id': 'post_hoc',
        'review_score': 'post_hoc',
        'review_comment_title': 'post_hoc',
        'review_comment_message': 'post_hoc',
        'review_creation_date': 'post_hoc',
        'review_answer_timestamp': 'post_hoc',
    },
    'olist_products_dataset.csv': {
        'product_id': 'id',
        'product_category_name': 'feature',
        'product_name_lenght': 'feature',
        'product_description_lenght': 'feature',
        'product_photos_qty': 'feature',
        'product_weight_g': 'feature',
        'product_length_cm': 'feature',
        'product_height_cm': 'feature',
        'product_width_cm': 'feature',
    },
    'olist_sellers_dataset.csv': {
        'seller_id': 'id',
        'seller_zip_code_prefix': 'feature',
        'seller_city': 'feature',
        'seller_state': 'feature',
    },
    'product_category_name_translation.csv': {
        'product_category_name': 'feature',            # join 键
        'product_category_name_english': 'feature',
    },
}

print('\n=== 字段白名单统计 ===')
from collections import Counter
cnt = Counter()
for t, cols in WHITELIST.items():
    for c, k in cols.items():
        cnt[k] += 1
print(dict(cnt))
print('\n=== 明细(禁用/需审查/标签基准) ===')
for t, cols in WHITELIST.items():
    for c, k in cols.items():
        if k in ('post_hoc', 'review', 'label'):
            print(f'  [{k:9}] {t}.{c}')
