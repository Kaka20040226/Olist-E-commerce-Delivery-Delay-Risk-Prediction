# -*- coding: utf-8 -*-
"""Olist 数据集 — 数据审计脚本（第 7 周 · 数据处理）
对 workspace 内 9 张 CSV 做完整性/准确性/一致性/合规性检查，输出审计结果。
"""
import pandas as pd
import numpy as np
import json, os, re
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES = [
    'olist_orders_dataset.csv',
    'olist_customers_dataset.csv',
    'olist_geolocation_dataset.csv',
    'olist_order_items_dataset.csv',
    'olist_order_payments_dataset.csv',
    'olist_order_reviews_dataset.csv',
    'olist_products_dataset.csv',
    'olist_sellers_dataset.csv',
    'product_category_name_translation.csv',
]

report = {}          # 汇总报告
detail = {}          # 每表每列明细

def pct(n, d):
    return round(100.0 * n / d, 3) if d else 0.0

# 时间列识别
DATE_COLS = {
    'olist_orders_dataset.csv': ['order_purchase_timestamp', 'order_approved_at',
                                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                                 'order_estimated_delivery_date'],
    'olist_order_reviews_dataset.csv': ['review_creation_date', 'review_answer_timestamp'],
    'olist_order_items_dataset.csv': ['shipping_limit_date'],
}

print('=' * 90)
print('一、逐表审计（行数 / 列数 / 重复行 / 每列缺失与类型）')
print('=' * 90)

for t in TABLES:
    path = os.path.join(DATA_DIR, t)
    df = pd.read_csv(path, low_memory=False)
    n_rows, n_cols = df.shape
    dup_rows = int(df.duplicated().sum())

    # 列名是否有纯空白/隐藏空格
    bad_cols = [c for c in df.columns if c != c.strip()]
    # 空字符串计数（对象列）
    empty_str = {}
    for c in df.columns:
        if df[c].dtype == object:
            es = int((df[c].astype(str).str.strip() == '').sum())
            if es:
                empty_str[c] = es

    rep = {
        'rows': n_rows, 'cols': n_cols, 'duplicated_rows': dup_rows,
        'dup_pct': pct(dup_rows, n_rows),
        'columns_with_whitespace': bad_cols,
        'empty_string_cols': empty_str,
    }
    report[t] = rep

    print(f'\n【{t}】 {n_rows} 行 × {n_cols} 列 | 完全重复行 {dup_rows} ({rep["dup_pct"]}%)')
    if bad_cols:
        print(f'   ! 列名含空白: {bad_cols}')
    if empty_str:
        print(f'   ! 含空字符串的列: {empty_str}')

    col_detail = {}
    for c in df.columns:
        nulls = int(df[c].isna().sum())
        uniq = int(df[c].nunique(dropna=True))
        dt = str(df[c].dtype)
        col_detail[c] = {'dtype': dt, 'nulls': nulls, 'null_pct': pct(nulls, n_rows), 'unique': uniq}
        flag = ''
        if nulls:
            flag += f' [缺失 {nulls} ({pct(nulls,n_rows)}%)]'
        if df[c].dtype == object and uniq == 1:
            flag += ' [仅1个唯一值]'
        print(f'     {c:<34} {dt:<9} null={nulls:<6} uniq={uniq}{flag}')
    detail[t] = col_detail

print('\n' + '=' * 90)
print('二、关联键完整性（外键能否在父表找到）')
print('=' * 90)

orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
customers = pd.read_csv(os.path.join(DATA_DIR, 'olist_customers_dataset.csv'))
items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_payments_dataset.csv'))
reviews = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_reviews_dataset.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(DATA_DIR, 'olist_sellers_dataset.csv'))
geoloc = pd.read_csv(os.path.join(DATA_DIR, 'olist_geolocation_dataset.csv'))
trans = pd.read_csv(os.path.join(DATA_DIR, 'product_category_name_translation.csv'))

fk_checks = {}

def fk_check(name, child_df, child_col, parent_df, parent_col):
    child_vals = set(child_df[child_col].dropna())
    parent_vals = set(parent_df[parent_col].dropna())
    missing = child_vals - parent_vals
    n_missing = len(missing)
    n_orphan_rows = int((~child_df[child_col].isin(parent_vals)).sum())
    fk_checks[name] = {'child': f'{child_col}', 'parent': f'{parent_col}',
                       'orphan_rows': n_orphan_rows, 'distinct_missing_keys': n_missing,
                       'sample_missing': list(missing)[:5]}
    print(f'  {name:<40} 孤立行 {n_orphan_rows:<8} 缺失键 {n_missing:<6}'
          f'{" 示例:"+str(list(missing)[:3]) if missing else ""}')

fk_check('items.order_id -> orders.order_id', items, 'order_id', orders, 'order_id')
fk_check('items.product_id -> products.product_id', items, 'product_id', products, 'product_id')
fk_check('items.seller_id -> sellers.seller_id', items, 'seller_id', sellers, 'seller_id')
fk_check('payments.order_id -> orders.order_id', payments, 'order_id', orders, 'order_id')
fk_check('reviews.order_id -> orders.order_id', reviews, 'order_id', orders, 'order_id')
fk_check('orders.customer_id -> customers.customer_id', orders, 'customer_id', customers, 'customer_id')

# 品类名 -> 翻译表
cat_used = set(products['product_category_name'].dropna())
cat_trans = set(trans['product_category_name'].dropna())
cat_missing = cat_used - cat_trans
print(f'  {"products.category_name -> translation":<40} 未翻译品类 {len(cat_missing)}'
      f'{" 示例:"+str(list(cat_missing)[:5]) if cat_missing else ""}')
fk_checks['products.category_name -> translation'] = {
    'distinct_missing_keys': len(cat_missing), 'sample_missing': list(cat_missing)[:5]}

print('\n' + '=' * 90)
print('三、关键字段异常值与一致性')
print('=' * 90)

anomaly = {}

# 价格/运费/支付金额：负值、零值、极端值
for name, df, cols in [
    ('items', items, ['price', 'freight_value']),
    ('payments', payments, ['payment_value']),
]:
    for c in cols:
        s = df[c]
        neg = int((s < 0).sum())
        zero = int((s == 0).sum())
        q99 = s.quantile(0.99)
        extreme = int((s > s.quantile(0.999)).sum())
        anomaly[f'{name}.{c}'] = {'neg': neg, 'zero': zero, 'max': float(s.max()),
                                  'p999': float(s.quantile(0.999)), 'n_gt_p999': extreme}
        print(f'  {name}.{c:<20} 负值 {neg:<5} 零值 {zero:<6} max={s.max():.2f}'
              f'  p999={s.quantile(0.999):.2f} (>p999 共 {extreme})')

# 商品物理属性：0 值或负值
for c in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
    s = products[c]
    zero = int((s == 0).sum())
    neg = int((s < 0).sum())
    anomaly[f'products.{c}'] = {'zero': zero, 'neg': neg, 'max': float(s.max())}
    print(f'  products.{c:<20} 零值 {zero:<5} 负值 {neg:<5} max={s.max()}')

# 订单状态分布
print('\n  订单状态分布:')
print(orders['order_status'].value_counts(dropna=False).to_string())

# 支付方式分布
print('\n  支付方式分布:')
print(payments['payment_type'].value_counts(dropna=False).to_string())

# 州缩写合规性（巴西 27 州）
BR_STATES = {'AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC','SP','SE','TO'}
for name, df, col in [('customers', customers, 'customer_state'),
                      ('sellers', sellers, 'seller_state'),
                      ('geolocation', geoloc, 'geolocation_state')]:
    vals = set(df[col].dropna().astype(str))
    bad = vals - BR_STATES
    print(f'  {name}.{col} 非法州值: {sorted(bad) if bad else "无"}')
    anomaly[f'{name}.{col}_invalid'] = sorted(bad)

print('\n' + '=' * 90)
print('四、时间字段格式与逻辑')
print('=' * 90)

date_issues = {}
for t, cols in DATE_COLS.items():
    df = pd.read_csv(os.path.join(DATA_DIR, t), low_memory=False)
    for c in cols:
        s = df[c].astype(str)
        nulls = int(df[c].isna().sum())
        # 尝试解析 datetime
        parsed = pd.to_datetime(df[c], errors='coerce')
        unparseable = int((parsed.isna() & df[c].notna()).sum())
        date_issues[f'{t}.{c}'] = {'nulls': nulls, 'unparseable': unparseable,
                                   'min': str(parsed.min()), 'max': str(parsed.max())}
        print(f'  {t}.{c:<30} 空 {nulls:<6} 无法解析 {unparseable:<5} '
              f'范围 {parsed.min()} ~ {parsed.max()}')

# 订单表时间逻辑（delivered 状态下）
d = orders[orders['order_status'] == 'delivered'].copy()
for col in ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']:
    d[col] = pd.to_datetime(d[col], errors='coerce')
bad_seq_approved = int((d['order_approved_at'] < d['order_purchase_timestamp']).sum())
bad_seq_carrier = int((d['order_delivered_carrier_date'] < d['order_approved_at']).sum())
bad_seq_customer = int((d['order_delivered_customer_date'] < d['order_delivered_carrier_date']).sum())
print(f'\n  时间顺序异常（delivered 内）: approved<purchase {bad_seq_approved}; '
      f'carrier<approved {bad_seq_carrier}; customer<carrier {bad_seq_customer}')
anomaly['orders_time_seq'] = {'approved_lt_purchase': bad_seq_approved,
                              'carrier_lt_approved': bad_seq_carrier,
                              'customer_lt_carrier': bad_seq_customer}

# 标签可构造性复核（与计划口径一致：delivered 且两时间字段完整）
delivered = orders[orders['order_status'] == 'delivered']
has_both = delivered[['order_delivered_customer_date','order_estimated_delivery_date']].notna().all(axis=1)
n_has = int(has_both.sum())
late = (pd.to_datetime(delivered.loc[has_both,'order_delivered_customer_date']) >
        pd.to_datetime(delivered.loc[has_both,'order_estimated_delivery_date'])).sum()
print(f'\n  标签复核: delivered={len(delivered)}, 两时间字段完整={n_has}, '
      f'延迟={int(late)}, 延迟率={pct(int(late), n_has)}%')

# 汇总输出到 JSON
out = {'generated_at': datetime.now().isoformat(), 'tables': report,
       'fk_checks': fk_checks, 'anomaly': anomaly, 'date_issues': date_issues}
out_path = os.path.join(DATA_DIR, 'audit_result.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print('\n审计结果已写入: ' + out_path)
