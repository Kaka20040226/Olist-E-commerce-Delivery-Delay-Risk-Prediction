# -*- coding: utf-8 -*-
"""Olist 延迟配送预测 — 第 4 步：多表连接聚合（9 表 → 每订单一行的建模数据集）
仅使用字段白名单中的 feature 字段（下单时点可得），事后字段一律不进入。
一对多表(order_items/order_payments)先聚合到订单层级，避免重复计数。
"""
import pandas as pd
import numpy as np
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned')
orders = pd.read_csv(os.path.join(DATA, 'olist_orders_dataset.csv'))
customers = pd.read_csv(os.path.join(DATA, 'olist_customers_dataset.csv'))
items = pd.read_csv(os.path.join(DATA, 'olist_order_items_dataset.csv'))
payments = pd.read_csv(os.path.join(DATA, 'olist_order_payments_dataset.csv'))
products = pd.read_csv(os.path.join(DATA, 'olist_products_dataset.csv'))
sellers = pd.read_csv(os.path.join(DATA, 'olist_sellers_dataset.csv'))
centroid = pd.read_csv(os.path.join(DATA, 'geolocation_zip_centroid.csv'))
label = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'label_is_late.csv'))

# ---------- 1. 样本：delivered 且含标签 ----------
df = orders.merge(label[['order_id', 'is_late']], on='order_id', how='inner').copy()
print(f'样本订单数: {len(df)} (期望 96470)')

# ---------- 2. order_items 聚合（先 join products 取品类/重量/尺寸） ----------
items_p = items.merge(products[['product_id', 'product_category_name', 'product_weight_g',
                                'product_length_cm', 'product_height_cm', 'product_width_cm']],
                      on='product_id', how='left')
items_p['item_volume_cm3'] = (items_p['product_length_cm'] * items_p['product_height_cm'] * items_p['product_width_cm'])
items_p['item_value'] = items_p['price'] + items_p['freight_value']

item_agg = items_p.groupby('order_id').agg(
    item_count=('order_item_id', 'count'),
    distinct_product_count=('product_id', 'nunique'),
    distinct_seller_count=('seller_id', 'nunique'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
    total_item_value=('item_value', 'sum'),
    total_weight_g=('product_weight_g', 'sum'),
    total_volume_cm3=('item_volume_cm3', 'sum'),
    category_count=('product_category_name', 'nunique'),
).reset_index()

# 主品类 / 主卖家：按 price 降序取第一（该订单金额最大的商品）
items_sorted = items_p.sort_values('price', ascending=False)
main_item = items_sorted.drop_duplicates('order_id', keep='first')[
    ['order_id', 'seller_id', 'product_category_name']].rename(
    columns={'seller_id': 'main_seller_id', 'product_category_name': 'main_category'})

# ---------- 3. order_payments 聚合 ----------
pay_agg = payments.groupby('order_id').agg(
    payment_count=('payment_sequential', 'count'),
    total_installments=('payment_installments', 'max'),
    total_payment_value=('payment_value', 'sum'),
).reset_index()
main_pay = payments.sort_values('payment_value', ascending=False).drop_duplicates(
    'order_id', keep='first')[['order_id', 'payment_type']].rename(
    columns={'payment_type': 'main_payment_type'})

# ---------- 4. join 维度表 ----------
df = df.merge(customers[['customer_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']],
              on='customer_id', how='left')
df = df.merge(item_agg, on='order_id', how='left')
df = df.merge(main_item, on='order_id', how='left')
df = df.merge(pay_agg, on='order_id', how='left')
df = df.merge(main_pay, on='order_id', how='left')
df = df.merge(sellers[['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state']],
              left_on='main_seller_id', right_on='seller_id', how='left',
              suffixes=('', '_seller')).drop(columns=['seller_id'])

# ---------- 5. 地理距离（主卖家 zip → 客户 zip，haversine；zip 缺失时州级坐标兜底） ----------
zc = centroid.set_index('geolocation_zip_code_prefix')
df = df.merge(zc[['geolocation_lat', 'geolocation_lng']].rename(
    columns={'geolocation_lat': 'cust_lat', 'geolocation_lng': 'cust_lng'}),
    left_on='customer_zip_code_prefix', right_index=True, how='left')
df = df.merge(zc[['geolocation_lat', 'geolocation_lng']].rename(
    columns={'geolocation_lat': 'sell_lat', 'geolocation_lng': 'sell_lng'}),
    left_on='seller_zip_code_prefix', right_index=True, how='left')

# 州级代表坐标（zip 缺失时兜底）
state_centroid = centroid.groupby('geolocation_state').agg(
    state_lat=('geolocation_lat', 'mean'),
    state_lng=('geolocation_lng', 'mean')).reset_index()
df = df.merge(state_centroid, left_on='customer_state', right_on='geolocation_state', how='left')
df['cust_lat'] = df['cust_lat'].fillna(df['state_lat'])
df['cust_lng'] = df['cust_lng'].fillna(df['state_lng'])
df = df.drop(columns=['geolocation_state', 'state_lat', 'state_lng'])
df = df.merge(state_centroid, left_on='seller_state', right_on='geolocation_state', how='left')
df['sell_lat'] = df['sell_lat'].fillna(df['state_lat'])
df['sell_lng'] = df['sell_lng'].fillna(df['state_lng'])
df = df.drop(columns=['geolocation_state', 'state_lat', 'state_lng'])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

valid = df[['cust_lat', 'cust_lng', 'sell_lat', 'sell_lng']].notna().all(axis=1)
df['seller_customer_distance_km'] = np.nan
df.loc[valid, 'seller_customer_distance_km'] = haversine(
    df.loc[valid, 'cust_lat'], df.loc[valid, 'cust_lng'],
    df.loc[valid, 'sell_lat'], df.loc[valid, 'sell_lng'])

# ---------- 5.5 缺支付订单规范化（1 单无支付记录） ----------
n_no_pay = int(df['payment_count'].isna().sum())
df['payment_count'] = df['payment_count'].fillna(0)
df['main_payment_type'] = df['main_payment_type'].fillna('unknown')
# total_installments / total_payment_value 保留 NaN（表示"无记录"，区别于 0）
print(f'缺支付订单规范化: {n_no_pay} 单 → payment_count=0, main_payment_type=unknown')

# ---------- 6. 整理最终列 ----------
final_cols = [
    'order_id', 'is_late',
    'order_purchase_timestamp',                     # 供时间切分与 W8 时间特征
    'item_count', 'distinct_product_count', 'distinct_seller_count',
    'total_price', 'total_freight', 'total_item_value',
    'total_weight_g', 'total_volume_cm3',
    'main_category', 'category_count',
    'payment_count', 'main_payment_type', 'total_installments', 'total_payment_value',
    'customer_zip_code_prefix', 'customer_city', 'customer_state',
    'seller_zip_code_prefix', 'seller_city', 'seller_state',
    'seller_customer_distance_km',
]
df = df[final_cols]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modeling_dataset.csv')
df.to_csv(out, index=False)
print(f'建模数据集已保存: {out} | 形状 {df.shape}')

# ---------- 质量核查 ----------
print('\n=== 质量核查 ===')
print('总行数:', len(df), '| 唯一 order_id:', df['order_id'].nunique())
print('is_late 分布:', df['is_late'].value_counts().to_dict(), '| 延迟率:', round(df['is_late'].mean()*100, 4), '%')
print('缺失值统计（>0 的列）:')
miss = df.isna().sum()
print(miss[miss > 0].to_string() if (miss > 0).any() else '  无缺失')
print('\n数值列描述:')
print(df[['item_count','total_price','total_freight','total_weight_g','total_volume_cm3',
          'total_installments','total_payment_value','seller_customer_distance_km']].describe().round(2).to_string())
print('\n类别字段取值数: main_category=', df['main_category'].nunique(),
      '| main_payment_type=', df['main_payment_type'].nunique(),
      '| customer_state=', df['customer_state'].nunique(),
      '| seller_state=', df['seller_state'].nunique())
