# -*- coding: utf-8 -*-
"""Olist 延迟配送预测 — 第 3 步：数据清洗
处理审计报告中的高/中/低危问题，产出清洗后的各表 + 清洗留痕。
清洗策略（含留痕）：
  H1 geolocation 完全重复行去重 + 每 zip_prefix 聚合成唯一代表坐标
  H2 order_items.shipping_limit_date 4 行日期越界(2020年) → 置 NaN
  H3 products 610 商品缺品类 → 填 'unknown'
  M3 category_translation 补 2 个缺失翻译
  L1 products 列名拼写 lenght → length
  L2 products weight=0(4行) 与物理属性缺失(2行) → 同品类中位数回填
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(DATA_DIR, 'cleaned')
os.makedirs(CLEAN_DIR, exist_ok=True)

log = []  # 清洗留痕
def note(s):
    log.append(s)
    print(s)

# ============ 1. products ============
products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))

# L1: 列名重命名
products = products.rename(columns={
    'product_name_lenght': 'product_name_length',
    'product_description_lenght': 'product_description_length'})
note(f'[products] L1 列名重命名 lenght→length (2 列)')

# H3: 品类缺失 → unknown
n_cat_null = int(products['product_category_name'].isna().sum())
products['product_category_name'] = products['product_category_name'].fillna('unknown')
note(f'[products] H3 品类缺失 {n_cat_null} → "unknown"')

# L2: weight=0 → NaN；物理属性缺失 → 同品类中位数回填
n_w0 = int((products['product_weight_g'] == 0).sum())
products.loc[products['product_weight_g'] == 0, 'product_weight_g'] = np.nan
note(f'[products] L2 weight=0 共 {n_w0} 行 → NaN')

for c in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
    n_null = int(products[c].isna().sum())
    if n_null:
        # 同品类中位数，品类 unknown 的用全局中位数兜底
        products[c] = products.groupby('product_category_name')[c].transform(
            lambda s: s.fillna(s.median()))
        global_med = products[c].median()
        still = int(products[c].isna().sum())
        products[c] = products[c].fillna(global_med)
        note(f'[products] L2 {c} 缺失 {n_null} → 品类中位数回填(兜底后仍缺 {still})')

products.to_csv(os.path.join(CLEAN_DIR, 'olist_products_dataset.csv'), index=False)
note('[products] 已保存 cleaned/olist_products_dataset.csv')

# ============ 2. geolocation ============
geo = pd.read_csv(os.path.join(DATA_DIR, 'olist_geolocation_dataset.csv'))
n_before = len(geo)
geo_dedup = geo.drop_duplicates()
n_dup = n_before - len(geo_dedup)
note(f'[geolocation] H1 完全重复行去重: {n_before} → {len(geo_dedup)} (去重 {n_dup} 行, {n_dup/n_before*100:.2f}%)')

# 每个 zip_prefix 聚合为唯一代表坐标（lat/lng 均值）
centroid = geo_dedup.groupby('geolocation_zip_code_prefix').agg(
    geolocation_lat=('geolocation_lat', 'mean'),
    geolocation_lng=('geolocation_lng', 'mean'),
    geolocation_city=('geolocation_city', 'first'),
    geolocation_state=('geolocation_state', 'first'),
).reset_index()
note(f'[geolocation] H1 zip_prefix 聚合: {len(centroid)} 个唯一 zip 代表坐标')

geo_dedup.to_csv(os.path.join(CLEAN_DIR, 'olist_geolocation_dataset.csv'), index=False)
centroid.to_csv(os.path.join(CLEAN_DIR, 'geolocation_zip_centroid.csv'), index=False)
note('[geolocation] 已保存 cleaned/olist_geolocation_dataset.csv 与 geolocation_zip_centroid.csv')

# ============ 3. order_items ============
items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
sl = pd.to_datetime(items['shipping_limit_date'], errors='coerce')
n_bad = int((sl > '2019-01-01').sum())
items.loc[sl > '2019-01-01', 'shipping_limit_date'] = np.nan
note(f'[order_items] H2 shipping_limit_date 日期越界 {n_bad} 行 → NaN')
items.to_csv(os.path.join(CLEAN_DIR, 'olist_order_items_dataset.csv'), index=False)
note('[order_items] 已保存 cleaned/olist_order_items_dataset.csv')

# ============ 4. category_translation ============
trans = pd.read_csv(os.path.join(DATA_DIR, 'product_category_name_translation.csv'))
missing_trans = {'portateis_cozinha_e_preparadores_de_alimentos': 'portable kitchen and food preparers',
                 'pc_gamer': 'gaming pc'}
for k, v in missing_trans.items():
    if k not in set(trans['product_category_name']):
        trans = pd.concat([trans, pd.DataFrame(
            {'product_category_name': [k], 'product_category_name_english': [v]})], ignore_index=True)
note(f'[translation] M3 补充 {len(missing_trans)} 个品类翻译')
trans.to_csv(os.path.join(CLEAN_DIR, 'product_category_name_translation.csv'), index=False)
note('[translation] 已保存 cleaned/product_category_name_translation.csv')

# ============ 5. 其余表（无需结构清洗，复制）============
for t in ['olist_orders_dataset.csv', 'olist_customers_dataset.csv',
          'olist_order_payments_dataset.csv', 'olist_order_reviews_dataset.csv',
          'olist_sellers_dataset.csv']:
    df = pd.read_csv(os.path.join(DATA_DIR, t))
    df.to_csv(os.path.join(CLEAN_DIR, t), index=False)
    note(f'[{t}] 无需结构清洗，已复制到 cleaned/')

# ============ 写清洗留痕 ============
log_path = os.path.join(DATA_DIR, 'cleaning_log.md')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('# 数据清洗留痕\n\n')
    f.write(f'清洗日期：2026-10-19 ｜ 复现脚本：`03_data_cleaning.py`\n\n')
    f.write('| # | 表 | 操作 | 影响行数 |\n|---|---|---|---|\n')
    f.write('| H1 | geolocation | 完全重复行去重 | ' + str(n_dup) + ' 行 (26.18%) |\n')
    f.write('| H1 | geolocation | zip_prefix 聚合为代表坐标 | 保留 ' + str(len(centroid)) + ' 个 zip |\n')
    f.write('| H2 | order_items | shipping_limit_date 日期越界置 NaN | ' + str(n_bad) + ' 行 |\n')
    f.write('| H3 | products | 品类缺失填 unknown | ' + str(n_cat_null) + ' 行 |\n')
    f.write('| M3 | translation | 补充缺失翻译 | 2 个品类 |\n')
    f.write('| L1 | products | 列名 lenght→length | 2 列 |\n')
    f.write('| L2 | products | weight=0 及物理属性缺失中位数回填 | ' + str(n_w0) + '+2 行 |\n')
    f.write('\n清洗后数据统一存放于 `cleaned/` 目录。\n')
print('\n清洗留痕已写入: ' + log_path)
