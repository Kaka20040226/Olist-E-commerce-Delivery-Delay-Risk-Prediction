# -*- coding: utf-8 -*-
"""Olist 延迟配送预测 — 第 5 步：时间切分（train/valid/test）
按 order_purchase_timestamp 时间顺序切分，避免随机切分掩盖时间漂移。
比例：train 前 70% / valid 中 15% / test 最近 15%。
"""
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, 'modeling_dataset.csv'))
df['ts'] = pd.to_datetime(df['order_purchase_timestamp'])
df = df.sort_values('ts').reset_index(drop=True)

# 时间分位数切分点
q70 = df['ts'].quantile(0.70)
q85 = df['ts'].quantile(0.85)

train = df[df['ts'] <= q70].copy()
valid = df[(df['ts'] > q70) & (df['ts'] <= q85)].copy()
test = df[df['ts'] > q85].copy()

# 统计
def stat(name, sub):
    n = len(sub)
    late = int(sub['is_late'].sum())
    return (f'{name:<6} 样本 {n:>6} ({n/len(df)*100:.1f}%)  |  时间 {sub["ts"].min()} ~ {sub["ts"].max()}'
            f'  |  延迟 {late:>5}  |  延迟率 {late/n*100:.2f}%')

print('切分点: train≤', q70, ' < valid≤', q85, ' < test')
print('=' * 100)
print(stat('train', train))
print(stat('valid', valid))
print(stat('test', test))
print('=' * 100)
print('合计:', len(train) + len(valid) + len(test), '(应为', len(df), ')')
print('时间重叠检查: train.max =', train['ts'].max(), '| valid.min =', valid['ts'].min(),
      '| valid.max =', valid['ts'].max(), '| test.min =', test['ts'].min())

# 保存（去掉辅助列 ts）
split_dir = os.path.join(BASE, 'splits')
os.makedirs(split_dir, exist_ok=True)
for name, sub in [('train', train), ('valid', valid), ('test', test)]:
    sub.drop(columns=['ts']).to_csv(os.path.join(split_dir, f'{name}.csv'), index=False)
print('\n已保存: splits/train.csv, splits/valid.csv, splits/test.csv')

# 切分说明写入
summary = {
    'train': {'n': len(train), 'late': int(train['is_late'].sum()),
              'start': str(train['ts'].min()), 'end': str(train['ts'].max())},
    'valid': {'n': len(valid), 'late': int(valid['is_late'].sum()),
              'start': str(valid['ts'].min()), 'end': str(valid['ts'].max())},
    'test': {'n': len(test), 'late': int(test['is_late'].sum()),
             'start': str(test['ts'].min()), 'end': str(test['ts'].max())},
}
import json
with open(os.path.join(BASE, 'time_split_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print('切分说明已写入: time_split_summary.json')
