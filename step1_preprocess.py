# step1_preprocess.py
# 数据预处理与特征工程（含标准化输出）

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("✅ 库导入成功")

# --- 1. 读取数据 ---
file_path = r"E:\MathorCup\Personal-Formal\data\附件1：样例数据.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

print("数据形状:", df.shape)
print("缺失值统计:\n", df.isnull().sum().sum())  # 总缺失数

# --- 2. 目标变量与初始特征选择 ---
y = df['高血脂症二分类标签']

feature_cols = [
    '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
    'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）',
    '空腹血糖', '血尿酸', 'BMI',
    '年龄组', '性别', '吸烟史', '饮酒史'
]

X = df[feature_cols].copy()
print("特征矩阵形状:", X.shape)

# --- 3. 特征工程 ---
# 3.1 血脂异常标记（注意：这些特征后续会被标记为泄露特征，在建模时删除）
X['TC_abnormal'] = ((X['TC（总胆固醇）'] < 3.1) | (X['TC（总胆固醇）'] > 6.2)).astype(int)
X['TG_abnormal'] = ((X['TG（甘油三酯）'] < 0.56) | (X['TG（甘油三酯）'] > 1.7)).astype(int)
X['LDL_abnormal'] = ((X['LDL-C（低密度脂蛋白）'] < 2.07) | (X['LDL-C（低密度脂蛋白）'] > 3.1)).astype(int)
X['HDL_abnormal'] = ((X['HDL-C（高密度脂蛋白）'] < 1.04) | (X['HDL-C（高密度脂蛋白）'] > 1.55)).astype(int)

# 3.2 血脂异常项数
X['abnormal_lipid_count'] = X['TC_abnormal'] + X['TG_abnormal'] + X['LDL_abnormal'] + X['HDL_abnormal']

# 3.3 痰湿质交互特征
X['tanshi_BMI'] = X['痰湿质'] * X['BMI']
X['tanshi_activity'] = X['痰湿质'] * X['活动量表总分（ADL总分+IADL总分）']

# 3.4 主导体质强度
tizhi_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
X['dominant_tizhi'] = X[tizhi_cols].max(axis=1)

# 3.5 痰湿质分析特征
X['tanshi_TG'] = X['痰湿质'] * X['TG（甘油三酯）']
X['tanshi_TC'] = X['痰湿质'] * X['TC（总胆固醇）']
X['tanshi_HDL-C'] = X['痰湿质'] * X['HDL-C（高密度脂蛋白）']
X['tanshi_LDL-C'] = X['痰湿质'] * X['LDL-C（低密度脂蛋白）']
X['tanshi_血尿酸'] = X['痰湿质'] * X['血尿酸']
X['tanshi_空腹血糖'] = X['痰湿质'] * X['空腹血糖']

'''
# 3.5+ 批量构造体质与关键指标的交互特征（增强痰湿质与其他体质的对比）
tizhi_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '湿热质', '血瘀质', '气郁质', '特禀质']
# 需要交互的指标（可根据需要增减）
interact_features = [
    'TG（甘油三酯）', 'TC（总胆固醇）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
    '空腹血糖', '血尿酸', 'BMI', '活动量表总分（ADL总分+IADL总分）'
]

for tizhi in tizhi_cols:
    for feat in interact_features:
        new_col = f'{tizhi}_x_{feat}'   # 例如 "平和质_x_TG（甘油三酯）"
        X[new_col] = X[tizhi] * X[feat]

print("特征工程后形状:", X.shape)
'''

# --- 4. 编码类别变量 ---
categorical_cols = ['年龄组', '性别', '吸烟史', '饮酒史']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("编码后特征形状:", X.shape)

# --- 5. 划分训练集和测试集 ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
print(f"训练集正样本比例: {y_train.mean():.2%}, 测试集正样本比例: {y_test.mean():.2%}")

# --- 6. 标准化（在划分后进行，仅用训练集 fit）---
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

print("✅ 标准化完成")

# --- 7. 保存数据 ---
output_dir = r"E:\MathorCup\Personal-Formal\data"
os.makedirs(output_dir, exist_ok=True)

# 保存原始未标准化数据（备用）
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False,encoding='utf-8-sig')
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False,encoding='utf-8-sig')

# 保存标准化后的数据（用于建模）
X_train_scaled.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False,encoding='utf-8-sig')
X_test_scaled.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False,encoding='utf-8-sig')

# 保存标签
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False,encoding='utf-8-sig')
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False,encoding='utf-8-sig')

print(f"✅ 所有数据已保存至 {output_dir}")