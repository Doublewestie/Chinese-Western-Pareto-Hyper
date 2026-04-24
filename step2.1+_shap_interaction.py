# step2.1+_mlp_permutation.py
# 神经网络 (MLP) + 排列重要性：用于问题一的交叉验证

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# --- 设置中文字体与输出目录 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

data_dir = r"E:\MathorCup\Personal-Formal\data"
output_dir = os.path.join(data_dir, "2.5+-picture")
os.makedirs(output_dir, exist_ok=True)

print("✅ 库导入成功，输出目录已创建")

# --- 1. 加载标准化数据 ---
X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")

# 删除泄露特征
leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
X_train = X_train.drop(columns=[col for col in leaky_cols if col in X_train.columns])
X_test  = X_test.drop(columns=[col for col in leaky_cols if col in X_test.columns])

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 从训练集中划分一小部分作为验证集（用于早停）
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
print(f"训练子集: {X_tr.shape}, 验证子集: {X_val.shape}")

# --- 2. 构建并训练 MLP 神经网络 ---
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # 两个隐藏层
    activation='relu',
    solver='adam',
    alpha=0.0001,                   # L2 正则化系数
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,                   # 最大迭代次数
    early_stopping=True,            # 启用早停
    validation_fraction=0.2,        # 内部验证集比例（但我们已经手动划分，这里设为0以禁用内部划分）
    n_iter_no_change=10,            # 连续10轮不提升则停止
    random_state=42,
    verbose=False
)

print("开始训练 MLP 神经网络...")
# 使用手动划分的验证集进行早停监控
mlp.fit(X_tr, y_tr)
print("✅ MLP 训练完成")

# --- 3. 模型性能评估 ---
y_prob = mlp.predict_proba(X_test)[:, 1]
y_pred = mlp.predict(X_test)

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nMLP 测试集性能:")
print(f"  AUC: {auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1-score: {f1:.4f}")

# 保存性能指标
with open(os.path.join(output_dir, "mlp_performance.txt"), 'w', encoding='utf-8') as f:
    f.write(f"AUC: {auc:.4f}\nAccuracy: {acc:.4f}\nF1-score: {f1:.4f}\n")

# --- 4. 计算排列重要性（重复10次取平均，增强稳定性）---
print("\n计算排列重要性（重复10次，以 AUC 为评分指标）...")
n_repeats = 10
perm_importance = permutation_importance(
    mlp, X_test, y_test,
    scoring='roc_auc',
    n_repeats=n_repeats,
    random_state=42,
    n_jobs=-1
)

# 提取重要性均值和标准差
importance_mean = perm_importance.importances_mean
importance_std = perm_importance.importances_std

# 构建 DataFrame
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': importance_mean,
    'std': importance_std
}).sort_values('importance', ascending=False)

print("✅ 排列重要性计算完成")

# --- 5. 可视化：特征重要性条形图 ---
plt.figure(figsize=(12, 10))
top_n = min(25, len(importance_df))
top_features = importance_df.head(top_n)
plt.barh(range(top_n), top_features['importance'], xerr=top_features['std'], color='steelblue')
plt.yticks(range(top_n), top_features['feature'])
plt.gca().invert_yaxis()
plt.xlabel('排列重要性 (AUC 下降均值)')
plt.title('MLP 神经网络特征重要性 (排列重要性)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mlp_permutation_importance.png"), dpi=300, bbox_inches='tight')
plt.show()
print("✅ 特征重要性图已保存")

# 保存完整特征重要性表
importance_df.to_csv(os.path.join(output_dir, "mlp_feature_importance.csv"), index=False,encoding='utf-8-sig')

# --- 6. 输出 Top 15 重要特征与九种体质贡献度 ---
print("\n🔝 MLP Top 15 重要特征:")
print(importance_df.head(15)[['feature', 'importance']])

tizhi_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
tizhi_importance = importance_df[importance_df['feature'].isin(tizhi_cols)].copy()
tizhi_importance = tizhi_importance.sort_values('importance', ascending=False)

print("\n📊 MLP 九种体质贡献度排名 (排列重要性):")
print(tizhi_importance[['feature', 'importance']])

# 保存九种体质排名到 CSV
tizhi_importance.to_csv(os.path.join(output_dir, "mlp_tizhi_importance.csv"), index=False,encoding='utf-8-sig')

print(f"\n✅ 所有分析结果已保存至 {output_dir}")