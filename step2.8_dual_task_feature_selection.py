# step2.8_dual_task_feature_selection.py
# 双任务随机森林：同时预测痰湿质分数与高血脂风险
# 基于 step2.5 的 western_activity 特征集，输出综合特征重要性及重合度分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score, f1_score
import joblib
import matplotlib
matplotlib.use('Agg')   # 避免 GUI 后端问题

warnings.filterwarnings("ignore")

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\results\dual_task"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_PATH = r"E:\MathorCup\Personal-Formal\data\附件1：样例数据.xlsx"

# 特征列表（完全与 step2.5 中的 WESTERN_ACTIVITY_COLS 一致）
WESTERN_ACTIVITY_COLS = [
    'TG（甘油三酯）', 'TC（总胆固醇）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
    '空腹血糖', '血尿酸', 'BMI',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）'
]

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("双任务随机森林特征筛选")
print("=" * 60)

# 标准化后的特征数据
X_train_scaled = pd.read_csv(os.path.join(DATA_DIR, "X_train_scaled.csv"))
X_test_scaled  = pd.read_csv(os.path.join(DATA_DIR, "X_test_scaled.csv"))

# 高血脂标签（分类任务）
y_train_hyper = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze("columns")
y_test_hyper  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze("columns")

# 原始数据（用于获取痰湿质分数，并保证与 step1 相同的划分）
df_raw = pd.read_excel(RAW_DATA_PATH, sheet_name="Sheet1")
# 原始高血脂标签（用于分层）
y_raw_hyper = df_raw['高血脂症二分类标签']
# 原始痰湿质分数
y_raw_tizhi = df_raw['痰湿质']

# 重新划分训练/测试集（与 step1 完全一致：test_size=0.3, stratify=y_raw_hyper, random_state=42）
_, _, y_train_tizhi, y_test_tizhi = train_test_split(
    y_raw_tizhi, y_raw_tizhi,   # 自变量占位，实际只用标签
    test_size=0.3,
    stratify=y_raw_hyper,
    random_state=42
)

print(f"痰湿质训练集样本数: {len(y_train_tizhi)}, 测试集: {len(y_test_tizhi)}")
print(f"高血脂训练集样本数: {len(y_train_hyper)}, 测试集: {len(y_test_hyper)}")

# ==================== 2. 提取特征子集 ====================
# 确保特征列存在于标准化数据中
available_cols = [col for col in WESTERN_ACTIVITY_COLS if col in X_train_scaled.columns]
if len(available_cols) != len(WESTERN_ACTIVITY_COLS):
    missing = set(WESTERN_ACTIVITY_COLS) - set(available_cols)
    print(f"警告：以下特征在数据中不存在，将跳过：{missing}")

X_train_dual = X_train_scaled[available_cols].copy()
X_test_dual  = X_test_scaled[available_cols].copy()
print(f"实际使用的特征数量: {X_train_dual.shape[1]}")
print(f"特征列表: {available_cols}")

# ==================== 3. 任务1：随机森林回归（预测痰湿质） ====================
print("\n" + "-" * 50)
print("任务1：随机森林回归 → 预测痰湿质分数")
print("-" * 50)

# 超参数参考 step2.5 中的最佳参数（分类器调整后用于回归）
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train_dual, y_train_tizhi)

# 预测与评估
y_train_tizhi_pred = rf_reg.predict(X_train_dual)
y_test_tizhi_pred  = rf_reg.predict(X_test_dual)

train_r2 = r2_score(y_train_tizhi, y_train_tizhi_pred)
test_r2  = r2_score(y_test_tizhi, y_test_tizhi_pred)
train_mae = mean_absolute_error(y_train_tizhi, y_train_tizhi_pred)
test_mae  = mean_absolute_error(y_test_tizhi, y_test_tizhi_pred)

print(f"训练集 R² = {train_r2:.4f}, MAE = {train_mae:.2f}")
print(f"测试集 R² = {test_r2:.4f}, MAE = {test_mae:.2f}")

# 特征重要性（基尼重要性）
reg_importance = rf_reg.feature_importances_
reg_importance_df = pd.DataFrame({
    'feature': available_cols,
    'importance_reg': reg_importance
}).sort_values('importance_reg', ascending=False)

print("\n预测痰湿质 Top 5 重要特征:")
print(reg_importance_df.head(5))

# ==================== 4. 任务2：随机森林分类（预测高血脂） ====================
print("\n" + "-" * 50)
print("任务2：随机森林分类 → 预测高血脂风险")
print("-" * 50)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_dual, y_train_hyper)

# 预测概率与类别
y_train_pred_prob = rf_clf.predict_proba(X_train_dual)[:, 1]
y_test_pred_prob  = rf_clf.predict_proba(X_test_dual)[:, 1]
y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
y_test_pred  = (y_test_pred_prob >= 0.5).astype(int)

train_auc = roc_auc_score(y_train_hyper, y_train_pred_prob)
test_auc  = roc_auc_score(y_test_hyper, y_test_pred_prob)
train_acc = accuracy_score(y_train_hyper, y_train_pred)
test_acc  = accuracy_score(y_test_hyper, y_test_pred)
train_f1  = f1_score(y_train_hyper, y_train_pred)
test_f1   = f1_score(y_test_hyper, y_test_pred)

print(f"训练集 AUC = {train_auc:.4f}, Acc = {train_acc:.4f}, F1 = {train_f1:.4f}")
print(f"测试集 AUC = {test_auc:.4f}, Acc = {test_acc:.4f}, F1 = {test_f1:.4f}")

# 分类特征重要性
clf_importance = rf_clf.feature_importances_
clf_importance_df = pd.DataFrame({
    'feature': available_cols,
    'importance_clf': clf_importance
}).sort_values('importance_clf', ascending=False)

print("\n预测高血脂 Top 5 重要特征:")
print(clf_importance_df.head(5))

# ==================== 5. 保存性能指标 ====================
with open(os.path.join(OUTPUT_DIR, "performance_dual.txt"), 'w', encoding='utf-8') as f:
    f.write("双任务随机森林性能报告\n")
    f.write("=" * 50 + "\n")
    f.write("任务1：回归（痰湿质分数）\n")
    f.write(f"  训练集 R²: {train_r2:.4f}, MAE: {train_mae:.2f}\n")
    f.write(f"  测试集 R²: {test_r2:.4f}, MAE: {test_mae:.2f}\n\n")
    f.write("任务2：分类（高血脂）\n")
    f.write(f"  训练集 AUC: {train_auc:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}\n")
    f.write(f"  测试集 AUC: {test_auc:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}\n")

# ==================== 6. 综合特征重要性 ====================
# 合并两个重要性
combined = reg_importance_df.merge(clf_importance_df, on='feature', how='outer').fillna(0)

# Min-Max 归一化
def normalize(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s

combined['imp_reg_norm'] = normalize(combined['importance_reg'])
combined['imp_clf_norm'] = normalize(combined['importance_clf'])

# 综合得分
combined['score_arith'] = (combined['imp_reg_norm'] + combined['imp_clf_norm']) / 2
combined['score_geo']   = np.sqrt(combined['imp_reg_norm'] * combined['imp_clf_norm'])

# 按几何平均排序
combined_sorted = combined.sort_values('score_geo', ascending=False)

print("\n" + "=" * 50)
print("综合特征重要性排名（几何平均，越高表示同时预测两个任务的能力越强）")
print("=" * 50)
print(combined_sorted[['feature', 'importance_reg', 'importance_clf', 'score_geo', 'score_arith']].head(10))

# 保存综合重要性表
combined_sorted.to_csv(os.path.join(OUTPUT_DIR, "dual_combined_importance.csv"), index=False,encoding='utf-8-sig')
reg_importance_df.to_csv(os.path.join(OUTPUT_DIR, "rf_importance_reg.csv"), index=False,encoding='utf-8-sig')
clf_importance_df.to_csv(os.path.join(OUTPUT_DIR, "rf_importance_clf.csv"), index=False,encoding='utf-8-sig')

# ==================== 7. 可视化 ====================
# 7.1 回归任务特征重要性柱状图（Top 15）
if len(reg_importance_df) >= 5:
    top_reg = reg_importance_df.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top_reg['feature'][::-1], top_reg['importance_reg'][::-1], color='forestgreen')
    plt.xlabel('基尼重要性')
    plt.title('随机森林回归：预测痰湿质 - 特征重要性 (Top 15)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_gini_importance_reg.png"), dpi=300)
    plt.close()

# 7.2 分类任务特征重要性柱状图（Top 15）
if len(clf_importance_df) >= 5:
    top_clf = clf_importance_df.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top_clf['feature'][::-1], top_clf['importance_clf'][::-1], color='steelblue')
    plt.xlabel('基尼重要性')
    plt.title('随机森林分类：预测高血脂 - 特征重要性 (Top 15)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_gini_importance_clf.png"), dpi=300)
    plt.close()

# 7.3 综合重要性柱状图（几何平均，Top 15）
top_combined = combined_sorted.head(15)
plt.figure(figsize=(10, 6))
plt.barh(top_combined['feature'][::-1], top_combined['score_geo'][::-1], color='teal')
plt.xlabel('综合重要性得分 (几何平均)')
plt.title('双任务随机森林：同时预测痰湿质与高血脂的关键特征')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dual_combined_importance.png"), dpi=300)
plt.close()

# 7.4 新增：特征重合度并列条形图（每个特征两个任务归一化重要性）
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(combined_sorted['feature']))
width = 0.35
ax.barh(y_pos - width/2, combined_sorted['imp_reg_norm'], width, label='回归任务 (痰湿质)', color='forestgreen')
ax.barh(y_pos + width/2, combined_sorted['imp_clf_norm'], width, label='分类任务 (高血脂)', color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(combined_sorted['feature'])
ax.invert_yaxis()
ax.set_xlabel('归一化重要性')
ax.set_title('特征在双任务中的重要性对比')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dual_feature_overlap.png"), dpi=300)
plt.close()

# 7.5 新增：散点图（回归重要性 vs 分类重要性）
plt.figure(figsize=(8, 8))
plt.scatter(combined_sorted['imp_reg_norm'], combined_sorted['imp_clf_norm'], alpha=0.7, c='darkred', edgecolors='k')
for _, row in combined_sorted.iterrows():
    plt.annotate(row['feature'], (row['imp_reg_norm'], row['imp_clf_norm']), 
                 fontsize=8, alpha=0.7, xytext=(3, 3), textcoords='offset points')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='对角线 (两者相等)')
plt.xlabel('回归任务归一化重要性 (预测痰湿质)')
plt.ylabel('分类任务归一化重要性 (预测高血脂)')
plt.title('双任务特征重要性散点图\n右上角区域为双高特征')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dual_scatter_overlap.png"), dpi=300)
plt.close()

# ==================== 8. 保存模型 ====================
joblib.dump(rf_reg, os.path.join(OUTPUT_DIR, "rf_reg_tizhi.pkl"))
joblib.dump(rf_clf, os.path.join(OUTPUT_DIR, "rf_clf_hyperlipid.pkl"))

print("\n" + "=" * 60)
print(f"✅ 双任务分析完成！所有结果已保存至：{OUTPUT_DIR}")
print("   - 性能报告: performance_dual.txt")
print("   - 特征重要性: rf_importance_reg.csv, rf_importance_clf.csv, dual_combined_importance.csv")
print("   - 图表: 各自重要性图 + 综合重要性图 + 重合度并列条形图 + 散点图")
print("=" * 60)