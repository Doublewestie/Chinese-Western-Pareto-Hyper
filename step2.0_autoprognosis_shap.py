# step2_autoprognosis_shap.py
# 最终版：读取标准化数据、移除泄露特征、使用标准shap、修复中文乱码

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import warnings
from pathlib import Path

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from autoprognosis.studies.classifiers import ClassifierStudy
import shap

# --- 设置 Matplotlib 中文字体，解决中文乱码问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 屏蔽无关警告
warnings.filterwarnings("ignore")

print("✅ 库导入成功")

# --- 1. 加载标准化后的数据 ---
data_dir = r"E:\MathorCup\Personal-Formal\data"

X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集正样本比例: {y_train.mean():.2%}")

# --- 2. 删除泄露特征（防止AUC虚高）---
leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
X_train = X_train.drop(columns=[col for col in leaky_cols if col in X_train.columns])
X_test  = X_test.drop(columns=[col for col in leaky_cols if col in X_test.columns])

print(f"删除泄露特征后，训练集特征数: {X_train.shape[1]}")

# 保存特征名供后续使用
feature_names = X_train.columns.tolist()

# --- 3. 创建临时工作目录与合并数据 ---
workspace = Path(tempfile.mkdtemp())
study_name = "mathorcup_c_final"

df_train = X_train.copy()
df_train["target"] = y_train.values

# --- 4. AutoPrognosis 分类器研究（正式参数）---
study = ClassifierStudy(
    study_name=study_name,
    dataset=df_train,
    target="target",
    num_iter=15,
    num_study_iter=2,
    timeout=150,
    metric="aucroc",
    workspace=workspace,
    classifiers=["logistic_regression", "lgbm","random_forest"],
)

print("开始训练 AutoPrognosis 模型（使用标准化数据），请耐心等待...")
model = study.fit()
print("✅ 模型训练完成")

# --- 5. 定义安全的预测函数（返回一维 NumPy 数组）---
def predict_proba_positive(data):
    """
    返回正类概率的一维 NumPy 数组，兼容 shap。
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)
    prob_raw = model.predict_proba(data)
    if hasattr(prob_raw, 'values'):
        prob_array = prob_raw.values
    else:
        prob_array = prob_raw
    if prob_array.ndim == 2:
        return prob_array[:, -1]
    else:
        return prob_array

# --- 6. 模型评估 ---
y_prob = predict_proba_positive(X_test)
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAutoPrognosis 测试集性能（标准化数据）:")
print(f"  AUC: {auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1-score: {f1:.4f}")

# --- 7. SHAP 分析（使用标准化数据 + 分层背景样本）---
print("\n使用标准 shap 库计算 SHAP 值...")

# 背景样本：从训练集正负样本中各选25个
pos_idx = y_train[y_train == 1].index[:25]
neg_idx = y_train[y_train == 0].index[:25]
background_idx = list(pos_idx) + list(neg_idx)
background = X_train.loc[background_idx]

# 测试样本：取前100个（正式运行）
X_test_sample = X_test.iloc[:100]

explainer = shap.KernelExplainer(predict_proba_positive, background)
shap_values_raw = explainer.shap_values(X_test_sample, nsamples=100)

if isinstance(shap_values_raw, list):
    shap_values_pos = shap_values_raw[1]  # 正类的 SHAP 值
else:
    shap_values_pos = shap_values_raw

print("✅ SHAP 值计算完成")

# --- 8. 可视化与特征重要性排名 ---
shap.summary_plot(shap_values_pos, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "shap_feature_importance.png"), dpi=300)
plt.show()
print("✅ SHAP 特征重要性图已保存")

shap_importance = np.abs(shap_values_pos).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print("\n🔝 Top 15 重要特征:")
print(importance_df.head(15))

# --- 9. 九种体质贡献度对比 ---
tizhi_cols = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
tizhi_importance = importance_df[importance_df['feature'].isin(tizhi_cols)].copy()
tizhi_importance = tizhi_importance.sort_values('shap_importance', ascending=False)

print("\n📊 九种体质 SHAP 贡献度排名:")
print(tizhi_importance)

# --- 10. 保存特征重要性表 ---
importance_df.to_csv(os.path.join(data_dir, "feature_importance_shap.csv"), index=False,encoding='utf-8-sig')

print(f"\n✅ 所有结果已保存至 {data_dir}")