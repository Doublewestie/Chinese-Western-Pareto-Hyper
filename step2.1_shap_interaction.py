# step2.1_shap_interaction.py （最终修正版）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import shap

# --- 设置中文字体 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 库导入成功")

# --- 1. 加载标准化数据和最佳模型 ---
data_dir = r"E:\MathorCup\Personal-Formal\data"
output_dir = os.path.join(data_dir, "2.5-picture")

X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
X_test  = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))

# 删除泄露特征
leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
X_train = X_train.drop(columns=[col for col in leaky_cols if col in X_train.columns])
X_test  = X_test.drop(columns=[col for col in leaky_cols if col in X_test.columns])

model = joblib.load(os.path.join(data_dir, "best_model_for_shap.pkl"))
print(f"✅ 模型加载成功: {type(model).__name__}")

# --- 2. 构建解释器并计算 SHAP ---
explainer = shap.TreeExplainer(model)
X_sample = X_test.copy().reset_index(drop=True)
shap_values_raw = explainer.shap_values(X_sample)

# 处理 SHAP 格式：随机森林可能返回 [n_samples, n_features, n_classes]
if isinstance(shap_values_raw, list):
    # 取正类（索引1）
    shap_values_pos = shap_values_raw[1]
else:
    shap_values_pos = shap_values_raw

# 如果是三维，取正类的切片
if shap_values_pos.ndim == 3:
    shap_values_pos = shap_values_pos[:, :, 1]

print(f"SHAP 值形状: {shap_values_pos.shape}, 样本数: {X_sample.shape[0]}")

# --- 3. 全局重要性图 ---
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_pos, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_interaction_importance.png"), dpi=300)
plt.show()
print("✅ 特征重要性图已保存")

# --- 4. 痰湿质交互效应分析 ---
target_feature = '痰湿质'
interaction_features = ['BMI', 'TG（甘油三酯）', '活动量表总分（ADL总分+IADL总分）']

for feat in interaction_features:
    if feat not in X_sample.columns:
        continue

    plt.figure(figsize=(8, 6))
    # 显式传入一维的 SHAP 值（对应该特征）
    shap.dependence_plot(
        target_feature,
        shap_values_pos,
        X_sample,
        interaction_index=feat,
        show=False
    )
    plt.title(f"{target_feature} 与 {feat} 的交互效应")
    plt.tight_layout()
    plt.savefig(os.path.join(doutput_dir, f"shap_interaction_tanshi_vs_{feat}.png"), dpi=300)
    plt.show()
    print(f"✅ 交互图已保存: {target_feature} vs {feat}")

# --- 5. 中医特征贡献度汇总 ---
tcm_features = ['痰湿质', 'tanshi_BMI', 'tanshi_activity', '气虚质', '湿热质',
                '活动量表总分（ADL总分+IADL总分）', '平和质', '阳虚质', '阴虚质']
shap_importance = np.abs(shap_values_pos).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_sample.columns,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)

print("\n📊 中医相关特征 SHAP 贡献度:")
print(importance_df[importance_df['feature'].isin(tcm_features)])

print(f"\n✅ 所有交互分析结果已保存至 {output_dir}")