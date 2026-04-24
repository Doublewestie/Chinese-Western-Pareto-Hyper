# step2.+_model_benchmark.py （修正版）
# 多模型对比：手动计算 AUC，确保正确选出最佳模型

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import joblib

warnings.filterwarnings("ignore")
print("✅ 库导入成功")

# --- 1. 加载标准化数据 ---
data_dir = r"E:\MathorCup\Personal-Formal\data"
X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")

print(f"训练集: {X_train.shape}, 正样本比例: {y_train.mean():.2%}")

# --- 2. 删除泄露特征 ---
leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
X_train = X_train.drop(columns=[col for col in leaky_cols if col in X_train.columns])
print(f"删除泄露特征后特征数: {X_train.shape[1]}")

# --- 3. 定义候选模型 ---
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
        use_label_encoder=False, eval_metric='logloss', verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
        class_weight='balanced', verbose=-1
    ),
}

# --- 4. 手动 5 折交叉验证（确保 AUC 正确计算）---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n开始 5 折交叉验证...")
for name, model in models.items():
    auc_list, f1_list = [], []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        y_prob = model_clone.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        auc_list.append(roc_auc_score(y_val, y_prob))
        f1_list.append(f1_score(y_val, y_pred))
    
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)
    results.append({
        'Model': name,
        'AUC_mean': auc_mean,
        'AUC_std': auc_std,
        'F1_mean': f1_mean,
        'F1_std': f1_std
    })
    print(f"{name}: AUC = {auc_mean:.4f} (±{auc_std:.4f}), F1 = {f1_mean:.4f} (±{f1_std:.4f})")

# --- 5. 选出最佳模型 ---
results_df = pd.DataFrame(results).sort_values('AUC_mean', ascending=False)
print("\n📊 模型性能排名 (按 AUC 降序):")
print(results_df.to_string(index=False,encoding='utf-8-sig'))

best_model_name = results_df.iloc[0]['Model']
best_auc = results_df.iloc[0]['AUC_mean']
print(f"\n🏆 最佳模型: {best_model_name} (AUC = {best_auc:.4f})")

# --- 6. 训练并保存最佳模型 ---
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

model_save_path = os.path.join(data_dir, "best_model_for_shap.pkl")
joblib.dump(best_model, model_save_path)
print(f"✅ 最佳模型已保存至 {model_save_path}")