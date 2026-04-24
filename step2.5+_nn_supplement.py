# step2.5+_rf_main.py
# 元代码 A：随机森林主体分析（支持多特征子集切换，已函数化封装）
# 可作为独立脚本运行，也可被总控脚本调用

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 全局常量定义 ====================
# 血常规 + 活动量表特征列表
WESTERN_ACTIVITY_COLS = [
    'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）', '空腹血糖', '血尿酸', 'BMI',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）'
]

# 九种体质积分列表
TIZHI_COLS = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']

# 基础信息特征（One-Hot编码后可能出现的列）
BASIC_COLS = ['年龄组_2', '年龄组_3', '年龄组_4', '年龄组_5', '性别_1', '吸烟史_1', '饮酒史_1']

# 泄露特征列表（建模时需删除）
LEAKY_COLS = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']

# ==================== 特征选择函数 ====================
def select_features(X, subset_name):
    """根据子集名称返回筛选后的特征DataFrame"""
    if subset_name == "all":
        return X
    elif subset_name == "no_TCTG":
        remove_cols = ["TC（总胆固醇）", "TG（甘油三酯）"]
        keep_cols = [col for col in X.columns if col not in remove_cols]
        return X[keep_cols]
    elif subset_name == "western_activity":
        keep_cols = WESTERN_ACTIVITY_COLS
        available = [col for col in keep_cols if col in X.columns]
        return X[available]
    elif subset_name == "tizhi_only":
        available = [col for col in TIZHI_COLS if col in X.columns]
        return X[available]
    else:
        raise ValueError(f"未知的特征子集: {subset_name}")

# ==================== 主分析函数 ====================
def run_analysis(feature_set, data_dir, rf_importance_path, output_dir):
    """
    执行随机森林分析

    参数:
        feature_set: str, 特征子集名称，可选 "all", "no_TCTG", "western_activity", "tizhi_only"
        data_dir: str, 标准化数据所在目录
        output_dir: str, 结果输出目录
    """
    print(f"当前特征子集: {feature_set}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载标准化数据 ---
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")

    # --- 2. 删除泄露特征（全局操作）---
    X_train = X_train.drop(columns=[col for col in LEAKY_COLS if col in X_train.columns])
    X_test  = X_test.drop(columns=[col for col in LEAKY_COLS if col in X_test.columns])

    # --- 3. 根据配置选择特征子集 ---
    X_train_sub = select_features(X_train, feature_set)
    X_test_sub  = select_features(X_test, feature_set)
    print(f"特征子集维度: 训练集 {X_train_sub.shape}, 测试集 {X_test_sub.shape}")

    # --- 4. 随机森林超参数优化（网格搜索）---
    print("正在进行超参数网格搜索...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, 10],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf_base, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_sub, y_train)

    best_rf = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证 AUC: {grid_search.best_score_:.4f}")

    # --- 5. 测试集评估 ---
    y_prob = best_rf.predict_proba(X_test_sub)[:, 1]
    y_pred = best_rf.predict(X_test_sub)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"测试集性能: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # 保存性能指标
    with open(os.path.join(output_dir, "performance.txt"), 'w') as f:
        f.write(f"AUC: {auc:.4f}\nAccuracy: {acc:.4f}\nF1-score: {f1:.4f}\n")

    # --- 6. 特征重要性计算 ---
    # 基尼重要性
    gini_importance = best_rf.feature_importances_
    # 置换重要性
    perm_importance = permutation_importance(best_rf, X_test_sub, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importance_mean = perm_importance.importances_mean

    importance_df = pd.DataFrame({
        'feature': X_train_sub.columns,
        'gini_importance': gini_importance,
        'permutation_importance': perm_importance_mean
    }).sort_values('gini_importance', ascending=False)

    print("\n🔝 Top 15 特征重要性 (基尼):")
    print(importance_df.head(15))
    importance_df.to_csv(os.path.join(output_dir, "rf_importance.csv"), index=False,encoding='utf-8-sig')

    # --- 7. 可视化（仅当特征数足够时）---
    if X_train_sub.shape[1] >= 5:
        # 基尼重要性柱状图（前15）
        top_features = importance_df.head(15)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['feature'][::-1], top_features['gini_importance'][::-1], color='forestgreen')
        plt.xlabel('基尼重要性')
        plt.title(f'随机森林特征重要性 (Top 15) - {feature_set}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rf_gini_importance.png"), dpi=300)
        plt.close()

        # 置换重要性柱状图（前15）
        top_perm = importance_df.sort_values('permutation_importance', ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        plt.barh(top_perm['feature'][::-1], top_perm['permutation_importance'][::-1], color='steelblue')
        plt.xlabel('置换重要性 (AUC下降)')
        plt.title(f'随机森林置换重要性 (Top 15) - {feature_set}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rf_permutation_importance.png"), dpi=300)
        plt.close()

    # --- 8. 九种体质贡献度（如果存在）---
    tizhi_present = [col for col in TIZHI_COLS if col in X_train_sub.columns]
    if tizhi_present:
        tizhi_imp = importance_df[importance_df['feature'].isin(tizhi_present)].copy()
        tizhi_imp = tizhi_imp.sort_values('gini_importance', ascending=False)
        print("\n📊 九种体质贡献度 (基尼重要性):")
        print(tizhi_imp[['feature', 'gini_importance']])
        tizhi_imp.to_csv(os.path.join(output_dir, "rf_tizhi_importance.csv"), index=False,encoding='utf-8-sig')

        # 绘制体质贡献度柱状图
        plt.figure(figsize=(10, 6))
        plt.barh(tizhi_imp['feature'][::-1], tizhi_imp['gini_importance'][::-1], color='darkorange')
        plt.xlabel('基尼重要性')
        plt.title(f'九种体质贡献度 - {feature_set}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rf_tizhi_importance.png"), dpi=300)
        plt.close()

    # --- 9. 保存模型 ---
    joblib.dump(best_rf, os.path.join(output_dir, "rf_model.pkl"))

    print(f"✅ 随机森林分析完成，结果保存至 {output_dir}")

# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    # 默认使用全特征子集，可根据需要修改
    run_analysis(
        feature_set="all",
        data_dir=r"E:\MathorCup\Personal-Formal\data",
        output_dir=r"E:\MathorCup\Personal-Formal\results\rf_analysis\all"
    )