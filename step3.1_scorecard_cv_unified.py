# step3.1_scorecard_cv_unified.py
# 评分卡模型交叉验证（支持原始版和plus版）
# 对两个版本分别进行5折交叉验证，输出AUC均值、标准差及箱线图对比

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\results\risk_model\scorecard_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交叉验证折数
N_SPLITS = 5
RANDOM_STATE = 42

# ==================== 特征定义 ====================
# 原始版特征（11个）
BASE_FEATURES = [
    'TG（甘油三酯）', 'TC（总胆固醇）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
    '空腹血糖', '血尿酸', 'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
    '痰湿质', 'BMI'
]

# plus版特征（包含交互特征）
PLUS_FEATURES = BASE_FEATURES + [
    'tanshi_BMI', 'tanshi_activity', 'tanshi_TG', 'tanshi_TC',
    'tanshi_HDL-C', 'tanshi_LDL-C', 'tanshi_血尿酸', 'tanshi_空腹血糖'
]

# 原始特征手动分箱边界
MANUAL_BINS = {
    'TG（甘油三酯）': [0, 1.7, 2.3, np.inf],
    'TC（总胆固醇）': [0, 5.2, 6.2, np.inf],
    'HDL-C（高密度脂蛋白）': [0, 1.04, 1.55, np.inf],
    'LDL-C（低密度脂蛋白）': [0, 2.6, 3.4, np.inf],
    '空腹血糖': [0, 6.1, 7.0, np.inf],
    '血尿酸': [0, 420, np.inf],
    'ADL总分': [0, 14, 20, np.inf],
    'IADL总分': [0, 8, 16, np.inf],
    '活动量表总分（ADL总分+IADL总分）': [0, 40, 60, np.inf],
    '痰湿质': [0, 40, 60, np.inf],
    'BMI': [0, 24, 28, np.inf],
}

# plus版交互特征列表（用于自动分箱）
INTERACTION_FEATS = [
    'tanshi_BMI', 'tanshi_activity', 'tanshi_TG', 'tanshi_TC',
    'tanshi_HDL-C', 'tanshi_LDL-C', 'tanshi_血尿酸', 'tanshi_空腹血糖'
]

# ==================== 辅助函数 ====================
def auto_bins_from_quantiles(series, n_bins=3):
    """
    基于等频分位数生成分箱边界（自动处理边界覆盖）
    返回边界列表，保证最小值、最大值、无穷边界
    """
    if series.nunique() < n_bins:
        uniq = sorted(series.dropna().unique())
        step = max(1, len(uniq) // n_bins)
        boundaries = [uniq[0]] + [uniq[i*step] for i in range(1, n_bins)] + [uniq[-1]]
    else:
        quantiles = np.linspace(0, 100, n_bins+1)
        boundaries = np.percentile(series.dropna(), quantiles)
        boundaries = np.unique(boundaries)
        if len(boundaries) < n_bins+1:
            boundaries = np.linspace(series.min(), series.max(), n_bins+1)
    if boundaries[0] > series.min():
        boundaries[0] = series.min()
    if boundaries[-1] < series.max():
        boundaries[-1] = series.max()
    bins = boundaries.tolist()
    bins.append(np.inf)
    bins = sorted(set(bins))
    return bins

def woe_encoding_train(df, feature, bins, target):
    """基于训练集计算 WOE 映射"""
    df['bin'] = pd.cut(df[feature], bins=bins, right=False, include_lowest=True)
    cross = pd.crosstab(df['bin'], target, margins=False)
    cross['total'] = cross.sum(axis=1)
    cross['pct_pos'] = cross[1] / cross['total']
    cross['pct_neg'] = cross[0] / cross['total']
    cross['woe'] = np.log((cross['pct_pos'] + 1e-10) / (cross['pct_neg'] + 1e-10))
    return cross['woe'].to_dict()

def apply_woe(df, feature, bins, woe_map):
    """将数据转换为 WOE 值"""
    bins_cut = pd.cut(df[feature], bins=bins, right=False, include_lowest=True)
    woe_vals = bins_cut.map(woe_map)
    woe_vals.fillna(0, inplace=True)
    return woe_vals

def cross_validate_scorecard(features, bins_dict, X, y, n_splits=5, random_state=42, interaction_feats=None):
    """
    对给定特征集和分箱字典进行交叉验证
    返回：auc_scores列表
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # 构建该折的 WOE 数据
        X_train_woe = pd.DataFrame(index=X_train_fold.index)
        X_val_woe = pd.DataFrame(index=X_val_fold.index)
        
        # 对每个特征：确定分箱边界和 WOE 映射
        for feat in features:
            # 如果是交互特征，则在该折内动态分箱（基于训练折）
            if interaction_feats and feat in interaction_feats:
                bins = auto_bins_from_quantiles(X_train_fold[feat], n_bins=3)
            else:
                bins = bins_dict[feat]
            # 计算 WOE 映射（基于训练折）
            woe_map = woe_encoding_train(X_train_fold, feat, bins, y_train_fold)
            # 转换训练折和验证折
            X_train_woe[feat] = apply_woe(X_train_fold, feat, bins, woe_map)
            X_val_woe[feat] = apply_woe(X_val_fold, feat, bins, woe_map)
        
        # 训练逻辑回归
        lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, random_state=random_state)
        lr.fit(X_train_woe, y_train_fold)
        
        # 预测验证集
        y_prob = lr.predict_proba(X_val_woe)[:, 1]
        auc = roc_auc_score(y_val_fold, y_prob)
        auc_scores.append(auc)
    
    return auc_scores

# ==================== 主函数 ====================
def run_cv():
    print("=" * 60)
    print("评分卡模型交叉验证（原始版 vs plus版）")
    print("=" * 60)

    # 加载数据
    X = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze("columns")
    
    # 删除泄露特征
    leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
    X = X.drop(columns=[col for col in leaky_cols if col in X.columns])
    
    print(f"总样本数: {X.shape[0]}, 正样本比例: {y.mean():.2%}")
    
    # --- 1. 原始版交叉验证 ---
    print("\n正在执行原始版（11个特征）5折交叉验证...")
    X_base = X[BASE_FEATURES].copy()
    auc_base = cross_validate_scorecard(
        features=BASE_FEATURES,
        bins_dict=MANUAL_BINS,
        X=X_base,
        y=y,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        interaction_feats=None
    )
    mean_base = np.mean(auc_base)
    std_base = np.std(auc_base)
    print(f"原始版: 平均 AUC = {mean_base:.4f} ± {std_base:.4f}")
    print(f"各折 AUC: {[f'{x:.4f}' for x in auc_base]}")
    
    # --- 2. plus版交叉验证 ---
    print("\n正在执行plus版（含交互特征）5折交叉验证...")
    X_plus = X[PLUS_FEATURES].copy()
    auc_plus = cross_validate_scorecard(
        features=PLUS_FEATURES,
        bins_dict=MANUAL_BINS,
        X=X_plus,
        y=y,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        interaction_feats=INTERACTION_FEATS
    )
    mean_plus = np.mean(auc_plus)
    std_plus = np.std(auc_plus)
    print(f"plus版: 平均 AUC = {mean_plus:.4f} ± {std_plus:.4f}")
    print(f"各折 AUC: {[f'{x:.4f}' for x in auc_plus]}")
    
    # --- 3. 保存结果 ---
    with open(os.path.join(OUTPUT_DIR, "cv_results_unified.txt"), 'w', encoding='utf-8') as f:
        f.write("评分卡模型交叉验证结果对比\n")
        f.write("=" * 50 + "\n")
        f.write(f"原始版（{len(BASE_FEATURES)}个特征）:\n")
        f.write(f"  平均 AUC: {mean_base:.4f}\n")
        f.write(f"  标准差:   {std_base:.4f}\n")
        f.write(f"  各折 AUC: {[f'{x:.4f}' for x in auc_base]}\n\n")
        f.write(f"plus版（{len(PLUS_FEATURES)}个特征）:\n")
        f.write(f"  平均 AUC: {mean_plus:.4f}\n")
        f.write(f"  标准差:   {std_plus:.4f}\n")
        f.write(f"  各折 AUC: {[f'{x:.4f}' for x in auc_plus]}\n")
    
    # --- 4. 绘制箱线图对比 ---
    plt.figure(figsize=(6, 5))
    data = [auc_base, auc_plus]
    bp = plt.boxplot(data, labels=['原始版', 'plus版'], patch_artist=True)
    # 设置颜色
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.title('评分卡模型 5折交叉验证 AUC 对比')
    plt.ylabel('AUC')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.95, 1.005)  # 根据实际范围调整
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cv_comparison_boxplot.png"), dpi=300)
    plt.close()
    
    # 绘制折线图（各折AUC趋势）
    plt.figure(figsize=(8, 5))
    folds = range(1, N_SPLITS+1)
    plt.plot(folds, auc_base, 'o-', label='原始版', color='blue')
    plt.plot(folds, auc_plus, 's-', label='plus版', color='red')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('折数')
    plt.ylabel('AUC')
    plt.title('各折 AUC 变化趋势')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cv_fold_trend.png"), dpi=300)
    plt.close()
    
    print(f"\n✅ 交叉验证完成，结果保存至 {OUTPUT_DIR}")
    print("=" * 60)

# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    run_cv()