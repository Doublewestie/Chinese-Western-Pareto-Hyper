# step3.1_bootstrap_ci.py
# Bootstrap 重采样计算评分卡模型 AUC 的 95% 置信区间
# 支持原始版（11个特征）和 plus 版（19个特征）
# 不绘图，仅输出文本结果

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\results\risk_model\scorecard_bootstrap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOTSTRAP = 100
RANDOM_SEED = 42

# ==================== 特征与分箱定义 ====================
FEATURES_BASIC = [
    'TG（甘油三酯）', 'TC（总胆固醇）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
    '空腹血糖', '血尿酸', 'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）',
    '痰湿质', 'BMI'
]

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

FEATURES_PLUS = FEATURES_BASIC + [
    'tanshi_BMI', 'tanshi_activity', 'tanshi_TG', 'tanshi_TC',
    'tanshi_HDL-C', 'tanshi_LDL-C', 'tanshi_血尿酸', 'tanshi_空腹血糖'
]

INTERACTION_FEATS = [
    'tanshi_BMI', 'tanshi_activity', 'tanshi_TG', 'tanshi_TC',
    'tanshi_HDL-C', 'tanshi_LDL-C', 'tanshi_血尿酸', 'tanshi_空腹血糖'
]

# ==================== 辅助函数 ====================
def auto_bins_from_quantiles(series, n_bins=3):
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
    return sorted(set(bins))

def woe_encoding_train(df, feature, bins, target):
    df = df.reset_index(drop=True)
    target = target.reset_index(drop=True)
    df_bin = pd.cut(df[feature], bins=bins, right=False, include_lowest=True)
    cross = pd.crosstab(df_bin, target, margins=False)
    if cross.shape[0] == 0:
        return {}
    cross['total'] = cross.sum(axis=1)
    cross['pct_pos'] = cross[1] / cross['total']
    cross['pct_neg'] = cross[0] / cross['total']
    cross['woe'] = np.log((cross['pct_pos'] + 1e-10) / (cross['pct_neg'] + 1e-10))
    return cross['woe'].to_dict()

def apply_woe(df, feature, bins, woe_map):
    binned = pd.cut(df[feature], bins=bins, right=False, include_lowest=True)
    woe_vals = binned.map(woe_map)
    woe_vals.fillna(0, inplace=True)
    return woe_vals

def build_scorecard_and_auc(X, y, features, manual_bins, interaction_feats=None):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_woe = pd.DataFrame(index=X.index)
    for feat in features:
        if interaction_feats is not None and feat in interaction_feats:
            bins = auto_bins_from_quantiles(X[feat], n_bins=3)
        else:
            bins = manual_bins[feat]
        woe_map = woe_encoding_train(X, feat, bins, y)
        if not woe_map:
            X_woe[feat] = 0.0
        else:
            X_woe[feat] = apply_woe(X, feat, bins, woe_map)
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_woe, y)
    y_prob = lr.predict_proba(X_woe)[:, 1]
    auc = roc_auc_score(y, y_prob)
    return auc

# ==================== 主函数 ====================
def run_bootstrap():
    print("=" * 60)
    print("Bootstrap 重采样计算 AUC 置信区间")
    print("=" * 60)
    
    X = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze("columns")
    
    leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
    X = X.drop(columns=[col for col in leaky_cols if col in X.columns])
    
    for feat in FEATURES_BASIC:
        if feat not in X.columns:
            raise ValueError(f"特征 {feat} 不存在于数据中")
    
    n_samples = X.shape[0]
    print(f"原始训练集样本数: {n_samples}")
    print(f"正样本比例: {y.mean():.2%}")
    
    aucs_basic = []
    aucs_plus = []
    
    np.random.seed(RANDOM_SEED)
    for i in range(N_BOOTSTRAP):
        idx = resample(np.arange(n_samples), replace=True, n_samples=n_samples, random_state=i)
        X_boot = X.iloc[idx].reset_index(drop=True)
        y_boot = y.iloc[idx].reset_index(drop=True)
        
        auc_basic = build_scorecard_and_auc(
            X_boot, y_boot, FEATURES_BASIC, MANUAL_BINS, interaction_feats=None
        )
        aucs_basic.append(auc_basic)
        
        auc_plus = build_scorecard_and_auc(
            X_boot, y_boot, FEATURES_PLUS, MANUAL_BINS, interaction_feats=INTERACTION_FEATS
        )
        aucs_plus.append(auc_plus)
        
        if (i+1) % 100 == 0:
            print(f"已完成 {i+1}/{N_BOOTSTRAP} 次重采样")
    
    def compute_stats(auc_list, name):
        auc_array = np.array(auc_list)
        mean_auc = np.mean(auc_array)
        std_auc = np.std(auc_array)
        ci_lower = np.percentile(auc_array, 2.5)
        ci_upper = np.percentile(auc_array, 97.5)
        print(f"\n{name}:")
        print(f"  平均 AUC: {mean_auc:.6f}")
        print(f"  标准差:   {std_auc:.6f}")
        print(f"  95% CI:   [{ci_lower:.6f}, {ci_upper:.6f}]")
        return mean_auc, std_auc, ci_lower, ci_upper
    
    stats_basic = compute_stats(aucs_basic, "原始版（11个特征）")
    stats_plus = compute_stats(aucs_plus, "plus版（19个特征）")
    
    # 保存结果
    with open(os.path.join(OUTPUT_DIR, "bootstrap_results.txt"), 'w', encoding='utf-8') as f:
        f.write("Bootstrap 重采样结果 (n=1000)\n")
        f.write("=" * 50 + "\n")
        f.write("原始版（11个特征）:\n")
        f.write(f"  平均 AUC: {stats_basic[0]:.6f}\n")
        f.write(f"  标准差:   {stats_basic[1]:.6f}\n")
        f.write(f"  95% CI:   [{stats_basic[2]:.6f}, {stats_basic[3]:.6f}]\n\n")
        f.write("plus版（19个特征）:\n")
        f.write(f"  平均 AUC: {stats_plus[0]:.6f}\n")
        f.write(f"  标准差:   {stats_plus[1]:.6f}\n")
        f.write(f"  95% CI:   [{stats_plus[2]:.6f}, {stats_plus[3]:.6f}]\n")
    
    print(f"\n✅ Bootstrap 分析完成，结果保存至 {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_bootstrap()