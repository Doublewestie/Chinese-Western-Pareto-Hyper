# step3_scorecard.py
# 基于评分卡的高血脂风险预警模型（问题2）
# 输出：特征分值表、三级风险等级、阈值依据、高风险人群核心特征组合

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import joblib

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\results\risk_model\scorecard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 选择的特征（基于随机森林重要性和临床意义）
FEATURES = [
    'TG（甘油三酯）',
    'TC（总胆固醇）',
    'HDL-C（高密度脂蛋白）',
    'LDL-C（低密度脂蛋白）',
    '空腹血糖', 
    '血尿酸', 
    'ADL总分', 
    'IADL总分', 
    '活动量表总分（ADL总分+IADL总分）',
    '痰湿质',
    'BMI'
]

# 特征分箱边界（手动设定，结合临床阈值和数据分布）
# 每个特征的分箱边界列表，例如 [0, 1.7, 2.3, np.inf]
BINS = {
    # 血脂指标
    'TG（甘油三酯）': [0, 1.7, 2.3, np.inf],           # 正常 <1.7, 边缘 1.7-2.3, 高 ≥2.3
    'TC（总胆固醇）': [0, 5.2, 6.2, np.inf],           # 正常 <5.2, 边缘 5.2-6.2, 高 ≥6.2
    'HDL-C（高密度脂蛋白）': [0, 1.04, 1.55, np.inf], # 低 <1.04, 正常 1.04-1.55, 高 ≥1.55
    'LDL-C（低密度脂蛋白）': [0, 2.6, 3.4, np.inf],    # 理想 <2.6, 边缘 2.6-3.4, 高 ≥3.4

    # 其他生化指标
    '空腹血糖': [0, 6.1, 7.0, np.inf],                # 正常 <6.1, 空腹血糖受损 6.1-7.0, 糖尿病 ≥7.0
    '血尿酸': [0, 420, np.inf],                       # 男性正常 <420, 女性 <360, 此处取 420 作为通用阈值

    # 活动能力指标
    'ADL总分': [0, 14, 20, np.inf],                   # 根据 ADL 量表通常 0-14 表示不同程度依赖，可参考数据分布
    'IADL总分': [0, 8, 16, np.inf],                   # IADL 通常 0-8 低功能，8-16 中等，>16 高功能（实际范围 0-40）
    '活动量表总分（ADL总分+IADL总分）': [0, 40, 60, np.inf],  # 低活动 <40, 中 40-60, 高 ≥60

    # 体质与形态
    '痰湿质': [0, 40, 60, np.inf],                   # 低积分 <40, 中 40-60, 高 ≥60
    'BMI': [0, 24, 28, np.inf],                      # 正常 <24, 超重 24-28, 肥胖 ≥28
}

# 风险等级总分阈值（低/中/高）
RISK_THRESHOLDS = (30, 60)   # 总分 <30 低风险，30-49 中风险，≥50 高风险

# ==================== 辅助函数 ====================
def woe_encoding(df, feature, bins, target):
    """
    对单个特征进行分箱并计算WOE值
    返回：woe_map (箱索引 -> WOE值), 分箱后的列名
    """
    # 分箱
    df['bin'] = pd.cut(df[feature], bins=bins, right=False, include_lowest=True)
    # 计算每个箱的正负样本数
    cross = pd.crosstab(df['bin'], target, margins=False)
    cross['total'] = cross.sum(axis=1)
    cross['pct_pos'] = cross[1] / cross['total']
    cross['pct_neg'] = cross[0] / cross['total']
    # 计算WOE = ln(pct_pos / pct_neg)
    cross['woe'] = np.log((cross['pct_pos'] + 1e-10) / (cross['pct_neg'] + 1e-10))
    woe_map = cross['woe'].to_dict()
    return woe_map, cross

def scorecard_scaling(coef, woe_map, intercept, target_score=600, target_odds=50, pdo=20):
    """
    将逻辑回归系数转换为评分卡分数
    参考标准评分卡刻度方法：Score = A - B * ln(odds)
    target_score: 在目标 odds 时的分数
    target_odds: 目标 odds
    pdo: odds 翻倍时分数增加量
    """
    # 计算 B 和 A
    B = pdo / np.log(2)
    A = target_score - B * np.log(target_odds)
    # 每个特征的基础分数 = - (系数 * WOE) * B + (A / 特征数) 的分配可简化
    # 简化：直接计算每个分箱的分数 = -coef * woe * B
    scores = {}
    base_score = A / len(coef)  # 平均分配截距
    for feat, woe in woe_map.items():
        scores[feat] = -coef[feat] * woe * B
    return scores, base_score, B

def calculate_total_score(df, feature_scores, base_score):
    """根据特征分箱分数计算每个样本的总分"""
    total = base_score
    for feat, score_dict in feature_scores.items():
        # 获取样本的分箱索引（这里需要重新分箱，简化：直接用原始特征匹配）
        # 更简单：在训练时记录每个样本的分箱，然后映射分数
        pass
    return total

# ==================== 主分析函数 ====================
def run_scorecard(data_dir, output_dir):
    print("=" * 60)
    print("评分卡风险预警模型构建")
    print("=" * 60)

    # --- 1. 加载原始数据（未标准化）---
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")

    # 删除泄露特征（如果存在）
    leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
    X_train = X_train.drop(columns=[col for col in leaky_cols if col in X_train.columns])
    X_test  = X_test.drop(columns=[col for col in leaky_cols if col in X_test.columns])

    # 只保留选定的特征
    X_train = X_train[FEATURES].copy()
    X_test  = X_test[FEATURES].copy()

    print(f"训练集样本数: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    print(f"特征列表: {FEATURES}")

    # --- 2. 对每个特征进行分箱并计算 WOE ---
    woe_maps = {}
    bin_tables = {}
    X_train_woe = pd.DataFrame(index=X_train.index)
    X_test_woe = pd.DataFrame(index=X_test.index)

    for feat in FEATURES:
        bins = BINS[feat]
        # 计算 WOE（基于训练集）
        woe_map, cross = woe_encoding(X_train, feat, bins, y_train)
        woe_maps[feat] = woe_map
        bin_tables[feat] = cross
        # 将训练集和测试集的分箱替换为 WOE 值
        # 对训练集
        bins_cut = pd.cut(X_train[feat], bins=bins, right=False, include_lowest=True)
        X_train_woe[feat] = bins_cut.map(woe_map)
        # 对测试集
        bins_cut_test = pd.cut(X_test[feat], bins=bins, right=False, include_lowest=True)
        X_test_woe[feat] = bins_cut_test.map(woe_map)
        # 处理缺失值（如果有）
        X_train_woe[feat].fillna(0, inplace=True)
        X_test_woe[feat].fillna(0, inplace=True)

        print(f"\n特征 {feat} 分箱及 WOE:")
        print(cross)

    # --- 3. 训练逻辑回归模型（以 WOE 值为输入）---
    lr = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000, random_state=42)
    lr.fit(X_train_woe, y_train)

    # 模型系数和截距
    coef = dict(zip(FEATURES, lr.coef_[0]))
    intercept = lr.intercept_[0]
    print(f"\n逻辑回归系数: {coef}")
    print(f"截距: {intercept:.4f}")

    # 预测测试集概率
    y_prob = lr.predict_proba(X_test_woe)[:, 1]
    y_pred = lr.predict(X_test_woe)

    # 评估性能
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n评分卡模型测试集性能:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    # 保存性能
    with open(os.path.join(output_dir, "scorecard_performance.txt"), 'w', encoding='utf-8') as f:
        f.write("评分卡模型性能\n")
        f.write("=" * 40 + "\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # --- 4. 计算每个特征分箱的得分（评分卡刻度）---
    # 使用标准评分卡刻度方法：Score = A - B * ln(odds)
    # 设定目标分数 600 对应 odds=50（即高风险概率 ~0.98），PDO=20（odds每翻倍分数增加20）
    # 但为了更直观，我们直接计算每个特征分箱的贡献分数
    # 总分数 = 截距分数 + Σ(系数 * WOE) * 缩放因子
    # 为方便理解，将总分范围映射到 0-100 分
    # 先计算线性预测值 lp = intercept + Σ(coef_i * woe_i)
    # 分数 = 100 / (max_lp - min_lp) * (lp - min_lp)  （线性映射到 0-100）
    # 或者使用标准评分卡公式，但为了直观，使用线性映射

    # 计算训练集的线性预测值
    lp_train = lr.decision_function(X_train_woe)
    min_lp = lp_train.min()
    max_lp = lp_train.max()
    print(f"\n训练集线性预测值范围: [{min_lp:.4f}, {max_lp:.4f}]")

    # 映射函数：lp -> 0-100 分
    def lp_to_score(lp):
        return 100 * (lp - min_lp) / (max_lp - min_lp)

    # 计算每个样本的总分
    train_score = lp_to_score(lp_train)
    test_score = lp_to_score(lr.decision_function(X_test_woe))

    # 确定风险等级阈值（基于总分）
    # 使用训练集总分的 33% 和 67% 分位数，或自定义阈值
    low_thresh = np.percentile(train_score, 33)
    mid_thresh = np.percentile(train_score, 67)
    print(f"\n训练集总分分位数: 33% = {low_thresh:.2f}, 67% = {mid_thresh:.2f}")

    # 也可以使用固定阈值（如 30, 50），但根据数据分布调整
    # 这里使用自定义阈值（可在代码中修改）
    # 为了与之前随机森林一致，使用分位数阈值
    risk_level = []
    for s in test_score:
        if s < low_thresh:
            risk_level.append('低')
        elif s < mid_thresh:
            risk_level.append('中')
        else:
            risk_level.append('高')
    risk_level = pd.Series(risk_level, index=y_test.index)

    # 保存风险等级及总分
    risk_df = pd.DataFrame({
        'sample_id': y_test.index,
        'score': test_score,
        'risk_level': risk_level,
        'risk_probability': y_prob
    })
    risk_df.to_csv(os.path.join(output_dir, "risk_levels_scorecard.csv"), index=False, encoding='utf-8-sig')

    # 保存阈值依据
    with open(os.path.join(output_dir, "risk_thresholds_scorecard.txt"), 'w', encoding='utf-8') as f:
        f.write("评分卡风险分层阈值依据\n")
        f.write("=" * 40 + "\n")
        f.write(f"总分范围: 0-100 分\n")
        f.write(f"低风险 ↔ 中风险 总分阈值: {low_thresh:.2f} (训练集33%分位数)\n")
        f.write(f"中风险 ↔ 高风险 总分阈值: {mid_thresh:.2f} (训练集67%分位数)\n")
        f.write("阈值选择理由：基于训练集总分的分位数，确保每个风险等级有足够的样本量。\n")
        f.write("注意：总分越高，高血脂风险越大。\n")

    # --- 5. 计算每个特征分箱的贡献分数（用于解释）---
    # 每个特征每个分箱的得分 = -coef * woe * (100 / (max_lp - min_lp))
    scale = 100 / (max_lp - min_lp)
    feature_score_maps = {}
    for feat in FEATURES:
        woe_map = woe_maps[feat]
        score_map = {}
        for bin_label, woe_val in woe_map.items():
            # 该分箱的得分贡献 = -coef[feat] * woe_val * scale + 截距的分配（截距平均分配到所有分箱）
            # 简化：只展示系数贡献，截距部分单独加在总分中
            score = -coef[feat] * woe_val * scale
            score_map[bin_label] = score
        feature_score_maps[feat] = score_map

    # 生成特征分值表（每个特征的分箱及对应分数）
    scorecard_table = []
    for feat in FEATURES:
        bins = BINS[feat]
        # 获取分箱区间字符串
        bin_intervals = pd.cut(X_train[feat], bins=bins, right=False, include_lowest=True).cat.categories
        woe_map = woe_maps[feat]
        for interval, woe in woe_map.items():
            # 找到对应的分数
            score = feature_score_maps[feat][interval]
            scorecard_table.append({
                '特征': feat,
                '分箱区间': str(interval),
                'WOE': f"{woe:.4f}",
                '分数贡献': f"{score:.2f}"
            })
    scorecard_df = pd.DataFrame(scorecard_table)
    scorecard_df.to_csv(os.path.join(output_dir, "scorecard_feature_scores.csv"), index=False, encoding='utf-8-sig')
    print("\n特征分值表已保存至 scorecard_feature_scores.csv")

    # 生成真实贡献表
    feature_contribution = {}
    for feat in FEATURES:
        scores = list(feature_score_maps[feat].values())
        contribution = max(scores) - min(scores)
        feature_contribution[feat] = contribution

    contrib_df = pd.DataFrame(list(feature_contribution.items()), columns=['feature', 'contribution'])
    contrib_df = contrib_df.sort_values('contribution', ascending=False)
    contrib_df.to_csv(os.path.join(output_dir, "scorecard_feature_contribution.csv"), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    plt.barh(contrib_df['feature'][::-1], contrib_df['contribution'][::-1], color='teal')
    plt.xlabel('特征对总分的最大贡献（分数变化范围）')
    plt.title('评分卡特征真实贡献排序')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scorecard_feature_contribution.png"), dpi=300)
    plt.close()

    # --- 6. 高风险人群核心特征组合挖掘（基于总分高风险样本）---
    high_risk_mask = risk_level == '高'
    if high_risk_mask.sum() >= 5:
        from mlxtend.frequent_patterns import apriori, association_rules
        from sklearn.preprocessing import KBinsDiscretizer
        
        # 选择用于规则挖掘的特征（连续特征离散化）
        rule_features = ['痰湿质', 'BMI', '活动量表总分（ADL总分+IADL总分）', 'TG（甘油三酯）', 'TC（总胆固醇）']
        available_rule = [f for f in rule_features if f in X_test.columns]
        X_high = X_test.loc[high_risk_mask, available_rule].copy()
        # 离散化
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        X_high_disc = pd.DataFrame(discretizer.fit_transform(X_high), columns=available_rule, index=X_high.index)
        for col in available_rule:
            X_high_disc[col] = X_high_disc[col].map({0: '低', 1: '中', 2: '高'})
        # One-hot 编码
        X_high_onehot = pd.get_dummies(X_high_disc)
        # 挖掘频繁项集
        frequent_itemsets = apriori(X_high_onehot, min_support=0.1, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
            rules = rules.sort_values('support', ascending=False)
            core_rules = rules[['antecedents', 'consequents', 'support', 'confidence']].head(5).copy()
            core_rules['antecedents'] = core_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            core_rules['consequents'] = core_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            core_rules.to_csv(os.path.join(output_dir, "core_feature_combinations_scorecard.csv"), index=False, encoding='utf-8-sig')
            print("\n高风险人群核心特征组合（关联规则）已保存。")
        else:
            print("未发现满足最小支持度的频繁项集。")
    else:
        print(f"高风险样本仅 {high_risk_mask.sum()} 个，跳过关联规则挖掘。")

    # --- 7. 可视化：风险分布、特征重要性（系数绝对值）---
    # 特征重要性（逻辑回归系数绝对值）
    coef_df = pd.DataFrame({
        'feature': FEATURES,
        'coefficient': [coef[f] for f in FEATURES],
        'abs_coef': [abs(coef[f]) for f in FEATURES]
    }).sort_values('abs_coef', ascending=False)
    coef_df.to_csv(os.path.join(output_dir, "scorecard_coefficients.csv"), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    plt.barh(coef_df['feature'][::-1], coef_df['abs_coef'][::-1], color='teal')
    plt.xlabel('逻辑回归系数绝对值')
    plt.title('评分卡特征重要性（基于WOE逻辑回归系数）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scorecard_feature_importance.png"), dpi=300)
    plt.close()

    # 风险分布直方图（按等级着色）
    plt.figure(figsize=(8, 5))
    colors = {'低': 'green', '中': 'orange', '高': 'red'}
    for level in ['低', '中', '高']:
        mask = risk_level == level
        if mask.sum() > 0:
            plt.hist(test_score[mask], bins=20, alpha=0.5, color=colors[level], label=level, density=True)
    plt.xlabel('总分 (0-100)')
    plt.ylabel('密度')
    plt.title('评分卡风险等级对应的总分分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scorecard_risk_distribution.png"), dpi=300)
    plt.close()

    # 校准曲线
    from sklearn.calibration import calibration_curve
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='评分卡')
        plt.plot([0, 1], [0, 1], linestyle='--', label='完美校准')
        plt.xlabel('平均预测概率')
        plt.ylabel('实际阳性比例')
        plt.title('校准曲线')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scorecard_calibration_curve.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"校准曲线生成失败: {e}")

    # --- 8. 保存模型 ---
    joblib.dump(lr, os.path.join(output_dir, "scorecard_model.pkl"))
    # 保存分箱信息和 WOE 映射，以便新数据使用
    import pickle
    with open(os.path.join(output_dir, "scorecard_woe_maps.pkl"), 'wb') as f:
        pickle.dump({'woe_maps': woe_maps, 'bins': BINS, 'min_lp': min_lp, 'max_lp': max_lp}, f)

    print(f"\n✅ 评分卡模型构建完成，所有结果保存至 {output_dir}")
    print("=" * 60)

# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    run_scorecard(DATA_DIR, OUTPUT_DIR)