# step3.0_rf_risk.py
# 随机森林风险预警模型 + 自定义三级风险 + 决策树规则提取
# 解决问题2：输出低/中/高三级风险，明确特征分层阈值，识别核心特征组合

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 特征分组定义（与 step2.5 一致） ====================
WESTERN_ACTIVITY_COLS = [
    'TG（甘油三酯）', 'TC（总胆固醇）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
    '空腹血糖', '血尿酸', 'BMI',
    'ADL总分', 'IADL总分', '活动量表总分（ADL总分+IADL总分）'
]

TIZHI_COLS = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
BASIC_COLS = ['年龄组_2', '年龄组_3', '年龄组_4', '年龄组_5', '性别_1', '吸烟史_1', '饮酒史_1']
LEAKY_COLS = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']

TANSHI_INTERACTION_COLS = [
    'tanshi_BMI', 'tanshi_activity', 'tanshi_TG', 'tanshi_TC',
    'tanshi_HDL-C', 'tanshi_LDL-C', 'tanshi_血尿酸', 'tanshi_空腹血糖'
]
OTHER_COLS = ['dominant_tizhi']

# ==================== 特征选择函数 ====================
def select_features(X, subset_name):
    if subset_name == "all":
        return X
    elif subset_name == "all-minus":
        exclude_tizhi = [col for col in TIZHI_COLS if col != '痰湿质']
        keep_cols = [col for col in X.columns if col not in exclude_tizhi]
        return X[keep_cols]
    elif subset_name == "western_activity":
        keep_cols = [col for col in WESTERN_ACTIVITY_COLS if col in X.columns]
        return X[keep_cols]
    else:
        raise ValueError(f"未知的特征子集: {subset_name}")

# ==================== 辅助函数 ====================
def compute_feature_direction(X, y):
    pos_mean = X[y == 1].mean()
    neg_mean = X[y == 0].mean()
    diff = pos_mean - neg_mean
    direction = diff.apply(lambda x: '正相关' if x > 0 else ('负相关' if x < 0 else '无'))
    return pd.DataFrame({'feature': diff.index, 'diff': diff.values, 'direction': direction.values})

def get_leaf_risk_mapping(dt, X_train, y_train_3class):
    """计算每个叶子节点中多数类别，用于预测"""
    leaf_ids = dt.apply(X_train)
    leaf_majority = {}
    for leaf in np.unique(leaf_ids):
        labels = y_train_3class[leaf_ids == leaf]
        if len(labels) == 0:
            leaf_majority[leaf] = '中'  # 默认
        else:
            leaf_majority[leaf] = labels.mode()[0]
    return leaf_majority

# ==================== 主分析函数 ====================
def run_analysis(feature_set, data_dir, output_dir, risk_threshold_method='custom', custom_thresholds=(0.33, 0.67)):
    """
    随机森林风险预警模型 + 可解释三级风险规则

    参数:
        feature_set: str, "all", "all-minus", "western_activity"
        data_dir: str, 数据目录
        output_dir: str, 输出目录
        risk_threshold_method: 'quantile' 或 'custom'
            - 'quantile': 使用训练集概率分位数（33%, 67%）
            - 'custom': 使用自定义概率阈值 (low_mid, mid_high)
        custom_thresholds: tuple, 当 method='custom' 时指定 (低-中阈值, 中-高阈值)
    """
    print(f"随机森林风险模型 - 特征子集: {feature_set}")
    print(f"风险阈值方法: {risk_threshold_method}")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载原始数据（未标准化）---
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")

    # 删除泄露特征
    X_train = X_train.drop(columns=[col for col in LEAKY_COLS if col in X_train.columns])
    X_test  = X_test.drop(columns=[col for col in LEAKY_COLS if col in X_test.columns])

    # 特征子集选择
    X_train_sub = select_features(X_train, feature_set)
    X_test_sub  = select_features(X_test, feature_set)
    print(f"特征子集维度: 训练集 {X_train_sub.shape}, 测试集 {X_test_sub.shape}")

    # --- 2. 训练随机森林（保留原参数）---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_sub, y_train)

    # --- 3. 随机森林预测与评估 ---
    y_prob = rf.predict_proba(X_test_sub)[:, 1]
    y_pred = rf.predict(X_test_sub)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n随机森林测试集性能:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    with open(os.path.join(output_dir, "risk_performance_rf.txt"), 'w', encoding='utf-8') as f:
        f.write(f"特征子集: {feature_set}\n")
        f.write("随机森林风险预警模型性能\n")
        f.write("=" * 40 + "\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # --- 4. 确定三级风险等级（基于随机森林概率）---
    if risk_threshold_method == 'quantile':
        train_prob = rf.predict_proba(X_train_sub)[:, 1]
        low_mid_thresh = np.percentile(train_prob, 33)
        mid_high_thresh = np.percentile(train_prob, 67)
        print(f"\n基于训练集概率分位数的风险阈值: 33% = {low_mid_thresh:.4f}, 67% = {mid_high_thresh:.4f}")
    else:  # custom
        low_mid_thresh, mid_high_thresh = custom_thresholds
        print(f"\n使用自定义风险阈值: 低-中 = {low_mid_thresh}, 中-高 = {mid_high_thresh}")

    # 为训练集和测试集生成三级标签（用于决策树学习）
    train_prob_rf = rf.predict_proba(X_train_sub)[:, 1]
    y_train_3class = []
    for p in train_prob_rf:
        if p < low_mid_thresh:
            y_train_3class.append('低')
        elif p < mid_high_thresh:
            y_train_3class.append('中')
        else:
            y_train_3class.append('高')
    y_train_3class = np.array(y_train_3class)

    test_prob_rf = y_prob
    y_test_3class = []
    for p in test_prob_rf:
        if p < low_mid_thresh:
            y_test_3class.append('低')
        elif p < mid_high_thresh:
            y_test_3class.append('中')
        else:
            y_test_3class.append('高')
    y_test_3class = np.array(y_test_3class)

    # 保存随机森林的三级风险等级（基于所选阈值）
    risk_df = pd.DataFrame({
        'sample_id': y_test.index,
        'risk_probability': y_prob,
        'risk_level_rf': y_test_3class
    })
    risk_df.to_csv(os.path.join(output_dir, "risk_levels_rf.csv"), index=False, encoding='utf-8-sig')

    # 保存阈值依据
    with open(os.path.join(output_dir, "risk_thresholds_rf.txt"), 'w', encoding='utf-8') as f:
        f.write(f"风险分层阈值依据（{risk_threshold_method}）\n")
        f.write("=" * 40 + "\n")
        f.write(f"低风险 ↔ 中风险 概率阈值: {low_mid_thresh:.4f}\n")
        f.write(f"中风险 ↔ 高风险 概率阈值: {mid_high_thresh:.4f}\n")
        if risk_threshold_method == 'quantile':
            f.write("阈值选择理由：基于训练集风险概率的33%和67%分位数，确保每个风险等级有足够的样本量。\n")
        else:
            f.write("阈值选择理由：自定义阈值，可根据临床需求调整（例如0.33/0.67或0.5/0.8）。\n")

    # --- 5. 随机森林特征重要性及方向（不变）---
    importance_df = pd.DataFrame({
        'feature': X_train_sub.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    direction_df = compute_feature_direction(X_train_sub, y_train)
    importance_df = importance_df.merge(direction_df, on='feature', how='left')
    importance_df = importance_df[['feature', 'importance', 'direction', 'diff']]
    importance_df.to_csv(os.path.join(output_dir, "feature_importance_rf.csv"), index=False, encoding='utf-8-sig')

    plt.figure(figsize=(10, 6))
    top10 = importance_df.head(10)
    plt.barh(top10['feature'][::-1], top10['importance'][::-1], color='steelblue')
    plt.xlabel('基尼重要性')
    plt.title(f'随机森林特征重要性 (Top 10) - {feature_set}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_rf.png"), dpi=300)
    plt.close()

    # --- 6. 训练浅层决策树，直接输出三级风险规则（可解释）---
    print("\n训练浅层决策树（多分类）用于提取三级风险规则...")
    dt = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        random_state=42
    )
    dt.fit(X_train_sub, y_train_3class)  # 使用三级标签

    # 提取规则
    rules_text = export_text(dt, feature_names=X_train_sub.columns.tolist())
    with open(os.path.join(output_dir, "decision_tree_3class_rules.txt"), 'w', encoding='utf-8') as f:
        f.write(rules_text)

    # 可视化决策树
    plt.figure(figsize=(20, 12))
    plot_tree(dt, feature_names=X_train_sub.columns.tolist(), 
              class_names=['低', '中', '高'], filled=True, rounded=True, fontsize=10)
    plt.title('决策树三级风险规则（可解释）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_tree_3class.png"), dpi=300)
    plt.close()

    # 预测测试集的三级风险（决策树）
    dt_pred = dt.predict(X_test_sub)
    dt_risk_df = pd.DataFrame({
        'sample_id': y_test.index,
        'risk_level_dt': dt_pred
    })
    dt_risk_df.to_csv(os.path.join(output_dir, "risk_levels_decision_tree.csv"), index=False, encoding='utf-8-sig')

    # 输出决策树规则中的阈值（即特征分裂点）
    # 提取树结构中的阈值信息
    tree_ = dt.tree_
    feature_names_arr = np.array(X_train_sub.columns.tolist())
    thresholds = []
    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:  # 非叶子节点
            feat = feature_names_arr[tree_.feature[i]]
            thresh = tree_.threshold[i]
            thresholds.append((feat, thresh))
    thresholds_df = pd.DataFrame(thresholds, columns=['特征', '分裂阈值']).drop_duplicates()
    thresholds_df.to_csv(os.path.join(output_dir, "decision_tree_thresholds.csv"), index=False, encoding='utf-8-sig')
    print("\n决策树关键分裂阈值（特征分级依据）已保存至 decision_tree_thresholds.csv")

    # --- 7. 高风险人群核心特征组合（基于决策树的高风险路径）---
    # 获取预测为高风险的叶子节点
    leaf_ids = dt.apply(X_test_sub)
    high_risk_leaves = []
    for leaf in np.unique(leaf_ids):
        # 该叶子节点在训练集中的多数类别
        train_leaves = dt.apply(X_train_sub)
        majority = pd.Series(y_train_3class[train_leaves == leaf]).mode()
        if len(majority) > 0 and majority[0] == '高':
            high_risk_leaves.append(leaf)
    if high_risk_leaves:
        # 从规则文本中提取包含高风险的行
        lines = rules_text.split('\n')
        high_rules = []
        for line in lines:
            if 'class: 高' in line:
                high_rules.append(line)
        with open(os.path.join(output_dir, "high_risk_core_combinations.txt"), 'w', encoding='utf-8') as f:
            f.write("高风险人群核心特征组合（决策树路径）：\n")
            f.write("\n".join(high_rules))
            f.write("\n\n说明：以上每条路径代表一种高风险模式，例如 'TG <= 1.68 且 TC > 6.18' 等。\n")
    else:
        print("未找到高风险叶子节点，请调整决策树参数。")

    # 可选：关联规则挖掘（补充）
    high_risk_mask = np.array(y_test_3class) == '高'
    if high_risk_mask.sum() >= 5:
        rule_features = ['痰湿质', 'BMI', '活动量表总分（ADL总分+IADL总分）', 'TG（甘油三酯）', 'TC（总胆固醇）']
        available_rule = [f for f in rule_features if f in X_test_sub.columns]
        X_high = X_test_sub.loc[high_risk_mask, available_rule].copy()
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        X_high_disc = pd.DataFrame(discretizer.fit_transform(X_high), columns=available_rule, index=X_high.index)
        for col in available_rule:
            X_high_disc[col] = X_high_disc[col].map({0: '低', 1: '中', 2: '高'})
        X_high_onehot = pd.get_dummies(X_high_disc)
        frequent_itemsets = apriori(X_high_onehot, min_support=0.1, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
            rules = rules.sort_values('support', ascending=False)
            core_rules = rules[['antecedents', 'consequents', 'support', 'confidence']].head(5).copy()
            core_rules['antecedents'] = core_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            core_rules['consequents'] = core_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            core_rules.to_csv(os.path.join(output_dir, "core_feature_combinations_rf.csv"), index=False, encoding='utf-8-sig')
            print("\n高风险人群核心特征组合（关联规则）已保存。")
        else:
            print("未发现满足最小支持度的频繁项集。")
    else:
        print(f"高风险样本仅 {high_risk_mask.sum()} 个，跳过关联规则挖掘。")

    # --- 8. 随机森林的校准曲线与风险分布（保留）---
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='随机森林')
        plt.plot([0, 1], [0, 1], linestyle='--', label='完美校准')
        plt.xlabel('平均预测概率')
        plt.ylabel('实际阳性比例')
        plt.title('校准曲线')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "calibration_curve_rf.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"校准曲线生成失败: {e}")

    plt.figure(figsize=(8, 5))
    colors = {'低': 'green', '中': 'orange', '高': 'red'}
    for level in ['低', '中', '高']:
        mask = np.array(y_test_3class) == level
        if mask.sum() > 0:
            plt.hist(y_prob[mask], bins=20, alpha=0.5, color=colors[level], label=level, density=True)
    plt.xlabel('预测风险概率')
    plt.ylabel('密度')
    plt.title(f'风险等级对应的概率分布 - {feature_set}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_distribution_rf.png"), dpi=300)
    plt.close()

    # --- 9. 保存模型 ---
    joblib.dump(rf, os.path.join(output_dir, "rf_risk_model.pkl"))
    joblib.dump(dt, os.path.join(output_dir, "decision_tree_3class.pkl"))
    print(f"\n✅ 分析完成，结果保存至 {output_dir}")

# ==================== 独立运行入口 ====================
if __name__ == "__main__":
    # 可修改以下参数
    feature_subset = "all"   # 可选: "all", "all-minus", "western_activity"
    # 风险阈值方法：'quantile'（基于分位数）或 'custom'（自定义）
    method = 'custom'   # 建议使用 custom，并设置合理的阈值
    # 自定义阈值（低-中, 中-高）
    # 可根据需要调整，例如 (0.3, 0.7) 或 (0.5, 0.8)
    custom_th = (0.33, 0.67)   # 这里使用常用的33%和67%分位数，但您也可以改成 (0.5, 0.8)
    output_subdir = os.path.join(r"E:\MathorCup\Personal-Formal\results\risk_model\rf", feature_subset)
    run_analysis(
        feature_set=feature_subset,
        data_dir=r"E:\MathorCup\Personal-Formal\data",
        output_dir=output_subdir,
        risk_threshold_method=method,
        custom_thresholds=custom_th
    )