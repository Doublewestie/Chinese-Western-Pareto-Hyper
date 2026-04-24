# step4.4_extract_rules.py
# 提取“患者特征 → 最优方案”匹配规律（决策树规则）
# 改进：构造 S0 的交互特征，处理类别不平衡，提供定量评估

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
MODEL_DIR = r"E:\MathorCup\Personal-Formal\models"
INTERVENTION_DIR = r"E:\MathorCup\Personal-Formal\results\intervention"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\results\intervention\matching_pattern"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "附件1：样例数据.xlsx")
MODEL_PATH = os.path.join(MODEL_DIR, "tabpfn_regressor.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
BEST_PLAN_CSV = os.path.join(INTERVENTION_DIR, "all_patients_best_plan.csv")

# 成本参数
TIOLI_COST = {1: 30, 2: 80, 3: 130}
TRAIN_COST_PER_SESSION = {1: 3, 2: 5, 3: 8}
MONTHS = 6
WEEKS = 24
MAX_COST = 2000
MAX_FREQ = 10          # 依从性约束（默认无约束）
BATCH_SIZE = 500
RANDOM_SEED = 42

# 决策树参数
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 5
TEST_SIZE = 0.3

# ==================== 辅助函数（与 step4.3 一致） ====================
def get_feasible_intensities(age_group, activity_score):
    if age_group <= 2:
        possible = [1,2,3]
    elif age_group <= 4:
        possible = [1,2]
    else:
        possible = [1]
    if activity_score < 40:
        allowed = [1]
    elif activity_score < 60:
        allowed = [1,2]
    else:
        allowed = [1,2,3]
    return [s for s in possible if s in allowed]

def compute_total_cost(c, s, f):
    return MONTHS * TIOLI_COST[c] + WEEKS * f * TRAIN_COST_PER_SESSION[s]

def construct_feature_vector(S0, age_group, A0, c, s, f, feature_cols):
    base = {
        'S0': S0, 'age_group': age_group, 'A0': A0,
        'c': c, 's': s, 'f': f,
        'S0_c': S0 * c, 'S0_s': S0 * s, 'A0_s': A0 * s
    }
    return [base[name] for name in feature_cols]

def compute_best_plans():
    """若 best_plan CSV 不存在，则动态计算所有痰湿质患者的最佳方案"""
    print("未找到最佳方案 CSV，正在重新计算所有患者的最佳方案...")
    # 加载模型和特征列
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)

    df_raw = pd.read_excel(RAW_DATA_PATH, sheet_name="Sheet1")
    df_tanshi = df_raw[df_raw['体质标签'] == 5].copy()
    print(f"痰湿质患者数量: {len(df_tanshi)}")

    best_plans = []
    all_rows = []
    patient_info = []

    for idx, row in tqdm(df_tanshi.iterrows(), total=len(df_tanshi), desc="枚举方案"):
        pid = row['样本ID']
        S0 = row['痰湿质']
        age_group = int(row['年龄组'])
        A0 = row['活动量表总分（ADL总分+IADL总分）']
        feasible_s = get_feasible_intensities(age_group, A0)
        if not feasible_s:
            continue

        start = len(all_rows)
        for c in [1,2,3]:
            for s in feasible_s:
                for f in range(1, MAX_FREQ+1):
                    cost = compute_total_cost(c, s, f)
                    if cost > MAX_COST:
                        continue
                    vec = construct_feature_vector(S0, age_group, A0, c, s, f, feature_cols)
                    all_rows.append({
                        'ID': pid, 'S0': S0, 'age_group': age_group, 'A0': A0,
                        'c': c, 's': s, 'f': f, 'cost': cost, 'feature_vec': vec
                    })
        end = len(all_rows)
        patient_info.append((pid, S0, age_group, A0, start, end))

    if not all_rows:
        raise RuntimeError("无任何可行方案，请检查数据或参数")

    # 批量预测
    X_all = np.array([row['feature_vec'] for row in all_rows])
    predictions = []
    for i in range(0, len(X_all), BATCH_SIZE):
        batch = X_all[i:i+BATCH_SIZE]
        pred = model.predict(batch)
        predictions.extend(pred)
    for i, row in enumerate(all_rows):
        row['S6_pred'] = predictions[i]
        row['reduction'] = row['S0'] - row['S6_pred']

    # 选择性价比最佳方案
    for (pid, S0, age_group, A0, start, end) in patient_info:
        pat_rows = all_rows[start:end]
        if not pat_rows:
            continue
        df_pat = pd.DataFrame(pat_rows)
        df_valid = df_pat[df_pat['reduction'] > 0].copy()
        if len(df_valid) == 0:
            df_valid = df_pat.loc[[df_pat['reduction'].idxmax()]]
        df_valid['ratio'] = df_valid['reduction'] / df_valid['cost']
        best = df_valid.loc[df_valid['ratio'].idxmax()]
        best_plans.append({
            'ID': pid,
            'S0': S0,
            'age_group': age_group,
            'A0': A0,
            'c': int(best['c']),
            's': int(best['s']),
            'f': int(best['f']),
            'cost': best['cost'],
            'reduction': best['reduction']
        })

    df_best = pd.DataFrame(best_plans)
    df_best.to_csv(BEST_PLAN_CSV, index=False, encoding='utf-8-sig')
    print(f"最佳方案已保存至 {BEST_PLAN_CSV}")
    return df_best

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("step4.4: 提取患者特征 → 最优方案匹配规律（含S0交互特征）")
    print("=" * 60)

    # 1. 加载最佳方案数据（若不存在则计算）
    if os.path.exists(BEST_PLAN_CSV):
        df_best = pd.read_csv(BEST_PLAN_CSV)
        print(f"加载已有最佳方案数据: {len(df_best)} 位患者")
    else:
        df_best = compute_best_plans()

    # 2. 构造特征（原始 + S0交互特征）
    df_best['S0_age'] = df_best['S0'] * df_best['age_group']
    df_best['S0_A0'] = df_best['S0'] * df_best['A0']

    feature_cols = ['S0', 'age_group', 'A0', 'S0_age', 'S0_A0']
    feature_names = ['痰湿质', '年龄组', '活动量表总分', '痰湿质×年龄组', '痰湿质×活动量表总分']

    X = df_best[feature_cols].values
    y = df_best.apply(lambda r: f"{int(r['c'])}_{int(r['s'])}_{int(r['f'])}", axis=1).values

    print(f"特征维度: {X.shape[1]}")
    print("类别分布:\n", pd.Series(y).value_counts())

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    # 4. 训练决策树（类别平衡）
    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
    clf.fit(X_train, y_train)

    # 5. 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    report = classification_report(y_test, y_pred, target_names=clf.classes_)

    print(f"\n测试集准确率: {acc:.4f}")
    print("混淆矩阵:\n", cm)
    print("分类报告:\n", report)

    # 保存评估报告
    eval_path = os.path.join(OUTPUT_DIR, "decision_tree_evaluation.txt")
    with open(eval_path, 'w', encoding='utf-8') as f:
        f.write("决策树模型评估报告（含S0交互特征）\n")
        f.write("=" * 50 + "\n")
        f.write(f"训练集样本数: {len(X_train)}\n")
        f.write(f"测试集样本数: {len(X_test)}\n")
        f.write(f"准确率: {acc:.4f}\n\n")
        f.write("混淆矩阵:\n")
        f.write(str(pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)) + "\n\n")
        f.write("分类报告:\n")
        f.write(report)
    print(f"评估报告已保存: {eval_path}")

    # 特征重要性
    imp = dict(zip(feature_names, clf.feature_importances_))
    print("\n特征重要性:")
    for name, val in imp.items():
        print(f"  {name}: {val:.4f}")

    imp_df = pd.DataFrame({'特征': list(imp.keys()), '重要性': list(imp.values())})
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False, encoding='utf-8-sig')

    # 导出规则
    rules = export_text(clf, feature_names=feature_names)
    rules_path = os.path.join(OUTPUT_DIR, "matching_rules.txt")
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write("患者特征 → 最优干预方案 (c_s_f) 匹配规则\n")
        f.write("=" * 50 + "\n")
        f.write(rules)
    print(f"规则文本已保存: {rules_path}")

    # 可视化决策树
    plt.figure(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, class_names=list(clf.classes_),
              filled=True, rounded=True, fontsize=10)
    plt.title("决策树：患者特征 → 最优干预方案", fontsize=16)
    plt.tight_layout()
    tree_img = os.path.join(OUTPUT_DIR, "matching_tree.png")
    plt.savefig(tree_img, dpi=300)
    plt.close()
    print(f"决策树图已保存: {tree_img}")

    print("\n✅ step4.4 执行完成！")

if __name__ == "__main__":
    main()