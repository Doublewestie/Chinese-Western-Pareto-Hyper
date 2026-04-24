# step2.7_tanshi_indirect_effect.py
# 综合量化痰湿质的间接效应：相关性 + SHAP交互强度 + 中介分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.stats import pearsonr, spearmanr
import joblib
import shap
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
RESULT_DIR=r"E:\MathorCup\Personal-Formal\results"
ORIGINAL_DATA_PATH = r"E:\MathorCup\Personal-Formal\data\附件1：样例数据.xlsx"
MODEL_PATH = os.path.join(DATA_DIR, "best_model_for_shap.pkl")  # 来自 step2.5/2.6
X_TEST_SCALED_PATH = os.path.join(DATA_DIR, "X_test_scaled.csv")
OUTPUT_DIR = os.path.join(RESULT_DIR, "tanshi_indirect_effect")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("开始痰湿质间接效应综合分析")
print("=" * 60)

# ==================== 第一部分：相关性分析 ====================
print("\n[1/3] 相关性分析...")

# 读取原始数据
df_raw = pd.read_excel(ORIGINAL_DATA_PATH, sheet_name="Sheet1")

# 关键指标列表
target_features = [
    'TC（总胆固醇）', 'TG（甘油三酯）', 'HDL-C（高密度脂蛋白）',
    'LDL-C（低密度脂蛋白）', 'BMI', '活动量表总分（ADL总分+IADL总分）'
]
analysis_cols = ['痰湿质'] + target_features
df_corr = df_raw[analysis_cols].dropna()

# 计算皮尔逊相关系数
pearson_corr = {}
spearman_corr = {}
for feat in target_features:
    pearson_corr[feat] = pearsonr(df_corr['痰湿质'], df_corr[feat])
    spearman_corr[feat] = spearmanr(df_corr['痰湿质'], df_corr[feat])

# 汇总相关性结果
corr_results = pd.DataFrame({
    'feature': target_features,
    'pearson_r': [pearson_corr[f][0] for f in target_features],
    'pearson_p': [pearson_corr[f][1] for f in target_features],
    'spearman_r': [spearman_corr[f][0] for f in target_features],
    'spearman_p': [spearman_corr[f][1] for f in target_features]
})
corr_results.to_csv(os.path.join(OUTPUT_DIR, "tanshi_correlation.csv"), index=False,encoding='utf-8-sig')
print("相关性结果已保存")

# 绘制热力图
corr_matrix = df_corr.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('痰湿质与代谢/活动指标的相关性热力图 (Spearman)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tanshi_correlation_heatmap.png"), dpi=300)
plt.close()
print("相关性热力图已保存")

# ==================== 第二部分：SHAP交互强度 ====================
print("\n[2/3] 计算 SHAP 交互强度...")

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ 模型文件不存在: {MODEL_PATH}")
    print("请先运行 step2.5 或 step2.6 生成 best_model_for_shap.pkl")
    shap_available = False
else:
    shap_available = True

if shap_available:
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_SCALED_PATH)

    # 删除泄露特征（与训练时保持一致）
    leaky_cols = ['TC_abnormal', 'TG_abnormal', 'LDL_abnormal', 'HDL_abnormal', 'abnormal_lipid_count']
    X_test = X_test.drop(columns=[col for col in leaky_cols if col in X_test.columns])

    # 使用前500个样本
    X_sample = X_test.iloc[:500]
    feature_names = X_sample.columns.tolist()

    if '痰湿质' not in feature_names:
        print("⚠️ 特征 '痰湿质' 不在测试集列中，跳过SHAP交互分析。")
    else:
        # 构建解释器
        if hasattr(model, 'get_booster') or hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_sample)

        # 计算SHAP交互值
        shap_interaction_values = explainer.shap_interaction_values(X_sample)
        if isinstance(shap_interaction_values, list):
            shap_interaction_values = shap_interaction_values[1]  # 二分类取正类

        tanshi_idx = feature_names.index('痰湿质')
        n_features = len(feature_names)

        # 方法：计算痰湿质与每个特征的平均绝对交互强度
        # shap_interaction_values[:, i, j] 表示特征i和j在样本上的交互值
        # 痰湿质与特征j的交互强度 = |shap_interaction_values[:, tanshi_idx, j]| 的均值
        interaction_strength_list = []
        for j in range(n_features):
            # 取出痰湿质与特征j的交互值（一个长度为n_samples的数组）
            inter_vals = shap_interaction_values[:, tanshi_idx, j]
            mean_abs = np.abs(inter_vals).mean()
            interaction_strength_list.append(mean_abs)

        interaction_strength = np.array(interaction_strength_list)

        # 确保长度一致
        assert len(interaction_strength) == len(feature_names), "长度不匹配"

        interaction_df = pd.DataFrame({
            'feature': feature_names,
            'interaction_strength': interaction_strength
        }).sort_values('interaction_strength', ascending=False)

        interaction_df.to_csv(os.path.join(OUTPUT_DIR, "tanshi_shap_interaction.csv"), index=False,encoding='utf-8-sig')
        print("SHAP交互强度表已保存")

        # 输出关键指标交互强度
        target_interaction = interaction_df[interaction_df['feature'].isin(target_features)]
        print("\n痰湿质与关键指标的 SHAP 交互强度:")
        print(target_interaction.to_string(index=False,encoding='utf-8-sig'))
else:
    print("跳过 SHAP 交互分析。")

# ==================== 第三部分：中介效应分析 (手动Bootstrap) ====================
print("\n[3/3] 中介效应分析 (手动Bootstrap法)...")

# 准备数据
df_med = df_raw[['痰湿质', 'BMI', 'TG（甘油三酯）', '高血脂症二分类标签']].dropna()

def mediation_bootstrap(X, M, Y, n_boot=500, random_state=42):
    """
    手动 Bootstrap 中介效应检验
    X: 自变量 (标准化后)
    M: 中介变量 (标准化后)
    Y: 因变量 (二分类)
    返回: 间接效应、直接效应、总效应、中介比例、p值
    """
    np.random.seed(random_state)
    n = len(X)
    
    # 全样本效应
    # 总效应: Y ~ X
    total_model = sm.Logit(Y, sm.add_constant(X)).fit(disp=0)
    total_effect = total_model.params[1]  # X 的系数
    
    # 间接效应: M ~ X 的系数 * (Y ~ X + M 中 M 的系数)
    model_m = sm.OLS(M, sm.add_constant(X)).fit()
    a_effect = model_m.params[1]  # X -> M
    
    model_y = sm.Logit(Y, sm.add_constant(pd.DataFrame({'X': X, 'M': M}))).fit(disp=0)
    b_effect = model_y.params[2]  # M -> Y (控制 X)
    c_prime_effect = model_y.params[1]  # X -> Y (控制 M)
    
    indirect_effect = a_effect * b_effect
    direct_effect = c_prime_effect
    prop_mediated = indirect_effect / total_effect if total_effect != 0 else 0
    
    # Bootstrap 置信区间和 p 值
    boot_indirect = []
    boot_direct = []
    boot_total = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_boot, M_boot, Y_boot = X[idx], M[idx], Y[idx]
        
        try:
            m_boot = sm.OLS(M_boot, sm.add_constant(X_boot)).fit()
            a_boot = m_boot.params[1]
            y_boot = sm.Logit(Y_boot, sm.add_constant(pd.DataFrame({'X': X_boot, 'M': M_boot}))).fit(disp=0)
            b_boot = y_boot.params[2]
            c_boot = y_boot.params[1]
            total_boot = sm.Logit(Y_boot, sm.add_constant(X_boot)).fit(disp=0).params[1]
            
            boot_indirect.append(a_boot * b_boot)
            boot_direct.append(c_boot)
            boot_total.append(total_boot)
        except:
            continue
    
    boot_indirect = np.array(boot_indirect)
    boot_direct = np.array(boot_direct)
    boot_total = np.array(boot_total)
    
    # 计算 p 值 (双侧)
    p_indirect = 2 * min(np.mean(boot_indirect >= 0), np.mean(boot_indirect <= 0))
    p_direct = 2 * min(np.mean(boot_direct >= 0), np.mean(boot_direct <= 0))
    
    # 置信区间
    ci_indirect = np.percentile(boot_indirect, [2.5, 97.5])
    ci_direct = np.percentile(boot_direct, [2.5, 97.5])
    
    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'prop_mediated': prop_mediated,
        'p_indirect': p_indirect,
        'p_direct': p_direct,
        'ci_indirect': ci_indirect,
        'ci_direct': ci_direct
    }

# 对 BMI 进行中介分析
print("\n分析中介变量: BMI")
X_raw = df_med['痰湿质'].values
M_raw = df_med['BMI'].values
Y_raw = df_med['高血脂症二分类标签'].values

X_std = (X_raw - X_raw.mean()) / X_raw.std()
M_std = (M_raw - M_raw.mean()) / M_raw.std()

result_bmi = mediation_bootstrap(X_std, M_std, Y_raw, n_boot=500)

print(f"  总效应: {result_bmi['total_effect']:.4f}")
print(f"  直接效应: {result_bmi['direct_effect']:.4f} (p={result_bmi['p_direct']:.4f})")
print(f"  间接效应: {result_bmi['indirect_effect']:.4f} (p={result_bmi['p_indirect']:.4f}, 95% CI: [{result_bmi['ci_indirect'][0]:.4f}, {result_bmi['ci_indirect'][1]:.4f}])")
print(f"  中介比例: {result_bmi['prop_mediated']:.2%}")

# 对 TG 进行中介分析
print("\n分析中介变量: TG（甘油三酯）")
M_tg_raw = df_med['TG（甘油三酯）'].values
M_tg_std = (M_tg_raw - M_tg_raw.mean()) / M_tg_raw.std()

result_tg = mediation_bootstrap(X_std, M_tg_std, Y_raw, n_boot=500)

print(f"  总效应: {result_tg['total_effect']:.4f}")
print(f"  直接效应: {result_tg['direct_effect']:.4f} (p={result_tg['p_direct']:.4f})")
print(f"  间接效应: {result_tg['indirect_effect']:.4f} (p={result_tg['p_indirect']:.4f}, 95% CI: [{result_tg['ci_indirect'][0]:.4f}, {result_tg['ci_indirect'][1]:.4f}])")
print(f"  中介比例: {result_tg['prop_mediated']:.2%}")

# 保存结果
with open(os.path.join(OUTPUT_DIR, "mediation_results.txt"), 'w', encoding='utf-8') as f:
    f.write("中介效应分析 (手动Bootstrap, n=500)\n")
    f.write("=" * 50 + "\n")
    f.write("中介变量: BMI\n")
    f.write(f"总效应: {result_bmi['total_effect']:.4f}\n")
    f.write(f"直接效应: {result_bmi['direct_effect']:.4f} (p={result_bmi['p_direct']:.4f})\n")
    f.write(f"间接效应: {result_bmi['indirect_effect']:.4f} (p={result_bmi['p_indirect']:.4f})\n")
    f.write(f"间接效应95% CI: [{result_bmi['ci_indirect'][0]:.4f}, {result_bmi['ci_indirect'][1]:.4f}]\n")
    f.write(f"中介比例: {result_bmi['prop_mediated']:.2%}\n\n")
    f.write("中介变量: TG（甘油三酯）\n")
    f.write(f"总效应: {result_tg['total_effect']:.4f}\n")
    f.write(f"直接效应: {result_tg['direct_effect']:.4f} (p={result_tg['p_direct']:.4f})\n")
    f.write(f"间接效应: {result_tg['indirect_effect']:.4f} (p={result_tg['p_indirect']:.4f})\n")
    f.write(f"间接效应95% CI: [{result_tg['ci_indirect'][0]:.4f}, {result_tg['ci_indirect'][1]:.4f}]\n")
    f.write(f"中介比例: {result_tg['prop_mediated']:.2%}\n")

print("\n" + "=" * 60)
print(f"所有分析完成，结果保存至: {OUTPUT_DIR}")
print("=" * 60)