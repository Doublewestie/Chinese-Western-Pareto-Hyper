# step4.1_generate_synthetic_data.py
# 生成虚拟数据：仅包含6个原始特征 + 3个强交互特征（S0_c, S0_s, A0_s）
# 使用降低后的噪声和非线性参数，提升数据确定性

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ==================== 配置 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "附件1：样例数据.xlsx")
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\data\synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_TRAIN = os.path.join(OUTPUT_DIR, "synthetic_training_data_train.csv")
OUTPUT_CSV_VAL = os.path.join(OUTPUT_DIR, "synthetic_training_data_val.csv")
OUTPUT_BENCHMARK = os.path.join(OUTPUT_DIR, "linear_benchmark.txt")

# 每个患者每个方案生成的样本数（不同噪声种子）
N_REPEATS = 3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==================== 参数随机化范围 ====================
# 非线性调节因子（每10分增加活动下降率的系数）
S0_MODULATION_RANGE = (0.002, 0.005)      # 0.2% ~ 0.5%
# 交互强度（痰湿质 × 活动强度等级）
INTERACTION_RANGE = (0.0, 0.0005)         # 0 ~ 0.0005
# 基础噪声标准差（大幅降低）
NOISE_BASE_RANGE = (0.0005, 0.001)        # 0.05% ~ 0.1%
# 噪声随 S0 增加系数
S0_NOISE_FACTOR_RANGE = (0.0001, 0.0003)  # 每分增加 0.01% ~ 0.03%

# 物理模型基础参数（题目给定）
TIOLI_DECAY = {1: 0.01, 2: 0.02, 3: 0.03}
ACTIVITY_BASE_DECAY = {1: 0.00, 2: 0.03, 3: 0.06}
EXTRA_PER_WEEK = 0.01

# 地板效应起始值
FLOOR_START = 15

# ==================== 辅助函数 ====================
def get_feasible_intensities(age_group, activity_score):
    """根据年龄和活动量表确定允许的活动强度"""
    if age_group <= 2:      # 40-59岁
        possible = [1,2,3]
    elif age_group <= 4:    # 60-79岁
        possible = [1,2]
    else:                   # 80-89岁
        possible = [1]
    if activity_score < 40:
        allowed = [1]
    elif activity_score < 60:
        allowed = [1,2]
    else:
        allowed = [1,2,3]
    return [s for s in possible if s in allowed]

def compute_base_decay_rate(c, s, f, S0, modulation, interaction):
    """
    计算理论下降率（含非线性调节）
    """
    r_tiaoli = TIOLI_DECAY[c]
    r_act_base = ACTIVITY_BASE_DECAY[s]
    extra = EXTRA_PER_WEEK * max(0, f - 5) if f >= 5 else 0
    r_act = r_act_base + extra
    
    # 非线性1：痰湿质越高，活动效果越明显
    s0_factor = (S0 / 100) * modulation * 10   # 归一化调节
    r_act *= (1 + s0_factor)
    
    # 非线性2：交互效应（痰湿质 × 活动强度）
    r_act += interaction * (S0 / 10) * s
    
    r_total = r_tiaoli + r_act
    return np.clip(r_total, 0.0, 0.3)

def simulate_S6_with_floor(S0, decay_rate, months=6, noise_std=None):
    """
    模拟6个月后的积分，考虑地板效应
    """
    if noise_std is None:
        noise_std = 0.005
    epsilon = np.random.normal(0, noise_std)
    actual_decay = max(0, decay_rate + epsilon)
    
    def effective_decay(S_current, base_decay):
        if S_current <= FLOOR_START:
            factor = max(0, S_current / FLOOR_START)
        else:
            factor = 1.0
        return base_decay * factor
    
    S = S0
    for _ in range(months):
        dec = effective_decay(S, actual_decay)
        S = S * (1 - dec)
        if S < 0.5:
            S = 0.5
            break
    return max(0.5, S)

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("生成虚拟数据（6原始 + 3强交互特征）")
    print("=" * 60)
    
    # 1. 读取原始数据
    df_raw = pd.read_excel(RAW_DATA_PATH, sheet_name="Sheet1")
    df_tanshi = df_raw[df_raw['体质标签'] == 5].copy()
    print(f"痰湿质患者数: {len(df_tanshi)}")
    
    features_raw = ['痰湿质', '年龄组', '活动量表总分（ADL总分+IADL总分）']
    df_tanshi = df_tanshi[features_raw].dropna()
    
    records = []
    records_linear = []   # 用于线性基准（无噪声、无非线性）
    
    total_combinations = 0
    
    for idx, row in tqdm(df_tanshi.iterrows(), total=len(df_tanshi), desc="生成数据"):
        S0 = row['痰湿质']
        age_group = int(row['年龄组'])
        A0 = row['活动量表总分（ADL总分+IADL总分）']
        
        feasible_s = get_feasible_intensities(age_group, A0)
        if not feasible_s:
            continue
        
        for c in [1,2,3]:
            for s in feasible_s:
                for f in range(1, 11):
                    total_combinations += 1
                    
                    # 为每条样本独立采样非线性参数和噪声参数
                    modulation = np.random.uniform(*S0_MODULATION_RANGE)
                    interaction = np.random.uniform(*INTERACTION_RANGE)
                    noise_base = np.random.uniform(*NOISE_BASE_RANGE)
                    s0_noise_factor = np.random.uniform(*S0_NOISE_FACTOR_RANGE)
                    noise_std = noise_base + s0_noise_factor * S0
                    
                    # 计算基础下降率（含非线性）
                    base_rate = compute_base_decay_rate(c, s, f, S0, modulation, interaction)
                    
                    for rep in range(N_REPEATS):
                        S6 = simulate_S6_with_floor(S0, base_rate, months=6, noise_std=noise_std)
                        
                        # 构建特征字典（6原始 + 3强交互）
                        feat_dict = {
                            'S0': S0,
                            'age_group': age_group,
                            'A0': A0,
                            'c': c,
                            's': s,
                            'f': f,
                            'S0_c': S0 * c,
                            'S0_s': S0 * s,
                            'A0_s': A0 * s,
                            'S6': S6
                        }
                        records.append(feat_dict)
                    
                    # ===== 线性基准数据（无噪声、无非线性） =====
                    linear_rate = compute_base_decay_rate(c, s, f, S0, modulation=0, interaction=0)
                    S6_linear = S0 * (1 - linear_rate) ** 6
                    feat_dict_lin = {
                        'S0': S0,
                        'age_group': age_group,
                        'A0': A0,
                        'c': c,
                        's': s,
                        'f': f,
                        'S0_c': S0 * c,
                        'S0_s': S0 * s,
                        'A0_s': A0 * s,
                        'S6': S6_linear
                    }
                    records_linear.append(feat_dict_lin)
    
    df_synthetic = pd.DataFrame(records)
    print(f"总生成样本数（含重复）: {len(df_synthetic)}")
    
    # 2. 划分训练集和验证集 (80/20)
    df_train, df_val = train_test_split(df_synthetic, test_size=0.2, random_state=RANDOM_SEED)
    df_train.to_csv(OUTPUT_CSV_TRAIN, index=False, encoding='utf-8-sig')
    df_val.to_csv(OUTPUT_CSV_VAL, index=False, encoding='utf-8-sig')
    print(f"✅ 训练集保存至: {OUTPUT_CSV_TRAIN} ({len(df_train)} 条)")
    print(f"✅ 验证集保存至: {OUTPUT_CSV_VAL} ({len(df_val)} 条)")
    
    # 3. 训练线性基准模型（基于简化物理模型数据，9个特征）
    df_linear = pd.DataFrame(records_linear).drop_duplicates()
    # 特征列（9个：S0, age_group, A0, c, s, f, S0_c, S0_s, A0_s）
    feature_cols = [col for col in df_linear.columns if col != 'S6']
    X_linear = df_linear[feature_cols].values
    y_linear = df_linear['S6'].values
    
    X_train_lin, X_val_lin, y_train_lin, y_val_lin = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=RANDOM_SEED
    )
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_lin, y_train_lin)
    y_pred_lin = lin_reg.predict(X_val_lin)
    mse_lin = mean_squared_error(y_val_lin, y_pred_lin)
    r2_lin = lin_reg.score(X_val_lin, y_val_lin)
    
    # 保存基准结果
    with open(OUTPUT_BENCHMARK, 'w', encoding='utf-8') as f:
        f.write("线性基准模型（基于简化物理数据，9个特征）性能\n")
        f.write("=" * 50 + "\n")
        f.write(f"验证集 MSE: {mse_lin:.6f}\n")
        f.write(f"验证集 R²:  {r2_lin:.6f}\n")
        f.write("\n注：此线性模型基于题目物理公式（无噪声、无非线性），\n")
        f.write("使用6个原始特征 + 3个强交互特征。\n")
        f.write("用于与后续 TabPFN 模型对比。\n")
    print(f"✅ 线性基准结果保存至: {OUTPUT_BENCHMARK}")
    print(f"   验证集 MSE = {mse_lin:.6f}, R² = {r2_lin:.6f}")
    
    # 输出简单统计
    print("\n生成数据统计:")
    print(f"  S0 范围: [{df_synthetic['S0'].min():.1f}, {df_synthetic['S0'].max():.1f}]")
    print(f"  S6 范围: [{df_synthetic['S6'].min():.1f}, {df_synthetic['S6'].max():.1f}]")
    print(f"  总特征数: {len(feature_cols)} (6原始 + 3交互)")
    print("=" * 60)

if __name__ == "__main__":
    main()