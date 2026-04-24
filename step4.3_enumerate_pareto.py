# step4.3_enumerate_pareto.py
# 枚举所有可行干预方案，使用 TabPFN 模型预测 S6，求解帕累托前沿
# 支持依从性约束：通过 MAX_FREQ 参数控制每周最大训练次数
# 包含优化后的帕累托前沿绘图（专业配色、标注、布局）

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"
MODEL_DIR = r"E:\MathorCup\Personal-Formal\models"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\results\intervention"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "附件1：样例数据.xlsx")
MODEL_PATH = os.path.join(MODEL_DIR, "tabpfn_regressor.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

# 目标患者 ID（问题3要求）
TARGET_IDS = [1, 2, 3]

# 成本参数
TIOLI_COST = {1: 30, 2: 80, 3: 130}          # 元/月
TRAIN_COST_PER_SESSION = {1: 3, 2: 5, 3: 8}  # 元/次
MONTHS = 6
WEEKS = 24

# 预测批次大小（防止显存溢出）
BATCH_SIZE = 500

# 成本上限（题目建议 ≤2000 元）
MAX_COST = 2000

# ==================== 依从性约束开关 ====================
# 每周最大训练次数，默认 10（无约束）。可修改为 5 或 7 来模拟依从性约束场景
MAX_FREQ = 10   # 改为 5 则只枚举 f=1..5，展示依从性约束下的方案

# ==================== 辅助函数 ====================
def get_feasible_intensities(age_group, activity_score):
    """根据年龄组和活动量表总分确定允许的活动强度"""
    if age_group <= 2:      # 40-59岁
        possible = [1, 2, 3]
    elif age_group <= 4:    # 60-79岁
        possible = [1, 2]
    else:                   # 80-89岁
        possible = [1]

    if activity_score < 40:
        allowed = [1]
    elif activity_score < 60:
        allowed = [1, 2]
    else:
        allowed = [1, 2, 3]

    return [s for s in possible if s in allowed]

def compute_total_cost(c, s, f):
    """计算6个月总成本（调理+训练）"""
    tiaoli = MONTHS * TIOLI_COST[c]
    train = WEEKS * f * TRAIN_COST_PER_SESSION[s]
    return tiaoli + train

def construct_feature_vector(S0, age_group, A0, c, s, f, feature_cols):
    """按给定的特征列顺序构造单个样本的特征向量（list）"""
    base = {
        'S0': S0,
        'age_group': age_group,
        'A0': A0,
        'c': c,
        's': s,
        'f': f
    }
    # 添加3个强交互特征
    base['S0_c'] = S0 * c
    base['S0_s'] = S0 * s
    base['A0_s'] = A0 * s
    # 按特征列顺序取值
    vec = [base[name] for name in feature_cols]
    return vec

def pareto_frontier(df):
    """
    计算帕累托前沿（最小化成本，最大化降低量）
    df: DataFrame with columns 'cost', 'reduction'
    返回前沿 DataFrame
    """
    df_sorted = df.sort_values(['cost', 'reduction'], ascending=[True, False]).reset_index(drop=True)
    frontier = []
    best_reduction = -np.inf
    for _, row in df_sorted.iterrows():
        if row['reduction'] > best_reduction:
            frontier.append(row)
            best_reduction = row['reduction']
    return pd.DataFrame(frontier)

# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("step4.3: 枚举干预方案 + 帕累托优化")
    print(f"依从性约束：每周最大训练次数 = {MAX_FREQ}" + (" (无约束)" if MAX_FREQ >= 10 else " (有约束)"))
    print("=" * 60)

    # 1. 加载模型和特征列
    print("\n[1/4] 加载模型和特征列...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_COLS_PATH, 'rb') as f:
        feature_cols = pickle.load(f)
    print(f"  特征数量: {len(feature_cols)}")

    # 2. 读取原始数据
    print("\n[2/4] 读取原始数据...")
    df_raw = pd.read_excel(RAW_DATA_PATH, sheet_name="Sheet1")
    patients = []
    for pid in TARGET_IDS:
        row = df_raw[df_raw['样本ID'] == pid].iloc[0]
        patients.append({
            'ID': pid,
            'S0': row['痰湿质'],
            'age_group': int(row['年龄组']),
            'A0': row['活动量表总分（ADL总分+IADL总分）']
        })
    print(f"  目标患者: {[p['ID'] for p in patients]}")

    # 3. 为每个患者枚举可行方案
    print("\n[3/4] 枚举可行方案并构造特征矩阵...")
    all_rows = []  # 存储所有患者的所有方案
    patient_plan_indices = []  # 记录每个患者的方案在 all_rows 中的起止索引

    for pat in patients:
        pid = pat['ID']
        S0 = pat['S0']
        age_group = pat['age_group']
        A0 = pat['A0']

        feasible_s = get_feasible_intensities(age_group, A0)
        if not feasible_s:
            print(f"  警告: 患者 {pid} 无可行活动强度，跳过")
            continue

        start_idx = len(all_rows)
        count = 0
        for c in [1, 2, 3]:
            for s in feasible_s:
                for f in range(1, MAX_FREQ + 1):
                    cost = compute_total_cost(c, s, f)
                    if cost > MAX_COST:
                        continue
                    vec = construct_feature_vector(S0, age_group, A0, c, s, f, feature_cols)
                    all_rows.append({
                        'ID': pid,
                        'c': c, 's': s, 'f': f,
                        'cost': cost,
                        'S0': S0,
                        'feature_vec': vec
                    })
                    count += 1
        end_idx = len(all_rows)
        patient_plan_indices.append((pid, start_idx, end_idx))
        print(f"  患者 {pid}: 生成 {count} 个可行方案（成本≤{MAX_COST}，f≤{MAX_FREQ}）")

    if not all_rows:
        print("错误: 无任何可行方案，请检查数据或成本上限。")
        return

    # 4. 批量预测（分批防止 OOM）
    print("\n[4/4] 批量预测 S6...")
    X_all = np.array([row['feature_vec'] for row in all_rows])
    num_samples = X_all.shape[0]
    predictions = []
    for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="预测进度"):
        batch = X_all[i: min(i + BATCH_SIZE, num_samples)]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)
    # 将预测结果写回 all_rows
    for i, row in enumerate(all_rows):
        row['S6_pred'] = predictions[i]
        row['reduction'] = row['S0'] - row['S6_pred']

    # 5. 按患者分组，计算帕累托前沿并输出
    print("\n[5/5] 计算帕累托前沿并保存结果...")
    summary = []  # 存储最终推荐方案

    # 根据是否开启依从性约束，添加文件名后缀
    suffix = f"_freq{MAX_FREQ}" if MAX_FREQ < 10 else ""

    for pid, start, end in patient_plan_indices:
        pat_rows = all_rows[start:end]
        if not pat_rows:
            continue

        df_plans = pd.DataFrame(pat_rows)
        # 只保留有效降低（reduction > 0）
        df_valid = df_plans[df_plans['reduction'] > 0].copy()
        if len(df_valid) == 0:
            df_valid = df_plans.loc[[df_plans['reduction'].idxmax()]].copy()
            print(f"  患者 {pid}: 所有方案均无有效降低，选择降低最大方案")

        frontier = pareto_frontier(df_valid[['cost', 'reduction', 'c', 's', 'f', 'S6_pred']])

        # 保存前沿 CSV
        frontier_csv = os.path.join(OUTPUT_DIR, f"pareto_frontier_patient_{pid}{suffix}.csv")
        frontier.to_csv(frontier_csv, index=False, encoding='utf-8-sig')
        print(f"  患者 {pid}: 帕累托前沿包含 {len(frontier)} 个方案，已保存至 {frontier_csv}")

        # 标注特殊点
        min_cost = frontier.loc[frontier['cost'].idxmin()]
        max_red = frontier.loc[frontier['reduction'].idxmax()]
        frontier['ratio'] = frontier['reduction'] / frontier['cost']
        best_ratio = frontier.loc[frontier['ratio'].idxmax()]

        # ==================== 改进后的绘图 ====================
        plt.figure(figsize=(8, 6), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')          # 浅灰背景

        # 所有可行方案（浅灰色，半透明，小点）
        ax.scatter(df_valid['cost'], df_valid['reduction'],
                   alpha=0.6, s=30, c='#7f8c8d', edgecolors='none', label='所有可行方案')

        # 帕累托前沿（深红色，粗线，大标记）
        ax.plot(frontier['cost'], frontier['reduction'],
                color='#d62728', linewidth=2.5, marker='o', markersize=8,
                markerfacecolor='#d62728', markeredgecolor='white', markeredgewidth=0.5,
                label='帕累托前沿')

        # 三个特殊点
        ax.scatter(min_cost['cost'], min_cost['reduction'],
                   s=220, marker='s', c='#2ca02c', edgecolors='grey', linewidth=1.5,
                   zorder=5, label='最小成本')
        ax.scatter(max_red['cost'], max_red['reduction'],
                   s=220, marker='^', c='#1f77b4', edgecolors='grey', linewidth=1.5,
                   zorder=5, label='最大降低')
        ax.scatter(best_ratio['cost'], best_ratio['reduction'],
                   s=260, marker='*', c='#ff7f0e', edgecolors='grey', linewidth=0.5,
                   zorder=5, label='性价比最佳')

        # 文字标注
        ax.annotate(f'最小成本\n{min_cost["cost"]:.0f}元\n{min_cost["reduction"]:.1f}分',
                    xy=(min_cost['cost'], min_cost['reduction']),
                    xytext=(15, -15), textcoords='offset points',
                    fontsize=9, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax.annotate(f'最大降低\n{max_red["cost"]:.0f}元\n{max_red["reduction"]:.1f}分',
                    xy=(max_red['cost'], max_red['reduction']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        ax.annotate(f'性价比最佳\n{best_ratio["cost"]:.0f}元\n{best_ratio["reduction"]:.1f}分',
                    xy=(best_ratio['cost'], best_ratio['reduction']),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # 坐标轴标签加粗
        ax.set_xlabel('总成本 (元)', fontsize=12, fontweight='bold')
        ax.set_ylabel('痰湿积分降低量 (分)', fontsize=12, fontweight='bold')

        # 标题
        ax.set_title(f'患者 ID={pid} 干预方案帕累托前沿 (f ≤ {MAX_FREQ})',
                     fontsize=14, fontweight='bold', pad=15)

        # 网格优化
        ax.grid(True, linestyle=':', alpha=0.5, color='gray')

        # 图例置于右侧外部
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True,
                  fancybox=True, shadow=False, fontsize=10)

        # 调整边距，防止裁剪
        plt.margins(x=0.05, y=0.05)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # 保存图片
        fig_path = os.path.join(OUTPUT_DIR, f"pareto_frontier_patient_{pid}{suffix}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  帕累托前沿图已保存: {fig_path}")

        # 记录推荐方案
        summary.append({
            'ID': pid,
            'S0': pat_rows[0]['S0'],
            '最小成本方案': f"c={int(min_cost['c'])}, s={int(min_cost['s'])}, f={int(min_cost['f'])}",
            '最小成本': f"{min_cost['cost']:.0f}元",
            '最小成本降低量': f"{min_cost['reduction']:.1f}分",
            '最大降低方案': f"c={int(max_red['c'])}, s={int(max_red['s'])}, f={int(max_red['f'])}",
            '最大降低成本': f"{max_red['cost']:.0f}元",
            '最大降低量': f"{max_red['reduction']:.1f}分",
            '性价比最佳方案': f"c={int(best_ratio['c'])}, s={int(best_ratio['s'])}, f={int(best_ratio['f'])}",
            '性价比成本': f"{best_ratio['cost']:.0f}元",
            '性价比降低量': f"{best_ratio['reduction']:.1f}分"
        })

        # 控制台输出
        print(f"\n患者 ID={pid} (初始痰湿积分 {pat_rows[0]['S0']:.1f}) 推荐方案 (f ≤ {MAX_FREQ}):")
        print(f"  - 最小成本: {min_cost['c']}级调理 + {min_cost['s']}级活动 × {min_cost['f']}次/周 → 成本{min_cost['cost']:.0f}元, 降低{min_cost['reduction']:.1f}分")
        print(f"  - 最大降低: {max_red['c']}级调理 + {max_red['s']}级活动 × {max_red['f']}次/周 → 成本{max_red['cost']:.0f}元, 降低{max_red['reduction']:.1f}分")
        print(f"  - 性价比最佳: {best_ratio['c']}级调理 + {best_ratio['s']}级活动 × {best_ratio['f']}次/周 → 成本{best_ratio['cost']:.0f}元, 降低{best_ratio['reduction']:.1f}分")

    # 保存汇总表
    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(OUTPUT_DIR, f"recommended_plans_summary{suffix}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f"\n汇总推荐方案已保存至: {summary_csv}")

    print("\n" + "=" * 60)
    print("✅ step4.3 执行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()