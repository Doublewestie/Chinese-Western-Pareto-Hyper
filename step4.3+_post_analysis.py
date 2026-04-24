# step4.3+_post_analysis.py
# 边际效益分析（基于帕累托前沿 CSV）
# 不包含 SHAP 分析，避免兼容性问题

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置路径 ====================
INTERVENTION_DIR = r"E:\MathorCup\Personal-Formal\results\intervention"
OUTPUT_DIR = os.path.join(INTERVENTION_DIR, "post_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 患者 ID 列表
PATIENT_IDS = [1, 2, 3]

# ==================== 边际效益分析函数 ====================
def marginal_benefit_analysis(csv_path, patient_id, output_dir):
    """
    读取帕累托前沿 CSV，计算边际效益，绘图并返回最佳方案
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values('cost').reset_index(drop=True)

    costs = df['cost'].values
    reductions = df['reduction'].values

    # 计算边际效益 (分/元)
    marginal = [np.nan]
    for i in range(1, len(costs)):
        delta_cost = costs[i] - costs[i-1]
        delta_reduction = reductions[i] - reductions[i-1]
        if delta_cost > 0:
            marginal.append(delta_reduction / delta_cost)
        else:
            marginal.append(0.0)
    df['marginal_benefit'] = marginal

    # 找出边际效益最大的点（排除第一个 NaN）
    best_idx = df.iloc[1:]['marginal_benefit'].idxmax()
    best_row = df.loc[best_idx]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marginal, 'o-', color='steelblue', linewidth=2, markersize=8, label='边际效益')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.scatter(best_row['cost'], best_row['marginal_benefit'],
                color='red', s=150, zorder=5, label='边际效益最高点')
    plt.xlabel('总成本 (元)')
    plt.ylabel('边际效益 (分/元)')
    plt.title(f'患者 ID={patient_id} 边际效益曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_img = os.path.join(output_dir, f'marginal_benefit_patient_{patient_id}.png')
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"  边际效益图已保存: {out_img}")

    return best_row

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("step4.3+ 后分析：边际效益分析")
    print("=" * 60)

    # ---------- 边际效益分析 ----------
    print("\n[1/1] 边际效益分析...")
    best_marginal_list = []
    for pid in PATIENT_IDS:
        # 自动查找帕累托前沿文件（支持带后缀的依从性约束文件）
        base_path = os.path.join(INTERVENTION_DIR, f"pareto_frontier_patient_{pid}.csv")
        if os.path.exists(base_path):
            csv_path = base_path
        else:
            import glob
            pattern = os.path.join(INTERVENTION_DIR, f"pareto_frontier_patient_{pid}_freq*.csv")
            files = glob.glob(pattern)
            if files:
                csv_path = files[0]
                print(f"  使用文件: {os.path.basename(csv_path)}")
            else:
                print(f"  警告: 患者 {pid} 的帕累托前沿文件不存在，跳过")
                continue

        best = marginal_benefit_analysis(csv_path, pid, OUTPUT_DIR)
        best_marginal_list.append({
            'ID': pid,
            'cost': best['cost'],
            'reduction': best['reduction'],
            'marginal': best['marginal_benefit'],
            'c': int(best['c']),
            's': int(best['s']),
            'f': int(best['f'])
        })

    if best_marginal_list:
        df_marginal = pd.DataFrame(best_marginal_list)
        marginal_csv = os.path.join(OUTPUT_DIR, "marginal_best_summary.csv")
        df_marginal.to_csv(marginal_csv, index=False, encoding='utf-8-sig')
        print(f"\n边际效益最佳方案汇总已保存: {marginal_csv}")
        print(df_marginal.to_string(index=False))

    print("\n" + "=" * 60)
    print("✅ 边际效益分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()