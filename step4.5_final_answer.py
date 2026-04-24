# step4.5_final_answer.py
# 生成问题三的最终答案：匹配规律 + 三位患者的最优干预方案
# 读取 matching_rules.txt 并嵌入，输出到 intervention/final_results

import os
import pandas as pd

# ==================== 配置 ====================
INTERVENTION_DIR = r"E:\MathorCup\Personal-Formal\results\intervention"
RULES_TXT_DIR= os.path.join(INTERVENTION_DIR, "matching_pattern")
OUTPUT_DIR = os.path.join(INTERVENTION_DIR, "final_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入文件
SUMMARY_CSV = os.path.join(INTERVENTION_DIR, "recommended_plans_summary.csv")
RULES_TXT = os.path.join(RULES_TXT_DIR, "matching_rules.txt")

# 输出文件
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "final_answer_Q3.txt")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "final_plans_table.csv")
OUTPUT_MD = os.path.join(OUTPUT_DIR, "final_answer_Q3.md")

# ==================== 辅助函数：将 DataFrame 转为 Markdown 表格 ====================
def df_to_markdown(df):
    """将 DataFrame 转换为 Markdown 表格字符串"""
    lines = []
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(v) for v in row.values) + " |"
        lines.append(row_str)
    return "\n".join(lines)

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("step4.5: 生成问题三最终答案")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 检查必要文件
    if not os.path.exists(SUMMARY_CSV):
        print(f"错误: 找不到 {SUMMARY_CSV}，请先运行 step4.3 生成汇总表")
        return

    # 读取汇总表
    df_summary = pd.read_csv(SUMMARY_CSV)
    target_ids = [1, 2, 3]
    df_target = df_summary[df_summary['ID'].isin(target_ids)].copy()
    if df_target.empty:
        print("错误: 汇总表中没有 ID=1,2,3 的记录")
        return

    # 解析性价比最佳方案（格式如 "c=1, s=1, f=1"）
    def parse_plan(plan_str):
        parts = plan_str.split(',')
        c = int(parts[0].split('=')[1])
        s = int(parts[1].split('=')[1])
        f = int(parts[2].split('=')[1])
        return c, s, f

    results = []
    for _, row in df_target.iterrows():
        pid = int(row['ID'])
        S0 = row['S0']
        plan_str = row['性价比最佳方案']
        cost_str = row['性价比成本']
        reduction_str = row['性价比降低量']
        c, s, f = parse_plan(plan_str)
        cost = float(cost_str.replace('元', ''))
        reduction = float(reduction_str.replace('分', ''))
        S6 = S0 - reduction
        results.append({
            'ID': pid,
            '初始痰湿积分': S0,
            '调理等级(c)': c,
            '活动强度(s)': s,
            '每周次数(f)': f,
            '总成本(元)': cost,
            '积分降低量(分)': reduction,
            '6个月后积分': S6
        })

    df_final = pd.DataFrame(results)

    # 读取匹配规则
    if os.path.exists(RULES_TXT):
        with open(RULES_TXT, 'r', encoding='utf-8') as f:
            matching_rules = f.read().strip()
        print("已读取匹配规则文件")
    else:
        matching_rules = "（未生成，请先运行 step4.4）"
        print(f"警告: 找不到 {RULES_TXT}，将使用占位符")

    # 保存文本答案
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("问题三 最终答案\n")
        f.write("=" * 60 + "\n\n")
        f.write("一、患者特征-最优方案匹配规律\n")
        f.write("-" * 40 + "\n")
        f.write(matching_rules)
        f.write("\n\n")
        f.write("二、样本ID=1,2,3的最优干预方案（性价比最佳）\n")
        f.write("-" * 40 + "\n")
        f.write(df_final.to_string(index=False))
        f.write("\n\n")
        f.write("注：以上方案基于6个月周期，总成本包含调理费用（按月）和训练费用（按周）。\n")
        f.write("性价比最佳定义为积分降低量/总成本最大。\n")

    print(f"文本答案已保存至: {OUTPUT_TXT}")

    # 保存 CSV 表格
    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"表格已保存至: {OUTPUT_CSV}")

    # 保存 Markdown 格式
    md_content = "# 问题三 最终答案\n\n"
    md_content += "## 一、患者特征-最优方案匹配规律\n\n```\n" + matching_rules + "\n```\n\n"
    md_content += "## 二、样本ID=1,2,3的最优干预方案\n\n"
    md_content += df_to_markdown(df_final) + "\n\n"
    md_content += "*注：性价比最佳定义为积分降低量/总成本最大。*"
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Markdown答案已保存至: {OUTPUT_MD}")

    # 控制台打印预览
    print("\n" + "=" * 60)
    print("最终答案预览：")
    print("\n--- 匹配规律（前500字符）---")
    print(matching_rules[:500] + "..." if len(matching_rules) > 500 else matching_rules)
    print("\n--- 三位患者方案 ---")
    print(df_final.to_string(index=False))
    print("\n✅ step4.5 执行完成！")

if __name__ == "__main__":
    main()