# run_all_analysis.py
# 总控脚本：自动遍历所有特征子集，依次调用元代码A和元代码B
# 使用方法：直接运行，无需手动修改参数

import os
import sys
import importlib.util
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置区 ====================
# 定义要运行的特征子集列表
FEATURE_SUBSETS = ["all", "no_TCTG", "western_activity", "tizhi_only"]

# 元代码文件路径（请根据实际文件名调整）
META_A_PATH = "step2.5_rf_main.py"
META_B_PATH = "step2.5+_nn_supplement.py"

# 数据目录
DATA_DIR = r"E:\MathorCup\Personal-Formal\data"

# 结果根目录
RF_RESULTS_DIR = r"E:\MathorCup\Personal-Formal\results\rf_analysis"
NN_RESULTS_DIR = r"E:\MathorCup\Personal-Formal\results\nn_supplement"

# ==================== 动态导入元代码中的主函数 ====================
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入元代码A和B（假设它们内部定义了名为 run_analysis 的函数）
module_a = import_module_from_path("meta_a", META_A_PATH)
module_b = import_module_from_path("meta_b", META_B_PATH)

# ==================== 主循环 ====================
print("=" * 60)
print("开始批量分析...")
print(f"特征子集列表: {FEATURE_SUBSETS}")
print("=" * 60)

for subset in FEATURE_SUBSETS:
    print(f"\n>>> 正在处理特征子集: {subset}")

    # 创建输出目录（元代码内部也会创建，这里提前创建确保存在）
    rf_out = os.path.join(RF_RESULTS_DIR, subset)
    nn_out = os.path.join(NN_RESULTS_DIR, subset)
    os.makedirs(rf_out, exist_ok=True)
    os.makedirs(nn_out, exist_ok=True)

    # 调用元代码A（随机森林分析）
    print(f"   [1/2] 运行随机森林分析...")
    try:
        module_a.run_analysis(
            feature_set=subset,
            data_dir=DATA_DIR,
            output_dir=rf_out
        )
        print(f"   ✅ 随机森林分析完成，结果保存至 {rf_out}")
    except Exception as e:
        print(f"   ❌ 随机森林分析失败: {e}")
        continue  # 如果A失败，跳过B

    # 调用元代码B（神经网络补充）
    print(f"   [2/2] 运行神经网络补充分析...")
    try:
        module_b.run_analysis(
            feature_set=subset,
            data_dir=DATA_DIR,
            rf_importance_path=os.path.join(rf_out, "rf_importance.csv"),
            output_dir=nn_out
        )
        print(f"   ✅ 神经网络分析完成，结果保存至 {nn_out}")
    except Exception as e:
        print(f"   ❌ 神经网络分析失败: {e}")

print("\n" + "=" * 60)
print("所有分析任务执行完毕！")
print("=" * 60)