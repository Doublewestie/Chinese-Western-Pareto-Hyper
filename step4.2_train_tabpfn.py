# step4.2_train_tabpfn.py
# 使用 TabPFN 基础回归模型（v2.5），适配 9 个特征（6原始 + 3强交互）
# 优化：训练集下采样、降低集成数、分批预测、内存节省模式

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")

# ==================== 1. 环境变量与路径配置 ====================
os.environ["TABPFN_MODEL_CACHE_DIR"] = r"C:\Users\26011\AppData\Roaming\tabpfn"
os.environ["TABPFN_NO_BROWSER"] = "1"

DATA_DIR = r"E:\MathorCup\Personal-Formal\data\synthetic"
OUTPUT_DIR = r"E:\MathorCup\Personal-Formal\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "synthetic_training_data_train.csv")
VAL_PATH = os.path.join(DATA_DIR, "synthetic_training_data_val.csv")

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "tabpfn_regressor.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "tabpfn_performance.txt")
SCATTER_PLOT_PATH = os.path.join(OUTPUT_DIR, "tabpfn_pred_vs_true.png")

# TabPFN 配置
RANDOM_SEED = 42
DEVICE = 'cuda'                     # 有 GPU 用 'cuda'，否则 'cpu'
N_ENSEMBLE = 16                     # 集成配置数（降低以加速）
MAX_TRAIN_SIZE = 10000              # 训练集最大样本数（下采样）
BATCH_SIZE = 1000                   # 分批预测的批次大小

# ==================== 2. 加载数据 ====================
print("=" * 60)
print("TabPFN 回归模型训练（9个特征：6原始+3强交互）")
print(f"设备: {DEVICE}")
print(f"集成配置数: {N_ENSEMBLE}")
print("=" * 60)

print("\n[1/5] 加载数据...")
df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)

print(f"原始训练集: {len(df_train)} 条样本")
print(f"验证集: {len(df_val)} 条样本")

# 训练集下采样（加速训练）
if len(df_train) > MAX_TRAIN_SIZE:
    df_train = df_train.sample(n=MAX_TRAIN_SIZE, random_state=RANDOM_SEED)
    print(f"训练集下采样至 {MAX_TRAIN_SIZE} 条（随机种子 {RANDOM_SEED}）")
else:
    print(f"训练集未下采样（{len(df_train)} ≤ {MAX_TRAIN_SIZE}）")

# 自动识别特征列（排除目标列 'S6'）
target_col = 'S6'
feature_cols = [col for col in df_train.columns if col != target_col]
print(f"特征列数: {len(feature_cols)}")
print(f"特征列: {feature_cols}")

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_val = df_val[feature_cols].values
y_val = df_val[target_col].values

print(f"特征维度: {X_train.shape[1]}")

# ==================== 3. 训练 TabPFN 回归模型 ====================
print("\n[2/5] 训练 TabPFN 回归模型...")
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

model = TabPFNRegressor.create_default_for_version(
    ModelVersion.V2_5,
    device=DEVICE,
    random_state=RANDOM_SEED,
    n_estimators=N_ENSEMBLE,
    softmax_temperature=0.8,            # 预测置信度
    ignore_pretraining_limits=True,      # 突破限制（样本>10000）
    memory_saving_mode=True,             # 内存节省模式
)

print("  模型初始化完成，开始训练...")
model.fit(X_train, y_train)
print("  训练完成！")

# ==================== 4. 验证集评估（分批预测） ====================
print("\n[3/5] 验证集评估（分批预测，防止 OOM）...")

num_samples = X_val.shape[0]
predictions = []
for i in range(0, num_samples, BATCH_SIZE):
    batch = X_val[i: min(i + BATCH_SIZE, num_samples)]
    batch_pred = model.predict(batch)
    predictions.append(batch_pred)
y_pred = np.concatenate(predictions)

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

relative_errors = np.abs((y_val - y_pred) / (y_val + 1e-6)) * 100
mean_rel_err = np.mean(relative_errors)
median_rel_err = np.median(relative_errors)

print(f"\n  TabPFN 验证集性能:")
print(f"    MSE:  {mse:.4f}")
print(f"    RMSE: {rmse:.4f}")
print(f"    MAE:  {mae:.4f}")
print(f"    R²:   {r2:.4f}")
print(f"    平均相对误差: {mean_rel_err:.2f}%")
print(f"    中位数相对误差: {median_rel_err:.2f}%")

# 计算训练集 R²（过拟合检查）
print("\n  计算训练集 R²（过拟合检查，分批预测）...")
num_train = X_train.shape[0]
train_preds = []
for i in range(0, num_train, BATCH_SIZE):
    batch = X_train[i: min(i + BATCH_SIZE, num_train)]
    batch_pred = model.predict(batch)
    train_preds.append(batch_pred)
y_train_pred = np.concatenate(train_preds)
r2_train = r2_score(y_train, y_train_pred)
print(f"    训练集 R²: {r2_train:.4f}")
print(f"    验证集 R²: {r2:.4f}")
if r2_train - r2 > 0.05:
    print("    ⚠️ 警告：训练集 R² 明显高于验证集，可能存在轻微过拟合。")
else:
    print("    ✅ 训练集与验证集 R² 接近，泛化良好。")

# ==================== 5. 线性基准对比 ====================
print("\n[4/5] 训练线性基准模型进行对比...")

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_val)
r2_lin = r2_score(y_val, y_pred_lin)
mse_lin = mean_squared_error(y_val, y_pred_lin)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)
r2_ridge = r2_score(y_val, y_pred_ridge)
mse_ridge = mean_squared_error(y_val, y_pred_ridge)

print(f"\n  线性回归:")
print(f"    MSE: {mse_lin:.4f}, R²: {r2_lin:.4f}")
print(f"  Ridge 回归 (α=1.0):")
print(f"    MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")
print(f"\n  TabPFN 性能提升 (vs 线性回归):")
print(f"    相对 MSE 降低: {(mse_lin - mse) / mse_lin * 100:.1f}%")
print(f"    R² 提升: {r2 - r2_lin:.4f}")

# ==================== 6. 可视化与保存 ====================
print("\n[5/5] 生成散点图...")
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_pred, alpha=0.3, s=10, c='steelblue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='理想预测')
plt.xlabel('真实 S6')
plt.ylabel('预测 S6')
plt.title(f'TabPFN 预测 vs 真实 (R²={r2:.3f})')
plt.legend()
plt.tight_layout()
plt.savefig(SCATTER_PLOT_PATH, dpi=300)
plt.close()
print(f"  散点图已保存至: {SCATTER_PLOT_PATH}")

# 保存模型
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"  模型已保存至: {MODEL_SAVE_PATH}")

# 保存性能报告
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("TabPFN 回归模型性能报告（9个特征：6原始+3强交互）\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"设备: {DEVICE}\n")
    f.write(f"集成配置数: {N_ENSEMBLE}\n")
    f.write(f"随机种子: {RANDOM_SEED}\n")
    f.write(f"特征数量: {len(feature_cols)}\n")
    f.write(f"特征列: {feature_cols}\n")
    f.write(f"训练集样本数（下采样后）: {len(df_train)}\n")
    f.write(f"验证集样本数: {len(df_val)}\n\n")
    
    f.write("TabPFN 性能:\n")
    f.write(f"  训练集 R²: {r2_train:.6f}\n")
    f.write(f"  验证集 MSE:  {mse:.6f}\n")
    f.write(f"  验证集 RMSE: {rmse:.6f}\n")
    f.write(f"  验证集 MAE:  {mae:.6f}\n")
    f.write(f"  验证集 R²:   {r2:.6f}\n")
    f.write(f"  平均相对误差: {mean_rel_err:.2f}%\n")
    f.write(f"  中位数相对误差: {median_rel_err:.2f}%\n\n")
    
    f.write("线性基准对比:\n")
    f.write(f"  线性回归 R²: {r2_lin:.6f}\n")
    f.write(f"  Ridge 回归 R²: {r2_ridge:.6f}\n")
    f.write(f"  TabPFN R² 提升: {r2 - r2_lin:.4f}\n")

print(f"  性能报告已保存至: {REPORT_PATH}")

# 保存特征列顺序（供 step4.3 使用）
feature_cols_path = os.path.join(OUTPUT_DIR, "feature_columns.pkl")
with open(feature_cols_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"  特征列顺序已保存至: {feature_cols_path}")

# 可选：保存原始特征名
base_feat_names = ['S0', 'age_group', 'A0', 'c', 's', 'f']
with open(os.path.join(OUTPUT_DIR, "base_feature_names.pkl"), 'wb') as f:
    pickle.dump(base_feat_names, f)

print("\n" + "=" * 60)
print("✅ TabPFN 模型训练完成！")
print("=" * 60)