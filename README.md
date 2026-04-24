# Chinese-Western-Pareto-Hyper

> 基于中西医融合与帕累托优化的高血脂风险预警及个性化干预研究

## 📌 项目简介

本项目针对中老年人群高血脂症的风险预警与痰湿体质个性化干预问题，融合**中医体质分型**（九种体质积分）、**中老年人活动能力评分**及**血常规代谢指标**，构建了一套涵盖特征筛选、风险分层与干预方案优化的完整数学模型。

主要亮点：
- 首次揭示痰湿质通过**非线性交互**（尤其与 TG）间接影响高血脂的核心机制；
- 构建 **WOE 评分卡** 输出 0–100 分直观风险等级（低/中/高），AUC 达 1.000；
- 采用 **TabPFN**（Nature 2025 表格基础模型）预测干预效果，R² = 0.810；
- 通过**帕累托多目标优化**提供最小成本、最大降低、性价比最佳三类方案；
- 利用**决策树**提取“患者特征→最优方案”匹配规则，测试集准确率 100%。

## 🗂️ 仓库结构

```
├── code/ # 核心建模代码
│ ├── step1_preprocess.py # 数据预处理、特征工程（交互特征构造）
│ ├── step2_question1_model.py # 问题一：特征筛选、双任务随机森林、SHAP分析
│ ├── step3_question2_scorecard.py # 问题二：WOE分箱、评分卡建模、交叉验证
│ ├── step4_question3_optimization.py # 问题三：虚拟数据生成、TabPFN、帕累托优化、决策树
│ └── utils/ # 辅助函数（分箱、可视化等）
├── results/ # 输出结果
│ ├── scorecard_feature_scores.csv # 评分卡各特征分值表
│ ├── risk_thresholds.txt # 三级风险阈值
│ ├── pareto_frontiers/ # 每位患者的帕累托前沿图
│ └── matching_rules.txt # 决策树提取的匹配规则
├── paper/ # 论文原文及附件（PDF）
├── README.md
└── requirements.txt # Python 依赖
```

# 项目data与研究基于2026年第十六届MathorCup数学应用挑战赛C题

## 若本项目对您的研究有启发，请引用：

```
@misc{ChineseWesternParetoHyper2026,
  title = {基于中西医融合与帕累托优化的高血脂风险预警及个性化干预研究},
  author = {Chinese-Western-Pareto-Hyper Team},
  year = {2026},
  howpublished = {\url{https://github.com/Doublewestie/Chinese-Western-Pareto-Hyper}}
}
```
