# 数据服务方案选择的条件Logit分析

本项目保存虚构数据共享及产品运营的合成表格与Python工具。`src/choice_reader.py` 只按材料读取与显示；
`src/choice_model.py` 与 `src/choice_analysis.py` 新增离散选择分析：
在给定候选集合下解释价格与更新频率的条件Logit，不推断个人敏感特征。

本仓库中的地区、机构、产品与人数均为合成。所有输出都是模型在给定候选集合下的概率与情景算术，
**不是社会价值、支付意愿或因果价格效应**；机构标签只用于整组划分与整组重抽样。

## 模型与约定

- 每个 `choice_id` 给出完整 `alternatives` 与恰好一个 `chosen`；`chosen` 必须是该集合中的键。
- 条件Logit：`P(chosen=j) = exp(beta·x_ij) / Σ_k exp(beta·x_ik)`，无方案特定截距。
- 属性为 `price` 与 `update_frequency`；`outside` 若出现是正式备选，**不会把未展示方案补成未选**。
- 价格单位换算（可选 `price_scale` / `price_unit`）**先于**标准化；
  标准化均值/标准差只在 `fit` 划分上估计，原样应用到 `validation`。
- 同一 `choice_id` 重复记为输入错误并拒绝；同一机构不得跨 `fit`/`validation`。
- 单候选集合没有集合内对比，因此没有偏好识别信息，但仍然计数并照常报告概率 1。
- 报告中的 `beta` 为标准化坐标；`beta_raw` 按 fit 划分的 sigma 与价格单位换算，
  sigma=0 的坐标标为 `not_identified`，不会被当成有限系数。
- 同一选择内所有方案的属性整体平移不改变概率（条件Logit 的平移不变性）。

## 输出与诊断

分析报告（JSON）区分：

1. **识别诊断**：集合内属性差矩阵的秩、完全分离、单候选、是否提供 outside、属性是否超出 fit 支撑。
2. **估计**：damped Newton 最大化条件似然；可选非负 `ridge`（仅惩罚系数平方）。
   报告同时给出惩罚目标值、未惩罚对数似然与惩罚量；`sa_penalized` 标记是否为惩罚拟合。
   优化收敛不等于所有系数被识别——秩或分离诊断未通过时不得宣称识别。
3. **预测**：集合内概率、validation 平均对数损失。
4. **情景**：每个方案自身价格变化的弹性；删除某方案后的概率重算（IIA 情景，非真实替代率），
   删除被选中方案单列；机构 bootstrap 整组重拟合，失败单列并保留原因码。

此外报告固定包含：极大效用/概率贴边标记（阈值 0.99）、outside 缺失标记、属性外推标记、
平移不变性检查、解析梯度/海森与有限差分的对照误差。bootstrap 同时给出
`identified`/`unidentified_rate` 与仅通过识别诊断的 `identified_only` 汇总，
不会用未识别重拟合冒充抽样分布。

## 运行

Python 3.11 以上，标准库即可，无需第三方依赖。

    python -m src.choice_reader fixtures/choices.json
    python -m src.choice_analysis fixtures/choices.json fixtures/choice.json
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

`src.choice_analysis` 也可 `--output` 写 JSON；未给 config 时按 fixture 中的 config 块读取。

## 固定容器

    docker build -t conditional-choice:local .
    docker run --rm conditional-choice:local

容器镜像以完整 digest 固定；新增分析功能仍只用 Python 标准库，未登记第三方依赖。
