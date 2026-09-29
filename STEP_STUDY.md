# 论文任务适配：条件多能源场景生成的扩散步数实验

当前入口针对论文配套代码 HEEWDailyStage1Dataset 的任务：
给定**目标日天气与日历条件**，联合生成当天 24 小时的
**Electricity、Heat、Cooling、PV**。模型不读取历史能源序列，
也不读取目标日的真实能源值作为条件。

默认 T=[10,20,50,100,200]。每个 T 独立训练 Direct CSDI 与
DLinear + Residual CSDI，并执行完整 T 步 DDPM。T 不是固定模型下的 DDIM K。

## 任务依据与实现范围

任务依据为 wenzi0731/wenzia 仓库 commit
`4dcb890a35c6597faeeecdcdfd47192c9a737e65` 的 `2-stages/`：

- dataset.py 中的 HEEWDailyStage1Dataset；
- configs/train_stage1.yaml、configs/train_stage2.yaml；
- dataset_stage2.py、model_stage2.py、metrics_stage2.py。

以当前数据加载器为准，不沿用早期 README 中的比例划分。

对齐内容包括：四个目标通道及顺序、条件通道、完整 24 小时样本、
2014–2020 年训练、2021 年验证、2022 年测试、训练集统计量归一化，
以及条件生成而非历史预测。
时间编码为 month/dayofyear/weekday/hour 的 sin/cos，共 8 个通道。
默认 pv10 气象通道为：

Temperature, Dew Point, Humidity, Wind Speed, Pressure, Precip,
ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, PV_CLEARNESS_RATIO, PV_IS_DAYLIGHT。

保留本对照实验指定的 DLinear + CSDI，不把论文中的 LCGT/PCMCI 等完整
第一阶段或 center/scale/factor 第二阶段搬入此基线。本分支的目的仍是
检验两阶段残差化对共享总扩散步数的影响，不是复现论文全部架构。

## 两组模型实际读取什么

令 C 为 [24,18] 的天气和日历条件，Y 为 [24,4] 的目标能源曲线，
z=(year−2014)/6 为单独传入的 PV 年份坐标。

| 模型 | 条件 | 扩散目标 | 最终样本 |
| --- | --- | --- | --- |
| Direct CSDI | C；PV 输出头另收 z | 标准化 Y | 逆标准化 generated Y |
| DLinear + Residual CSDI | C 和可选 mu(C,z)；PV 输出头另收 z | 标准化残差 R | mu(C,z)+残差均值+残差尺度×generated R |

z 不拼入共享天气/日历条件。DLinear 仅在 PV 输出应用年份 FiLM，
CSDI 仅在 PV 噪声预测头应用年份 FiLM。2021、2022 的 z 为 7/6、8/6。
这是“年份的显式注入仅位于 PV 头”，不保证联合采样中其他变量完全
不受间接影响：变量注意力及含 PV 的基准曲线仍允许跨变量信息传播。

DLinear 每个种子训练一次后冻结，各 T 复用。由于原版 DLinear 假定输入
和输出是同一组时间序列，这里明确采用 **Conditional DLinear 改编**：
先分解每个外生条件通道的平滑项/剩余项，再做时间线性映射（24→24），
最后通过可学习的条件通道→四个能源通道线性投影生成基准曲线。
它不使用历史能源，不能描述为未经改动的原版 DLinear。

两组 CSDI 的网络架构和参数量相同，天气与时间在每个去噪块作为 side
information 注入，同时保留时间注意力和变量注意力。
Direct 的第二个辅助输入为零；Residual 默认显式接收冻结的 mu(C)，
既提供外生条件也提供第一阶段结构。辅助输入不是真实能源观测。

默认 model.condition_on_skeleton=true。改为 false 可进行重要消融：
残差组仍学习 Y-mu(C)、最终加回 mu(C)，但两组仅接收相同的 C。
这样可以进一步区分“残差化”和“显式基准曲线引导”的影响。
此设置改变了实验定义，需使用新输出目录重新训练。

默认 data.residual_normalization=train_residual：冻结 DLinear 后，
仅用训练日的物理量纲残差拟合每变量均值及样本标准差（加 1e-6），
与论文第二阶段的残差归一化口径一致。验证、测试复用训练统计量。
生成后先还原残差并加回基准曲线，最后对完整能源序列评分。

设为 target_scale 可关闭残差重新标准化：残差均值固定为 0，尺度
使用原目标训练标准差。这是建议报告的消融，区分残差化与重新缩放
的贡献；与 condition_on_skeleton=false 组合可进一步控制显式结构条件。
每种配置均需新输出目录、独立训练，不能混入一条曲线。
没有额外施加非负裁剪或 PV 夜间强制归零，避免后处理改变曲线归因。

## 使用真实论文数据

在仓库根目录，Python 3.10+：

```bash
python -m pip install -r requirements-step-study.txt
python -m unittest discover -s tests -v

python -m step_study.run \
  --energy-path /path/to/wenzia/2-stages/Data/CN03_energy_cleaned.csv \
  --weather-path /path/to/wenzia/2-stages/Data/weather_cleaned.csv \
  --diffusion-steps 10 20 50 100 200 \
  --device cuda:0 --seeds 42 43 44 \
  --output step_study_runs/paper_conditioned
```

两个 CSV 都须包含 Year,Month,Day,Hour，并包含上述目标/条件字段。
按时间戳交集连接，允许两张表行序不同；重复时间戳直接报错。
记录两表未匹配小时数，丢弃交集中非完整 24 小时的日，
在 manifest 记录丢弃数量。随后按配置中的自然年固定划分；
目标和天气的均值/标准差只用训练日拟合，标准差加 1e-6。
日历周期编码不再拟合 scaler。

data.weather_feature_set 支持 pv10（默认）、base6；
只有天气列改变，始终保留 8 个日历条件。默认所有四个能源目标联合生成，
输出顺序为 Electricity,Heat,Cooling,PV，不依赖 CSV 内的列排列顺序。
--plot-variables 可改变四个子图的显示顺序。

快速检查（可以直接使用真实数据路径）：

```bash
python -m step_study.run \
  --energy-path /path/to/wenzia/2-stages/Data/CN03_energy_cleaned.csv \
  --weather-path /path/to/wenzia/2-stages/Data/weather_cleaned.csv \
  --smoke --seeds 42 --output step_study_runs/paper_smoke

python -m step_study.run --synthetic --smoke --seeds 42 \
  --output step_study_runs/condition_demo
```

smoke 保持 24 小时日样本、18 条件及独立 PV 年份、4 目标、自然年划分和训练 scaler，
仅缩小模型、训练一轮，并在原 split 内确定性抽取最多 16/4/4 日；
默认 T=[10,20,40]，4 个生成样本。图片标注 DEMONSTRATION ONLY。
这用于数据接口和流程测试，不能据其效果得出论文结论。

使用相同参数加 --evaluate-only 可以重跑保存权重的评估。
历史预测版本、旧 DDIM 版本以及旧比例划分版本的配置与权重不能用于
当前 v4 入口。省略 --seeds 时默认 42；正式实验建议至少三个种子。
初次训练不覆盖非空输出目录，数据与配置指纹用于检查评估复现。
旧 --data/--columns/NumPy 历史预测入口已从主命令移除。

无需运行模型、只重画保存的曲线：

```bash
python -m step_study.plotting --run step_study_runs/paper_conditioned
```

## 图与指标

每种图均输出 PNG 和 PDF。每张四变量图为 2×2 子图，
每格对比 Direct CSDI 和 DLinear + Residual CSDI。

| 文件 | 含义 |
| --- | --- |
| ncrps_vs_diffusion_steps_test | 测试集横轴 T，纵轴 nCRPS |
| relative_loss_vs_diffusion_steps_test | 测试集横轴 T，纵轴每变量相对损失百分比 |
| ncrps_vs_diffusion_steps_val | 验证集 nCRPS 版本 |
| relative_loss_vs_diffusion_steps_val | 验证集相对损失版本 |
| shared_step_relative_loss_val | 验证集四变量平均相对损失 |

曲线是种子均值，阴影为总体标准差而非置信区间，低于零部分截至零。
所有评分针对最终完整能源序列，不用残差自身的评分替代。

nCRPS_i(T) = sum_j CRPS(F_ij,T,y_ij) / (sum_j |y_ij| + 1e-8)。

j 遍历评估日和小时；分子/分母均在原能源量纲计算。该归一化与原
CSDI 的目标绝对值口径一致，这里使用精确的经验分布 CRPS，而不是
原实现的 19 分位点近似。nCRPS 不是 CRPS/training_std，也不乘100。
分母加 1e-8 与论文指标实现一致。某变量目标全零或量级极小时，
稳定化后的分数也可能非常大，应结合保存的分母与原尺度 CRPS 解读。

r_m,i,s(T) = [M_m,i,s(T)-min_T' M_m,i,s(T')]
             / [min_T' M_m,i,s(T')+epsilon]，M=nCRPS，epsilon=1e-8。

每个方法、变量、种子、split 分别以自己的最小值为基准，之后跨种子
计算均值和标准差。测试网格最小值仅作描述性敏感性分析，图中明确标明；
正式共享 T 和逐变量 T_i* 从验证集选择，再在测试集评价。

summary.json 还保存 std(log T_i*)、2% 近最优区间及其交集、
共享损失 G=min_T mean_i r_i(T)，以及验证集选择的 T 在测试集上的表现。
最小 mean(nCRPS) 与最小 mean(relative loss) 的共享 T 分别报告。

## 扩散控制

使用同一连续 VP 日程 alpha_bar(u)=exp[-b_min*u-
0.5*(b_max-b_min)*u^2] 在 u=t/T 上离散化，
beta_t=1-alpha_bar(t/T)/alpha_bar((t-1)/T)。
默认 b_min=0.1、b_max=20，各 T 的终端 alpha_bar≈4.32e-5。
时间嵌入使用固定参考尺度 1000*u，以对齐不同 T 的噪声时刻。

每个 T 独立初始化并训练两组扩散网络；骨干参数量、训练更新预算、
条件编码、配对训练随机数相同。验证噪声损失选择 checkpoint，
随后固定该权重完整采样 T 步 DDPM，最后一步不加反向噪声。
初始 Gaussian 跨 T/方法复用，同一 T 两组的反向噪声也配对；
不宣称不同 T 的中间随机轨迹相同。

DLinear 的额外参数/训练成本属于两阶段系统，不计为扩散骨干容量。
训练残差使用第一阶段样本内输出，过拟合时可另做交叉拟合检查。

## 保存内容

- manifest.json：任务类型、条件列、目标列、日划分、实际评估日期、
  数据指纹、scaler、配置、种子与环境。
- seed_*/dlinear.pt：条件 DLinear 权重，每个种子一份。
- seed_*/residual_scaler.json：训练残差均值/尺度及归一化模式。
- seed_*/T_*/direct.pt、residual.pt：每个 T 独立的扩散模型权重。
- seed_*/T_*/diffusion_config.json：T、日程、终端 alpha_bar。
- *.history.json：训练及验证损失。
- per_variable.csv：nCRPS、原尺度 CRPS、标准化 CRPS/RMSE、归一化分母。
- per_variable_relative_loss.csv：每变量/方法/种子/T 的相对损失及参考最优值。
- joint_scores.csv：联合轨迹 energy score、采样耗时（残差组含 DLinear）。
- summary.json、plot_metadata.json：步数诊断和绘图元信息。

原始能源与天气数据不包含在分支中。较低共享损失支持“共享总步数折中
减小”，但不能单独证明生成难度异质性下降；需结合绝对生成质量、联合
指标和上述消融解释。
