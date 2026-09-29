# 总扩散步数实验：Direct CSDI vs DLinear + Residual CSDI

本实验比较总扩散步数 **T**，不是固定模型下的 DDIM 采样预算 K。
每个 T 重新训练一对联合模型，并执行完整的 T 步 DDPM 反向采样。
任务仍为“历史窗口 → 多变量未来序列”的条件生成。

## 运行

在仓库根目录，Python 3.10+：

```bash
python -m pip install -r requirements-step-study.txt
python -m unittest discover -s tests -v
python -m step_study.run --synthetic --smoke --seeds 42 43 \
  --output step_study_runs/total_T_smoke
```

合成数据默认包含四个变量。smoke 使用小骨干、单轮训练、T=[10,20,40]，
仅检查流程；图片明确标注 DEMONSTRATION ONLY，不能作为论文实验结果。

真实数据示例（把 var1 等替换为真实的四个列名）：

```bash
python -m step_study.run --data data/my_series.csv \
  --columns var1,var2,var3,var4 \
  --diffusion-steps 10 20 50 100 200 \
  --device cuda:0 --seeds 42 43 44 \
  --output step_study_runs/four_variables
```

数据须按时间排序、等间隔、无缺失；CSV 需显式指定数值列以排除时间戳。
也支持 .npy，形状为 [time, variables]，列名自动为 variable_0 等。
若联合建模的变量多于四个，使用 --columns 指定全部输入列，再用
--plot-variables var1,var2,var3,var4 指定画图的四列；其余变量仍参与联合建模，
CSV 输出和共享损失统计仍包含全部变量。默认输入不是四列时要求显式选择，
不会自动挑选结果最好的四个变量。

历史长度、预测长度、训练轮数、样本数等在 config/step_study.yaml 中设置。
study.diffusion_steps 是扫描列表，命令行 --diffusion-steps 可覆盖它。
每个 T 和每个种子均训练两套扩散模型，训练成本随 T 配置数增加。
DLinear 每个种子训练一次并冻结，在全部 T 下复用。

使用相同参数加 --evaluate-only 可加载所有 T 的权重重新评估。
旧版 DDIM 配置和 checkpoint 不兼容，新入口会明确拒绝混用。
初次训练拒绝覆盖非空输出目录。

不运行模型、只从保存的 CSV 重画图片：

```bash
python -m step_study.plotting --run step_study_runs/four_variables
```

该命令可加 --plot-variables 选择其他四个已评估变量。

## 用户要求的图

每种图均输出 PNG 和 PDF，变量顺序遵循输入或 --plot-variables。

| 文件 | 布局与含义 |
| --- | --- |
| ncrps_vs_diffusion_steps_test | 2×2 四变量图；横轴总扩散步数 T，纵轴 nCRPS；每格对比 Direct CSDI 与 DLinear + Residual CSDI |
| relative_loss_vs_diffusion_steps_test | 2×2 四变量图；横轴 T，纵轴每变量相对损失 r_i(T)，以百分比显示 |
| ncrps_vs_diffusion_steps_val | 验证集版本 |
| relative_loss_vs_diffusion_steps_val | 验证集相对损失版本 |
| shared_step_relative_loss_val | 验证集所有变量平均相对损失随 T 变化的辅助图 |

横轴为线性数值轴，刻度明确显示实际训练过的 T。曲线是训练种子均值，
阴影为总体标准差，非置信区间；阴影低于零的部分截至零。
每个 T 的评分均针对最终完整序列：
Direct 直接生成 Y；Two-stage 输出 DLinear(history) + 生成残差。
不对残差自身评分来替代完整序列质量。

## nCRPS 的明确口径

在每个变量 i、每个评估集合内，对所有预测窗口和预测时刻 j：

nCRPS_i(T) = sum_j CRPS(F_ij,T, y_ij) / sum_j |y_ij|。

分子、分母均在原始物理量纲上计算，两个方法和全部 T 共用同一目标分母。
这与原 CSDI utils.py 的目标绝对值归一化口径一致；这里使用精确的有限
经验分布 CRPS，而原 CSDI 使用 19 个分位点近似积分，数值不保证完全相同。
本指标不是上一版的 CRPS / training_std，也没有额外乘 100。

经验 CRPS = mean_s |x_s-y| - 0.5 mean_(s,s') |x_s-x_s'|。
实现通过样本排序精确计算该有限样本表达式。
归一化分母为零（某变量评估目标全为零）时报错，不把未定义结果画成零。
nCRPS 对乘法单位换算不变，但对平移不保持不变，因此先还原训练均值。

## 每变量相对损失

对每个方法 m、变量 i、种子 s、评估集合分别计算：

r_m,i,s(T) = [M_m,i,s(T) - min_T' M_m,i,s(T')]
             / [min_T' M_m,i,s(T') + epsilon]，

其中 M=nCRPS，epsilon 默认为 1e-8。**两种方法各自以自己的最优值为基准**。
先在每个种子内计算相对损失，再跨种子取均值和标准差。因此图中的平均曲线
不一定有恰好为零的点（不同种子的最优 T 可能不同）。10% 意味着比该方法、
该变量、该种子在扫描范围内的最好 nCRPS 高约 10%。

测试图使用测试网格最小值，仅作描述性敏感性分析，并在图上标明。
不据此选择最终部署或报告的最优 T；步数选择使用验证集，随后在测试集评价。

summary.json 同时报告验证集逐变量最优 T、std(log T_i*)、2% 近最优区间、
区间交集和共享损失 G = min_T mean_i r_i(T)。
最优 mean(nCRPS) 的共享 T 与最小 mean(relative loss) 的 T 分别保存，
二者可能不同。测试集上另报验证集选出的共享 T 与逐变量 T_i* 的性能差，
这个测试差值允许为负。

## 控制变量与可复现性

- 两组复用原版 diff_CSDI 时间/变量注意力、相同参数量、历史条件、
  初始化、训练批次和每个 T 内配对的训练噪声；无变量子采样。
- 同一连续 VP 日程离散化：alpha_bar(u)=exp[-b_min*u -
  0.5*(b_max-b_min)*u^2]，u∈[0,1]；beta_t=1-alpha_bar(t/T)/
  alpha_bar((t-1)/T)。默认 b_min=0.1、b_max=20，
  各 T 的 terminal alpha_bar 均约 4.32e-5。较小 T 的单步噪声自然更大。
- 时间嵌入使用同一归一化时间 u（固定参考尺度 1000*u），避免相同整数
  t 在不同 T 下代表不同噪声水平却使用相同嵌入。
- 每个 T 均执行完整 ancestral DDPM，用标准后验方差添加反向噪声，
  最后一步不加噪；恰好 T 次去噪网络调用。
- 初始 Gaussian 跨 T、跨方法复用；反向噪声使用独立随机流，同一个 T
  的两组方法逐次配对。不同 T 的链长度不同，不宣称中间随机轨迹相同。
- 每个 T 使用相同训练更新预算，验证集固定噪声损失选 checkpoint。
  不复用其他 T 的训练权重。DLinear 选验证集 MSE 最优权重后冻结。
- 原时间轴按 60/20/20 划分；目标窗口不跨 split，历史可使用先前已知
  观测（rolling-origin）。归一化仅拟合训练数据，残差不另行标准化。
- DLinear 增加整体参数量和训练成本，只有扩散骨干容量相同。训练残差
  使用 DLinear 的样本内预测；若第一阶段过拟合，需要另做交叉拟合检查。

## 其他输出

- manifest.json：实验类型、配置、数据指纹、scaler、种子、变量、运行环境。
- seed_*/dlinear.pt：每个种子唯一的第一阶段权重。
- seed_*/T_*/direct.pt、residual.pt：每个 T 独立训练的权重。
- seed_*/T_*/diffusion_config.json：T、日程、终端 alpha_bar 与采样器。
- *.history.json：训练及验证损失。
- per_variable.csv：nCRPS、原尺度 CRPS、标准化 CRPS/RMSE、归一化分母、
  diffusion_steps 和 sampling_steps（均为 T）。
- per_variable_relative_loss.csv：逐变量/方法/种子/T 的相对损失及参考最优值。
- joint_scores.csv：联合轨迹 energy score 和采样耗时（残差组包含 DLinear）。
- summary.json：最优 T、共享损失、近最优区间等。
- plot_metadata.json：画图变量、轴含义、聚合方式和是否为示范数据。

更平坦的曲线及更低的相对损失支持“共享总步数的折中减小”，但单独不能
证明生成难度异质性下降：残差幅值与第一阶段占比也会影响曲线。
需结合绝对 nCRPS、联合指标、多种子结果及适当消融解释。

## 参考

- [CSDI](https://github.com/ermongroup/CSDI)：复用本仓库去噪骨干。
- [DLinear / LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)：趋势/季节分解加线性预测设计。
- [DDPM](https://arxiv.org/abs/2006.11239)：完整反向采样过程。
