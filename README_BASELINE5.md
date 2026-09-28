# Baseline 5：CSDI 条件联合源荷场景生成

## 1. 改了什么，保留了什么

在本仓库上游提交 `7f24a436f08d98853a6b43d4f7f04e5a65ecdf27` 基础上适配。
论文中建议写 **CSDI (condition-only adaptation)**，不要声称是原论文数据集实验的原样复现。

- 直接继承原版 `main_model.CSDI_base`，复用原版 `diff_models.diff_CSDI`。
- 保留交替的时间 Transformer、变量 Transformer、残差/skip 连接、扩散步嵌入、特征嵌入、掩码侧信息和噪声预测 MSE。
- 将 `[Electricity, Heat, Cooling, PV, 18个天气/日历条件, year]` 组成 `[B,23,24]`。
  条件掩码始终为前4个通道0、后19个通道1。训练和测试的掩码一致，不随机暴露能源真值。
- 保留原版双输入通道：`[mask * observed, (1-mask) * noisy]`。
  能源真值仅用于训练时构造带噪目标和监督，不进入已知条件输入；测试生成函数没有目标参数。
- 损失仅作用于四个能源通道；通过原版变量注意力联合建模，不是四个独立模型。
- 原版 `main_model.py` 不改动。`diff_models.py` 仅把可选的线性注意力依赖改为延迟导入；默认 `is_linear=false` 不需要该依赖，网络计算不变。
- 新增批量分块采样与独立随机数生成器，反向更新仍是原版全步 DDPM。单场景固定随机数下已通过与上游采样逐值一致的测试。
- 单阶段训练，无第一阶段预测器，无 OOF。

这是给定目标日天气和时间的条件生成任务，不是使用历史能源序列的预测任务。
数据中的目标日天气条件不能自动称为“前一天发布的天气预报”。

## 2. 统一数据协议

| 项目 | 设置 |
|---|---|
| 训练 / 验证 / 测试 | 2014–2020 / 2021 / 2022 |
| 一个样本 | 完整24小时，按时间戳合并能源和天气数据 |
| 四个目标顺序 | Electricity、Heat、Cooling、PV |
| 天气 | pv10：Temperature、Dew Point、Humidity、Wind Speed、Pressure、Precip、ALLSKY_SFC_SW_DWN、CLRSKY_SFC_SW_DWN、PV_CLEARNESS_RATIO、PV_IS_DAYLIGHT |
| 日历 | 月、年内日、星期、小时各一对 sin/cos；另加 year=(Year-2014)/6 |
| 归一化 | 仅完整训练日的各通道 mean/std；验证测试复用，不重新拟合 |
| 能源历史 | 不使用 |
| 验证 / 测试场景数 | 每日20 / 100 |
| 后处理 | 反归一化后仅将 PV 负值截断为0，与之前 baseline 保持一致 |

默认数据文件：`Data/CN03_energy_cleaned.csv` 和 `Data/weather_cleaned.csv`。
两者均须有 `Year,Month,Day,Hour` 列，小时为0–23。数据不上传仓库。
数据适配代码沿用 baseline3/4，目标/天气归一化统计和两份 CSV 的 SHA256 随 checkpoint 保存。
测试路径可以变，但文件内容必须与训练时一致；不同数据会明确报错。

## 3. 环境和运行位置

在仓库根目录运行：

```bash
pip install -r requirements_baseline5.txt
python -m pytest -q
```

推荐 Python 3.10+、PyTorch 2.x。实际运行环境版本会写入 `environment.json`。
路径默认相对于仓库根目录解析。支持 `python -m baseline5.train` 和直接运行 `baseline5/train.py`。
只加载自己训练或可信来源的 checkpoint（PyTorch checkpoint 可能包含 pickle 对象）。

## 4. 严格六组调参

| ID | 扩散网络宽度 channels | 初始 learning_rate |
|---|---:|---:|
| c01_w64_lr1e3 | 64 | 0.001 |
| c02_w64_lr1e4 | 64 | 0.0001 |
| c03_w96_lr1e3 | 96 | 0.001 |
| c04_w96_lr1e4 | 96 | 0.0001 |
| c05_w128_lr1e3 | 128 | 0.001 |
| c06_w128_lr1e4 | 128 | 0.0001 |

保留原版默认：4层残差块、8头、时间嵌入128、特征嵌入16、扩散嵌入128；
**T=50，采样也完整50步**，二次 beta 调度 `1e-4 → 0.5`。
这是原版 CSDI 的设置，并非因为其他 diffusion baseline 使用1000步就统一改成1000步。
此实现不支持独立跳步采样；不要添加 `sampling_steps` 伪参数。改 T 需要重新训练，且不能根据测试集表现挑选。

Adam，batch_size16，weight_decay1e-6，默认不做 EMA 平滑；保留原版在最大轮数75%和90%处乘0.1的学习率调度。
实验协议统一为最多300轮、每5轮验证一次、12次验证不提升则早停，梯度裁剪1。
六组只改变宽度和学习率；选择标准是2021验证集 **macro_nCRPS最小**。
验证采样使用独立固定随机流，不消耗训练随机流。六组等预算不等同于等 GPU 时间，采样时间和参数量另行记录。

执行六组搜索（YAML不是可执行文件）：

```bash
python -m baseline5.sweep --seed 42
```

若数据放在其他位置：

```bash
python -m baseline5.sweep --seed 42 \
  --set data.energy_path=/data/HEEW/CN03_energy_cleaned.csv \
  --set data.weather_path=/data/HEEW/weather_cleaned.csv
```

脚本不会测试2022数据。输出到 `experiments/baseline5/tuning_budget_6/`：

- `protocol.yaml`：本次搜索的完整协议；不允许同目录混用不同协议。
- `tuning_results.csv`：六组按验证分数排序。
- `best_config.json`：赢家编号、验证指标、最佳epoch和checkpoint路径。
- `best_config.yaml`：已经合并赢家参数及数据路径，可直接用于最终训练。

各组 checkpoint 位于 `experiments/baseline5/csdi_tune_<ID>_seed42/best.pt`。
`--skip-completed` 只跳过配置哈希匹配且已完整结束的运行，不是中断训练的checkpoint续训。
中断留下的非空未完成目录会报错；请先备份该目录，或换新输出根目录再运行。默认不会覆盖结果。

## 5. 选定配置后五种子复现

```bash
for seed in 42 123 777 2024 3407; do
  python -m baseline5.train \
    --config experiments/baseline5/tuning_budget_6/best_config.yaml \
    --run-name csdi_final --set run.seed=$seed
done
```

固定赢家超参数，各种子仍按同一验证规则选checkpoint。不要按测试结果选择种子/步数/超参数。
这五次是固定配置复现，不是另加五组超参数搜索。主表报告五种子均值±样本标准差。

## 6. 测试及输出位置

```bash
python -m baseline5.evaluate \
  --checkpoint experiments/baseline5/csdi_final_seed42/best.pt
```

默认对2022所有完整日，每日生成100个四通道联合场景。
输出在该 checkpoint 同级的 `evaluation_test/`，上述命令对应：

```text
experiments/baseline5/csdi_final_seed42/evaluation_test/
  global_metrics.csv          # 全部指标，metric/value形式
  global_metrics.json
  channel_metrics.csv         # 每通道一行，适合整理主表
  daily_scores.csv            # 每日ES/VS及各通道CRPS、IS，便于配对统计
  metric_definitions.json     # 指标公式、标准化、VS权重和区间水平
  baseline5_scenarios.npz      # physical [day,scenario,channel,hour]
  global_pearson.png
  pearson/                    # 真实/生成相关性及绝对差异图
  random_timeseries_50/       # 固定随机挑选最多50日的曲线图
```

五种子测试：

```bash
for seed in 42 123 777 2024 3407; do
  python -m baseline5.evaluate \
    --checkpoint experiments/baseline5/csdi_final_seed${seed}/best.pt
done
```

`--seed` 默认取checkpoint训练seed，实际采样随机种子为 `seed+90000`；验证为 `seed+50000`。
同一checkpoint、seed、硬件/库版本、batch_size和sample_chunk_size下复现；改变分块形状可能改变随机数分配及数值。
使用 `--outdir` 指定新目录；`--no-plots` 只跳过画图，不跳过指标；`--max-days` 仅用于调试，JSON会标记 partial_test。
所有最终比较必须使用相同测试日期和场景数。GPU跨平台不保证逐bit完全一致。

## 7. 新增指标的严格定义

以下 R² 越大越好，IS、ES、VS 越小越好。

### 每通道 R²

用100个场景的均值作为点估计，合并所有测试日和小时：

`R2_c = 1 - sum((y_c - mean_s x_sc)^2) / sum((y_c - mean(y_c))^2)`。

不是 Pearson 相关系数平方，也不是先计算每日 R² 再平均；R²可以为负。
常数目标通道的 R²未定义，JSON写null、CSV留空，不强行变成0或1。

### 每通道 IS：95% interval score

`l=q0.025, u=q0.975, alpha=0.05`，分位数使用线性插值：

`IS = (u-l) + (2/alpha)*(l-y)*1[y<l] + (2/alpha)*(y-u)*1[y>u]`。

对所有测试日和小时取均值。`<channel>_IS` 与 `<channel>_IS95` 同义，保留原单位；
`<channel>_IS_Z = IS / training_std_c`，便于不同通道比较，不能用测试集std归一化。
95%水平与现有 `CR`、`IW` 对齐；同时保留原有90% coverage/width，不将90%和95%混用。

### 联合 ES：energy score

先用每通道训练mean/std标准化，再把每个场景按通道优先顺序展开为96维向量。
逐日计算：

`ES_d = (1/S)*sum_s ||x_s-y||_2 - (1/(2*S^2))*sum_s sum_r ||x_s-x_r||_2`。

输出为逐日得分的等权平均。使用经验分布/V-statistic（包含零对角），不是分母 `S*(S-1)` 的fair版本。
`ES` 与 `ES_Z` 同义；不额外除以96或sqrt(96)，因此比较必须统一维度、标准化和场景数。

### 联合 VS：variogram score

同一个96维标准化向量上，`p=0.5`：

`VS_d = (1/M)*sum_{i<j} (|y_i-y_j|^p - (1/S)*sum_s |x_si-x_sj|^p)^2`，
其中 `M=96*95/2=4560`，所有无序维度对等权，权重总和1。
覆盖跨时间和跨通道的维度对；输出逐日平均，`VS` 与 `VS_Z` 同义。
有些论文不除以M，或对有序对求和，数值不可直接对照。这里把约定显式固定。

保留原有 MAE/RMSE、MAE_Z/RMSE_Z、CRPS/nCRPS、Precision_Z/Recall_Z、CR/IW。
`mean_nCRPS` 是所有通道混合后的比值，`macro_nCRPS` 是四个通道比值的平均；选参用后者。
Precision/Recall 沿用其他 baseline 的最近中心半径近似算法，不是完整 kNN 球并集判据；论文中应如实说明。

评分参考：

- Gneiting & Raftery (2007), [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437)。
- Scheuerer & Hamill (2015), [Variogram-Based Proper Scoring Rules for Probabilistic Forecasts of Multivariate Quantities](https://journals.ametsoc.org/abstract/journals/mwre/143/4/mwr-d-14-00269.1.xml)。

## 8. 给其他baseline已有场景补算相同指标

为使主表可比较，应对所有方法用同一评分公式。对于已保存兼容NPZ的模型，可离线补算，无需重训：

```bash
python -m baseline5.score_npz \
  --npz /absolute/path/to/baseline4_scenarios.npz \
  --outdir results/baseline5/rescore_baseline4_seed42 --seed 42
```

必须包含 `scenarios [N,S,4,24]`、`targets [N,4,24]`、`dates`、`target_mean`、`target_std`。
输入必须是已经完成一致后处理的物理单位数据，通道顺序严格为电/热/冷/PV，统计量来自训练集。
脚本不会猜测缺失的统计量、从测试集重拟合，或再次改变后处理。
请自己核对不同模型的日期、条件、归一化和场景数是否一致；此脚本不能证明旧实验协议相同。

## 9. 验证与限制

自动测试包括：上游损失/采样一致性、条件梯度、无目标条件泄漏、固定seed复现、训练集归一化、严格六组搜索、
微型训练→选参→完整测试文件/图输出、离线重评分一致性，以及R²/IS/ES/VS的手算与暴力公式核对。
这些测试证明实现流程可运行，不代表已经完成真实数据六组训练或证明模型优劣。
初始搜索空间是预先声明的预算，不保证是理论全局最优。不要在看过测试结果后增加搜索。
新增代码遵循仓库MIT许可证；使用时请引用原版CSDI论文（原README保留）。
