# TDL ↔ UMi/UMa 信道泛化实验实施计划

> 状态：设计评审稿，当前只定义实验和代码改造方案，不实现代码。
>
> 适用项目：`~/projects/phase_invariant_receiver_sionna`
>
> 当前环境：Sionna-no-RT 2.0.1、PyTorch 2.13、SISO-OFDM/QPSK。

## 1. 研究目标

本阶段不再继续扩展接收机结构，固定比较两个参数严格一致的模型：

- **A：`single_branch_n0_gate`**，使用 Hermitian 读出，严格公共相位不变；
- **C：`strict_matched_complex_p_n0_gate`**，主干、投影和参数量与 A 相同，但读出保留绝对 Re/Im 信息，不严格相位不变。

主要回答两个方向的问题：

1. **TDL → UMi/UMa**：在 TDL-A/B/C/D/E 混合分布上训练，能否零样本泛化到更真实的 UMi 和 UMa 系统级信道？
2. **UMi/UMa → TDL**：在 UMi、UMa 或二者混合分布上训练，能否零样本泛化到 TDL-A/B/C/D/E？

次要问题是：A 的结构约束是否能在跨信道分布时带来更小的性能退化。不能预设 A 必然优于 C，需要使用严格配对实验验证。

## 2. 本阶段的范围边界

### 2.1 第一阶段：归一化小尺度泛化（主实验）

第一阶段只比较小尺度信道结构，统一采用：

- 单 BS、单 UT；
- 上行链路；
- UT 和 BS 都使用 `1×1`、单极化 V、omni 天线；
- 载频 3.5 GHz；
- 与当前项目相同的 14×72 OFDM 网格、30 kHz SCS、QPSK 和 DMRS；
- `enable_pathloss=False`；
- `enable_shadow_fading=False`；
- `normalize_channel=True`；
- 保持当前按接收信号功率定义 N0 的 measured-SNR 方式。

这样得到的结论是“信道结构/PDP/几何随机性变化下的泛化”，不包含覆盖距离和大尺度接收功率差异。

### 2.2 第二阶段：真实覆盖泛化（后续实验）

只有第一阶段结果稳定后，才考虑：

- 开启 pathloss 和 shadow fading；
- 关闭 channel normalization；
- 固定发射功率、热噪声密度、带宽和接收机噪声系数；
- 统计 BER/BLER 随距离、LoS/NLoS 和室内外状态的变化。

当前生成器使用：

```text
N0 = 当前样本接收信号功率 / 10^(SNR/10)
```

若直接开启路径损耗，N0 会随接收功率一起降低，距离造成的 SNR 差异会被抵消。因此第二阶段必须先新增绝对功率噪声模型，不能与第一阶段混跑。

### 2.3 暂不纳入

- MIMO、多用户和多流；
- H 去噪器、复数 Attention 或新网络结构；
- waveform 级 ICI/ISI；
- 真实 DMRS 在线学习；
- UMi/UMa 到达角带来的多天线增益。

## 3. 信道分布定义

## 3.1 TDL 混合训练分布

第一版采用有限、可复现的 TDL profile bank，避免每 batch 临时构造任意信道对象：

```text
TDL model：A、B、C、D、E
delay spread：10、30、100、300 ns
max Doppler：200 Hz
```

共 20 个 profile。训练时按 batch 级别均衡采样，每个 epoch 中各 profile 的 batch 数最多相差 1。

选择 batch 级而不是逐样本混合的原因：

- Sionna 的 channel model/OFDMChannel 是对象级配置；
- UMi/UMa 的 topology batch size 必须与实际 batch 一致；
- batch 级切换更容易保证确定性和统计记录；
- 避免把多个不同 backend 的输出手工拼接成一个 batch。

正式评估时不使用混合结果掩盖差异，而是分别输出 20 个 TDL profile 的结果，并额外汇总：

- TDL-A–E 总体；
- NLOS：A/B/C；
- LOS：D/E；
- delay spread：10/30/100/300 ns。

## 3.2 UMi/UMa 分布

当前安装的 Sionna 2.0.1 使用：

```python
PanelArray(...)
UMi(...) / UMa(...)
gen_single_sector_topology(...)
channel_model.set_topology(...)
OFDMChannel(channel_model=channel_model, resource_grid=...)
```

主实验配置：

```text
direction：uplink
o2i_model：low
num_ut：1
UT array：1×1, single, V, omni
BS array：1×1, single, V, omni
indoor_probability：0.0
min_ut_velocity：0 m/s
max_ut_velocity：约 17.13 m/s
pathloss：关闭
shadow fading：关闭
normalize_channel：开启
```

17.13 m/s 对应 3.5 GHz 下约 200 Hz 最大 Doppler，与当前 TDL 设置对齐。

每个 batch 重新生成 topology。训练和评估分别使用独立的 topology seed；同一评估 seed 在不同模型间必须复用相同 topology、bits、噪声和公共相位。

定义三个训练 profile：

- `umi_normalized`：只训练 UMi；
- `uma_normalized`：只训练 UMa；
- `umi_uma_mix_normalized`：以 1:1 的 batch 比例训练 UMi 和 UMa。

## 4. 正式实验矩阵

## 4.1 主矩阵：零样本泛化

| 训练分布 | 测试 TDL-A–E | 测试 UMi | 测试 UMa | 用途 |
|---|---:|---:|---:|---|
| TDL mix | 域内 | 跨域 | 跨域 | TDL → UMi/UMa 主实验 |
| UMi only | 跨域 | 域内 | 跨城市模型 | specialist 对照 |
| UMa only | 跨域 | 跨城市模型 | 域内 | specialist 对照 |
| UMi+UMa mix | 跨域 | 域内组成 | 域内组成 | UMi/UMa → TDL 主实验 |

每个训练分布都训练 A/C 两个模型。

正式设置：

```text
训练 seed：0、1、2
评估 seed：777000、888000
模型：A、C
epochs：50
每 epoch 训练样本：10000
验证样本：2000
batch size：64
训练 SNR：-10 到 20 dB
验证 phase mode：uniform
BER 每 SNR 样本：4096
BER SNR：-10,-8,...,20 dB
```

主矩阵正式训练总量：

```text
4 个训练分布 × 2 个模型 × 3 个 seed = 24 个 checkpoint
```

如果计算成本需要分阶段，优先顺序是：

1. `TDL mix` 和 `UMi+UMa mix`，共 12 个 checkpoint；
2. 确认主方向后补 `UMi only`、`UMa only` specialist；
3. 最后补 LDPC BLER。

## 4.2 LDPC BLER

只对通过 BER 筛选的正式 checkpoint 运行：

```text
码率：0.5
K/N：864/1728（默认网格）
decoder iterations：20
Eb/N0：2,2.5,3,3.5,4,4.5,5,5.5,6 dB
每点目标错误块：100
每点最大 block：10000
```

同时保留：

- neural A；
- neural C；
- LS-LMMSE；
- Perfect-CSI LMMSE 上界。

LMMSE 需要在每个测试信道 profile 上独立评估，不与训练分布绑定。

## 4.3 可选的第二轮：少样本适配

零样本矩阵完成前不运行。若存在明显 domain gap，再复用现有 few-shot 逻辑：

```text
预算：0、16、64、256、1024、4096
adapt epochs：5
lr：1e-4
每个预算都从同一 source checkpoint 重启
```

重点比较：

- 相同预算下的绝对 BER；
- target-domain 恢复速度；
- source-domain 遗忘；
- A/C 在 UMi/UMa 与 TDL 之间是否仍有约 4 倍遗忘差异。

这仍然是 oracle-label few-shot，不应表述为真实在线 DMRS 学习。

## 5. 主要指标与判断方式

## 5.1 主指标

1. 每个 SNR 的 pooled BER；
2. 每个 Eb/N0 的 pooled BLER；
3. 跨域相对退化：

```text
generalization gap = cross-domain BER - target specialist BER
```

4. A/C 配对差异：使用相同训练 seed、评估 seed 和测试样本；
5. 等效 SNR/Eb/N0 位移，只在错误数充分的曲线区间插值。

## 5.2 诊断指标

- LS `H_hat` 相对真实 H 的 NMSE；
- UMi/UMa 的 LoS/NLoS 样本比例；
- UT 距离、速度分布；
- 每个训练 epoch 实际抽到的 backend/profile 数量；
- 每个 BLER 点的错误块数和是否达到停止目标；
- 单 batch 生成耗时和显存峰值。

## 5.3 推荐结论标准

- 不使用单 seed 或单 SNR 点下结论；
- A/C 的差异若小于统计波动，应表述为等效；
- 高 SNR/低 BLER 处错误数量不足时，不报告精确 dB 增益；
- TDL→UMi 和 UMi/UMa→TDL 必须分别解释，不能假设方向对称；
- LOS 与 NLOS 分开汇总，避免 D/E 抵消 A/B/C 的趋势。

## 6. 计划修改与新增的文件

## 6.1 新增 `data/sionna_channel_backends.py`

职责：将资源网格之外的信道差异封装成统一 backend。

计划提供：

```text
TDLChannelBackend
SystemLevelChannelBackend（UMi/UMa）
MixedChannelBackend / ChannelProfileSampler
```

统一接口建议：

```text
reset(seed)
prepare_batch(batch_size)
apply(x_rg) -> y_clean_full, h_full
metadata() -> 当前 backend/profile/topology 信息
```

具体修改：

- TDL backend 缓存有限 profile bank；
- UMi/UMa backend 创建 1×1 `PanelArray`；
- 每个 batch 调用 `gen_single_sector_topology()`；
- topology batch size 使用实际 batch size，正确处理最后一个不足完整 batch 的情况；
- `set_topology()` 后调用对应 `OFDMChannel`；
- profile sampler 使用独立随机状态，不能与 bits/noise RNG 相互污染；
- 支持 batch 级均衡采样和 profile 计数。

## 6.2 修改 `data/sionna_ofdm_generator.py`

保留现有资源网格、QPSK、DMRS、AWGN、LS 估计和 batch contract，将当前写死的 TDL 构造移到 backend。

修改后生成器应：

1. 从 channel profile 构造 backend；
2. 在 `generate_batch(actual_batch_size)` 中先准备 topology/profile；
3. 调用 backend 得到 `y_clean_full, h_full`；
4. 继续复用当前 N0、AWGN、LS 和公共相位旋转；
5. 保持模型输入完全不变：`Y/H_hat/P/N0`；
6. 在 batch 中增加不参与模型输入的元数据：

```text
channel_backend
channel_profile_id
tdl_model / delay_spread_ns
scenario
topology_seed
los_state（可获得时）
ut_distance_m（可获得时）
ut_speed_mps
```

`reset(seed)` 必须同时恢复：

- bits/SNR/phase RNG；
- Sionna channel/noise RNG；
- profile sampler RNG；
- topology RNG。

## 6.3 新增 JSON 配置

不引入 YAML 依赖，建议新增：

```text
configs/channel_profiles/tdl_mix_normalized.json
configs/channel_profiles/umi_normalized.json
configs/channel_profiles/uma_normalized.json
configs/channel_profiles/umi_uma_mix_normalized.json
configs/channel_suites/generalization_normalized.json
```

示例字段：

```json
{
  "name": "umi_normalized",
  "components": [
    {
      "backend": "umi",
      "weight": 1.0,
      "direction": "uplink",
      "o2i_model": "low",
      "enable_pathloss": false,
      "enable_shadow_fading": false,
      "normalize_channel": true,
      "indoor_probability": 0.0,
      "min_ut_velocity_mps": 0.0,
      "max_ut_velocity_mps": 17.13
    }
  ],
  "sampling": "balanced_batch"
}
```

测试 suite 应展开为 20 个 TDL 固定 profile、UMi 和 UMa，而不是只输出一个混合平均值。

## 6.4 修改 `training/train_sionna.py`

新增计划接口：

```text
--train_channel_profile PATH
--val_channel_profile PATH
```

修改内容：

- profile 参数优先于旧 `--tdl_model/--delay_spread_s`；
- 保留旧参数兼容已有 checkpoint 和脚本；
- 每个 epoch 重置 profile sampler，并记录实际 profile batch 数；
- validation 使用固定 seed 和固定 profile 顺序；
- checkpoint 新增：

```text
channel_profile_schema_version
train_channel_profile
val_channel_profile
channel_profile_hash
```

- 控制台和训练历史中打印各 backend 的样本数与耗时。

## 6.5 修改 `evaluation/eval_ber_sionna.py`

新增计划接口：

```text
--eval_channel_profile PATH
--eval_channel_suite PATH
```

修改内容：

- 默认仍可使用 checkpoint 的训练配置；
- 指定 profile/suite 时覆盖信道分布，但不改变 OFDM/QPSK/DMRS 语义；
- 每个固定测试 domain 单独生成 CSV；
- CSV 增加：`train_profile/test_profile/backend/scenario/tdl_model/delay_ns`；
- `--common_random_numbers` 对 UMi/UMa 也必须复现 topology；
- A/C 在同一 test profile 下复用相同样本。

## 6.6 修改 `evaluation/eval_bler_sionna.py`

与 BER 评估使用相同 profile/suite 覆盖机制。

确认：

- neural、LS-LMMSE、Perfect-LMMSE 都读取同一个 generator batch；
- UMi/UMa 的 H/H_hat 维度被压成当前 SISO batch contract；
- LLR bit 顺序与现有 5G LDPC decoder 保持一致；
- 结果记录 topology 和停止统计。

## 6.7 新增泛化实验 runner

建议新增：

```text
scripts/run_channel_generalization.sh
experiments/channel_generalization/aggregate_generalization.py
```

runner 负责：

- 训练 A/C、多 seed；
- 自动跳过已有 checkpoint/CSV；
- 对 suite 中所有测试 domain 运行 BER；
- 可选运行 BLER；
- 保存完整运行命令和 Git commit；
- 输出 train-profile × test-profile 矩阵。

聚合器分组键至少包括：

```text
model
train_profile
test_profile
snr_db / ebno_db
```

不要直接修改旧聚合器的输出格式，以免破坏已有 runs 的复现；新实验使用独立聚合器更安全。

## 6.8 修改 `utils/checkpoints.py`

- 兼容旧 `sionna_config` checkpoint；
- 新 checkpoint 同时校验 OFDM 网格配置和 channel profile schema；
- 评估时只允许覆盖信道 profile，不允许静默改变 modulation、DMRS 或网格尺寸。

## 6.9 测试文件

新增：

```text
tests/test_channel_profiles.py
```

扩展：

```text
tests/test_sionna_generator.py
```

测试内容见第 8 节。

## 6.10 文档

更新：

```text
README_SIONNA.md
experiments/channel_generalization/README.md
```

需要明确区分 normalized small-scale experiment 与 realistic coverage experiment。

## 7. 预计实现后的运行命令

以下命令定义计划中的最终 CLI；当前代码尚未实现这些参数，因此现在不要执行。

### 7.1 接口和单元测试

```bash
source ~/venvs/sionna-pi/bin/activate
cd ~/projects/phase_invariant_receiver_sionna

python -m unittest \
  tests.test_channel_profiles \
  tests.test_sionna_generator \
  tests.test_model_factory -v
```

### 7.2 UMi/UMa 最小 smoke

```bash
python -m training.train_sionna \
  --model single_branch_n0_gate \
  --train_channel_profile configs/channel_profiles/umi_normalized.json \
  --val_channel_profile configs/channel_profiles/umi_normalized.json \
  --hidden 64 \
  --hidden_complex 32 \
  --zero_complex 32 \
  --branch_layers 2 \
  --epochs 1 \
  --num_train 128 \
  --num_val 64 \
  --batch_size 16 \
  --snr_db_min -10 \
  --snr_db_max 20 \
  --seed 0 \
  --save_dir runs/smoke_umi_single
```

将 profile 换成 `uma_normalized.json` 再跑一次。smoke 只验证接口，不分析性能。

### 7.3 TDL mix → UMi/UMa 正式实验

```bash
TRAIN_PROFILE=configs/channel_profiles/tdl_mix_normalized.json \
TEST_SUITE=configs/channel_suites/generalization_normalized.json \
MODELS="single_branch_n0_gate strict_matched_complex_p_n0_gate" \
TRAIN_SEEDS="0 1 2" \
EVAL_SEEDS="777000 888000" \
EPOCHS=50 \
NUM_TRAIN=10000 \
NUM_VAL=2000 \
BATCH_SIZE=64 \
NUM_EVAL=4096 \
SNR_LIST="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20" \
RUN_ROOT=runs/generalization_tdl_mix_to_urban \
RUN_BER=1 \
RUN_BLER=0 \
SKIP_TRAINED=1 \
SKIP_EVALUATED=1 \
bash scripts/run_channel_generalization.sh \
  |& tee runs/generalization_tdl_mix_to_urban.log
```

### 7.4 UMi+UMa mix → TDL 正式实验

```bash
TRAIN_PROFILE=configs/channel_profiles/umi_uma_mix_normalized.json \
TEST_SUITE=configs/channel_suites/generalization_normalized.json \
MODELS="single_branch_n0_gate strict_matched_complex_p_n0_gate" \
TRAIN_SEEDS="0 1 2" \
EVAL_SEEDS="777000 888000" \
EPOCHS=50 \
NUM_TRAIN=10000 \
NUM_VAL=2000 \
BATCH_SIZE=64 \
NUM_EVAL=4096 \
SNR_LIST="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20" \
RUN_ROOT=runs/generalization_urban_mix_to_tdl \
RUN_BER=1 \
RUN_BLER=0 \
SKIP_TRAINED=1 \
SKIP_EVALUATED=1 \
bash scripts/run_channel_generalization.sh \
  |& tee runs/generalization_urban_mix_to_tdl.log
```

### 7.5 UMi-only 和 UMa-only specialist

```bash
for PROFILE in umi_normalized uma_normalized; do
  NAME="${PROFILE%_normalized}"
  TRAIN_PROFILE="configs/channel_profiles/${PROFILE}.json" \
  TEST_SUITE=configs/channel_suites/generalization_normalized.json \
  MODELS="single_branch_n0_gate strict_matched_complex_p_n0_gate" \
  TRAIN_SEEDS="0 1 2" \
  EVAL_SEEDS="777000 888000" \
  RUN_ROOT="runs/generalization_${NAME}_specialist" \
  RUN_BER=1 \
  RUN_BLER=0 \
  bash scripts/run_channel_generalization.sh \
    |& tee "runs/generalization_${NAME}_specialist.log"
done
```

### 7.6 正式 BLER

BER 检查通过后，对相同 run root 补跑：

```bash
EBNO_LIST="2,2.5,3,3.5,4,4.5,5,5.5,6" \
TARGET_BLOCK_ERRORS=100 \
MAX_BLOCKS=10000 \
RUN_BER=0 \
RUN_BLER=1 \
INCLUDE_LMMSE=1 \
INCLUDE_LMMSE_PERFECT=1 \
SKIP_TRAINED=1 \
SKIP_EVALUATED=1 \
RUN_ROOT=runs/generalization_tdl_mix_to_urban \
bash scripts/run_channel_generalization.sh \
  |& tee runs/generalization_tdl_mix_to_urban_bler.log
```

`urban_mix_to_tdl` 使用相同方式补跑。

## 8. 测试与验收计划

## 8.1 配置测试

- JSON schema 缺字段时明确失败；
- 权重为负、总权重为零、未知 backend 时失败；
- TDL profile 展开数量必须为 20；
- UMi/UMa 只允许当前 SISO 支持的 1×1 配置；
- profile hash 在相同文件内容下稳定。

## 8.2 生成器单元测试

分别对 TDL、UMi、UMa 检查：

- 输出键、shape、dtype 和 device 与旧生成器一致；
- `Y/H/H_hat/X` 都是 `(B,14,72)` 语义；
- `P/loss_mask/bits` 顺序不变；
- 最后一个 partial batch 可以运行；
- N0 与请求的 measured SNR 匹配；
- DMRS-LS 输出有限，不出现 NaN/Inf；
- UMi/UMa topology batch size 与实际 batch 一致。

## 8.3 确定性与 common-random-numbers

- 相同 profile 和 seed 生成完全相同 batch；
- 重置后可复现 TDL profile 顺序；
- 重置后可复现 UMi/UMa topology；
- 不同模型读取相同 test seed 时，bits/channel/noise/phase 一致；
- 同一 seed 跨 SNR 时只改变噪声尺度，不改变 topology 和无噪声信道。

## 8.4 物理 sanity checks

- `enable_pathloss=False + normalize_channel=True` 时，各 backend 平均信道功率同量级；
- UMi/UMa topology 中 UT 距离、速度和室外比例满足配置；
- UMi 与 UMa 的信道统计不应完全相同；
- TDL-A–E/不同 delay spread 的频率相关性应有可辨别差异；
- Perfect-CSI LMMSE 应优于或接近 LS-LMMSE；
- 无噪声或极高 SNR smoke 下 bit/LLR 顺序正确。

## 8.5 模型性质测试

- A 在 TDL、UMi、UMa batch 上的公共相位不变误差接近数值精度；
- C 的输出允许随公共相位变化；
- A/C 参数量仍严格相同；
- 新 backend 不改变模型 forward 接口；
- 旧 TDL checkpoint 仍可加载和复现旧 BER。

## 8.6 端到端 smoke 验收

每个 backend 至少完成：

```text
1 个 epoch
128 train samples
64 val samples
一次 BER CSV
一次 LDPC/LMMSE 小规模 CSV
```

通过标准：

- loss 可反向传播并下降；
- checkpoint 可重建；
- CSV 元数据完整；
- 聚合器能生成 train×test 矩阵；
- 显存无持续增长。

## 8.7 正式实验前的 seed-0 pilot

先只跑：

```text
2 个主训练 profile
2 个模型
1 个训练 seed
1 个评估 seed
每 SNR 1024 样本
```

只有满足以下条件才扩大到 3×2 seeds：

- UMi/UMa 生成耗时可接受；
- 没有 shape、seed 或 topology 泄漏；
- A/C 使用完全相同的评估样本；
- BER 曲线随 SNR 基本单调；
- 结果文件能区分每个固定测试 domain。

## 9. 实施顺序

### Milestone 1：backend 与 profile 基础设施

1. 新增 JSON profile loader；
2. 新增 TDL/UMi/UMa backend；
3. 保持旧 TDL generator 行为不变；
4. 完成单元测试和确定性测试。

### Milestone 2：训练与 BER

1. 让 `training/train_sionna.py` 接受 train/val profile；
2. checkpoint 保存 profile；
3. BER 评估支持 profile/suite；
4. 跑 seed-0 pilot；
5. 跑两个主方向的正式 BER。

### Milestone 3：specialist 与 BLER

1. 训练 UMi-only、UMa-only specialist；
2. 计算 generalization gap；
3. 对筛选后的 checkpoint 跑 LDPC；
4. 加 LS/Perfect-CSI LMMSE。

### Milestone 4：可选 few-shot

在零样本结果显示明确 domain gap 后，再扩展现有 few-shot 脚本到 profile 接口。

### Milestone 5：真实覆盖

单独设计绝对功率和热噪声模型，不复用 measured-SNR 结论。

## 10. 预期输出目录

```text
runs/generalization_tdl_mix_to_urban/
  checkpoints/
  eval_ber/
  eval_bler/
  manifests/
  ber_generalization_matrix.csv
  bler_generalization_matrix.csv
  profile_statistics.csv

runs/generalization_urban_mix_to_tdl/
  ...
```

每个 manifest 至少记录：

- Git commit 和 dirty 状态；
- 完整 CLI；
- profile JSON 原文和 hash；
- Sionna/PyTorch/CUDA 版本；
- train/eval seeds；
- profile 抽样计数；
- topology 参数；
- 模型参数量；
- 运行耗时。

## 11. 风险与规避

### UMi/UMa 生成速度较慢

先做 seed-0 pilot；profile 和 OFDMChannel 对象只创建一次，batch 间只更新 topology；不逐样本切换 backend。

### 混合训练采样不平衡

使用每 epoch 平衡 batch 列表并记录实际计数，而不是只依赖长期随机期望。

### pathloss 解释错误

第一阶段强制关闭 pathloss/shadow；第二阶段改绝对噪声模型后再开启。

### TDL/UMi/UMa 参数没有完全对齐

统一载频、最大速度、OFDM、天线和归一化；承认 TDL delay spread 与 UMi/UMa 几何统计不能一一对应，用统计诊断而不是宣称完全等价。

### 结果被 LOS/NLOS 构成混淆

输出总体结果之外，按 LOS/NLOS、距离和速度分组；TDL-D/E 单独汇总。

### 旧实验不可复现

保留旧 CLI 和 checkpoint schema 兼容测试；新结果使用独立 run root 和聚合器。

## 12. 建议的最终决策

建议按以下最小闭环开始：

```text
归一化 SISO
TDL-A–E × 4 delay spreads 混合训练
UMi+UMa 1:1 混合训练
A/C 两个模型
3 train seeds × 2 eval seeds
先 BER，后 BLER
```

这一闭环直接回答当前最重要的两个泛化方向，同时保持实验变量可解释。如果结果表明 A 只在 TDL 内部 NLOS 漂移中占优、但跨到 UMi/UMa 后优势消失，这同样是有价值的适用范围结论；如果 A 在两个方向都表现出更小的 generalization gap，则可以进一步进入真实覆盖和少样本适配实验。
