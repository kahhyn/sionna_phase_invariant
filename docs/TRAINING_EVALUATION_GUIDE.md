# 训练与评估使用指南

本文档说明当前项目中 Sionna 2.x 接收机的训练、BER/BLER 评估、跨信道泛化实验和
QuaDRiGa 轨迹评估流程。当前重点是比较：

- 模型 A：`single_branch_n0_gate`
- 模型 C：`strict_matched_complex_p_n0_gate`
- TDL-mix 训练后测试 UMi、UMa 和 QuaDRiGa UMi-NLOS
- UMi/UMa-mix 训练后测试 TDL-A 至 TDL-E 和 QuaDRiGa UMi-NLOS

## 1. 环境与约定

以下命令均在 WSL 项目根目录运行：

```bash
cd ~/projects/phase_invariant_receiver_sionna
source ~/venvs/sionna-pi/bin/activate
```

默认物理层配置如下：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| OFDM symbols | 14 | 每帧 OFDM 符号数 |
| FFT size | 72 | 子载波数 |
| Subcarrier spacing | 30 kHz | 子载波间隔 |
| Modulation | QPSK | 每个数据 RE 2 bit |
| DMRS symbols | 2, 11 | 0-based OFDM 符号索引 |
| Channel estimate | LS + linear interpolation | 神经接收机输入中的信道估计 |
| Carrier frequency | 3.5 GHz | TDL/UMi/UMa 默认载频 |
| Maximum Doppler | 200 Hz | 默认最大多普勒频移 |
| Train phase | fixed | 训练时公共相位模式 |
| Validation/evaluation phase | uniform | 验证和评估时公共相位均匀采样 |

这里的 `*_normalized` profile 关闭 pathloss 和 shadow fading，并对信道进行归一化，
因此主要衡量小尺度信道结构与分布泛化，不代表完整链路预算或覆盖性能。

## 2. 已发布的常用 checkpoint

Git 中保存了 A、C 两个模型在两个训练分布、三个训练 seed 下的 validation-best
checkpoint：

```text
checkpoints/generalization/
├── tdl_mix_normalized/
│   ├── single_branch_n0_gate_seed{0,1,2}.pt
│   └── strict_matched_complex_p_n0_gate_seed{0,1,2}.pt
└── umi_uma_mix_normalized/
    ├── single_branch_n0_gate_seed{0,1,2}.pt
    └── strict_matched_complex_p_n0_gate_seed{0,1,2}.pt
```

这些 checkpoint 均使用：50 epochs、10,000 个训练样本、2,000 个验证样本、batch
size 64、训练 SNR -10 至 20 dB。完整校验值见
[`../checkpoints/generalization/SHA256SUMS`](../checkpoints/generalization/SHA256SUMS)。

## 3. 推荐：一键训练和跨域评估

`scripts/run_channel_generalization.sh` 是当前推荐入口。它依次完成：

1. 对每个模型和训练 seed 训练 checkpoint；
2. 对测试 suite 中的每个信道域和评估 seed 计算 BER，可选计算 BLER；
3. 汇总为跨域矩阵和 pooled 结果。

### 3.1 冒烟测试

先用极小规模验证环境、profile 和输出路径：

```bash
TRAIN_PROFILE=configs/channel_profiles/tdl_mix_normalized.json \
TEST_SUITE=configs/channel_suites/generalization_normalized.json \
MODELS="single_branch_n0_gate strict_matched_complex_p_n0_gate" \
TRAIN_SEEDS="0" \
EVAL_SEEDS="777000" \
EPOCHS=1 \
NUM_TRAIN=128 \
NUM_VAL=64 \
BATCH_SIZE=32 \
NUM_EVAL=64 \
SNR_LIST="0,10" \
RUN_ROOT=runs/smoke_tdl_mix \
RUN_BER=1 \
RUN_BLER=0 \
SKIP_TRAINED=0 \
SKIP_EVALUATED=0 \
DEVICE=cuda \
bash scripts/run_channel_generalization.sh
```

### 3.2 TDL-mix 训练，测试 TDL/UMi/UMa

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
NUM_EVAL=1024 \
SNR_LIST="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20" \
RUN_ROOT=runs/pilot_tdl_mix_to_urban \
RUN_BER=1 \
RUN_BLER=0 \
SKIP_TRAINED=1 \
SKIP_EVALUATED=1 \
DEVICE=cuda \
bash scripts/run_channel_generalization.sh \
|& tee runs/pilot_tdl_mix_to_urban.log
```

### 3.3 UMi/UMa-mix 训练，测试 TDL/UMi/UMa

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
NUM_EVAL=1024 \
SNR_LIST="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20" \
RUN_ROOT=runs/pilot_urban_mix_to_tdl \
RUN_BER=1 \
RUN_BLER=0 \
SKIP_TRAINED=1 \
SKIP_EVALUATED=1 \
DEVICE=cuda \
bash scripts/run_channel_generalization.sh \
|& tee runs/pilot_urban_mix_to_tdl.log
```

### 3.4 一键脚本参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `TRAIN_PROFILE` | `tdl_mix_normalized.json` | 训练 profile JSON |
| `TEST_SUITE` | `generalization_normalized.json` | 评估域集合 JSON |
| `MODELS` | A 和 C | 空格分隔的模型名 |
| `TRAIN_SEEDS` | `0` | 空格分隔的训练 seed |
| `EVAL_SEEDS` | `777000` | 空格分隔的评估 seed |
| `EPOCHS` | `50` | 训练 epoch 数 |
| `NUM_TRAIN` | `10000` | 每个 epoch 生成的训练样本数 |
| `NUM_VAL` | `2000` | 每个 epoch 的验证样本数 |
| `BATCH_SIZE` | `64` | 训练 batch size |
| `NUM_EVAL` | `4096` | 每个 BER SNR 点的样本数 |
| `SNR_LIST` | `-10,-8,...,20` | BER SNR 点，逗号分隔 |
| `EBNO_LIST` | `2,2.5,...,6` | BLER Eb/N0 点，逗号分隔 |
| `TARGET_BLOCK_ERRORS` | `100` | 每个 BLER 点停止前目标错误块数 |
| `MAX_BLOCKS` | `10000` | 每个 BLER 点最大传输块数 |
| `RUN_ROOT` | `runs/channel_generalization` | 本次实验输出根目录 |
| `RUN_BER` | `1` | 是否运行 BER，1/0 |
| `RUN_BLER` | `0` | 是否运行 BLER，1/0 |
| `INCLUDE_LMMSE` | `0` | BER 是否加入 LS-LMMSE 基线 |
| `INCLUDE_LMMSE_PERFECT` | `0` | BER 是否加入 perfect-CSI LMMSE |
| `SKIP_TRAINED` | `1` | checkpoint 已存在时跳过训练 |
| `SKIP_EVALUATED` | `1` | 结果已存在时跳过评估 |
| `DEVICE` | `cuda` | `cuda` 或 `cpu` |

若需要从头重跑，应使用新的 `RUN_ROOT`，或显式设置 `SKIP_TRAINED=0` 和
`SKIP_EVALUATED=0`。正常续跑则保持二者为 1。

## 4. 直接调用训练脚本

需要单独控制某个模型时使用 `training.train_sionna`：

```bash
python -m training.train_sionna \
  --model single_branch_n0_gate \
  --train_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --val_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --train_phase_mode fixed \
  --val_phase_mode uniform \
  --num_train 10000 \
  --num_val 2000 \
  --epochs 50 \
  --batch_size 64 \
  --snr_db_min -10 \
  --snr_db_max 20 \
  --hidden 64 \
  --hidden_complex 32 \
  --zero_complex 32 \
  --branch_layers 2 \
  --lr 1e-3 \
  --seed 0 \
  --device cuda \
  --save_dir runs/manual_tdl_mix_A_seed0
```

关键训练参数：

| 参数 | 说明 |
|---|---|
| `--model` | 模型注册名；当前主比较使用 A 或 C |
| `--train_channel_profile` | 训练信道 profile；指定后覆盖单一 TDL 参数 |
| `--val_channel_profile` | 验证信道 profile |
| `--train_phase_mode` / `--val_phase_mode` | `fixed`、`uniform` 或 `narrow` |
| `--num_train` / `--num_val` | 每个 epoch 在线生成的样本量 |
| `--epochs` / `--batch_size` | epoch 数与 batch size |
| `--snr_db_min` / `--snr_db_max` | 每个样本随机训练 SNR 的范围 |
| `--num_ofdm_symbols` / `--fft_size` | OFDM 网格尺寸，默认 14/72 |
| `--subcarrier_spacing_hz` | 子载波间隔，默认 30000 Hz |
| `--dmrs_symbols` | DMRS OFDM 符号索引，默认 `2 11` |
| `--tdl_model` / `--delay_spread_s` | 未使用 profile 时的单一 TDL 配置 |
| `--carrier_frequency_hz` | 载频，默认 3.5e9 Hz |
| `--max_doppler_hz` | 最大多普勒，默认 200 Hz |
| `--hidden` | 实值主干宽度，正式实验为 64 |
| `--hidden_complex` / `--zero_complex` | 复值分支宽度，正式实验均为 32 |
| `--branch_layers` | 分支层数，正式实验为 2 |
| `--lr` / `--weight_decay` | Adam 学习率和权重衰减 |
| `--seed` | 训练数据、初始化等随机种子 |
| `--save_dir` | 保存 `best.pt`、日志和配置的目录 |
| `--device` | `cuda` 或 `cpu` |

完整参数以代码版本自带帮助为准：

```bash
python -m training.train_sionna --help
```

## 5. BER 评估

下面示例用已发布的 TDL-mix 模型 A seed 0 测试 normalized UMi：

```bash
python -m evaluation.eval_ber_sionna \
  --checkpoint checkpoints/generalization/tdl_mix_normalized/single_branch_n0_gate_seed0.pt \
  --eval_channel_profile configs/channel_profiles/umi_normalized.json \
  --eval_component_id umi_normalized \
  --phase_mode uniform \
  --snr_list=-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20 \
  --num_samples 4096 \
  --batch_size 128 \
  --seed 777000 \
  --common_random_numbers \
  --device cuda \
  --out_csv runs/eval_tdl_mix_A_seed0_umi.csv
```

| 参数 | 说明 |
|---|---|
| `--checkpoint` | 待评估的 `.pt` 文件 |
| `--eval_channel_profile` | 测试信道 profile JSON |
| `--eval_component_id` | 固定测试 profile 中某个 component；按 suite 调用时使用 |
| `--phase_mode` | 测试公共相位模式，泛化实验使用 `uniform` |
| `--snr_list` | 逗号分隔的 SNR；负数列表建议使用 `--snr_list=...` 形式 |
| `--num_samples` | 每个 SNR 点的总样本数 |
| `--batch_size` | 推理 batch size |
| `--seed` | 评估随机 seed |
| `--common_random_numbers` | 在各 SNR 点复用相同 bit、信道、单位方差噪声和相位，仅改变噪声缩放 |
| `--out_csv` | 输出逐 SNR BER/BCE CSV |

```bash
python -m evaluation.eval_ber_sionna --help
```

## 6. BLER 评估

BLER 使用编码链路，不能由未编码 BER 直接换算。示例：

```bash
python -m evaluation.eval_bler_sionna \
  --checkpoint checkpoints/generalization/tdl_mix_normalized/single_branch_n0_gate_seed0.pt \
  --receiver neural \
  --eval_channel_profile configs/channel_profiles/umi_normalized.json \
  --eval_component_id umi_normalized \
  --phase_mode uniform \
  --ebno_list=2,2.5,3,3.5,4,4.5,5,5.5,6 \
  --coderate 0.5 \
  --decoder_iterations 20 \
  --cn_update boxplus-phi \
  --batch_size 64 \
  --target_block_errors 100 \
  --max_blocks 10000 \
  --seed 777000 \
  --device cuda \
  --out_csv runs/bler_tdl_mix_A_seed0_umi.csv
```

| 参数 | 说明 |
|---|---|
| `--receiver` | `neural` 或脚本支持的传统接收机类型 |
| `--ebno_list` | 编码链路 Eb/N0 点 |
| `--coderate` | 信道编码码率，默认 0.5 |
| `--decoder_iterations` | LDPC 译码迭代次数 |
| `--cn_update` | 校验节点更新规则 |
| `--target_block_errors` | 达到该错误块数后停止当前 Eb/N0 点 |
| `--max_blocks` | 当前 Eb/N0 点最多评估的块数 |
| `--batch_size` | 每批传输块数 |
| `--seed` | 评估随机 seed |
| `--out_csv` | 输出 BLER CSV |

```bash
python -m evaluation.eval_bler_sionna --help
```

## 7. 汇总多 seed 和跨域矩阵

一键脚本会自动调用聚合器。手动聚合已有运行目录时：

```bash
python -m experiments.channel_generalization.aggregate_generalization \
  --input_root runs/pilot_tdl_mix_to_urban/eval_ber \
  --metric ber \
  --out_csv runs/pilot_tdl_mix_to_urban/ber_generalization_matrix.csv
```

聚合结果用于比较 A/C 的平均 BER、BCE、不同训练 seed 的波动，以及每个测试域的
性能。正式结论至少使用 3 个训练 seed；若计算资源允许，每个 checkpoint 使用 2 个
或更多评估 seed，并报告均值、离散程度和配对差值。

```bash
python -m experiments.channel_generalization.aggregate_generalization --help
```

## 8. QuaDRiGa 轨迹评估

`eval_quadriga_trajectory.py` 读取 MATLAB 生成的连续信道轨迹，并在同一条轨迹上比较
A/C checkpoint。示例使用 TDL-mix 的 A/C seed 0：

```bash
python eval_quadriga_trajectory.py \
  --channel_mat /path/to/quadriga_umi_nlos_seed0.mat \
  --invariant_checkpoint checkpoints/generalization/tdl_mix_normalized/single_branch_n0_gate_seed0.pt \
  --strict_checkpoint checkpoints/generalization/tdl_mix_normalized/strict_matched_complex_p_n0_gate_seed0.pt \
  --snr_list=-5,0,5,10,15,20 \
  --batch_size 128 \
  --window_frames 64 \
  --seed 777000 \
  --normalization checkpoint \
  --device cuda \
  --output_dir runs/quadriga_tdl_mix_seed0_traj0
```

MAT 文件至少需要：

- `H_real`、`H_imag`：形状 `[num_frames, 14, 72]`
- `positions_m`：形状 `[num_frames, 3]`
- `timestamps_s`：形状 `[num_frames]`
- 可选 `frame_index`

评估器会检查 checkpoint A/C 的 OFDM 网格、QPSK 配置与 MAT 数据是否一致。主要参数：

| 参数 | 说明 |
|---|---|
| `--channel_mat` | QuaDRiGa 导出的 MAT 文件 |
| `--invariant_checkpoint` / `--strict_checkpoint` | 配对比较的 A/C checkpoint |
| `--snr_list` | 逗号分隔的测试 SNR |
| `--batch_size` | 推理 batch size |
| `--window_frames` | 轨迹窗口长度，用于局部统计 |
| `--max_frames` | 最大读取帧数；0 表示全部 |
| `--seed` | 比特、噪声和相位随机 seed |
| `--independent_snr_randomness` | 不同 SNR 使用独立比特/噪声；默认在 SNR 间复用基础随机量 |
| `--phase_mode` | `fixed`、`uniform` 或 `narrow` |
| `--narrow_phase_range` | `narrow` 模式的相位范围，默认 pi/8 |
| `--normalization` | 使用 checkpoint 约定或脚本支持的归一化模式 |
| `--output_dir` | 结果目录 |

每次运行输出：

- `experiment_config.json`：输入、模型和随机设置
- `quadriga_umi_summary.csv`：各 SNR 的整体 BCE/BER
- `quadriga_umi_trajectory_windows.csv`：沿轨迹窗口的局部统计

对 5 条独立轨迹和 3 个训练 seed，应分别运行并保留每个组合的结果，再对“轨迹”和
“训练 seed”两个层次做统计，不要先拼接成一条轨迹后只报告单一总体 BER。

```bash
python eval_quadriga_trajectory.py --help
```

## 9. QuaDRiGa 双链路同频干扰

干扰轨迹的 MATLAB 脚本和生成数据统一放在 Windows：

```text
E:\quadriga_data
├── generate_quadriga_interference_trajectory.m
├── generate_quadriga_interference_suite.m
└── quadriga_umi_nlos_interference_suite\*.mat
```

在 MATLAB 中生成单条轨迹：

```matlab
addpath('E:\quadriga_data');
generate_quadriga_interference_trajectory( ...
    quadriga_path, ...
    'E:\quadriga_data\quadriga_umi_nlos_interference_10m_seed0.mat', ...
    0);
```

生成 seed 0 至 4 的五条轨迹：

```matlab
addpath('E:\quadriga_data');
generate_quadriga_interference_suite( ...
    quadriga_path, ...
    'E:\quadriga_data\quadriga_umi_nlos_interference_suite');
```

每个 MAT 文件包含时间对齐的两个 reciprocal SISO 链路：

```text
H_desired_real / H_desired_imag       [frame, 14, 72]
H_interferer_real / H_interferer_imag [frame, 14, 72]
desired_positions_m                    [frame, 3]
interferer_positions_m                 [frame, 3]
timestamps_s / frame_index             [frame, 1]
```

手动复制到 WSL 后，建议目录为：

```bash
mkdir -p data/quadriga_umi_nlos_interference_suite
# 将 Windows MAT 文件复制到上面的 WSL 目录。
```

单个组合的最小测试：

```bash
python -m evaluation.eval_quadriga_interference \
  --channel_mat data/quadriga_umi_nlos_interference_suite/quadriga_umi_nlos_interference_10m_seed0.mat \
  --invariant_checkpoint checkpoints/generalization/tdl_mix_normalized/single_branch_n0_gate_seed0.pt \
  --strict_checkpoint checkpoints/generalization/tdl_mix_normalized/strict_matched_complex_p_n0_gate_seed0.pt \
  --snr_list=10 \
  --sir_list=inf,20,10,5,0 \
  --interference_mode cochannel_full \
  --sir_normalization per_frame \
  --n0_mode thermal \
  --normalization checkpoint \
  --seed 777000 \
  --device cuda \
  --output_dir runs/quadriga_interference_smoke
```

干扰定义为：

```text
Y = H_desired X_desired
  + alpha H_interferer X_interferer
  + N
```

`SNR` 始终用期望信号功率与热噪声定义；`SIR` 用期望信号功率与缩放后的干扰功率定义。
默认 `N0=thermal` 只向模型报告热噪声，不把未知干扰伪装成 AWGN。

| 参数 | 说明 |
|---|---|
| `--sir_list` | `inf` 表示无干扰，其余为接收端目标 SIR |
| `--interference_mode cochannel_full` | 干扰占用完整网格并污染目标 DMRS |
| `--interference_mode data_only` | 干扰仅占目标数据 RE，DMRS 保持干净 |
| `--interference_mode partial_band` | 干扰占中心连续部分子载波 |
| `--partial_band_fraction` | partial-band 占总子载波的比例，默认 0.25 |
| `--sir_normalization per_frame` | 每帧精确固定 SIR，隔离干扰结构效应 |
| `--sir_normalization global` | 整条轨迹固定一次发射缩放，保留局部 SIR 变化 |
| `--n0_mode thermal` | 模型只知道热噪声功率，推荐的零样本设置 |
| `--n0_mode oracle_total` | 将干扰功率加进 N0，作为 oracle 消融 |
| `--independent_condition_randomness` | 每个 SNR/SIR 使用不同随机量；默认复用基础随机量 |

干扰符号使用独立随机流，因此 `SIR=inf` 不会额外消耗目标比特、热噪声或公共相位的
随机序列，可作为无干扰基线；同一条件中的 A/C 始终共享完整接收样本。

正式批量运行五条轨迹、两个训练分布、三个训练 seed：

```bash
CHANNEL_DIR=data/quadriga_umi_nlos_interference_suite \
TRAIN_PROFILES="tdl_mix_normalized umi_uma_mix_normalized" \
TRAIN_SEEDS="0 1 2" \
INTERFERENCE_MODES="cochannel_full data_only" \
SNR_LIST="10" \
SIR_LIST="inf,20,10,5,0" \
SIR_NORMALIZATION=per_frame \
N0_MODE=thermal \
RUN_ROOT=runs/quadriga_interference \
DEVICE=cuda \
bash scripts/run_quadriga_interference.sh \
|& tee runs/quadriga_interference.log
```

批量脚本结束后自动生成：

- `quadriga_interference_aggregate.csv`：每个 receiver 的 pooled/mean BER 和 BCE；
- `quadriga_interference_paired_ac.csv`：相同轨迹、训练 seed 和随机量下的 A−C 配对差；
- 每次运行的 summary、trajectory windows 和 `experiment_config.json`。

第一轮推荐固定 `SNR=10 dB` 扫描 SIR。第二轮再固定 `SIR=10 dB` 扫描
`SNR=0,5,10,15,20 dB`，避免一开始运行完整二维矩阵。

## 10. DICHASUS-0152 真实测量信道评估

DICHASUS-0152 是室内 LoS 测量数据，每条记录包含 32 根接收天线、1024 个子载波的
复 CSI，以及位置、时间、每根天线的测量 SNR 和 CFO。原始数据并不包含发送比特和
原始接收波形，因此本实验将测量 CSI 作为真实信道 `H`，再统一生成 QPSK、DMRS、AWGN
和 `H_hat`。结果应表述为“真实测量信道上的半合成零样本 BER”，而不是完全 OTA BER。

Windows 数据和转换脚本位于：

```text
E:\invariant_data\DICHASUS_data
├── dichasus-0152.tfrecords
├── convert_dichasus_tfrecord.py
└── dichasus-0152_72sc_30khz.h5
```

在 Windows PowerShell 中执行完整转换：

```powershell
cd E:\invariant_data\DICHASUS_data
C:\ProgramData\miniconda3\python.exe .\convert_dichasus_tfrecord.py
```

转换器按官方约定对 1024 子载波执行 `fftshift`，然后将中心频段复插值到当前模型的
72×30 kHz 网格。输出保留所有 13,496 条记录和 32 根天线，但不预先展开 14 个 OFDM
symbol。TFRecord 的物理记录顺序不是时间顺序，评估器会按 timestamp 稳定排序。

将 HDF5 手动复制到 WSL：

```bash
mkdir -p data/dichasus
# 目标文件：data/dichasus/dichasus-0152_72sc_30khz.h5
```

单天线最小评估：

```bash
python -m evaluation.eval_dichasus \
  --channel_h5 data/dichasus/dichasus-0152_72sc_30khz.h5 \
  --antenna_index 28 \
  --invariant_checkpoint checkpoints/generalization/tdl_mix_normalized/single_branch_n0_gate_seed0.pt \
  --strict_checkpoint checkpoints/generalization/tdl_mix_normalized/strict_matched_complex_p_n0_gate_seed0.pt \
  --snr_list=0,5,10 \
  --record_stride 16 \
  --max_records 128 \
  --phase_mode fixed \
  --normalization checkpoint \
  --device cuda \
  --output_dir runs/dichasus_smoke_single
```

每条测量 CSI 在一个 frame 的 14 个 OFDM symbol 内保持不变。不能用连续 14 条
DICHASUS 记录构造 14 个 symbol，因为测量记录间隔约 64 ms，而当前 OFDM frame 仅约
0.467 ms。

常用参数：

| 参数 | 说明 |
|---|---|
| `--antenna_index` | DICHASUS 原始天线编号 0–31 |
| `--record_stride` | 按时间排序后每隔多少条取一条 |
| `--max_records` | stride/filter 后最多使用多少条；0 表示全部 |
| `--min_measured_snr_db` | 按测量 CSI 的 SNR 过滤；不等于合成评估 SNR |
| `--snr_list` | 后续合成 QPSK 链路的 SNR |
| `--phase_mode` | `fixed` 隔离信道失配，`uniform` 测试相位失配叠加 |
| `--normalization checkpoint` | 沿用训练时的逐 frame 信道功率归一化设置 |

推荐的第一轮批量实验覆盖全部 32 根天线，但在时间轴上 stride 16，以控制运行量：

```bash
CHANNEL_H5=data/dichasus/dichasus-0152_72sc_30khz.h5 \
TRAIN_PROFILES="tdl_mix_normalized umi_uma_mix_normalized" \
TRAIN_SEEDS="0 1 2" \
ANTENNA_INDICES=all \
SNR_LIST="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20" \
RECORD_STRIDE=16 \
MAX_RECORDS=0 \
MIN_MEASURED_SNR_DB=-inf \
PHASE_MODE=fixed \
RUN_ROOT=runs/dichasus_fixed \
DEVICE=cuda \
bash scripts/run_dichasus.sh \
|& tee runs/dichasus_fixed.log
```

批量结束后生成：

- `dichasus_aggregate.csv`：跨天线和训练 seed 的 pooled/mean BER、BCE；
- `dichasus_paired_ac.csv`：相同天线、训练 seed 和随机样本下的 A−C 配对差；
- 各天线的 summary、时间窗口结果和实验 manifest。

固定相位结果完成后，换一个 `RUN_ROOT` 并设置 `PHASE_MODE=uniform`，测试真实信道失配
与公共相位失配同时存在时的效果。

## 11. 复现实验注意事项

1. A/C 对比应使用相同评估 seed，并保持 profile、样本数和 SNR 完全相同。BER 脚本的
   `--common_random_numbers` 还会固定各 SNR 点的基础随机量；QuaDRiGa 评估器内部对
   A/C 使用相同样本。
2. 至少比较 3 个训练 seed，避免将某次初始化差异解释成结构性优势。
3. `SKIP_TRAINED=1` 和 `SKIP_EVALUATED=1` 适合断点续跑；修改配置后应换
   `RUN_ROOT`，防止错误复用旧结果。
4. 命令行中的负 SNR 列表使用 `--snr_list=-10,...`，避免被参数解析器误认为新选项。
5. `runs/`、CSV 和普通 `.pt` 默认不进入 Git；只有经过选择的常用 checkpoint 放在
   `checkpoints/generalization/`。
6. QuaDRiGa 的 per-frame 归一化会移除大尺度功率变化。若要研究真实链路预算，应另设
   保留绝对路径增益、发射功率和接收机噪声标定的实验，不能与 normalized 结果混为一谈。
