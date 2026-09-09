# H200 量化流水线性能优化最终报告

更新时间：2026-09-03（Asia/Shanghai）  
目标环境：H200 / `10.2.29.180`  
对照环境：wrs / RTX 5090

## 1. 最终结论

本轮优化已经完成并通过两次真实 CUDA 标定全流程复测。相同模型、标定集 `aituosha_moon_500`、测试集 `aituosha_moon_840` 和预处理方式 `grayscale_r_channel` 下：

- 原始 H200 流水线耗时 `6252.678 s`（104 分 12.7 秒）。
- 优化后 CPU 标定耗时 `2070.566 s`（34 分 30.6 秒）。
- CUDA 标定冷启动耗时 `790.513 s`（13 分 10.5 秒）。
- CUDA 标定复测耗时 `725.451 s`（12 分 05.5 秒）。
- 两次 CUDA 平均耗时 `757.982 s`（12 分 38.0 秒）。

相对优化后 CPU 标定，CUDA 将量化阶段平均从 `1476.447 s` 降到 `241.499 s`，平均提速 **6.11 倍**；整条流水线平均提速 **2.73 倍**。相对原始 H200，最终整条流水线平均提速 **8.25 倍**；相对 wrs 历史任务平均快 **4.28 倍**。

两次 CUDA 任务的检测指标均与 CPU 基线一致，296 个交付 `.bin` 的路径、大小和 SHA-256 逐文件完全一致。CUDA 标定可以作为生产默认值，CPU 保留为显式回退选项。

## 2. A/B 任务

| 角色 | task_id | 标定 provider | 步骤合计 |
|---|---|---|---:|
| 原始 H200 | `20260902_065807_test_h200` | CPU | 6252.678 s |
| wrs 历史对照 | `20260727_151741_aituosh_moon_0727_2` | CPU | 3242.333 s |
| 优化后 CPU 对照 | `20260902_153103_perf_opt_h200_cuda_calib` | CPU | 2070.566 s |
| CUDA A/B 冷启动 | `20260902_165033_perf_opt_h200_true_cuda_ab1` | CUDA + CPU fallback | 790.513 s |
| CUDA A/B 复测 | `20260902_170544_perf_opt_h200_true_cuda_ab2` | CUDA + CPU fallback | 725.451 s |

旧任务 `20260902_153103_perf_opt_h200_cuda_calib` 虽然名称包含 `cuda_calib`，但它启动时生产配置已恢复成 CPU，因此只能作为 CPU 对照，不能作为 CUDA 证据。两条真实 CUDA 任务的 `quantize.log` 都明确记录：

```text
✓ 量化校准 providers: CUDAExecutionProvider, CPUExecutionProvider
```

## 3. 各步骤耗时

单位：秒。

| 步骤 | 原始 H200 | 优化后 CPU | CUDA A/B 1 | CUDA A/B 2 | CUDA 两次平均 |
|---|---:|---:|---:|---:|---:|
| PT 评估 | 288.924 | 268.067 | 262.326 | 258.304 | 260.315 |
| 转换与量化 | 1549.221 | 1476.447 | 267.789 | 215.208 | 241.499 |
| ONNX 评估 | 163.550 | 163.391 | 112.510 | 111.592 | 112.051 |
| FPGA 测试集打包 | 2444.461 | 54.312 | 55.814 | 53.789 | 54.802 |
| FPGA 评估 | 66.165 | 72.560 | 55.145 | 50.140 | 52.643 |
| 生成 BIN | 1724.334 | 20.569 | 21.264 | 21.519 | 21.392 |
| 合并 BIN | 2.761 | 2.472 | 2.526 | 2.316 | 2.421 |
| Bundle ZIP | 13.262 | 12.748 | 13.139 | 12.583 | 12.861 |
| **合计** | **6252.678** | **2070.566** | **790.513** | **725.451** | **757.982** |

CUDA A/B 1 从首步骤开始到 scratch 清理完成约 `795.444 s`；A/B 2 约 `730.093 s`。额外约 5 秒用于 ZIP 完整性校验和 NVMe 临时目录清理。

## 4. 正确性与确定性验证

### 4.1 检测指标

PT、ONNX、FPGA 三套评估在 CPU 与两次 CUDA 任务中均保持：

```text
TP=840, FP=1, FN=0
Precision=0.9988109393579072
Recall=1.0
F1=0.9994051160023795
mAP=0.9975124378109452
```

PT 指标 JSON 在 CPU 和两次 CUDA 任务中完全一致。ONNX/FPGA 的核心检测指标完全一致；像素误差均值存在约千分之一像素量级的运行间浮点波动，不改变任何 TP、FP、FN 或业务指标。

### 4.2 BIN 与 ZIP

逐文件读取所有 ZIP 后验证：

- 每个任务均有 296 个 `.bin`。
- BIN 总字节数均为 `200749312`。
- 两次 CUDA、优化后 CPU、原始 H200、wrs 五个任务的 296 个 BIN 路径、大小、SHA-256 全部一致。
- 所有 ZIP 均通过 `ZipFile.testzip()`，没有损坏成员。
- `ALL_WT.bin` 和 `ALL_BN.bin` 继续保持既有黄金哈希。

### 4.3 临时数据生命周期

两次 CUDA 任务均满足：

- 运行期 workspace 和 FPGA 测试包位于 `/data1/ywang/quantitize-scratch/<task_id>`。
- manifest、日志、指标和最终 ZIP 始终保存在 `/data3`。
- 最终 ZIP 完整性验证通过后，manifest 记录 `scratch_cleanup.status=cleaned`。
- 对应 `/data1` 任务目录已不存在，没有遗留历史临时数据。

## 5. 已实施优化

1. 通过 `TASK_SCRATCH_ROOT` 将运行期 workspace 和 FPGA 测试包放到 Memblaze NVMe `/data1`，历史查询仍只读取 `/data3`。
2. 成功任务仅在 ZIP 验证后清理 scratch；失败任务先将完整现场校验归档到 `/data3/failure_artifacts`，再清理 scratch。
3. PNG/BIN 12-bit 转换由逐像素 Python 循环改为 NumPy 向量化。
4. FPGA 测试集打包改为 8 个进程并行。
5. 权重、BN 和成对 int8 写入改为 NumPy 向量化，`generate_bin` 从约 28 分 44 秒降到约 21 秒。
6. PT 评估改为批量提交图片。
7. 为 ORT calibrator 设置 `CUDAExecutionProvider`，并保留 `CPUExecutionProvider` fallback。
8. manifest 增加阶段开始时间和准确的 `duration_seconds`。

## 6. 生产配置与回退

生产默认配置：

```yaml
QUANTIZE_CALIBRATION_PROVIDER: "${QUANTIZE_CALIBRATION_PROVIDER:-cuda}"
```

CUDA 只用于 calibration session 推理；ONNX 图改写、统计汇总及不支持的算子仍可能使用 CPU。需要紧急回退时，以环境变量覆盖为 `cpu` 并只重建 API 容器：

```bash
cd /data3/ywang/quantitize-platform/deploy
QUANTIZE_CALIBRATION_PROVIDER=cpu docker compose up -d --force-recreate --no-deps quantitize-api
```

任何服务重建前必须确认没有运行中的量化任务。

## 7. 后续优化建议

当前两次 CUDA 平均耗时中，PT 评估约占 34.3%，量化约占 31.9%，ONNX 评估约占 14.8%。下一阶段建议：

1. 将 provider、ORT/CUDA 版本和实际 fallback 统计写入每个任务 manifest，避免再依赖任务名或容器当前环境判断历史任务。
2. 为量化阶段加入 GPU 显存峰值和持续利用率采样；H200 是共享设备，不能仅依赖启动前单点 readiness。
3. 优化 PT 图片解码、预处理和数据装载，使批量推理不被 CPU 喂数限制。
4. 评估动态 batch 或固定 batch 8/16 的 ONNX 评估模型；交付 BIN 仍保留静态 batch=1 模型。
5. 如果业务要求像素误差结果逐次完全一致，固定 CUDA/ORT 算法和线程配置；当前 BIN 已严格确定，但评估浮点结果存在极小运行间差异。
6. 为 CUDA provider 初始化增加可观测计时，区分冷启动和稳态耗时。

## 8. 回归与服务验收

- 部署环境自动化测试：14 项全部通过。
- `pipeline/verify_runner.py`：`PASS: runner smoke`。
- Python `compileall`：通过。
- Compose 配置解析：通过。
- API：healthy、idle；Web：healthy。
- API 容器重建后仍显示 `QUANTIZE_CALIBRATION_PROVIDER=cuda`，证明生产默认值持久生效。

## 9. 最终状态

```yaml
task_type: performance_change
status: DONE
cuda_ab_runs: 2
cuda_provider_verified: true
core_metrics_regression: false
delivery_bin_regression: false
scratch_cleanup_verified: true
production_default: cuda
cpu_fallback_available: true
```
