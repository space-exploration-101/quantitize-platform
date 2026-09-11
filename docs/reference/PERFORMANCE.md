# H200 流水线性能基线

本文保存 2026-09-02 至 09-03 的性能优化证据。测试对象是旧 `[1,3,1280,1280]` 二类模型；数字不能直接外推到当前单通道 21 类代码。

## 1. 结论

同一二类模型、`aituosha_moon_500` 标定集、`aituosha_moon_840` 测试集下：

| 任务 | 标定 provider | 总耗时 |
|---|---|---:|
| `20260902_065807_test_h200` | CPU，优化前 | 6252.678 s |
| `20260902_153103_perf_opt_h200_cuda_calib` | 实际 CPU，优化后 | 2070.566 s |
| `20260902_165033_perf_opt_h200_true_cuda_ab1` | CUDA + CPU fallback | 790.513 s |
| `20260902_170544_perf_opt_h200_true_cuda_ab2` | CUDA + CPU fallback | 725.451 s |

CUDA 两次平均 757.982 s；相对优化后 CPU，量化阶段提速 6.11 倍、整条流水线 2.73 倍；相对优化前 H200，端到端 8.25 倍。

## 2. 正确性

两次 CUDA 与 CPU/历史基线的核心指标一致：

```text
TP=840, FP=1, FN=0
Precision=0.9988109393579072
Recall=1.0
F1=0.9994051160023795
mAP=0.9975124378109452
```

每个任务有 296 个交付 BIN，总计 200,749,312 字节；路径、大小和 SHA-256 完全一致。ZIP 完整性通过；scratch 在交付校验后清理。

## 3. 已实施优化

1. 用 `TASK_SCRATCH_ROOT` 把 workspace 和 FPGA 测包放到 `/data1` NVMe，永久元数据/结果留在 `/data3`。
2. 成功任务校验 ZIP 后清 scratch；失败任务校验归档现场后再清。
3. PNG/BIN 12-bit 转换和权重/BN 写入改为 NumPy 向量化。
4. FPGA 测试包使用 8 进程；PT 评测批量提交。
5. ORT calibration 默认 CUDA，并保留 CPU fallback。
6. manifest 记录阶段开始时间和 `duration_seconds`。

## 4. 优化前根因

原始任务约 66.7% 时间在 `fpga_test_pack` 和 `generate_bin`，属于 CPU/图像编解码/小文件和 SATA I/O，不是大规模 GPU 计算。静态 batch=1、少线程 Python、4 NUMA 节点和共享负载也限制 H200 吞吐。未发现 Docker CPU/内存硬限额。

## 5. 后续性能变更门禁

- 使用新的独立任务，不覆盖历史完成任务。
- 固定模型 SHA-256、标定/测试文件清单哈希、Git diff/commit、镜像 ID、runtime 版本、GPU UUID 和参数。
- 同时比较阶段耗时、PT/ONNX/回环指标、ZIP/BIN 清单与哈希。
- 运行期间采集 `nvidia-smi dmon`、进程 CPU/I/O、磁盘延迟和显存峰值；单点 GPU 利用率不能归因。
- 任何当前单通道性能测试必须先通过正确性；F1=0 或 BIN 失败时，耗时没有生产代表性。

原始任务 manifest、日志和结果目录仍保存在 `data/output_data`，它们是详细证据来源。
