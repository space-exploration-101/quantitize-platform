# Agent Context — Quantitize Platform

更新时间：2026-09-11（Asia/Shanghai）。本页是结构和最近核验事实；GPU、磁盘、容器与任务状态仍需实时复查。

## 1. 主机与边界

| 项 | 当前事实 |
|---|---|
| 控制端 | Windows，工作区 `E:\codex\quantitize` |
| 生产连接 | `ssh H200` 直连 `ywang@10.2.29.180`；禁止 `wrs` 跳板 |
| H200 主机 | `user-Rack-Server`，Ubuntu 22.04.5，x86_64 |
| 计算资源 | 2× Xeon Gold 6530、1 TiB RAM、4× NVIDIA H200 NVL（每卡约 143771 MiB），MIG 关闭 |
| 项目 | `/data3/ywang/quantitize-platform`，分支 `main` |
| 当前 Git | HEAD `4b857c1`；单通道切换代码尚未提交，工作树非净 |
| 数据盘 | 项目/结果在 `/data3` SATA；任务 scratch 在 `/data1/ywang/quantitize-scratch` NVMe |

共享机限制：不停止其他用户进程，不做全局 Docker/驱动/内核/APT 变更。GPU 设备号不是资源所有权；必须核对 UUID、进程用户和持续负载。

## 2. 当前服务快照

2026-09-11 14:39–14:44 核验：

| 服务 | 镜像 | 状态/入口 |
|---|---|---|
| API | `quantitize-platform-api:latest`，ID `sha256:4693858842e1…` | container healthy、worker idle；`8000` |
| Web | `quantitize-platform-web:latest`，ID `sha256:7aef48747d3f…` | healthy；`8088` |

Compose 使用 `restart: unless-stopped`，API 绑定分配 UUID `GPU-1610284d-3afe-6e14-b8cb-bbbe40a1f28b`（历史上对应物理 GPU 3，容器内通常显示 0）。门禁为至少 32768 MiB 空闲、利用率不超过 20%、30 秒轮询。

重要异常：API `/health` 正常，但 `/ready` 为 HTTP 503；API 容器中的 `nvidia-smi` 与 NVML 均报 `Failed to initialize NVML: Unknown Error`。所以当前创建任务可以成功，但开始后会等待 GPU。不要绕过门禁；先按 Runbook 诊断，任何容器重建都需确认无运行任务并获授权。

## 3. 镜像内运行环境

- API 基础镜像标签：`ywang/yolo-gray1-quant:ultralytics-8.3.98-v6`。
- Python：3.12.12；PyTorch：2.8.0+cu128；ONNX：1.17.0；ONNX Runtime：1.24.3；OpenCV headless：4.11.0.86；NumPy：1.26.4。
- Python：`/opt/ywang/yolo-runtime/bin/python`。
- `PYTHONPATH=/workspace/src/Yolo_fine/ultralytics`；宿主机 `apps/api` 与 `pipeline` 热挂载覆盖镜像代码。
- 当前 Compose 文件定义了 `YOLO_INPUT_CHANNELS=1`、`YOLO_INPUT_SEMANTICS=gray1`，但 2026-09-11 对运行中容器执行 `env` 未看到这两个变量，说明容器配置与工作树存在漂移；重建容器后才会同步（需先获授权并确认无任务）。
- `pip show ultralytics` 观察为 8.4.21 editable，而基础镜像标签写 8.3.98；运行时来源存在版本标识不一致，重建或回归时必须记录实际导入文件与版本。

## 4. 当前输入与输出 ABI

当前新任务固定：

```text
input:  images, FLOAT16, NCHW [1,1,1280,1280]
output: [1, 4 + nc + 6, 34000]
21类:   [1,31,34000]；FPGA 对齐可为 [34000,32] FP16
```

预处理只有两种：

- `passthrough`：灰度/HWC1 保留；三通道取 OpenCV 的 R 通道，不执行 BGR2GRAY。
- `color_to_gray`：彩图执行 BGR2GRAY。

两者都会 resize、除以 255、转 NCHW1。PT 第一层和 ONNX 输入若不是 1 通道会被拒绝。训练和 FPGA 几何的未对齐项见 `../reference/MODEL_INPUT.md`；21 类输出见 `../reference/CLASS21_FPGA.md`。

## 5. 流水线与验证边界

编排入口：`pipeline/runner/05_runner.py`。八步是：

```text
pt_eval -> quantize -> onnx_eval -> fpga_test_pack
-> fpga_input_roundtrip_eval -> generate_bin -> merge_bin -> bundle
```

当前单通道证据（任务 `20260909_191112_test_21_class_gray_input`）：PT F1 0.3418；ONNX 与输入回环 F1 0；`generate_bin` 有 21 层因 `KeyError: input` 失败；失败现场 506 个文件、约 3.4 GB，已校验归档。因此单通道 21 类尚未通过 ONNX 等价、BIN 生成或全流程验收。

历史三通道证据（任务 `20260904_031656_test_21_class_pipeline_03`）：八步完成，PT/ONNX/回环 F1 均 0.9420，84 张测试图；这是切换到 NCHW1 以前的 R-only 三通道版本，只能用于结构和结果对照。

历史二类性能回归（2026-09-02）：两次 CUDA 全流程分别 790.513 s 和 725.451 s，指标和 296 个 BIN 哈希一致。这证明当时的三通道流水线与性能优化，不证明当前单通道代码正确。见 `../CHANGELOG.md` 与 `../reference/PERFORMANCE.md`。

## 6. 存储与备份

- 共享输入：`data/shared_data`；永久任务元数据/结果：`data/output_data`。
- `TASK_SCRATCH_ROOT=/data1/ywang/quantitize-scratch`；成功且 ZIP 校验后清理，失败先归档现场。
- 系统备份排除 `data/output_data`。输出归档是独立任务，未经授权不得删除。
- NAS：`//10.2.26.26/902_data`，临时挂载 `/mnt/ywang-nas`；不是持久挂载。
- 最近记录的已验证系统快照：`snapshot-20260902-182606`，它是单通道切换前快照，不能直接代表当前生产版本。

## 7. 事实路由

操作命令见 `RUNBOOK.md`；流水线见 `../../pipeline/FLOW.md`；部署见 `../../deploy/README.md`；主题参考入口见 `../reference/README.md`；历史变更和测试见 `../CHANGELOG.md`。
