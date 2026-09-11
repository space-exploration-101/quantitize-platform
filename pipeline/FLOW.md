# 量化流水线

本文是当前代码的流水线、任务目录和输入输出约定。服务操作见 `../deploy/README.md`，单通道语义见 `../docs/reference/MODEL_INPUT.md`，21 类 FPGA 接口见 `../docs/reference/CLASS21_FPGA.md`。

## 1. 架构

```text
浏览器 -> apps/web -> apps/api -> pipeline/runner/05_runner.py
                                      |
                                      +-> runner：校验、PT/ONNX 评测、测包、打包
                                      +-> engine：ONNX 量化、逐层 BIN、合并
                                      +-> patches：项目定制 ONNX Runtime 量化逻辑
```

路径单一事实来源是 `pipeline/script_registry.py`。容器解释器由 `YOLOV8_PYTHON` 指定，当前为 `/opt/ywang/yolo-runtime/bin/python`；不要使用旧主机 `/home/rs/miniconda3/...` 路径。

## 2. 当前 ABI

新任务由 Web/API 固定为：

```text
imgsz=1280
input_channels=1
input_semantics=gray1
ONNX input: images / FLOAT16 / NCHW [1,1,1280,1280]
pose output: [1,4+nc+6,34000]
```

`passthrough` 和 `color_to_gray` 都输出单通道、resize、`/255`、NCHW。前者保留灰度或取三通道 R；后者执行 BGR2GRAY。旧模式名只作为兼容别名，不能用于推断旧三通道行为。

量化前后 `quantitize.py` 会校验 ONNX 输入必须是 `[1,1,1280,1280]` FP16；PT 评测会校验第一层卷积输入通道为 1。

## 3. Job 目录

永久目录：`data/output_data/<task_id>/`。运行期大文件可放到 `/data1/ywang/quantitize-scratch/<task_id>`，由 `TASK_SCRATCH_ROOT` 控制。

```text
job_config.json                    任务配置和 ABI
manifest.json                      八步状态、耗时、错误、scratch 生命周期
input/model.pt                     模型
input/cali                         标定图或共享集链接
input/test/images|labels           测试集
logs/<step>.log                    阶段日志
results/pt_eval                    PT 指标
results/onnx_eval                  ONNX 指标
results/fpga_input_roundtrip_eval  FPGA 输入回环指标
failure_artifacts                  失败任务的已校验现场
<task_id>_quantized_bundle.zip     成功任务交付包

scratch/workspace/
  <name>_fp16.onnx                 FP16 中间模型
  <name>.onnx                      量化模型
  <name>_output.onnx               逐层导出模型
  bin/                             分层 WT/BN
  all_bin/ALL_WT.bin, ALL_BN.bin   合并产物
scratch/fpga_test_pack/            FPGA 输入侧视测包
```

成功任务只有在 ZIP 完整性验证后才清 scratch；失败任务先复制并校验 `failure_artifacts`。不要手工删除历史任务目录。

## 4. 八步流程

| # | manifest | 程序 | 主要输出 |
|---|---|---|---|
| 1 | `pt_eval` | `runner/01_pt_eval.py` | `results/pt_eval/summary.json` |
| 2 | `quantize` | `engine/quantitize.py` | FP16、量化及 output ONNX |
| 3 | `onnx_eval` | `runner/02_onnx_eval.py --input-mode direct` | ONNX 指标 |
| 4 | `fpga_test_pack` | `runner/04_generate_fpga_test_pack.py` | 12-bit 单通道测包 |
| 5 | `fpga_input_roundtrip_eval` | `02_onnx_eval.py --input-mode fpga_side_view` | 输入回环指标 |
| 6 | `generate_bin` | `engine/check_certain_layer_multi.py` | 各层 WT/BN BIN |
| 7 | `merge_bin` | 4 个 rename/merge engine 脚本 | `ALL_WT.bin`、`ALL_BN.bin` |
| 8 | `bundle` | `runner/06_bundle.py` | ZIP |

输入检校 `runner/07_validate_input.py` 在八步前单独执行。旧材料中的 `fpga_eval` 是步骤 5 的旧名。

## 5. 推荐运行方式

先做实时 GPU 门禁和输入检校，再在新 job 上执行，不能覆盖完成任务：

```bash
cd /data3/ywang/quantitize-platform
PY=/opt/ywang/yolo-runtime/bin/python
JOB=data/output_data/<new_task_id>

$PY pipeline/runner/07_validate_input.py --job-dir "$JOB"
$PY pipeline/runner/05_runner.py --job-dir "$JOB"
```

逐步调试：

```bash
$PY pipeline/runner/01_pt_eval.py --job-dir "$JOB"
$PY pipeline/engine/quantitize.py "$JOB/workspace" <onnx_name> "$JOB/input/model.pt" --cali-dir "$JOB/input/cali" --preprocess-mode passthrough
$PY pipeline/runner/02_onnx_eval.py --job-dir "$JOB" --model "$JOB/workspace/<name>.onnx" --input-mode direct --batch-size 8
$PY pipeline/runner/04_generate_fpga_test_pack.py --job-dir "$JOB"
$PY pipeline/runner/02_onnx_eval.py --job-dir "$JOB" --model "$JOB/workspace/<name>.onnx" --input-mode fpga_side_view --batch-size 8
$PY pipeline/engine/check_certain_layer_multi.py "$JOB/workspace/<name>_output.onnx" "$JOB/workspace/bin"
$PY pipeline/runner/06_bundle.py --job-dir "$JOB"
```

当 scratch 启用时，实际 workspace 路径由 `JobConfig` 决定，不要假设它一定在 job 目录内。优先从 manifest、日志或以下代码获取：`runner/_lib/job_config.py`。

## 6. 成功判定

必须同时满足：八步均 completed；manifest error 为空；PT、ONNX 和输入回环指标在约定容差内；ZIP 完整；BIN 清单/尺寸/哈希符合接口；scratch 已正确清理或失败现场已归档。

当前单通道 21 类版本尚不满足：任务 `20260909_191112_test_21_class_gray_input` 的 ONNX/回环 F1 为 0，且 `generate_bin` 因 21 层 `KeyError: input` 失败。下一次验证必须先解决并解释 PT/ONNX 差异，再继续 BIN；不能引用 2026-09-04 的三通道成功任务作为当前验收。

## 7. 快速检查

```bash
$PY pipeline/verify_patches.py
$PY pipeline/script_registry.py
$PY pipeline/verify_runner.py
```

核心阅读顺序：`runner/_lib/runner_core.py` → `job_config.py` → `preprocess.py` → `engine/quantitize.py` → `engine/check_certain_layer_multi.py`。
