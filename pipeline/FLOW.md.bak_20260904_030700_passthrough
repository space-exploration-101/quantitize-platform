# Pipeline 设计与运行手册

面向接手者：先建立架构心智模型，再按步骤跑通一条 job。  
覆盖 **patches / engine / runner**（M2–M3）。Web 见 `apps/web`，API 见 M4。

---

## 1. 架构一眼看懂

```text
浏览器 → apps/web（展示）
            ↓ HTTP
       apps/api（接单 / worker）          ← M4
            ↓ subprocess
       pipeline/runner（05_runner 编排）  ← M3
            ↓ subprocess
       pipeline/engine（量化 / 抽 bin）   ← M2
            ↑
       patches/（ORT 量化补丁）           ← M2
```

| 目录 | 职责 | GPU |
|------|------|-----|
| `patches/` | 热替换 ORT 的 `quant_utils`、`conv` | 间接 |
| `pipeline/engine/` | pt→量化 onnx→层 bin→合并 | 是 |
| `pipeline/runner/` | 按 job 调引擎 + 评测 + 打包 | 是 |
| `data/` | `shared_data` / `output_data`（数据，不进镜像） | — |
| `apps/web` | 页面 | 否 |
| `apps/api` | HTTP + 启流水线 | 调 runner |

**路径单一事实来源：** `pipeline/script_registry.py`  
（`PLATFORM_ROOT`、`ENGINE_DIR`、`RUNNER_DIR`、`DATA_DIR`、`OUTPUT_DATA_ROOT`、`SHARED_DATA_ROOT`；`QUANTITIZE_DIR` 仅为兼容别名＝产品根。）

子进程环境（`runner/_lib/env.py`）：

- 解释器：强制 conda `yolov8`（可用 `YOLOV8_PYTHON` 覆盖）
- `PYTHONPATH`：`runner/_lib` + `pipeline/` + `engine/`
- 不要把旧 `quantitize/` 放进 `PYTHONPATH`

### Isaac Sim 污染（常见报错）

若 shell 里曾 `source` 过 Isaac Sim，会出现类似：

- `No module named 'numpy.core._multiarray_umath'`
- cv2/numpy 路径落在 `.../isaac-sim/.../pip_prebundle/...`

原因：Isaac 的 **cp311** prebundle 被写进了 `PYTHONPATH`，把 yolov8（cp39）的包盖住了。

**立刻可用：**

```bash
unset PYTHONPATH
# 或
env -u PYTHONPATH $PY pipeline/runner/07_validate_input.py --job-dir "$JOB"
```

Runner 脚本在 `bootstrap_platform_script()` 里也会自动剔除 `isaac-sim` 相关路径；仍建议日常跑 pipeline 时保持 `PYTHONPATH` 干净。

---

## 2. Job 目录约定

默认根：`data/output_data/<task_id>/`（可用环境变量 `OUTPUT_DATA_ROOT`）。

```text
job_config.json          # 任务配置（onnx_name、预处理、nc…）
input/
  model.pt
  cali/                  # 校准图（或指向 shared_data 的链接）
  test/images|labels/    # 测试集
workspace/               # 量化 onnx、bin、all_bin
fpga_test_pack/          # FPGA 侧视测包
results/                 # pt_eval / onnx_eval / fpga_eval
logs/<step>.log          # 全量 runner 写入
manifest.json            # 八步状态
*_quantized_bundle.zip   # 最终交付包
```

路径由 `JobConfig`（`runner/_lib/job_config.py`）拼出，例如：

| 属性 / 方法 | 路径 |
|-------------|------|
| `model_pt` | `input/model.pt` |
| `cali_dir` | `input/cali` |
| `fp16_onnx()` | `workspace/<onnx_name>_fp16.onnx` |
| `quantized_onnx()` | `workspace/<onnx_name>.onnx` |
| `output_onnx()` | `workspace/<onnx_name>_output.onnx` |
| `layer_bin_dir()` | `workspace/bin` |
| `all_bin_dir()` | `workspace/all_bin` |
| `bundle_zip_path()` | `<job_id>_quantized_bundle.zip` |

共享集：`data/shared_data/`（`SHARED_DATA_ROOT` 可覆盖）。  
从旧树灌数时设 `QUANTITIZE_LEGACY_DIR`，再跑 `09_bootstrap_shared_datasets.py`（只读旧目录）。

---

## 3. Patches（量化为何能跑出「项目定制」结果）

系统 ORT 仍来自 `yolov8`；项目只替换：

- `patches/onnxruntime/quantization/quant_utils.py`
- `patches/onnxruntime/quantization/operators/conv.py`

引擎脚本开头会：

1. `_pin_stdlib_platform` — 避免本地包挡住标准库 `platform`
2. `_load_local_onnxruntime_modules` / `_setup_local_onnxruntime` — 注入上述补丁

冒烟：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
/home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_patches.py
```

---

## 4. 完整 Pipeline：八步 + 可选检校

编排入口：`runner_core.run_full_pipeline` ← `05_runner.py`。  
`manifest.json` 中的步骤名如下（顺序固定）。

| # | manifest | 程序 | 层 |
|---|----------|------|----|
| 0 | （可选） | `07_validate_input.py` | runner |
| 1 | `pt_eval` | `01_pt_eval.py` | runner |
| 2 | `quantize` | `engine/quantitize.py` | engine |
| 3 | `onnx_eval` | `02_onnx_eval.py`（`direct`） | runner |
| 4 | `fpga_test_pack` | `04_generate_fpga_test_pack.py` | runner |
| 5 | `fpga_eval` | `02_onnx_eval.py`（`fpga_side_view`） | runner |
| 6 | `generate_bin` | `engine/check_certain_layer_multi.py` | engine |
| 7 | `merge_bin` | `01_rename_wt` → `02_merge_wt` → `03_rename_bn` → `04_merge_bn` | engine |
| 8 | `bundle` | `06_bundle.py` | runner |

```text
validate?
    → pt_eval
    → quantitize.py
    → onnx_eval (direct)
    → fpga_test_pack
    → onnx_eval (fpga_side_view)   # manifest: fpga_eval
    → check_certain_layer_multi
    → rename/merge WT + BN
    → bundle
```

引擎侧特殊层：`special_layer_processors.py`、`L02_*_bin_process.py`。  
辅助：`creat_bin.py`、`png_bin_converter.py`、`grayscale_preprocess.py`。

---

## 5. 如何逐步运行（推荐在独立副本上）

### 5.1 环境变量

在 **`quantitize-platform` 根目录** 执行。下面以已隔离副本为例  
（勿直接改旧 `quantitize/output_data/...`）：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
export PY=/home/rs/miniconda3/envs/yolov8/bin/python
export JOB=data/output_data/20260727_151741_aituosh_moon_0727_2
# 该 job：onnx_name=aituosh_moon_0727_2，preprocess_mode=grayscale_r_channel
```

一键全流程（会覆盖该 job 下 workspace/results/logs/manifest 等）：

```bash
$PY pipeline/runner/05_runner.py --job-dir "$JOB"
```

### 5.2 逐步命令与预期产物

#### 第 0 步 — 输入检校（建议先跑）

```bash
$PY pipeline/runner/07_validate_input.py --job-dir "$JOB"
```

**预期：** exit 0；写出检校报告（如 `input_validation.json`）；`model.pt`、cali/test 数量与标签配对合理。

#### 第 1 步 — `pt_eval`

```bash
$PY pipeline/runner/01_pt_eval.py --job-dir "$JOB"
```

**预期：** `$JOB/results/pt_eval/`；全量 runner 时还有 `$JOB/logs/pt_eval.log`。

#### 第 2 步 — `quantize`（最耗时，需 GPU）

```bash
$PY pipeline/engine/quantitize.py \
  "$JOB/workspace" \
  aituosh_moon_0727_2 \
  "$JOB/input/model.pt" \
  --cali-dir "$JOB/input/cali" \
  --preprocess-mode grayscale_r_channel
```

校准目录解析优先级（不再写死 `engine/cali_data`）：

1. `--cali-dir`
2. 环境变量 `QUANTITIZE_CALI_DIR`
3. 标准 job 布局：`workspace` 同级的 `input/cali`
4. 遗留回退：`pipeline/engine/cali_data`（若存在）

`--preprocess-mode` 会作用于校准 DataReader（`rgb` / `grayscale_uniform` / `grayscale_r_channel`）。
**预期（`workspace/`）：**

| 文件 | 含义 |
|------|------|
| `aituosh_moon_0727_2_fp16.onnx` | FP16 中间模型 |
| `aituosh_moon_0727_2.onnx` | 量化模型（后续评测用） |
| `aituosh_moon_0727_2_output.onnx` | 抽 bin 用 output 模型 |

日志中应出现已加载 `patches/.../quant_utils.py` 与 `conv.py`。

#### 第 3 步 — `onnx_eval`（direct）

```bash
$PY pipeline/runner/02_onnx_eval.py \
  --job-dir "$JOB" \
  --model "$JOB/workspace/aituosh_moon_0727_2.onnx" \
  --input-mode direct \
  --batch-size 8
```

**预期：** `$JOB/results/onnx_eval/`。

#### 第 4 步 — `fpga_test_pack`

```bash
$PY pipeline/runner/04_generate_fpga_test_pack.py --job-dir "$JOB"
```

**预期：** `$JOB/fpga_test_pack/`（由 test 图转换的侧视测包）。

#### 第 5 步 — `fpga_eval`（侧视）

```bash
$PY pipeline/runner/02_onnx_eval.py \
  --job-dir "$JOB" \
  --model "$JOB/workspace/aituosh_moon_0727_2.onnx" \
  --input-mode fpga_side_view \
  --batch-size 8
```

**预期：** `$JOB/results/fpga_eval/`。

#### 第 6 步 — `generate_bin`

```bash
mkdir -p "$JOB/workspace/bin"
$PY pipeline/engine/check_certain_layer_multi.py \
  "$JOB/workspace/aituosh_moon_0727_2_output.onnx" \
  "$JOB/workspace/bin"
```

**预期：** `$JOB/workspace/bin/` 下大量层 `*_wt.bin` / `*_bn.bin`。

#### 第 7 步 — `merge_bin`（四个子程序）

```bash
SRC_BIN="$JOB/workspace/bin"
OUT_ALL="$JOB/workspace/all_bin"
RENAMED="$JOB/workspace/renamed_weights_bin"
mkdir -p "$OUT_ALL" "$RENAMED"

$PY pipeline/engine/01_rename_wt_files.py "$SRC_BIN" "$RENAMED"
$PY pipeline/engine/02_merge_wt_files.py  "$RENAMED" "$OUT_ALL"
# runner 要求存在 ALL_WT.bin（若只有 ALL_wt.bin 会改名）
$PY pipeline/engine/03_rename_bn_files.py "$SRC_BIN" "$RENAMED"
$PY pipeline/engine/04_merge_bn.py        "$RENAMED" "$OUT_ALL"
```

**预期：** `$JOB/workspace/all_bin/ALL_WT.bin`，以及合并后的 BN 产物（如 `ALL_BN.bin`，以脚本实际输出为准）。

#### 第 8 步 — `bundle`

```bash
$PY pipeline/runner/06_bundle.py --job-dir "$JOB"
```

**预期：** `$JOB/20260727_151741_aituosh_moon_0727_2_quantized_bundle.zip`（及可能的 `README_bundle.md`）。

### 5.3 如何判断成功

1. 进程 exit code = 0  
2. 上表预期文件/目录出现或更新  
3. 全量 runner：`manifest.json` 中各 step 为 `"completed"`，总 `"status": "completed"`  
4. 失败先看 `$JOB/logs/<step>.log`（手跑可用 `tee` 自行落盘）

全量重跑预计数小时量级（量化 + 两轮评测最重）。

---

## 6. 冒烟与登记表（不跑完整量化）

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
$PY=/home/rs/miniconda3/envs/yolov8/bin/python

$PY pipeline/verify_patches.py      # 补丁是否从 patches/ 加载
$PY pipeline/script_registry.py     # engine + runner 脚本路径
$PY pipeline/verify_runner.py       # bootstrap / JobConfig / PYTHONPATH
```

---

## 7. runner 目录速查

| 路径 | 作用 |
|------|------|
| `05_runner.py` | `--job-dir` 跑全流程 |
| `07_validate_input.py` | 上传/副本检校 |
| `01_pt_eval.py` / `02_onnx_eval.py` | PT / ONNX 评测 |
| `04_generate_fpga_test_pack.py` | FPGA 测包 |
| `06_bundle.py` | 打 zip |
| `08` / `09` / `10` | 样例 job、灌数据集、归档 |
| `_lib/runner_core.py` | 八步编排实现 |
| `_lib/job_config.py` | 目录与文件名约定 |
| `_lib/env.py` | yolov8 子进程与环境 |
| `_lib/manifest.py` | 步骤状态 |

编号脚本开头统一：

```python
from bootstrap import bootstrap_platform_script
bootstrap_platform_script()
```

---

## 8. 与旧 `quantitize/` 的关系

- 旧目录**只读参考**；产品代码在 `quantitize-platform/`。  
- 旧 `quantitize/platform/*` ≈ 新 `pipeline/runner/*`；旧根目录引擎脚本 ≈ `pipeline/engine/`。  
- 测试请用 `data/output_data/` 下的**独立副本**（解引用 cali/test），避免覆盖旧产物。  
- 子进程 `PYTHONPATH` 已避开旧 `quantitize/`。

---

## 9. 建议阅读顺序

1. 本文 §1 → §2 → §4（架构、目录、步骤表）  
2. §5（对着副本逐步跑）  
3. `script_registry.py`  
4. `runner/_lib/runner_core.py`  
5. `runner/_lib/job_config.py`、`env.py`  
6. `engine/quantitize.py` 开头（补丁加载）
