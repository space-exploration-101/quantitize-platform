# H200 量化流水线性能分析与优化建议

更新日期：2026-09-02（Asia/Shanghai）  
目标主机：`H200` / `10.2.29.180`  
对照主机：`wrs` / RTX 5090  

最终实施和 CUDA A/B 验收结果见 [`PERFORMANCE_OPTIMIZATION_FINAL_20260903.md`](PERFORMANCE_OPTIMIZATION_FINAL_20260903.md)。

## 1. 结论摘要

同一模型和同一数据集配置下，H200 任务 `20260902_065807_test_h200` 的端到端耗时约为 **1 小时 44 分 13 秒**；wrs 历史任务 `20260727_151741_aituosh_moon_0727_2` 约为 **54 分 02 秒**。本次 H200 运行约为 wrs 的 **1.93 倍耗时**。

当前证据不支持“H200 GPU 故障”或“Docker CPU 限额”作为首要原因。更符合证据的解释是：

1. 流水线约三分之二时间花在 CPU、图像编解码和磁盘 I/O 密集阶段，GPU 型号对这些阶段几乎没有帮助。
2. ONNX 模型输入固定为 `[1, 3, 1280, 1280]`，实际最大 batch 为 1；命令行的 `batch=8` 会被拆成逐张推理，无法发挥 H200 的吞吐优势。
3. H200 当前工作目录位于 SATA SSD `/data3`，wrs 位于 NVMe；任务会产生大量大图片、BIN 和小文件，存储差异会被明显放大。
4. H200 的双路 Xeon Gold 6530 核数更多，但此流水线大量使用串行 Python、OpenCV、NumPy 或固定 4/8 线程。此类负载更依赖单核性能，i9-14900KF 并不弱，且通常具有更高的单核频率。
5. H200 是 4 NUMA 节点的共享服务器，容器未绑定 GPU0 对应的 NUMA 0；存在调度、跨 NUMA 内存访问和共享资源争用风险，但目前没有运行期间的监控数据可以定量归因。

最优先的优化方向不是更换 GPU，而是：**把临时工作集放到 NVMe、真正启用大 batch、并行化 FPGA 测试包生成、减少 generate_bin 的重复图处理与小文件写入，并建立运行期监控。**

## 2. 存储现状

### 2.1 Memblaze NVMe

设备信息：

```text
设备：/dev/nvme0n1
型号：MEMBLAZE P6531DT0384Y00
容量：3.5 TiB
分区：/dev/nvme0n1p1
文件系统：ext4
挂载点：/data1
挂载参数：rw,relatime
```

2026-09-02 检查时的空间：

```text
总容量：3,779,301,580,800 bytes
已用：3,018,470,080,512 bytes
可用：568,776,994,816 bytes
df -h 显示：3.5T total / 2.8T used / 530G available / 85%
```

注意事项：

- `/data1` 权限当前为 `drwxrwxrwx root:root`。
- `/data1/ywang` 当前不存在。
- `/data1` 是共享盘且已使用 85%；不要未经确认直接迁移长期数据。
- 建议经主机管理员或盘使用方确认后，为量化任务建立专用临时目录，例如 `/data1/ywang/quantitize-scratch`，只保存运行期工作集，完成后将必要产物归档回 `/data3`。

### 2.2 当前量化平台存储

当前项目位于 `/data3/ywang/quantitize-platform`，底层为：

```text
/dev/sdc1 -> /data3
设备型号：INTEL SSDSC2KB038TZ
接口：SATA
文件系统：ext4
可用空间：约 2.0 TiB
```

H200 基准任务目录约 12 GiB，其中：

```text
fpga_test_pack：7.3 GiB
workspace：3.7 GiB
results：22 MiB
logs：1.8 MiB
```

因此 `/data1` 当前 530 GiB 可容纳多次单任务受控测试，但必须设置清理策略，避免共享 NVMe 被长期输出占满。

## 3. 基准任务和输入一致性

H200 任务：

```text
task_id: 20260902_065807_test_h200
pipeline_start: 2026-09-02T07:53:20Z
pipeline_complete: 2026-09-02T09:37:32.677717Z
elapsed: 6252.678 s
GPU: NVIDIA H200 NVL, GPU0
```

wrs 对照任务：

```text
task_id: 20260727_151741_aituosh_moon_0727_2
input_manifest_created: 2026-07-27T07:17:56.527404Z
pipeline_complete: 2026-07-27T08:11:58.860605Z
elapsed: 3242.333 s
GPU: NVIDIA GeForce RTX 5090
```

共同配置：

```text
model.pt MD5: 7bcd8db3e75cd97b28f67e0dd2e69eed
preprocess: grayscale_r_channel
calibration dataset id: aituosha_moon_500
test dataset id: aituosha_moon_840
test image count: 840
imgsz: 1280
nc: 2
```

输入差异提醒：H200 量化日志显示扫描到 500 张标定图，旧 wrs 日志显示 496 张。两个任务的清单均声明使用 `aituosha_moon_500`，最终 296 个交付 `.bin` 也逐文件一致，但后续严格性能回归必须固定同一份文件列表并校验列表哈希，不能只比较数据集 ID。

## 4. 分阶段耗时

| 阶段 | H200 | wrs / RTX 5090 | H200 / wrs |
|---|---:|---:|---:|
| `pt_eval` | 288.924 s | 142.540 s | 2.03× |
| `quantize` | 1549.221 s | 1259.193 s | 1.23× |
| `onnx_eval` | 163.550 s | 57.307 s | 2.85× |
| `fpga_test_pack` | 2444.461 s | 1156.240 s | 2.11× |
| `fpga_eval` | 66.165 s | 28.676 s | 2.31× |
| `generate_bin` | 1724.334 s | 591.097 s | 2.92× |
| `merge_bin` | 2.761 s | 1.163 s | 2.37× |
| `bundle` | 13.262 s | 6.116 s | 2.17× |
| **总计** | **6252.678 s** | **3242.333 s** | **1.93×** |

H200 上 `fpga_test_pack` 与 `generate_bin` 合计约 4168.8 秒，占总时长约 **66.7%**。这两个阶段是第一优化目标。

## 5. 已确认的技术事实

### 5.1 ONNX 评测虽然请求 batch=8，实际为 batch=1

H200 和 wrs 日志均显示：

```text
ORT providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
输入 shape: [1, 3, 1280, 1280] (max_batch=1)
```

评测代码从模型静态输入读取 `max_batch=1`，随后把传入的 8 张拆成逐张 `session.run()`。NMS 还会把输出转换为 CPU Tensor 后执行。因此 H200 的大显存和大吞吐没有被利用。

日志内的纯评测循环耗时：

```text
onnx_eval direct:
  H200 97.1 s
  wrs  51.0 s

fpga_eval side-view:
  H200 49.4 s
  wrs  22.8 s
```

这说明即使在 GPU 推理阶段，当前 batch=1 的低延迟路径也更适合高频消费级 GPU，不能用 H200 的峰值吞吐规格直接推导端到端时间。

### 5.2 PT 评测逐图运行

`01_pt_eval.py` 对图片列表逐张循环，并对每张调用一次 `model.predict()`。该实现同样无法形成能让 H200 吃满的 batch。

### 5.3 FPGA 测试包是串行 CPU/I/O 流程

`04_generate_fpga_test_pack.py` 对 840 张图串行执行：

```text
源图片 -> png_to_bin(target_size=2000) -> bin_to_png(2000x2000 side view)
```

该阶段不使用 GPU，且生成约 7.3 GiB 文件，是 SATA SSD、单核图像处理和 PNG 压缩的组合瓶颈。

### 5.4 generate_bin 主要不是大规模 GPU 计算

`check_certain_layer_multi.py` 先做一次模型推理，然后进行 ONNX 图解析、逐层查找、NumPy 数据处理和大量文件写入。层处理固定使用最多 8 个 Python 线程。其 H200 耗时是 wrs 的 2.92 倍，符合 CPU 单线程/少线程、NUMA 和存储瓶颈特征。

### 5.5 H200 主机和容器配置

```text
CPU: 2 x Intel Xeon Gold 6530
CPU topology: 2 sockets / 64 physical cores / 128 threads / 4 NUMA nodes
GPU: 4 x NVIDIA H200 NVL
GPU0 CPU affinity: 0-15,64-79
GPU0 NUMA affinity: 0
container CPU quota: none
container cpuset: none
container memory limit: none
GPU allocation: GPU0 only
```

未发现 Docker 人为 CPU 限速。问题是“无限额”不等于“最优调度”：未设置 cpuset/NUMA 亲和性时，CPU 线程和内存可能远离 GPU0。

检查时 H200 load average 约 8.5，wrs 约 0.2。该数据只是检查时快照，不是任务运行历史，不能单独用作根因结论。后续必须采集运行期间监控。

## 6. 根因判断和置信度

| 判断 | 置信度 | 证据 |
|---|---|---|
| 流水线主要受 CPU/I/O 限制 | 高 | 66.7% 时间在 FPGA 包和 generate_bin；代码路径以串行处理和文件写入为主 |
| batch=1 导致 H200 吞吐优势无法发挥 | 高 | 模型输入及日志明确为 `max_batch=1`；代码逐张 `session.run()` |
| `/data3` SATA 相对 wrs NVMe 拖慢文件密集阶段 | 中高 | 实机块设备和阶段特征一致；尚缺同机 A/B 存储实测 |
| Xeon 多核未被当前少线程实现利用，i9 单核占优 | 中高 | 代码串行或固定 4/8 线程；CPU 拓扑差异明确 |
| 跨 NUMA 或共享负载进一步拖慢 H200 | 中 | 未绑 NUMA、共享主机；缺运行期 perf/numastat 证据 |
| H200 GPU 或 CUDA 配置故障 | 低 | CUDA、sm90 和 CUDAExecutionProvider 可用，任务结果正确且 GPU0 正常分配 |
| Docker CPU/内存硬限额 | 已排除 | `CpuQuota=0`、`NanoCpus=0`、`CpusetCpus=''`、`Memory=0` |

## 7. 优化路线图

### P0：先建立可信基线，不改算法

目标：让每次优化都能归因，避免只看总时长。

1. 新建专用性能任务，禁止覆盖或重新启动历史完成任务。
2. 固定并记录：
   - model MD5/SHA-256；
   - 完整标定/测试文件列表 SHA-256；
   - Git commit、Docker image ID；
   - Python、PyTorch、ONNX Runtime、CUDA/cuDNN 版本；
   - GPU UUID、CPU affinity 和 NUMA 节点。
3. 为每个阶段记录 wall time、user time、system time、max RSS、读写字节数。
4. 运行期间同步采集：
   - `nvidia-smi dmon`：GPU util、显存、功耗、时钟；
   - `pidstat -dur`：进程 CPU、I/O、上下文切换；
   - `iostat -xz 1`：SATA/NVMe 延迟、队列深度、吞吐；
   - `numastat -p <pid>`：本地/远端 NUMA 内存；
   - `mpstat -P ALL 1`：是否只占少数核心。
5. 每项 A/B 测试至少运行 3 次，报告中位数和离散程度；首次运行单独标记为冷缓存。

验收：能判断每个阶段主要等待 CPU、GPU 还是磁盘，并能复现基准到 ±5%。

### P1：同代码进行 SATA 与 NVMe A/B 测试

这是最小风险、最容易验证的改动。

建议方案：

1. 经管理员确认后创建 `/data1/ywang/quantitize-scratch`。
2. 不移动平台代码和长期归档，只把单次任务的临时工作目录放到 NVMe。
3. 可选择：
   - 为单个新任务把 job root 指向 `/data1/...`；或
   - 在 `/data3/.../output_data/<job>` 保留元数据，用明确的 bind mount/配置将 `workspace` 和 `fpga_test_pack` 指向 NVMe。
4. 任务完成并校验产物后，把最终 ZIP、manifest、指标和必要日志原子复制回 `/data3`。
5. 清理 NVMe 临时工作集；设置容量水位，例如可用空间低于 150 GiB 时拒绝启动新任务。

禁止事项：不要把历史任务目录直接移动到 `/data1`；不要在没有回滚方案时修改全局 Docker data-root；不要使用含糊的通配符清理共享盘。

验收：重点比较 `fpga_test_pack`、`generate_bin`、`bundle` 三阶段。若 NVMe 对这三项没有显著改善，再转向 CPU profiling，而不是继续猜测存储。

#### P1.1 推荐的双层存储架构

目标是让 `/data1` 永远只保存正在运行或短期诊断所需的数据，而让前端历史查询始终只依赖 `/data3`。

```text
/data3/ywang/quantitize-platform/data/output_data/<task_id>/
  job_config.json          # 永久；前端/API 的任务发现入口
  manifest.json            # 永久；运行状态和阶段时间
  input_manifest.md        # 永久
  input_validation.json    # 永久
  logs/                    # 永久；运行时可实时查询
  results/                 # 永久；指标和少量样例图
  <task_id>_quantized_bundle.zip  # 永久；历史下载和 inventory 来源
  failure_artifacts/       # 失败任务专用；完整保存 scratch 快照
  failure_inventory.json  # 失败快照的路径、大小和 SHA-256 清单

/data1/ywang/quantitize-scratch/<task_id>/
  input-stage/             # 可选；运行期模型/标定集/测试集 staging
  workspace/               # 临时；ONNX、逐层 BIN、合并 BIN
  fpga_test_pack/          # 临时；约 7.3 GiB，任务完成后不保留
  scratch_state.json       # 临时生命周期和归档状态
```

前端和 API 继续把 `OUTPUT_DATA_ROOT` 固定为 `/data3/.../output_data`，因此：

- 历史列表、任务状态、指标页、图片页和下载接口仍从 `/data3` 查询；
- `/data1` 暂时不可用或临时目录已清理，不影响历史任务；
- 不需要让前端扫描两个根目录，也不会把残留 scratch 错当成历史任务。

现有代码中 `JobConfig` 把所有路径都派生自 `job_root`。建议新增独立的 `TASK_SCRATCH_ROOT` 配置，并只让这些属性指向 scratch：

```text
workspace_dir
fpga_test_pack_dir
可选的 staged_input_dir
```

这些目录继续留在 `/data3`：

```text
manifest_path
logs_dir
results_dir
input_manifest_path
input_validation_path
bundle_zip_path
```

不要把整个 `/data3/.../<task_id>` 软链接到 `/data1`。当前 API 的 `_job_dir()` 会对任务目录执行 `resolve()` 并要求解析后的父目录仍为 `OUTPUT_DATA_ROOT`；顶层软链接会被路径安全检查拒绝。嵌套软链接也会增加 bundle、路径穿越保护和故障恢复的复杂度，不建议作为正式方案。

容器应增加一个稳定的 scratch bind mount，而不是为每个任务动态修改 Compose：

```yaml
environment:
  TASK_SCRATCH_ROOT: /scratch/quantitize
volumes:
  - /data1/ywang/quantitize-scratch:/scratch/quantitize
```

#### P1.2 任务生命周期和自动清理

推荐状态机：

```text
allocate_scratch
  -> running
  -> success:
       bundle_created_on_data3
       -> bundle_verified
       -> history_committed
       -> scratch_cleanup_pending
       -> scratch_cleaned
  -> failure:
       failure_marked
       -> failure_snapshot_copying
       -> failure_snapshot_verified
       -> failure_history_committed
       -> scratch_cleanup_pending
       -> scratch_cleaned
```

具体规则：

1. 创建任务时先在 `/data3` 写入最小任务记录，再为该 task ID 分配 `/data1` scratch。
2. manifest 和日志始终写 `/data3`，所以运行中前端监控不依赖 scratch 扫描。
3. `bundle` 阶段直接从 scratch 的 `workspace` 和 `/data3` 的 results/logs 读取，最终 ZIP 写入 `/data3` 临时文件。
4. ZIP 完成后执行完整性测试、成员清单检查和关键 BIN SHA-256 检查。
5. 验证通过后，将 ZIP 原子改名为正式历史文件，更新 `/data3/manifest.json` 为 completed。
6. 成功任务只有满足“正式 ZIP 存在且已验证、manifest 已持久化、任务不是当前运行任务”三个条件，才删除对应 scratch。
7. 失败任务不得直接按 TTL 删除 scratch。先把该任务在 `/data1` 的完整目录复制到 `/data3/.../<task_id>/failure_artifacts.partial/`，包括中间 ONNX、逐层 BIN、FPGA 测试包、临时配置和其他现场文件。
8. 为失败快照生成 `failure_inventory.json`，至少记录相对路径、文件类型、字节数、mtime 和 SHA-256；同时保存失败发生时的 manifest、错误栈、运行环境版本和最后阶段。
9. 复制完成后核对文件数量、总字节数和清单哈希；验证通过才把 `failure_artifacts.partial` 原子改名为 `failure_artifacts`，并在 `/data3/manifest.json` 写入 `failure_archive_status=verified`。
10. 只有失败快照已在 `/data3` 完整验证后，才允许清理 `/data1` 的对应 scratch。若复制、校验或原子提交失败，scratch 必须保留并标记 `failure_archive_status=pending`，由后台任务重试。
11. API 启动时扫描孤儿 scratch，但不得立即删除。只处理名称符合 task ID、状态非运行中且已有成功归档或已验证失败快照的目录。
12. 清理前必须解析绝对路径并确认其直属于配置的 `TASK_SCRATCH_ROOT`；禁止通配符和未解析环境变量递归删除。
13. 设置水位保护，例如 `/data1` 可用空间低于 150 GiB 时拒绝新任务，优先完成待归档失败任务和清理已经验证的 scratch。
14. 管理员可设置 `retain_scratch=true`，在 `/data3` 已有失败快照的同时继续临时保留 `/data1` 现场，但必须显示其空间占用和到期策略。

异常处理：

- 任意阶段失败：先将 manifest 标记为 failed 并保留 scratch，同时后台完整归档到 `/data3/failure_artifacts`；前端可立即查看日志，归档状态显示为 pending/copying/verified。
- 失败归档过程中服务重启：根据 `.partial` 目录和 `failure_inventory.json` 断点续传或安全重建清单，不得把未验证副本当成完整历史记录。
- `/data3` 空间不足：失败任务保持 scratch，不得删除；前端明确显示“失败现场待归档”，并阻止启动可能进一步耗尽空间的新任务。
- ZIP 已写入但进程在 manifest 更新前退出：重启恢复逻辑重新验证 ZIP，再决定提交或保留 scratch。
- manifest completed 但 scratch 清理失败：历史任务仍可查询；记录 `cleanup_pending` 并异步重试。
- `/data1` 丢失或未挂载：拒绝启动新任务，但不能影响 `/data3` 历史查询。

#### P1.3 历史保留范围

现有 H200 任务目录约 12 GiB，其中 7.3 GiB 是可再生的 `fpga_test_pack`。成功任务建议默认只长期保留：

```text
任务配置、manifest、输入清单和校验报告
八步日志
评测指标和问题/成功样例图
最终 bundle ZIP
bundle inventory
关键交付 BIN 哈希清单
```

成功任务的 `fpga_test_pack`、中间 ONNX、逐层临时模型和其他可再生 scratch 默认不进入历史存储。若业务要求支持完整重放，可为特定成功任务设置保留策略，或将完整工作集另行归档到 NAS，而不是长期占用 `/data1`。

**失败任务采用不同策略：默认将整个 scratch 完整归档到 `/data3/.../<task_id>/failure_artifacts`，不丢弃任何已经生成的中间文件。** 这样后续 Agent 可以复盘失败现场、从中间阶段恢复或比较异常产物。失败快照进入 `/data3` 后属于正式历史记录，不由 scratch TTL 清理；如需长期容量治理，应另设人工审核或 NAS 冷归档策略，不能与 `/data1` 自动清理混为一谈。

按当前完整任务约 12 GiB 估算，连续失败任务会较快消耗 `/data3`。必须在前端显示失败快照大小和归档状态，并为 `/data3` 设置独立容量告警；容量治理应优先迁移到 NAS，而不是自动删除失败证据。

### P2：让 ONNX 和 PT 真正批量推理

1. 导出动态 batch ONNX，或分别生成固定 batch 8/16/32 模型。
2. 修改 ONNX 评测，确保一次 `session.run()` 实际接收 N 张，而不是在内部拆成单张。
3. 修改 PT 评测，一次向 `model.predict()` 提交图片列表或预处理后的 batch。
4. 逐步测试 batch 1、4、8、16、32，记录吞吐、峰值显存和指标一致性。
5. 若动态 batch 影响 FPGA 抽 bin 模型，分离“评测模型”和“generate_bin 静态 batch=1 模型”，不要强迫一个 ONNX 同时服务所有用途。
6. 评估将 NMS 留在 GPU 或使用更少的数据拷贝；任何变更必须验证检测结果不变。

验收：H200 GPU util 明显提高；840 张评测吞吐优于 5090；Precision/Recall/F1/mAP 和问题样本数量在既定容差内。

### P3：并行化 FPGA 测试包生成

1. 将 840 张图片按文件并行，优先使用 `ProcessPoolExecutor`，避免 Python GIL 影响 OpenCV/编码路径。
2. 从 4、8、16 worker 逐步压测，结合 `iostat` 寻找 CPU 与磁盘的平衡点。
3. 检查是否真的需要 `PNG -> BIN -> PNG` 往返。如果 side-view 可直接由同一中间数组生成，则只转换一次。
4. 批量创建输出目录和 manifest，避免每张图重复路径解析和不必要的 metadata 操作。
5. 确保并行输出顺序不影响 manifest；最终按输入文件名稳定排序。

验收：输出 BIN/PNG 数量、文件名、内容或业务允许的像素结果与基线一致；阶段耗时目标先定为低于 15 分钟，再根据 I/O 实测调整。

### P4：优化 generate_bin

1. 用 `cProfile`/`py-spy` 和阶段内细粒度计时定位：ONNX 加载、图扫描、initializer 提取、NumPy 转换、文本/二进制写入分别占多少。
2. 确认是否对每层重复遍历完整 ONNX 图；建立一次性的 node/initializer 索引。
3. 避免重复创建 ORT Session、重复运行相同推理和重复序列化临时模型。
4. 将写文件与计算解耦，合并极小写操作；优先直接写最终二进制格式。
5. 线程数做 1/4/8/16 A/B；若主要是 Python CPU 代码，考虑进程池，但必须评估 100 MB 模型在进程间复制的成本。
6. 先在 NVMe 上运行，再归档到 SATA。

验收：所有 296 个交付 `.bin` 的路径、大小和 SHA-256 与基线一致；`ALL_WT.bin` 和 `ALL_BN.bin` 必须逐字节一致。

基线关键哈希：

```text
workspace/all_bin/ALL_WT.bin
size: 100205568
sha256: 94776567f1806a64191d5aa5ed76f788390e4570a53cb108afe0a89e3ab5ee1c

workspace/all_bin/ALL_BN.bin
size: 169088
sha256: 75d547659f6b64a3fc3b75676b7971d979dbd495e7ca19680eb8229db98503e0
```

### P5：CPU/NUMA 调优

仅在 P0 监控证明存在跨 NUMA 或调度问题后实施。

1. GPU0 对应 NUMA 0，CPU affinity 为 `0-15,64-79`。
2. 对单任务 A/B 测试 `cpuset_cpus: 0-15,64-79` 和 `cpuset_mems: 0`。
3. 根据 ORT、OpenMP、OpenCV 的实际并行行为设置线程数，避免 128 线程过度订阅。
4. 比较绑定前后 GPU H2D 延迟、远端 NUMA 内存、CPU 上下文切换和阶段耗时。
5. 不要直接假设“核心越多越快”；对串行阶段应优先减少工作量和 I/O。

验收：绑定后至少连续三次运行稳定改善；若没有改善则回退，不保留无效复杂度。

## 8. 推荐实施顺序

```text
P0 监控和严格基线
  -> P1 NVMe 单任务 A/B
  -> P3 FPGA 包并行化
  -> P4 generate_bin profiling/去重
  -> P2 真正的批量 GPU 推理
  -> P5 有证据后再做 NUMA/线程调优
```

P2 对 GPU 利用率最关键，但可能牵涉 ONNX 形状、评测逻辑和 FPGA 工具兼容性；P1/P3/P4 更直接针对当前 66.7% 的主要耗时，所以建议先处理。

## 9. 后续 Agent 的安全与回归要求

1. 不修改、删除或重新启动两个历史完成任务。
2. 所有性能测试使用新 task ID 和隔离目录。
3. 每次只改变一个主要变量，保留前后监控数据。
4. 不以 ZIP 哈希作为确定性验收条件；ZIP 内日志、时间戳和量化 ONNX 可能非确定。
5. 交付 BIN 必须逐文件比较路径、字节数和 SHA-256。
6. 指标至少校验 PT/ONNX/FPGA 的 Precision、Recall、F1、mAP、TP、FP、FN。
7. NVMe 临时目录必须有明确的任务级路径、空间水位和清理清单；删除前解析并验证绝对路径。
8. H200 是共享主机，压力测试前确认不会影响其他用户任务。

## 10. 复查命令（只读）

```bash
# NVMe 设备和挂载
lsblk -o NAME,PATH,SIZE,FSTYPE,MOUNTPOINTS,MODEL,TRAN
findmnt -T /data1 -o TARGET,SOURCE,FSTYPE,OPTIONS
df -hT /data1 /data3

# CPU / NUMA / GPU 拓扑
lscpu
nvidia-smi topo -m

# 容器限额和 GPU 分配
docker inspect quantitize-platform-api --format '{{json .HostConfig}}'

# 当前任务目录规模
du -sh /data3/ywang/quantitize-platform/data/output_data/<task_id>
du -sh /data3/ywang/quantitize-platform/data/output_data/<task_id>/*
```

以上命令只反映检查时状态。运行期瓶颈判断必须结合 P0 的持续监控，不要用任务结束后的单次 `nvidia-smi` 或 `uptime` 快照替代。
