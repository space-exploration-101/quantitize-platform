# 功能变更与量化测试记录

本文从当前 Git diff、任务 manifest、指标和日志整理。仓库目前只有一个快照提交，后续多项代码仍未提交，因此这里记录“观察到的开发变化”，不是完整 Git 提交史。

## 2026-09-11：文档与现状对齐

- 重建文档入口、Agent 路由和当前状态说明；移除重复的交接、迁移和中间文档。
- 明确区分当前单通道代码、正在运行的容器和历史三通道任务。
- 发现服务健康但 GPU 未就绪：API 容器 NVML 初始化失败，`/ready` 为 503。

## 2026-09-09 至 09-10：单通道 gray1 切换（未完成验收）

观察到的代码变化：

- 新任务固定 `input_channels=1`、`input_semantics=gray1`，模型输入 `[1,1,1280,1280]` FP16。
- Web 预处理精简为 `passthrough` 和 `color_to_gray`；前者取灰度/R，后者 BGR2GRAY。
- PT、ONNX、FPGA 测包、PNG/BIN 与量化校验统一为单通道。
- PT 评测拒绝第一层不是 1 通道的模型；量化前后校验 ONNX ABI。
- API 镜像切换到 gray1 基础镜像和 Python 3.12 runtime；增加兼容 Ultralytics 不同版本的 NMS 导入。

运行记录：

| task | 数据/目标 | 结果 |
|---|---|---|
| `20260909_191112_test_21_class_gray_input` | 21 类；标定 400+；测试 240 | PT F1 0.3418；ONNX/回环 F1 0；`generate_bin` 21 层 `KeyError: input`，任务失败；506 文件约 3.4 GB 失败现场已校验归档 |
| `20260909_191009/191030/191051_test_21_class_gray_input` | 21 类单通道 | 仅创建，八步均 pending |
| `20260909_180701_gray1_web_create_check` | Web 创建检查 | 仅创建，未运行 |

结论：单通道 PT 可执行，但精度低；ONNX 等价、输入回环和逐层 BIN 未通过。不能进入 FPGA 交付。

## 2026-09-03 至 09-04：21 类三通道 R-only 适配（历史基线）

- 类别数从 2 扩展到 21；输出由 `[1,12,34000]` 变为 `[1,31,34000]`。
- FPGA 输出按 32 通道对齐；类别位于 ch4–24，关键点 ch25–30，ch31 padding。
- 增加 21 类 Linux/Zynq 后处理：普通 class-aware NMS 后，业务过滤最多保留 1 个地标和 1 个月亮。
- 修复 R-only 图像在 PNG→BIN→side-view 中被再次 BGR2GRAY 导致压暗的问题。

`20260904_031656_test_21_class_pipeline_03` 八步完成：PT、ONNX、输入回环 F1 均 0.9420（84 张，TP 65、FP 6、FN 2）；阶段耗时依次 44.219、427.257、29.100、7.483、20.448、21.874、2.639、15.328 秒。该模型输入仍是 `[1,3,1280,1280]`，只用于历史对照。

## 2026-09-02 至 09-03：H200 性能与存储优化（历史二类基线）

功能变化：NVMe scratch；成功校验后清理、失败现场归档；PNG/BIN 和权重写入向量化；FPGA 测包 8 进程；PT 批处理；ORT CUDA calibration + CPU fallback；manifest 记录准确阶段耗时。

| task | provider | 总耗时 | 验证 |
|---|---|---:|---|
| `20260902_065807_test_h200` | CPU，优化前 | 6252.678 s | 八步完成 |
| `20260902_153103_perf_opt_h200_cuda_calib` | 实际 CPU | 2070.566 s | 优化后 CPU 对照 |
| `20260902_165033_perf_opt_h200_true_cuda_ab1` | CUDA + CPU fallback | 790.513 s | 八步完成 |
| `20260902_170544_perf_opt_h200_true_cuda_ab2` | CUDA + CPU fallback | 725.451 s | 八步完成 |

两次 CUDA 任务的二类检测指标一致：TP 840、FP 1、FN 0、F1 0.999405；296 个 BIN、总计 200,749,312 字节，路径/大小/SHA-256 与 CPU/历史基线一致。详细证据见 `reference/PERFORMANCE.md`。

## 待完成

1. 修复或解释单通道 ONNX F1=0，与 PT 输出逐层/逐样本对比。
2. 修复 `generate_bin` 对 gray1 ONNX 图中节点输入命名/拓扑的假设。
3. 对齐训练、软件预处理与 FPGA 的 2000→1000→pad140 几何。
4. 固化运行时 Ultralytics 版本与来源，消除基础镜像标签和 pip metadata 不一致。
5. 在获批 GPU 上建立新的单通道 21 类 golden task；通过前不替换历史交付基线。
