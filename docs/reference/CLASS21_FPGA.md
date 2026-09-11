# 21 类模型与 FPGA 接口

本文给出当前目标接口，并区分历史三通道成功证据与当前单通道未完成状态。

## 1. 输出结构

候选数和姿态关键点数量不变，类别从 2 增至 21：

| 项 | 历史 2 类 | 21 类 |
|---|---:|---:|
| 网络输出 | `[1,12,34000]` | `[1,31,34000]` |
| bbox | ch0–3 | ch0–3 |
| class score | ch4–5 | ch4–24 |
| keypoint | ch6–11 | ch25–30 |
| FPGA 对齐布局 | 32 通道 | 32 通道，ch31 padding |

公式：`4 bbox + nc class + 6 keypoint`。FPGA/NMS 代码不得硬编码 2 类偏移；padding 通道不能参与分类、阈值、NMS 或关键点解析。

## 2. 当前输入接口

当前平台和目标模型输入是 `[1,1,1280,1280]` FP16，不再是历史的 `[1,3,1280,1280]` R-only。FPGA 需要输出一个灰度通道；具体 12-bit→8-bit、2×2 降采样和 padding 约束见 `MODEL_INPUT.md`。

## 3. 权重和输出 BIN

历史三通道 21 类任务中：

```text
ALL_WT.bin = 102,381,568 bytes
历史2类 ALL_WT.bin = 100,205,568 bytes
L43_WT.bin = 2,176,000 bytes = 32 * 34000 * 2
```

这些尺寸说明输出层按 32 通道对齐，但不是当前单通道版本的验收值。输入通道改变会影响第一层权重和 ONNX 拓扑；必须以新的完整任务产物重新生成清单、大小和 SHA-256。

当前单通道任务 `20260909_191112_test_21_class_gray_input` 在 `generate_bin` 阶段有 21 层报 `KeyError: input`，尚无可交付的单通道合并 BIN。FPGA 开发不能把历史三通道 `ALL_WT.bin` 与当前单通道模型混用。

## 4. 后处理

现有参考实现：

- Linux 文件版：`postprocess/fpga_nms/nms_from_output21.c`
- Zynq 内存版：`postprocess/zynq_nms/nms_output21_zynq.c/.h`

期望输入为 `[34000][32]` FP16，即至少 `2,176,000` 字节；有效 ch0–30。处理顺序：

```text
34000 candidates
-> score threshold
-> class-aware NMS
-> business filter: 20 个地标最多保留最高分 1 个，月亮可另保留 1 个
-> 去除 pad140 并裁剪到 1000x1000 有效区域
```

当前假设月亮类别 ID 为 20，必须与训练数据 `names` 最终确认。Zynq 版使用数 MB 静态工作区；需检查 linker script 的 DDR/OCM 放置。DMA 输出由 CPU 读取前需要 cache invalidate。

## 5. 历史验证与当前缺口

历史任务 `20260904_031656_test_21_class_pipeline_03` 使用 `[1,3,1280,1280]` R-only，八步完成，PT/ONNX/输入回环 F1 均 0.9420。它验证了 21 类输出偏移、三通道测包和后处理基线。

当前 `[1,1,1280,1280]` 任务 PT F1 0.3418，ONNX/输入回环 F1 0，逐层 BIN 失败。因此需依次完成：

1. PT 与 FP16/量化 ONNX 等价。
2. 单通道 FPGA 输入回环等价。
3. 修复逐层节点解析并生成全部 WT/BN。
4. 核对输出转置 `[1,31,34000]` ↔ `[34000,32]`、FP16 字节序和 padding。
5. 编译后处理、mock BIN 冒烟，再使用真实 FPGA 输出验证阈值、NMS、业务过滤和坐标回映射。

上述全部完成前，单通道 21 类不能标记为 FPGA 适配完成。
