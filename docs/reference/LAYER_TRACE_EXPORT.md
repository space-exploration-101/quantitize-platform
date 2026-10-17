# 单张图片逐层输入输出导出

`pipeline/engine/export_layer_trace.py` 从最终 weight-only ONNX 导出一张 Gray1 图片在 FPGA 硬件层边界上的软件参考输入和输出。它是独立调试工具，不改变正式 ONNX，也不会自动扩大量化任务交付 ZIP；结果不代表 FPGA 实机计算验收通过。

## 使用方法

在独立验证环境准备包含 NumPy、ONNX、ONNX Runtime 和图像依赖的解释器，从仓库根目录执行：

```bash
"$YOLOV8_PYTHON" pipeline/engine/export_layer_trace.py --model /path/to/model_output.onnx --image /path/to/explicit-image.png --output-dir /path/to/new-output --preprocess-mode passthrough
```

工具使用 CPUExecutionProvider，不申请 GPU。容器运行时，模型、图片和输出必须使用容器可见路径。`passthrough` 适用于已准备的网络输入，其他输入使用匹配的任务预处理模式。图片必须显式传入，不会从标定集自动选择；输出目录必须不存在或为空，先写同级 `.partial.<pid>` 目录，校验后原子发布，不覆盖已有结果。

## 输出和布局

- `manifest.json`：模型/图片哈希、预处理、运行环境、层到 ONNX 节点映射及 tensor 元数据。
- `layers.csv`：人工检查使用的层索引。
- `source/network_input.bin`：预处理后的网络输入。
- `tensors/*.bin`：每个 ONNX tensor 只保存一次，由 manifest 引用。
- `SHA256SUMS`：结果目录的文件校验清单。

BIN 为小端 FP16。普通 `[1,C,H,W]` 特征变为 `[ceil(C/32),H,W,32]`，`[1,C,N]` 变为 `[N,ceil(C/32),32]`；不足 32 通道的 lane 补零。最终 `[1,31,34000]` 保存为 `[34000,32]`，第 32 个通道不能参与分类、NMS 或关键点解析。

计算层沿用 `T32_Lxx_sxx`；Concat 和 Head 解码等无权重路由节点使用 `T32_Lxx_rxx`，避免改变现有 WT/BN 层号。Add、Concat 等多输入算子的每个输入在 manifest 中独立列出。
