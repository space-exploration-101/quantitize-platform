# 单张图片 FPGA 逐层输入输出导出

`pipeline/engine/export_layer_trace.py` 从最终 weight-only ONNX 导出一张 Gray1 图片在
FPGA 硬件层边界上的输入和输出。它是独立调试工具，不改变正式 ONNX，也不会自动扩大
量化任务的交付 ZIP。

## 使用方法

```bash
cd /data3/ywang/quantitize-platform
docker exec -w /home/rs/wrs/onnxviewe/quantitize-platform \
  quantitize-platform-api \
  /opt/ywang/yolo-runtime/bin/python pipeline/engine/export_layer_trace.py \
  --model /container/path/to/model_output.onnx \
  --image /container/path/to/explicit-image.png \
  --output-dir /container/path/to/new-empty-output \
  --preprocess-mode passthrough
```

这里的路径必须是 API 容器内可见路径。当前项目、`data/output_data`、`data/shared_data` 和
`/scratch/quantitize` 已挂载到容器中；宿主机没有 `/opt/ywang/yolo-runtime` 解释器。

图片必须显式传入。工具不会从标定集自动选择图片。输出目录必须不存在或为空；生成时先
写入同级 `.partial.<pid>` 目录，全部校验通过后才原子改名。

## 输出和布局

- `manifest.json`：模型和图片哈希、预处理、运行环境、层到 ONNX 节点映射及 tensor 元数据。
- `layers.csv`：下游人工检查使用的扁平索引。
- `source/network_input.bin`：预处理后的网络输入。
- `tensors/*.bin`：每个 ONNX tensor 只保存一次，由 manifest 引用。
- `SHA256SUMS`：整个结果目录的文件校验值。

BIN 是小端 FP16。普通特征从 `[1,C,H,W]` 变换为
`[ceil(C/32),H,W,32]`；`[1,C,N]` 变换为 `[N,ceil(C/32),32]`。
不足 32 通道的 lane 补零。最终 `[1,31,34000]` 因此保存为 `[34000,32]`，
第 32 个通道不能参与分类、NMS 或关键点解析。

计算层沿用 `T32_Lxx_sxx`；Concat 和 Head 解码等无权重路由节点使用
`T32_Lxx_rxx`，避免改变现有 WT/BN 层号。Add、Concat 等多输入算子的每个输入都会在
manifest 中独立列出。
