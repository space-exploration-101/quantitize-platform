# YOLO pose gray1 训练遗留问题

与 `/data3/ywang/docs/agent/YOLO_GRAY1_TRAINING_LEFTOVERS.md` 同步。量化平台 agent 打开本仓库时优先读这一份。

## 必须在下一轮数据预处理里对齐

1. **训练图太少**：`20260909_gray1_21c_1680` 是集成检查点（约 1200 张 / 每类 50），不是生产基线。从零 50 epoch 过拟合。PT test F1 ~0.34 vs R-only 全量 Box mAP50 ~0.957。
2. **几何未对齐 FPGA**：训练是 4096/2048 → 长边 1280 letterbox；FPGA 是 2000×12bit → 8bit → 2×2 均值 1000 → pad 140 → 1280。预处理必须按 FPGA 路径做。
3. **权重**：有 R-only 权重时迁移除第一层卷积外的参数，不要再从零训 1 通道。
4. **Step14 ONNX F1=0** 是评测管线（中间 ONNX / CPU），不是 gray1 检不出。
5. **平台 ABI**：只接受 1 通道。`passthrough`（取灰度/R，不 BGR2GRAY）或 `color_to_gray`（BGR2GRAY）。旧 3 通道预处理已删除。
