# Quantitize Platform

YOLO 姿态模型量化与 FPGA 交付平台。Web/API 接收模型、标定集和测试集，Runner 依次完成 PT 评测、ONNX FP16/量化、ONNX 评测、FPGA 输入回环、逐层 BIN、合并与打包。

## 当前状态

- 生产主机：H200（`10.2.29.180`），项目：`/data3/ywang/quantitize-platform`。
- Web：<http://10.2.29.180:8088>；API：<http://10.2.29.180:8000>。
- 当前代码已切换到单通道灰度输入 ABI：`[1,1,1280,1280]`、FP16。
- 2026-09-11 检查时两个容器均为 healthy、worker idle，但 API 容器 NVML 初始化失败，`/ready` 返回 503；新任务会等待 GPU，不会开始流水线。
- 单通道 21 类版本尚未全流程验收：最近一次任务的 ONNX/输入回环 F1 为 0，`generate_bin` 因 `KeyError: input` 失败。历史三通道 21 类成功结果只能作为对照。

实时状态必须重新检查，不能把本页快照当作资源现状。

## 文档导航

| 需要了解 | 文档 |
|---|---|
| 当前主机、服务、版本、已知风险 | [Agent Context](docs/agent/CONTEXT.md) |
| 安全检查、服务和任务操作 | [Agent Runbook](docs/agent/RUNBOOK.md) |
| 流水线、目录、输入输出和逐步命令 | [Pipeline Flow](pipeline/FLOW.md) |
| 容器构建、启动和访问 | [Deploy](deploy/README.md) |
| 功能变更与运行过的量化测试 | [Change Log](docs/CHANGELOG.md) |
| 单通道预处理、训练与 FPGA 几何 | [Model Input](docs/reference/MODEL_INPUT.md) |
| 21 类输出、BIN 与后处理 | [21-class FPGA](docs/reference/CLASS21_FPGA.md) |
| 性能优化证据 | [Performance](docs/reference/PERFORMANCE.md) |
| 灾难恢复 | [Recovery](docs/reference/RECOVERY.md) |
| 环境重建 | [Rebuild Guide](rebuild/REBUILD_GUIDE.md) |

自动化 Agent 从 [AGENTS.md](AGENTS.md) 开始。文档不保存密码；`data/output_data` 不属于系统备份，历史任务清理必须单独授权并先验证归档。
