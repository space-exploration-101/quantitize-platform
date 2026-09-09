# 量化平台交接手册

面向第一次接手本项目的同事。读完后应能回答五个问题：平台解决什么问题、服务在哪里、数据在哪里、一次任务怎样流转、出现故障时从哪里开始。

## 1. 项目背景：什么是模型量化

训练得到的 PyTorch 模型通常以浮点形式运行，精度高但计算和存储成本较大。量化的目标是把模型转换成更适合部署硬件的表示，并用标定集确定数值范围，再用测试集检查转换前后的效果是否可接受。

本平台不是单纯执行一次格式转换，而是一条完整交付流水线：

1. `pt_eval`：评估原始 PyTorch 模型，形成基线。
2. `quantize`：生成量化相关 ONNX 和中间文件。
3. `onnx_eval`：评估量化后的 ONNX。
4. `fpga_test_pack`：生成 FPGA 测试输入和侧视数据。
5. `fpga_eval`：评估 FPGA 侧测试结果。
6. `generate_bin`：生成部署需要的分层 BIN。
7. `merge_bin`：合并最终权重和配置 BIN。
8. `bundle`：把配置、日志、结果和交付文件打成 ZIP。

标定集用于量化，不等同于测试集；测试集用于检查指标和问题样本。任务完成的判断依据不是“GPU 跑过”，而是八个阶段全部完成、无 worker/task error、最终 ZIP 存在。

已验证基线任务：

```text
task_id:       20260902_065807_test_h200
model:         model.pt
calibration:   aituosha_moon_500
test:          aituosha_moon_840
preprocess:    grayscale_r_channel
status:        completed
```

## 2. 人、机器和网络

### 2.1 机器角色

| 名称 | 地址 | 用户 | 角色 | 连接方式 |
|---|---|---|---|---|
| 本机 Windows | 局域网工作站 | `admin` | 文档、Agent、SSH 控制端 | 本地 |
| H200 | `10.2.29.180` | `ywang` | 当前生产量化平台，4×H200 NVL | `ssh H200` 直连 |
| wrs | `10.2.26.132` | `rs` | 旧环境和历史来源，仅用于追溯 | `ssh wrs` |
| NAS | `10.2.26.26` | NAS 账户 | 冷备份和 Agent 文档 | SMB/CIFS |

H200 不经过 wrs 跳转。密码不写入文档；SSH 使用本机密钥。

### 2.2 Astrill 与局域网路由

Astrill 会安装覆盖范围很大的 VPN 路由。为了避免 H200、wrs 和 NAS 流量误入 VPN，本机 Windows 已配置持久路由：

```text
10.2.29.0/24 -> 192.168.31.1  # H200 网段
10.2.26.0/24 -> 192.168.31.1  # wrs 和 NAS 网段
物理网卡 ifIndex: 19
```

Windows 路由按“最长前缀优先”选择，因此 `/24` 局域网路由会优先于 Astrill 的 `/1` 广域路由。排查连接时先看路由和 TCP 端口，不要把 IP 直连问题误判成 DNS 问题。

常用检查：

```powershell
Find-NetRoute -RemoteIPAddress 10.2.29.180
Test-NetConnection 10.2.29.180 -Port 22
ssh -o BatchMode=yes H200 "whoami; hostname"
```

## 3. H200 服务器基线

```text
OS:               Ubuntu 22.04.5 LTS
CPU:              2× Intel Xeon Gold 6530，128 逻辑 CPU，4 NUMA 节点
内存:             约 1 TiB
GPU:              4× NVIDIA H200 NVL，每张约 143771 MiB
驱动:             580.82.07
架构:             x86_64 / CUDA sm_90
项目盘:           /data3，SATA SSD
Docker data-root: /data1/docker，NVMe
Docker:           服务开机启动
```

`ywang` 在 Docker 组内，Docker 权限等价于高权限。H200 是公用服务器，GPU、CPU、磁盘和系统包都可能被其他用户使用。不要因为一张卡瞬时利用率为 0 就认为它空闲；还要看显存、进程和连续采样。

## 4. 数据和文件在哪里

### 4.1 本机 Windows

```text
E:\codex\quantitize\
├── README.md                 # 人类总入口
├── AGENTS.md                 # Agent 总入口
├── docs\                     # 当前交接、Agent 和参考文档
├── h200-worktree\            # H200 项目代码的本机镜像
├── rebuild\                 # 环境重建清单与校验脚本
└── model.pt                  # 本机保留的模型文件
```

### 4.2 H200

```text
/data3/ywang/quantitize-platform/
├── apps/api/                 # API、任务管理和 worker
├── apps/web/                 # Web 界面/BFF
├── pipeline/                 # 八阶段 runner 和量化逻辑
├── deploy/                   # Compose、Dockerfile、GPU 配置
├── rebuild/                  # 可重建环境证据
└── data/
    ├── shared_data/          # 标定集、测试集等共享输入，约 11GB
    └── output_data/          # 每个任务的工作目录和历史结果，持续增长
```

任务目录通常为：

```text
data/output_data/<task_id>/
├── job_config.json
├── manifest.json             # 总状态和阶段状态
├── input/                    # 该任务冻结后的模型与输入
├── logs/                     # 阶段日志
├── results/                  # PT/ONNX/FPGA 指标和样例
├── workspace/                # ONNX、BIN 等中间产物
├── fpga_test_pack/           # 大体积 FPGA 测试数据
└── <task_id>_quantized_bundle.zip
```

`output_data` 是容量增长最快的目录。系统备份明确不包含它；后续应通过单独的历史归档和保留策略迁移到 NAS，不能与系统恢复快照混在一起。

### 4.3 NAS

H200 上的挂载点：

```text
/mnt/ywang-nas -> //10.2.26.26/902_data
```

该挂载当前不是 `/etc/fstab` 持久挂载；服务器重启后，量化服务仍可从本地 `/data3` 启动，但备份或恢复前需要重新挂载 NAS。

系统快照：

```text
\\10.2.26.26\902_data\0-项目\13-专项\4-代码\量化平台\H200
```

最新已验证快照：

```text
snapshot-20260902-182606
```

它包含项目与 `shared_data`、API/Web 镜像、原始 Conda 环境包、环境清单和恢复手册，不包含 `output_data`。

Agent 文档镜像：

```text
\\10.2.26.26\902_data\0-项目\13-专项\4-代码\量化平台\agent_context
```

## 5. 平台架构

```mermaid
flowchart LR
    U[浏览器] -->|8088| W[Web / BFF]
    W -->|Compose 内网| A[API / Worker :8000]
    A --> R[05_runner.py]
    R --> P[八阶段 Pipeline]
    P --> G[分配的单张 H200 GPU]
    P --> S[(shared_data)]
    P --> O[(output_data / task_id)]
    D[deploy/.env] --> A
    C[Docker Compose] --> W
    C --> A
    O -.独立归档策略.-> N[NAS]
    B[代码+共享数据+镜像快照] --> N
```

核心关系：

- 宿主机 `8088` 是用户入口，`8000` 是 API。
- Web 通过 Compose 服务名访问 API，不直接执行量化。
- API 维护任务状态并启动一个 worker；runner 顺序执行八个阶段。
- API 镜像约 21GB，内含原始 yolov8/量化运行环境；Web 镜像较轻。
- `apps/api`、`pipeline`、`shared_data` 和 `output_data` 通过 bind mount 进入 API 容器，代码修改无需重建 21GB 镜像。
- `deploy/.env` 使用物理 GPU UUID。容器中被分配的物理 GPU 会显示为逻辑 GPU 0。
- GPU readiness 会检查空闲显存和利用率；API/Web 可以健康，但任务可能处于 `waiting_gpu`。

## 6. 日常使用

### 6.1 打开平台

浏览器访问：

```text
http://10.2.29.180:8088
```

创建任务时确认模型、标定集、测试集和预处理方式。不要重复启动已经 `running` 的任务。

### 6.2 查看服务和任务

```bash
ssh H200
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8088/health
```

查询具体任务：

```bash
curl -fsS http://127.0.0.1:8000/api/tasks/<task_id> | python3 -m json.tool
```

重点字段：`status`、`worker_status`、`task_error`、`worker_error`、`manifest.steps` 和 `has_zip`。

### 6.3 查看 GPU

```bash
nvidia-smi
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw --format=csv
nvidia-smi dmon -s pucm -c 5
```

正式任务前要确认：分配 UUID 正确、显存满足门槛、没有与其他用户冲突。不要停止不属于自己的进程。

### 6.4 重启服务

普通重启不需要从 NAS 恢复：

```bash
ssh H200
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

禁止使用 `docker compose down -v`。

## 7. 怎样判断任务成功

同时满足：

- API 总状态为 `completed`。
- 八个 `manifest.steps` 全部为 `completed`。
- `worker_error` 和 `task_error` 为空。
- `has_zip=true`，最终 bundle ZIP 存在。
- PT、ONNX、FPGA 评估报告存在且指标在业务容差内。

`quantize: pending` 只表示阶段尚未完成并写回 manifest；若 `quantitize.py` 进程仍在消耗 CPU，量化可能正在进行。当前量化引擎大量使用 CPU，GPU 利用率低不一定代表卡住。

## 8. 已知性能特征

H200 金标准任务耗时约 1 小时 44 分。约三分之二时间在 `fpga_test_pack` 和 `generate_bin`，它们主要受 CPU、图像编解码和 SATA I/O 限制。当前 ONNX 输入固定 batch=1，H200 的吞吐优势没有充分利用。

因此：

- 不要只根据 GPU 型号估算任务时间。
- 优化优先级是 NVMe 临时工作区、真实批量推理、并行 FPGA 包生成和减少小文件 I/O。
- 任何优化必须固定模型和数据清单，执行结果回归。

详细证据见 [reference/PERFORMANCE.md](reference/PERFORMANCE.md)。

## 9. 故障分流

| 现象 | 首先检查 | 继续阅读 |
|---|---|---|
| 浏览器打不开 | Windows 路由、端口 8088、Compose 状态 | 本文第 2、6 节 |
| API 健康但任务不启动 | `/ready`、GPU UUID、空闲显存、其他用户进程 | `agent/RUNBOOK.md` |
| 任务失败 | API 错误字段、任务日志、最后完成阶段 | `agent/RUNBOOK.md` |
| H200 重启 | Docker、GPU、Compose、health | `reference/RECOVERY.md` |
| 系统重装或 `/data3` 丢失 | NAS 快照校验、镜像加载、项目恢复 | `reference/RECOVERY.md` |
| 镜像损坏或平台变更 | 镜像 → Conda 包 → 依赖清单三级恢复 | `../rebuild/REBUILD_GUIDE.md` |

## 10. 下一步阅读

- 需要让 Agent 协助：读 [AGENT_GUIDE.md](AGENT_GUIDE.md)。
- 需要执行故障恢复：读 [reference/RECOVERY.md](reference/RECOVERY.md)。
- 需要优化耗时：读 [reference/PERFORMANCE.md](reference/PERFORMANCE.md)。
- 只想追溯迁移历史：读 [archive/README.md](archive/README.md)。

