# Agent Runbook — Quantitize Platform

先读 `CONTEXT.md`。命令默认通过 `ssh H200` 在 `/data3/ywang/quantitize-platform` 执行。历史快照不能代替实时检查。

## 1. 先按任务选范围

| 请求 | 默认动作 | 继续阅读 |
|---|---|---|
| 服务/GPU/任务状态 | 只读取证 | 本文 §2–4 |
| 故障诊断 | 只读，给原因与建议，不自动修复 | 对应阶段日志、`../../pipeline/FLOW.md` |
| 启动/重建服务 | 先确认无运行任务，再按授权修改 | `../../deploy/README.md` |
| 运行量化 | 校验输入、GPU 和新任务目录 | `../../pipeline/FLOW.md` |
| 单通道/训练/预处理 | 只读分析或明确授权后改 | `../reference/MODEL_INPUT.md` |
| FPGA/21 类 | 核对 ABI、BIN、后处理 | `../reference/CLASS21_FPGA.md` |
| 性能变更 | 固定基线并 A/B | `../reference/PERFORMANCE.md` |
| 恢复/重建 | 高风险，逐级门禁 | `../reference/RECOVERY.md`、`../../rebuild/REBUILD_GUIDE.md` |
| 备份/清理 | 默认 dry-run | 本文 §7 |

## 2. 通用只读基线

```bash
date
hostname
cd /data3/ywang/quantitize-platform
git status --short
git rev-parse --short HEAD
docker compose -f deploy/docker-compose.yml ps
curl -sS http://127.0.0.1:8000/health
curl -sS -i http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8088/health
df -h /data3 /data1
pgrep -af '05_runner.py|01_pt_eval.py|02_onnx_eval.py|quantitize.py|check_certain_layer_multi.py' || true
```

`/health` 只表示服务存活；`/ready` 才表示 GPU 门禁通过。遇到 `GPU query failed`：

```bash
curl -sS http://127.0.0.1:8000/api/gpu
docker exec quantitize-platform-api nvidia-smi -L
nvidia-smi -L
docker inspect quantitize-platform-api --format '{{json .HostConfig.DeviceRequests}}'
```

宿主机正常而容器 NVML 失败通常需要评估重建容器，但诊断请求不授权重建。

## 3. GPU 分配与使用门禁

```bash
sed -n '1,80p' deploy/.env
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,power.draw --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi dmon -s pucm -c 5
ps -o user,pid,etime,stat,cmd -p <gpu-pids>
```

必须同时报告 UUID、显存、利用率、进程用户和持续样本。不要停止他人进程。修改 `GPU_DEVICE_ID` 必须有分配授权并使用 UUID；容器只暴露一张卡，物理 GPU 3 在容器内通常是 0。

## 4. 任务状态与诊断

```bash
TASK_ID='<task_id>'
curl -fsS "http://127.0.0.1:8000/api/tasks/$TASK_ID" | python3 -m json.tool
sed -n '1,260p' "data/output_data/$TASK_ID/manifest.json"
du -sh "data/output_data/$TASK_ID"
find "data/output_data/$TASK_ID" -type f -mmin -10 -print
ls -lah "data/output_data/$TASK_ID/logs"
```

- `completed`：八步 completed、error 为空、ZIP 存在且完整。
- `waiting_gpu`：API 健康、`/ready` 不通过，任务尚未启动。
- `running`：进程、日志或文件时间持续变化；manifest 有些阶段仅完成时更新。
- `failed`：保留现场，记录最后成功步骤、错误、归档状态；不要自动重跑或删除。

当前步骤名是 `fpga_input_roundtrip_eval`；旧文档/调用可能写 `fpga_eval`。

## 5. 服务启动或重建

门禁：

```bash
pgrep -af '05_runner.py|pipeline/engine/quantitize.py' || true
curl -sS http://127.0.0.1:8000/health
```

只有确认无运行任务且授权覆盖服务变更后：

```bash
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose.resolved.yml
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
curl -fsS http://127.0.0.1:8000/ready
```

代码通过 bind mount 热更新，但依赖、基础镜像和设备挂载变化需要重建镜像/容器。禁止 `docker compose down -v`。

## 6. 变更后的验收

文档或静态代码：检查链接、`git diff --check`、Compose 解析。涉及 Python 时再做 compile/smoke；涉及输入 ABI、量化或镜像时必须在获批 GPU 上运行独立回归任务。

```bash
git diff --check
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose.resolved.yml
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

不要覆盖历史完成任务。固定模型、数据清单哈希、Git 状态、镜像 ID、参数和 GPU UUID；比较 PT/ONNX/回环指标、阶段时间、产物清单与哈希。

## 7. NAS、备份与 output_data 清理

```bash
findmnt -T /mnt/ywang-nas
df -h /mnt/ywang-nas
test -w /mnt/ywang-nas
```

大文件：写 `<name>.partial` → 内容/流测试 → SHA-256 → 同目录原子改名 → 记录 VERIFIED。失败时保留本地源。

`output_data` 默认只允许 dry-run。删除必须满足：明确授权；不是当前任务；NAS 是真实 SMB 挂载；完整归档及哈希验证；保留宽限期；解析后的绝对路径是 `/data3/ywang/quantitize-platform/data/output_data` 的直属任务目录。状态不明、失败现场未归档或 NAS 异常时删除数量为 0。

## 8. 结束记录

```yaml
task_type:
started_at:
authorization:
precheck:
changes: []
verification: []
artifacts: []
risks: []
remaining: []
status: DONE|BLOCKED|FAILED
```
