# Agent Runbook — Quantitize Platform

先读 `CONTEXT.md`。命令默认在 `ssh H200` 后执行。每次操作重新检查实时状态，不把 2026-09-02 的快照当作资源现状。

## 1. 任务分类

| task_type | 默认权限 | 必读补充 | 完成条件 |
|---|---|---|---|
| `status_check` | 只读 | 无 | 服务、GPU、磁盘、任务状态有证据 |
| `task_diagnosis` | 只读 | 项目阶段日志 | 给出故障阶段、证据、建议，不实施修复 |
| `service_restart` | 修改两个 Compose 服务 | 本文第 4 节 | 无运行任务，两个容器 healthy |
| `system_backup` | 写 NAS | `../reference/RECOVERY.md` | 哈希和流测试通过，无 `.partial` |
| `output_archive` | 写 NAS；删除需额外门禁 | 性能文档的存储策略 | NAS 验证后才清本地 |
| `disaster_recovery` | 高风险恢复 | `../reference/RECOVERY.md` | 镜像、服务、runtime、golden task 全通过 |
| `environment_rebuild` | 构建镜像 | `../../rebuild/REBUILD_GUIDE.md` | runtime 与 golden task 回归通过 |
| `performance_change` | 修改代码/路径 | `../reference/PERFORMANCE.md` | 固定基线、A/B、指标和产物回归 |

## 2. 通用只读基线

```bash
date
hostname
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready || true
curl -fsS http://127.0.0.1:8088/health
df -h /data3 /data1
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,power.draw --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

GPU 结论至少基于：连续采样、显存、计算进程、进程用户和运行时长。

```bash
nvidia-smi dmon -s pucm -c 5
ps -o user,pid,etime,stat,cmd -p <gpu-pids>
```

## 3. 任务状态与诊断

```bash
TASK_ID='<task_id>'
curl -fsS "http://127.0.0.1:8000/api/tasks/$TASK_ID" | python3 -m json.tool
pgrep -af "$TASK_ID|05_runner.py|01_pt_eval.py|quantitize.py|03_onnx_eval.py|04_fpga_eval.py" || true
du -sh "data/output_data/$TASK_ID"
find "data/output_data/$TASK_ID" -type f -mmin -10 -print
ls -lah "data/output_data/$TASK_ID/logs"
```

判定：

- `completed`：八步 completed、error 为空、`has_zip=true`。
- `running`：runner/阶段进程存在，CPU/GPU 或文件时间持续变化。
- `waiting_gpu`：API 健康但 `/ready` 不通过；检查分配 UUID 和资源。
- `failed`：记录 `task_error`、`worker_error`、最后成功阶段和日志；不要自动删除现场。
- manifest 长时间不变不等于卡住；部分阶段只在完成时更新 manifest。

诊断请求不授权修复。先报告原因和最小修复方案。

## 4. 服务重启

门禁：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/tasks/<known-running-task> 2>/dev/null || true
pgrep -af '05_runner.py|pipeline/engine/quantitize.py' || true
```

存在运行任务时不要重启，除非用户明确授权中断该任务。

执行：

```bash
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

如需重建单个轻量 Web 镜像，必须明确说明；不要意外重建 21GB API 镜像。禁止 `down -v`。

## 5. GPU 分配

读取当前配置：

```bash
sed -n '1,80p' deploy/.env
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
```

修改 `GPU_DEVICE_ID` 前必须得到资源分配授权。使用 UUID，不依赖序号。Compose 内仅暴露一张卡，因此物理 GPU 3 可能在容器内显示为 GPU 0。

GPU 资源在任务启动检查后仍可能被其他用户抢占。运行期间继续监控显存；如果低于任务峰值，报告 OOM 风险，不停止其他用户进程。

## 6. NAS 门禁

```bash
findmnt -T /mnt/ywang-nas
df -h /mnt/ywang-nas
test -w /mnt/ywang-nas
```

挂载不存在时按 `../reference/RECOVERY.md` 操作；凭据只从 root-only 文件读取。NAS 当前不是持久挂载。

大文件流程：

```text
write <name>.partial
-> tool-specific stream test
-> SHA-256
-> atomic rename in same directory
-> record VERIFIED
```

失败时保留本地源，不把 `.partial` 当作备份。

## 7. output_data 归档与清理门禁

默认只允许 dry-run。候选必须是 `output_data` 直属的完整 task 目录，并满足：

- 不属于当前运行任务。
- 目录内所有文件连续 14 天无修改。
- 任务已完成；失败任务需完整保存现场，不自动删除。
- NAS 目标是正确挂载的 SMB，而不是空的本地挂载点。
- 归档校验通过并记录哈希。
- 本地再保留 7 天宽限期。

删除前解析真实绝对路径，确认直属于：

```text
/data3/ywang/quantitize-platform/data/output_data
```

NAS 不可用、空间不足、校验失败或状态不明时，删除数量必须为 0。

## 8. 系统快照验收

系统快照包含：项目（排除 `output_data`）、`shared_data`、API/Web 镜像、原始 Conda 环境包、环境清单、恢复文档。

```bash
cd '<snapshot>'
sha256sum -c SHA256SUMS
zstd -q -t quantitize-platform-no-output.tar.zst
zstd -q -t docker-images-api-web.tar.zst
gzip -t yolov8_env.tar.gz
find . -name '*.partial' -print
```

最后一条必须无输出。不得覆盖已有 `VERIFIED` 快照。

## 9. 变更后的最小验收

```bash
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose.resolved.yml
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

涉及运行环境或 pipeline 行为时，额外执行 runtime 验证和受控 golden task。涉及性能时，固定模型、数据文件清单、镜像 ID 和参数，并比较阶段时间、指标、最终文件清单及哈希。

## 10. 结果记录

```yaml
task_type:
started_at:
target:
authorization:
precheck:
changes: []
verification: []
artifacts: []
risks: []
remaining: []
status: DONE|BLOCKED|FAILED
```

