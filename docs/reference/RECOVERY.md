# H200 量化平台恢复手册

当前事实先读 `../agent/CONTEXT.md`。恢复会改服务、镜像或文件，必须明确授权；共享机上不顺带修改全局 Docker、驱动、内核或 APT。

## 1. 恢复级别

| 场景 | 操作 |
|---|---|
| 普通重启 | 检查宿主机 GPU/Docker，Compose `up -d --no-build` |
| 容器丢失、项目仍在 | 用当前 Dockerfile 重建，或加载已核验镜像后做版本迁移 |
| 项目丢失 | 从 NAS 校验并解包系统快照，再应用 2026-09-02 之后的 Git/备份 |
| 系统重装 | 管理员先恢复驱动、Docker、NVIDIA Toolkit、CIFS，再恢复项目 |

## 2. 普通重启

```bash
ssh H200
systemctl is-active docker
nvidia-smi -L
cd /data3/ywang/quantitize-platform
pgrep -af '05_runner.py|pipeline/engine/quantitize.py' || true
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -sS -i http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8088/health
```

两个 health 成功但 `/ready` 503 时先诊断 GPU/NVML，不要因此重建整个平台。

## 3. 已记录快照及限制

NAS 位置：

```text
//10.2.26.26/902_data/0-项目/13-专项/4-代码/量化平台/H200/snapshot-20260902-182606
```

该快照已验证，但生成于单通道 gray1 切换之前：它恢复的是旧三通道代码、旧 API/Web 镜像和 Conda 环境，不是 2026-09-11 当前版本。只能作为灾难恢复底座；恢复后还必须取得当前 Git 工作树/提交、重建 gray1 API，并重新做单通道回归。

快照排除 `data/output_data`；历史任务必须从独立归档恢复。

## 4. NAS 和快照校验

凭据只允许交互写入 root-only 文件，不写文档。挂载后：

```bash
findmnt -T /mnt/ywang-nas
SNAPSHOT='/mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/H200/snapshot-20260902-182606'
cd "$SNAPSHOT"
sha256sum -c SHA256SUMS
zstd -q -t quantitize-platform-no-output.tar.zst
zstd -q -t docker-images-api-web.tar.zst
gzip -t yolov8_env.tar.gz
find . -name '*.partial' -print
```

任一失败即停止；最后一条必须无输出。已记录详细清单见 `backup/BACKUP_STATUS.md`。

## 5. 恢复项目

目标目录存在时不要覆盖。先保留现场，选择新目录或在明确授权后重命名。恢复到确认过的空目录：

```bash
PROJECT='/data3/ywang/quantitize-platform'
mkdir -p "$PROJECT"
zstd -dc "$SNAPSHOT/quantitize-platform-no-output.tar.zst" | tar -xpf - -C "$PROJECT"
mkdir -p "$PROJECT/data/output_data"
```

加载快照镜像：

```bash
zstd -dc "$SNAPSHOT/docker-images-api-web.tar.zst" | docker load
docker image inspect quantitize-platform-api:latest quantitize-platform-web:latest
```

注意：这些是 pre-gray1 镜像。若目标是恢复当前版本，继续按 `../../rebuild/REBUILD_GUIDE.md` 的当前 gray1 路径构建，而不是直接投产旧镜像。

## 6. GPU 配置和验收

`deploy/.env` 使用获批 UUID。启动后检查宿主机与容器 GPU、两个 health、`/ready`、实际 Python/Ultralytics/ORT 来源。最后在获批 GPU 上运行新的单通道 21 类 golden task。

恢复完成标准不是“容器能启动”，而是：输入 ABI `[1,1,1280,1280]`；PT 与 ONNX/回环指标等价；逐层和合并 BIN 成功；ZIP 完整；产物清单与接口一致。当前代码尚无满足这些条件的单通道 golden task。
