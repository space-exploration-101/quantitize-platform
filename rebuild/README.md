# 环境重建资料索引

首次接手先读 `../README.md`、`../docs/agent/CONTEXT.md` 和 `../docs/agent/RUNBOOK.md`。只有镜像丢失、硬件变化或依赖审计时再读 [REBUILD_GUIDE.md](REBUILD_GUIDE.md)。

## 当前与历史的边界

- 当前 API 使用 gray1 单通道基础镜像和 Python 3.12 runtime，构建入口是 `../deploy/Dockerfile.api`。
- 本目录的大多数 manifests、pinned Dockerfile 和 `yolov8_env.tar.gz` 来自 2026-09-01 的旧三通道 Python 3.9 环境。
- 历史资料能恢复 pre-gray1 底座，但不能单独重建 2026-09-11 当前 gray1 环境；恢复后必须重新固化运行时来源并完成单通道回归。

## 内容

```text
REBUILD_GUIDE.md        当前选择路径、门禁和验收
verify_host.sh          主机只读检查
verify_runtime.py       历史 runtime 验证脚本，使用前核对解释器和期望值
manifests/              旧环境依赖与审计清单
snapshots/              旧 API/Web Dockerfile 和 Compose
rebuild-manifest.yml    旧资产索引
SHA256SUMS              仓库内重建资料校验
```

已验证二进制快照记录在 `../docs/reference/RECOVERY.md`。系统快照不含 `data/output_data`，本目录不保存密码。
