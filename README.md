# 量化平台交接文档

这是文档总入口，面向第一次接触模型量化、不了解服务器和网络环境的同事。

## 谁应该读什么

| 读者 | 从哪里开始 | 用途 |
|---|---|---|
| 新接手同事 | [docs/HANDOFF.md](docs/HANDOFF.md) | 理解背景、服务器、网络、存储、平台架构和日常操作 |
| 使用 Agent 的同事 | 先读 HANDOFF，再读 [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) | 学会给 Agent 下达安全、可验证的任务 |
| 自动化 Agent | [AGENTS.md](AGENTS.md) | 获取强制阅读顺序、边界和当前事实来源 |
| 故障恢复人员 | [docs/reference/RECOVERY.md](docs/reference/RECOVERY.md) | H200 重启、系统重装或数据盘丢失后的恢复 |
| 环境重建人员 | [rebuild/REBUILD_GUIDE.md](rebuild/REBUILD_GUIDE.md) | 镜像不可用时从环境包或依赖清单重建 |
| 性能优化人员 | [docs/reference/PERFORMANCE.md](docs/reference/PERFORMANCE.md) | 理解 H200 性能瓶颈和优化路线 |

## 推荐阅读顺序

```text
README.md
  └─ docs/HANDOFF.md
       ├─ docs/AGENT_GUIDE.md
       ├─ docs/reference/RECOVERY.md
       ├─ docs/reference/PERFORMANCE.md
       └─ rebuild/REBUILD_GUIDE.md

AGENTS.md
  └─ docs/agent/CONTEXT.md
       └─ docs/agent/RUNBOOK.md
```

## 活跃文档与历史文档

- `README.md`、`AGENTS.md`、`docs/` 和 `rebuild/` 是当前有效文档。
- `docs/archive/` 只保存迁移过程、旧服务器说明和历史决策，不作为当前操作依据。
- `h200-worktree/` 是 H200 项目代码的本机镜像；其中的项目内说明用于开发，不是交接入口。
- 文档不保存 SSH、sudo 或 NAS 明文密码。

当前生产入口：<http://10.2.29.180:8088>。

