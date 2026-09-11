# Agent Instructions — Quantitize Platform

## 开始顺序

1. 读 `docs/agent/CONTEXT.md`：当前事实、已验证边界和风险。
2. 读 `docs/agent/RUNBOOK.md`：按任务类型执行门禁和验收。
3. 只在需要时缩小范围：
   - 流水线与任务目录：`pipeline/FLOW.md`
   - 容器与网页服务：`deploy/README.md`
   - 输入/训练/灰度：`docs/reference/MODEL_INPUT.md`
   - 21 类 FPGA 接口：`docs/reference/CLASS21_FPGA.md`
   - 性能：`docs/reference/PERFORMANCE.md`
   - 恢复：`docs/reference/RECOVERY.md`
   - 重建：`rebuild/REBUILD_GUIDE.md`
   - 变更与历史测试：`docs/CHANGELOG.md`

不要全仓库搜索后再判断；先用以上入口缩小范围。历史任务结果只作为证据，不代表当前代码已通过。

## 强制边界

- 仅用 `ssh H200` 直连；禁止配置或使用 `wrs` 跳板。
- H200 是公用服务器。未经明确授权，不停止他人进程，不改全局 Docker、驱动、内核、APT 或升级策略。
- 默认只读。启动任务、重启服务、挂载 NAS、备份、删除前确认授权。
- 使用 GPU 前检查利用率、显存、计算进程、进程用户并持续采样；单次 `0%` 不代表空闲。
- 不保存或复述明文密码。
- `data/output_data` 不在系统备份中；验证独立归档后才可清理。
- 禁止 `docker compose down -v`、删除 APT/DPKG lock、宽泛通配符递归删除共享数据。

## 工作要求

- 开始时说明检查项、是否用 GPU、是否改变状态。
- 先收集实时证据，再给结论；区分当前代码、当前容器和历史任务。
- 变更可回滚；大文件用 `.partial`、校验后原子改名。
- 结束时报告实际变更、验证结果、未完成项和风险。
