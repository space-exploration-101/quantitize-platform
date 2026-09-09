# Quantitize Platform Agent Context

> Current entry point for future Agents. Updated 2026-09-02 after successful H200 functional validation and NAS backup preparation.

## Current state

```yaml
production_host: H200
ssh: direct
proxy_jump: prohibited
project_root: /data3/ywang/quantitize-platform
service_state: healthy
api: http://10.2.29.180:8000
web: http://10.2.29.180:8088
golden_task: 20260902_065807_test_h200
golden_task_status: completed
nas_mount: /mnt/ywang-nas
nas_backup_root: /mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/H200
latest_verified_snapshot: snapshot-20260902-182606
output_data_backup_policy: excluded_from_system_backup
```

## Read in this order

1. `H200_RECOVERY.md` — canonical reboot, reinstall, restore, and verification runbook.
2. `BACKUP_STATUS.md` — latest immutable NAS snapshot, hashes, and validation result.
3. `h200-device.md` — host, SSH, network, GPU, Docker, and storage inventory.
4. `h200-migration.md` — historical migration execution record; completed through golden-task acceptance.
5. `rebuild/REBUILD_GUIDE.md` — rebuilding from Dockerfile, Conda pack, or dependency manifests.
6. `network.md` — Windows routes, Astrill bypass, SSH aliases, and NAS networking.
7. `server-migration.md` and `migration.md` — source-era historical context; do not treat old TODO fields as current state.

## Immediate operations

After an H200 reboot:

```bash
ssh H200
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

Before a quantization task, verify that the UUID in `deploy/.env` is assigned and sufficiently idle. This is a shared server; do not stop other users' processes.

## Security and scope

- Passwords are intentionally absent from all Agent documents.
- Do not route H200 through `wrs`.
- Do not alter global Docker, NVIDIA driver, kernel, APT, or automatic-upgrade policy without administrator approval.
- System backup excludes `data/output_data`; archive historical task output under a separate retention policy.
