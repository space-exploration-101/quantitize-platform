# H200 Quantitize Platform Recovery Guide

> Canonical disaster-recovery runbook for an automation Agent. Last updated: 2026-09-02. Never record NAS, sudo, or SSH passwords in this file.

## 1. Machine-readable state

```yaml
service: quantitize-platform
state: PRODUCTION_VALIDATED
host:
  ssh_alias: H200
  address: 10.2.29.180
  user: ywang
  hostname: user-Rack-Server
  os: Ubuntu 22.04.5 LTS
  kernel_observed: 6.8.0-90-generic
  architecture: x86_64
  gpu: 4 x NVIDIA H200 NVL
  driver_observed: 580.82.07
paths:
  project: /data3/ywang/quantitize-platform
  output_data: /data3/ywang/quantitize-platform/data/output_data
  nas_mount: /mnt/ywang-nas
  nas_backup_root: /mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/H200
services:
  api: http://10.2.29.180:8000
  web: http://10.2.29.180:8088
  compose_file: deploy/docker-compose.yml
  api_image: quantitize-platform-api:latest
  api_image_id: sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
  web_image: quantitize-platform-web:latest
  web_image_id: sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
gpu_allocation:
  physical_gpu_uuid: GPU-1610284d-3afe-6e14-b8cb-bbbe40a1f28b
  compose_env_file: deploy/.env
validation:
  golden_task: 20260902_065807_test_h200
  status: completed
  completed_at: 2026-09-02T09:37:32Z
  all_eight_steps_completed: true
  output_bundle_created: true
backup_policy:
  output_data_included: false
  secrets_in_documentation: false
latest_verified_snapshot:
  name: snapshot-20260902-182606
  status: VERIFIED
  project_sha256: 5d40756c8a3a5e333a58ddce881e56bd8c680b643bfde3ba1d538347c9e42d9c
  docker_images_sha256: 1ce12391b0b84abf4de369bd8f2a43f704cf9f131514c7e36fb25c049bad3262
  conda_environment_sha256: bb66795409a56195830641d24620b37f7d3617bc4a424517d2885545d1ac7251
```

## 2. Backup layout and recovery priority

Each immutable snapshot below the NAS backup root contains:

```text
snapshot-YYYYMMDD-HHMMSS/
├── H200_RECOVERY.md
├── BACKUP_CONTENTS.txt
├── SHA256SUMS
├── quantitize-platform-no-output.tar.zst
├── docker-images-api-web.tar.zst
├── yolov8_env.tar.gz
└── inventory/
    ├── host.txt
    ├── docker.txt
    ├── compose.resolved.yml
    ├── api-image-inspect.json
    ├── web-image-inspect.json
    └── golden-task.json
```

`quantitize-platform-no-output.tar.zst` contains source code, Compose files, Dockerfiles, rebuild manifests, calibration/test shared data, and documentation. It deliberately excludes every file below `data/output_data`.

Recovery priority:

1. Load `docker-images-api-web.tar.zst`; this is the fastest and most exact route.
2. If the image archive is unavailable, rebuild the API image from `yolov8_env.tar.gz` plus `rebuild/snapshots/Dockerfile.api.pinned-amd64`.
3. If both binary assets are unavailable, rebuild from `rebuild/manifests/`; this requires package repositories and full regression testing.

Always verify the snapshot before restoring:

```bash
cd '<snapshot-directory>'
sha256sum -c SHA256SUMS
zstd -q -t quantitize-platform-no-output.tar.zst
zstd -q -t docker-images-api-web.tar.zst
gzip -t yolov8_env.tar.gz
```

Do not continue if any check fails.

## 3. Recovery after an ordinary reboot

An ordinary reboot should not require archive restoration. The project and shared data are local under `/data3`, and both containers use `restart: unless-stopped`. Connect directly without `wrs`:

```bash
ssh H200
findmnt /mnt/ywang-nas || true
systemctl is-active docker
nvidia-smi -L
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

Acceptance:

- Docker is active.
- The expected H200 GPUs are visible.
- Both containers report `healthy`.
- Both health requests return success.
- `deploy/.env` points to a currently available GPU UUID.

The current NAS mount is not recorded in `/etc/fstab`; this does not prevent the service from starting. Remount NAS manually using section 4.2 before backup or disaster restore work.

Do not run `docker compose down -v`; the deployment uses host bind mounts and the command is unnecessary.

## 4. Recovery after OS reinstall or loss of `/data3`

### 4.1 Host gate

The host distribution may change, but the required architecture is Linux `x86_64`. Install and validate:

- NVIDIA driver compatible with H200 and the CUDA 12.8 runtime in the saved image.
- Docker Engine and Docker Compose v2.
- NVIDIA Container Toolkit configured for Docker.
- `cifs-utils`, `zstd`, `tar`, `gzip`, and `curl`.

Run:

```bash
uname -m
nvidia-smi
docker version
docker compose version
nvidia-ctk --version
docker info | sed -n '/Runtimes/,+2p'
df -h /data3 /var/lib/docker
```

Do not change Docker's global data root, kernel, driver, or unattended-upgrade policy on this shared server without administrator approval.

### 4.2 Mount NAS

Use a root-only credentials file; enter the secret interactively:

```bash
sudo install -d -m 0700 -o ywang -g ywang /mnt/ywang-nas
sudo install -m 0600 -o root -g root /dev/null /etc/samba/credentials-ywang
sudoedit /etc/samba/credentials-ywang
```

Credentials file format:

```ini
username=orange
password=<enter interactively>
```

Mount:

```bash
sudo mount -t cifs //10.2.26.26/902_data /mnt/ywang-nas \
  -o credentials=/etc/samba/credentials-ywang,vers=3.0,iocharset=utf8,uid=$(id -u ywang),gid=$(id -g ywang),file_mode=0600,dir_mode=0700,nosuid,nodev,noexec
findmnt /mnt/ywang-nas
```

### 4.3 Restore project and images

Select the newest fully verified snapshot; never restore from a `.partial` file:

```bash
SNAPSHOT='/mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/H200/<verified-snapshot>'
PROJECT_PARENT='/data3/ywang'

mkdir -p "$PROJECT_PARENT/quantitize-platform"
zstd -dc "$SNAPSHOT/quantitize-platform-no-output.tar.zst" \
  | tar -xpf - -C "$PROJECT_PARENT/quantitize-platform"
mkdir -p "$PROJECT_PARENT/quantitize-platform/data/output_data"

zstd -dc "$SNAPSHOT/docker-images-api-web.tar.zst" | docker load
docker image inspect quantitize-platform-api:latest quantitize-platform-web:latest
```

The archive stores the contents of the project root, not an extra enclosing directory.

Restore ownership if the numeric account changed after reinstall:

```bash
sudo chown -R ywang:ywang /data3/ywang/quantitize-platform
```

Do not restore old `output_data` as part of service recovery. Create it empty, then restore selected historical jobs separately only when explicitly required.

### 4.4 GPU allocation and startup

List stable UUIDs:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
```

Edit `deploy/.env` only if the assigned GPU changed. Keep a stable UUID rather than a numeric index:

```ini
GPU_DEVICE_ID=<approved-GPU-or-MIG-UUID>
GPU_READINESS_REQUIRED=1
GPU_MIN_FREE_MIB=32768
GPU_MAX_UTILIZATION_PERCENT=20
GPU_READY_POLL_SECONDS=30
WEB_DEMO_MODE=0
```

Then:

```bash
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose.resolved.yml
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

## 5. Runtime and functional verification

Container runtime gate:

```bash
cd /data3/ywang/quantitize-platform
docker run --rm --gpus all \
  -v "$PWD/rebuild:/rebuild:ro" \
  quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python /rebuild/verify_runtime.py
```

Expected evidence includes CUDA availability, `sm_90`, and ONNX Runtime `CUDAExecutionProvider`.

After basic health checks, run one controlled golden task only when a shared GPU is approved. The accepted H200 baseline is task `20260902_065807_test_h200`, using calibration dataset `aituosha_moon_500`, test dataset `aituosha_moon_840`, and preprocessing `grayscale_r_channel`. Acceptance requires all eight manifest steps completed, no worker/task error, and a generated bundle ZIP.

## 6. Agent rules and progress record

- Never store plaintext passwords or sudo input in project files, NAS documentation, command arguments, logs, or Git.
- Never delete or replace a known-good snapshot while creating a new one.
- Write large files as `.partial`, verify them, then rename atomically within the same NAS directory.
- Never infer that a GPU is free from a single utilization sample; inspect memory and processes and coordinate with other users.
- Backups exclude `data/output_data` by policy. Historical-output archiving is a separate lifecycle job.
- Preserve `rebuild/` because it contains pinned Dockerfiles, Conda/Pip manifests, package inventories, checksums, and verification programs.

Use this progress schema during recovery:

```yaml
recovery:
  snapshot_selected: TODO
  checksums_verified: TODO
  host_gate: TODO
  nas_mounted: TODO
  project_restored: TODO
  images_loaded_or_rebuilt: TODO
  gpu_uuid_selected: TODO
  compose_started: TODO
  health_checks: TODO
  runtime_verification: TODO
  golden_task: TODO
  final_status: TODO
  notes: []
```
