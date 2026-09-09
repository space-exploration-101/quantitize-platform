# Quantitize Platform Migration to H200

> Executable, progress-recording guide for an Agent. Target: direct SSH alias `H200`; never route through `wrs`. Secrets must be entered interactively and must not be written into this document, Git, shell history, `.env`, or command arguments.

## 1. Machine-readable state

```yaml
project: quantitize-platform
document_status: active
last_updated: 2026-09-02

source_backup:
  nas: 10.2.26.26
  protocol: SMB
  share: 902_data
  backup_subpath: 3-个人/16-王宇航/3-Project/2-DRO/模型量化/quantitize-platform-backup-20260901
  password_storage: prohibited

target:
  ssh_alias: H200
  connection: direct
  proxy_jump: prohibited
  address: 10.2.29.180
  user: ywang
  host_os: Ubuntu 22.04.5 LTS
  architecture: x86_64
  gpu: 4 x NVIDIA H200 NVL
  driver: 580.82.07
  user_data_root: /data3/ywang
  project_root: /data3/ywang/quantitize-platform
  staging_root: /data3/ywang/quantitize-migration

strategy:
  selected: level_1_load_saved_images
  fallback_1: level_2_rebuild_from_conda_pack
  fallback_2: level_3_rebuild_from_manifests

progress:
  H0_direct_ssh_and_inventory: DONE
  H1_host_permissions_and_tools: DONE
  H2_copy_and_verify_backup: DONE
  H3_restore_code_and_data: DONE
  H4_load_or_build_images: DONE
  H5_gpu_runtime_gate: DONE
  H6_start_compose: DONE
  H7_acceptance_and_golden_task: DONE

current_phase: COMPLETE
next_task: maintain_verified_backups_and_output_retention
blockers: []
```

### Executed migration result

```yaml
executed_at: 2026-09-02
transfer:
  mode: windows_relay
  local_staging: false
  bytes: 55600909362
  destination: /data3/ywang/quantitize-migration
  sha256_all_passed: true
  archive_stream_tests_all_passed: true
host_changes:
  apt_packages_installed: []
  cifs_utils_installed: true
  ywang_added_to_docker_group: true
  docker_access_model: docker_group
restore:
  project_root: /data3/ywang/quantitize-platform
  expanded_size_observed: 56 GiB
  shared_files: 2185
  shared_symlinks: 0
  historical_task_directories: 9
  historical_other_entries:
    - .gitkeep
  unreadable_data_files: 0
  legacy_wrs_mount_removed: true
  original_compose_backup: deploy/docker-compose.yml.pre-h200
images:
  api_id: sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
  web_id: sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
  ids_match_source: true
validation:
  gpu_count_visible_in_container: 4
  gpu_model: NVIDIA H200 NVL
  cuda_available: true
  torch_sm90: true
  onnx_cuda_provider: true
  verify_runtime: PASS
  verify_runner: PASS
  verify_patches: PASS
services:
  api: healthy
  web: healthy
  api_url: http://10.2.29.180:8000
  web_url: http://10.2.29.180:8088
  local_http_status: 200
  observed_continuous_uptime: 10 hours
golden_acceptance:
  golden_quantization_task: DONE
  task_id: 20260902_065807_test_h200
  completed_at: 2026-09-02T09:37:32Z
  all_steps_completed: true
  bundle_created: true
  worker_error: null
  task_error: null
post_migration_backup:
  status: VERIFIED
  snapshot: snapshot-20260902-182606
  nas_subpath: 0-项目/13-专项/4-代码/量化平台/H200/snapshot-20260902-182606
  output_data_included: false
  project_sha256: 5d40756c8a3a5e333a58ddce881e56bd8c680b643bfde3ba1d538347c9e42d9c
  docker_images_sha256: 1ce12391b0b84abf4de369bd8f2a43f704cf9f131514c7e36fb25c049bad3262
  conda_environment_sha256: bb66795409a56195830641d24620b37f7d3617bc4a424517d2885545d1ac7251
```

Allowed status values are `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, and `SKIPPED`. An Agent may mark a task `DONE` only after executing its verification and recording the result. At most one task should be `IN_PROGRESS`.

## 2. Known assets and immutable baselines

NAS directory:

```text
//10.2.26.26/902_data/3-个人/16-王宇航/3-Project/2-DRO/模型量化/quantitize-platform-backup-20260901
```

| Asset | Purpose | SHA-256 |
|---|---|---|
| `docker-images.tar.zst` | Preferred API and Web images | `1ce12391b0b84abf4de369bd8f2a43f704cf9f131514c7e36fb25c049bad3262` |
| `quantitize-platform-code.tar.zst` | Code, Compose and Dockerfiles | `84c180d8ea86089134c244f7ee4d2e273aedad59900d11d25da7f317ce2a4f8f` |
| `shared-data.tar.zst` | Dereferenced calibration/test data | `130a9add69b643a7752993e00d872c56dad68020c9108a893787077f31b0b33c` |
| `output-data.tar.zst` | Historical tasks and outputs | `0350f93c7db687ab74043cc7923dc917bece1bad814f55c9ce8fa4f17880f0aa` |
| `yolov8_env.tar.gz` | Level-2 API image rebuild input | `bb66795409a56195830641d24620b37f7d3617bc4a424517d2885545d1ac7251` |

Expected image IDs:

```text
quantitize-platform-api:latest sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
quantitize-platform-web:latest sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
```

The complete reconstruction evidence is stored locally on the Windows control machine at `E:\codex\quantitize\rebuild`. It was created after the formal NAS code archive and must therefore be copied separately to H200 before runtime verification or fallback rebuilding.

## 3. H0 — direct access and host gate

Status: `DONE` on 2026-09-01.

Control-machine verification:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 H200 "whoami; hostname"
ssh -G H200 | Select-String '^(hostname|user|identityfile|proxyjump) '
```

Acceptance evidence:

- Direct key authentication works as `ywang` on `user-Rack-Server`.
- `ssh -G H200` resolves to `10.2.29.180`, user `ywang`, key `~/.ssh/id_ed25519`, and no `proxyjump`.
- Windows persistent route `10.2.29.0/24 via 192.168.31.1 ifIndex 19` exists.
- Host is x86_64 Ubuntu 22.04.5 with 4 H200 NVL GPUs, driver 580.82.07, compute capability 9.0.
- `/data3` had 2.2 TiB free; ports 8000 and 8088 were unused at observation time.

Re-run before migration:

```bash
uname -m
nvidia-smi
df -h / /data3
ss -ltn | grep -E ':(8000|8088)([[:space:]]|$)' || true
```

## 4. H1 — grant access and install missing host tools

### H1-T1 Docker access

An administrator runs interactively on H200:

```bash
sudo usermod -aG docker ywang
```

Disconnect every SSH session and reconnect so group membership refreshes:

```powershell
ssh H200
```

Verify without sudo:

```bash
id
docker version
docker info --format 'root={{.DockerRootDir}} runtimes={{json .Runtimes}}'
docker compose version
```

Acceptance:

- `id` includes group `docker`.
- Docker client and server both report versions.
- Docker info is not denied by `/var/run/docker.sock`.
- Docker runtime list supports NVIDIA after H1-T2.

Security note: Docker-group membership is effectively root-equivalent. If policy forbids it, keep `ywang` out of the group and consistently use interactive `sudo docker ...`; record that decision.

### H1-T2 CIFS and NVIDIA runtime

Install the missing NAS mount helper:

```bash
sudo apt-get update
sudo apt-get install -y cifs-utils
```

The NVIDIA Container Toolkit is already present (`1.17.8`), but Docker integration must still be verified. If `docker info` does not show NVIDIA support, run:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
command -v mount.cifs
nvidia-ctk --version
systemctl is-active docker
docker info | sed -n '/Runtimes/,+2p'
```

`docker buildx` is currently missing. It is not needed for Level 1. Before Level 2, install the Buildx package appropriate to the configured Docker repository and verify `docker buildx version`; do not mix Docker CE and distribution packages blindly.

Progress record:

```yaml
task: H1
status: DONE
started_at: 2026-09-01
completed_at: 2026-09-02
docker_access: interactive_sudo
docker_access_current: docker_group
cifs_utils: installed_after_migration
nvidia_runtime: verified_by_docker_run_gpus_all
buildx:
verification_output: four H200 GPUs visible from CUDA 12.8.1 container
notes: Windows relay selected to avoid modifying the shared host; ywang was not added to the docker group.
```

## 5. H2 — copy backup directly from NAS

### H2-T1 Mount NAS read-only

Create a root-only credentials file interactively. Do not put the password in a shell command:

```bash
sudo install -d -m 0755 /mnt/quantitize-nas
sudo install -m 0600 /dev/null /root/.quantitize-nas-credentials
sudoedit /root/.quantitize-nas-credentials
```

Enter only:

```text
username=orange
password=<enter interactively from the user's secret>
```

Mount read-only:

```bash
sudo mount -t cifs //10.2.26.26/902_data /mnt/quantitize-nas \
  -o credentials=/root/.quantitize-nas-credentials,vers=3.0,iocharset=utf8,ro
findmnt /mnt/quantitize-nas
```

Set paths without using broad system variables:

```bash
NAS_BACKUP='/mnt/quantitize-nas/3-个人/16-王宇航/3-Project/2-DRO/模型量化/quantitize-platform-backup-20260901'
MIGRATION_STAGING='/data3/ywang/quantitize-migration'
install -d -m 0750 "$MIGRATION_STAGING"
ls -lah "$NAS_BACKUP"
```

### H2-T2 Copy with restart support

```bash
rsync -ah --info=progress2 --partial "$NAS_BACKUP/" "$MIGRATION_STAGING/"
```

If only the minimum runnable system is needed first, copy `docker-images.tar.zst`, `quantitize-platform-code.tar.zst`, `shared-data.tar.zst`, and `SHA256SUMS`; defer historical output and the environment archive. Do not mark the omitted files as checksum failures.

### H2-T3 Verify copied bytes

```bash
cd /data3/ywang/quantitize-migration
sha256sum -c SHA256SUMS
zstd -q -t docker-images.tar.zst
zstd -q -t quantitize-platform-code.tar.zst
zstd -q -t shared-data.tar.zst
zstd -q -t output-data.tar.zst
gzip -t yolov8_env.tar.gz
```

Acceptance: every copied file reports `OK`, and each copied archive test exits 0. If a transfer was interrupted, resume with `rsync` and repeat all verification for that file.

Progress record:

```yaml
task: H2
status: DONE
transfer_mode: windows_relay
files_copied:
  - docker-images.tar.zst
  - quantitize-platform-code.tar.zst
  - shared-data.tar.zst
  - output-data.tar.zst
  - yolov8_env.tar.gz
  - SHA256SUMS
  - server-migration.md
sha256_result: all five baseline assets OK
archive_test_result: all zstd and gzip streams OK
notes: 55,600,909,397 bytes including documentation were relayed directly from NAS UNC to H200 without local disk staging.
```

## 6. H3 — restore code and host data under `/data3`

```bash
MIGRATION_STAGING='/data3/ywang/quantitize-migration'
PROJECT_ROOT='/data3/ywang/quantitize-platform'
install -d -m 0750 "$PROJECT_ROOT"
zstd -dc "$MIGRATION_STAGING/quantitize-platform-code.tar.zst" | tar -xf - -C "$PROJECT_ROOT"
install -d -m 0750 "$PROJECT_ROOT/data/shared_data" "$PROJECT_ROOT/data/output_data"
zstd -dc "$MIGRATION_STAGING/shared-data.tar.zst" | tar -xf - -C "$PROJECT_ROOT"
zstd -dc "$MIGRATION_STAGING/output-data.tar.zst" | tar -xf - -C "$PROJECT_ROOT"
```

Historical output is optional for first startup. If deferred, leave `data/output_data` present and empty.

Verify archive layout before continuing:

```bash
test -f "$PROJECT_ROOT/deploy/docker-compose.yml"
test -f "$PROJECT_ROOT/deploy/Dockerfile.api"
test -f "$PROJECT_ROOT/apps/api/app.py"
test -f "$PROJECT_ROOT/apps/web/app.py"
test -f "$PROJECT_ROOT/pipeline/verify_runner.py"
find "$PROJECT_ROOT/data/shared_data" -type l -print
find "$PROJECT_ROOT/data/shared_data" -type f | wc -l
find "$PROJECT_ROOT/data/output_data" -mindepth 1 -maxdepth 1 -type d | wc -l
```

The symlink check must print nothing. The verified archive contains 9 top-level historical task directories plus `.gitkeep`; this corrects the earlier planning estimate of 10 task directories.

Before Compose validation, edit `$PROJECT_ROOT/deploy/docker-compose.yml` and remove this legacy source-host bind mount:

```yaml
- /home/rs/wrs/onnxviewe/quantitize/shared_data:/home/rs/wrs/onnxviewe/quantitize/shared_data:ro
```

Keep the relative code/data mounts. Their host paths resolve below `/data3/ywang/quantitize-platform`; the fixed container-side paths may remain unchanged. Then run:

```bash
cd "$PROJECT_ROOT"
docker compose -f deploy/docker-compose.yml config > /tmp/quantitize-compose.resolved.yml
grep -F '/home/rs/wrs/onnxviewe/quantitize/shared_data' /tmp/quantitize-compose.resolved.yml && exit 1 || true
```

Acceptance: required files exist, data is readable, no legacy host dependency remains, and Compose config exits 0.

Copy the locally retained rebuild evidence from the Windows control machine after the project root exists:

```powershell
scp -r E:\codex\quantitize\rebuild H200:/data3/ywang/quantitize-platform/
```

Verify on H200:

```bash
test -f /data3/ywang/quantitize-platform/rebuild/REBUILD_GUIDE.md
test -f /data3/ywang/quantitize-platform/rebuild/rebuild-manifest.yml
test -f /data3/ywang/quantitize-platform/rebuild/verify_runtime.py
cd /data3/ywang/quantitize-platform/rebuild && sha256sum -c SHA256SUMS
```

## 7. H4 — load images; rebuild only on failure

### Preferred Level 1

```bash
cd /data3/ywang/quantitize-migration
zstd -dc docker-images.tar.zst | docker load
docker image inspect quantitize-platform-api:latest --format '{{.Id}} {{.Architecture}} {{.Size}}'
docker image inspect quantitize-platform-web:latest --format '{{.Id}} {{.Architecture}} {{.Size}}'
```

Acceptance: architecture is `amd64` and IDs match the immutable baselines in section 2.

### Fallback Level 2

Use only if image loading fails or the image intentionally must be rebuilt. Read `rebuild/REBUILD_GUIDE.md` first. Ensure Buildx is installed, place the verified `yolov8_env.tar.gz` where the pinned Dockerfile expects it, then build with `rebuild/snapshots/Dockerfile.api.pinned-amd64`. This level should reproduce behavior but is not guaranteed to reproduce the same image ID.

### Disaster recovery Level 3

Use the Conda/Pip manifests only if both the saved image and conda-pack archive are unavailable. It requires external package repositories and full regression testing. The source environment's duplicate NumPy metadata must not be copied into a clean rebuild; runtime import should resolve to NumPy 2.0.1.

## 8. H5 — H200 container runtime gate

Run before Compose startup:

```bash
PROJECT_ROOT='/data3/ywang/quantitize-platform'
docker run --rm --gpus all quantitize-platform-api:latest nvidia-smi -L
docker run --rm --gpus all \
  -v "$PROJECT_ROOT/rebuild:/rebuild:ro" \
  quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python /rebuild/verify_runtime.py
```

Acceptance:

- Four expected H200 GPUs are visible unless GPU allocation policy intentionally restricts them.
- `cuda_available=true`.
- PyTorch architecture list contains `sm_90`.
- ONNX Runtime providers contain `CUDAExecutionProvider`.
- NumPy, OpenCV, Ultralytics, FastAPI, and Uvicorn imports pass.

Also run project checks from the restored project context according to its Dockerfile working directory:

```bash
cd "$PROJECT_ROOT"
docker compose -f deploy/docker-compose.yml run --rm api \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py
docker compose -f deploy/docker-compose.yml run --rm api \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_patches.py
```

Do not continue merely because `nvidia-smi` passes; Python CUDA and ONNX Runtime are separate gates.

## 9. H6 — start containers

```bash
cd /data3/ywang/quantitize-platform
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs --tail=200
```

Verify:

```bash
curl -fsS http://127.0.0.1:8000/ >/dev/null
curl -fsS http://127.0.0.1:8088/ >/dev/null
docker compose -f deploy/docker-compose.yml ps
```

If the API health endpoint differs, use the endpoint defined in Compose or application routes and record it. Acceptance requires both services running/healthy, no restart loop, no bind-mount permission error, and successful local HTTP checks.

## 10. H7 — functional acceptance

Status: `DONE` on 2026-09-02. A fixed H200 golden task completed every pipeline stage and produced the final bundle.

Record at minimum:

```yaml
task: H7
status: DONE
started_at: 2026-09-02T07:53:20Z
completed_at: 2026-09-02T09:37:32Z
source_model_sha256:
calibration_dataset:
test_dataset:
parameters:
gpu_used:
container_image_ids:
task_id: 20260902_065807_test_h200
exit_status: completed
key_metrics:
output_manifest: data/output_data/20260902_065807_test_h200/manifest.json
output_sha256:
comparison_to_source:
notes:
  - All eight stages completed.
  - worker_error and task_error were null.
  - Final quantized bundle ZIP was generated.
```

Migration is `DONE` only when the task completes, expected artifacts exist, logs contain no CUDA/ORT/permission failures, and results satisfy the agreed source tolerance. Import tests alone are insufficient.

## 11. Cleanup and rollback rules

- Keep NAS mounted read-only during migration.
- H7 is accepted. Keep staging archives until the post-migration NAS snapshot and its checksums are verified.
- Do not modify or delete the source project on `wrs`.
- Do not change Docker's global `data-root` on this shared H200 server without administrator approval; that affects all containers.
- If startup fails, collect `docker compose ps`, resolved Compose config, image IDs, mounts, GPU output, and logs before changing state.
- Before rollback, back up any new `/data3/ywang/quantitize-platform/data/output_data` content to NAS.
- Unmount NAS when finished: `sudo umount /mnt/quantitize-nas`. Retain or securely delete the root-only credential file according to administrator policy.

## 12. Next Agent action

Migration and functional acceptance are complete. Use `H200_RECOVERY.md` for reboot or disaster recovery. Maintain verified NAS snapshots, keep `data/output_data` under a separate retention policy, and do not repeat migration phases unless the host or storage is lost.
