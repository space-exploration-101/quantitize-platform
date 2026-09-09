# Quantitize Platform 跨服务器迁移执行文档

> 用途：指导 Agent 从已完成的 NAS 备份开始，在另一台 GPU 服务器上恢复 Docker 镜像、代码、共享数据和历史任务，并启动、验证容器。
>
> 执行规则：Agent 必须先读取“当前恢复点”，再核对实际状态。每完成一项，更新状态、时间、验证输出和备注。只有验收条件全部通过，任务才能标记为 `DONE`。

## 1. 机器可读状态

```yaml
project: quantitize-platform
document_status: active
last_updated: 2026-09-01

source_host:
  ssh_alias: wrs
  address: 10.2.26.132
  user: rs
  os: Ubuntu 24.04.3 LTS
  architecture: amd64
  project_root: /home/rs/wrs/onnxviewe/quantitize-platform
  environment_archive: /home/rs/wrs/onnxviewe/yolov8_env.tar.gz

nas:
  address: 10.2.26.26
  protocol: SMB
  share: 902_data
  username: orange
  password_storage: never_store_in_this_document
  backup_subpath: 3-个人/16-王宇航/3-Project/2-DRO/模型量化/quantitize-platform-backup-20260901
  windows_unc: '\\10.2.26.26\902_data\3-个人\16-王宇航\3-Project\2-DRO\模型量化\quantitize-platform-backup-20260901'

target_host:
  address: REQUIRED
  ssh_user: REQUIRED
  gpu: REQUIRED
  os: Ubuntu 24.04 recommended
  architecture: amd64 required
  project_root: /home/rs/wrs/onnxviewe/quantitize-platform
  staging_root: /srv/quantitize-migration

current_phase: D0
next_task: D0-T1
blocking_input:
  - target_host.address
  - target_host.ssh_user
```

状态取值：

- `TODO`：未开始。
- `IN_PROGRESS`：正在执行；同一时间最多一个任务处于此状态。
- `BLOCKED`：缺少权限、参数或外部条件；记录解除条件。
- `DONE`：操作及全部验证均通过。
- `SKIPPED`：经用户确认不需要执行；记录原因。

## 2. 当前备份结果

备份目录：

```text
\\10.2.26.26\902_data\3-个人\16-王宇航\3-Project\2-DRO\模型量化\quantitize-platform-backup-20260901
```

| 文件 | 字节数 | 约合 GiB | 内容 | 是否启动必需 |
|---|---:|---:|---|---|
| `docker-images.tar.zst` | 9,671,368,776 | 9.007 | API + Web Docker 镜像 | 是 |
| `quantitize-platform-code.tar.zst` | 138,679 | 0.000 | 平台代码、Compose、Dockerfile、文档 | 是 |
| `shared-data.tar.zst` | 10,659,895,315 | 9.928 | 已解引用的共享校准集和测试集 | 是 |
| `output-data.tar.zst` | 25,974,617,116 | 24.191 | 10 个现有任务及历史产物 | 否，但已备份 |
| `yolov8_env.tar.gz` | 9,294,862,253 | 8.657 | conda-pack 环境，供重新构建 API 镜像 | 否，但建议保留 |
| 合计 | 55,600,882,139 | 51.782 | 全部迁移资产 | — |

镜像基线：

```yaml
api_image:
  name: quantitize-platform-api:latest
  image_id: sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
  unpacked_size_bytes: 21072416494
  architecture: amd64
web_image:
  name: quantitize-platform-web:latest
  image_id: sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
  unpacked_size_bytes: 184723399
  architecture: amd64
```

SHA-256：

```text
1ce12391b0b84abf4de369bd8f2a43f704cf9f131514c7e36fb25c049bad3262 *docker-images.tar.zst
0350f93c7db687ab74043cc7923dc917bece1bad814f55c9ce8fa4f17880f0aa *output-data.tar.zst
84c180d8ea86089134c244f7ee4d2e273aedad59900d11d25da7f317ce2a4f8f *quantitize-platform-code.tar.zst
130a9add69b643a7752993e00d872c56dad68020c9108a893787077f31b0b33c *shared-data.tar.zst
bb66795409a56195830641d24620b37f7d3617bc4a424517d2885545d1ac7251 *yolov8_env.tar.gz
```

已完成验证：

- [x] 四个 `.zst` 文件均通过完整流校验。
- [x] NAS 环境包哈希与 `wrs` 原始环境包哈希一致。
- [x] 代码包包含 `apps/api`、`apps/web`、`pipeline`、`patches` 和 `deploy`。
- [x] 共享数据打包时使用 `--dereference`，旧项目绝对符号链接已转换为普通目录和文件。
- [x] 历史输出中的 840 个 `root:root 0600` 文件通过 API 容器读取后重新归档，成功流无权限错误。
- [x] 备份通过 SSH 流式写入 NAS，没有占用 `wrs` 的本地中间空间。

## 3. 已知架构和迁移约束

1. API 镜像基于 `nvidia/cuda:12.8.1-runtime-ubuntu24.04`，内部包含 `/home/rs/miniconda3/envs/yolov8`。
2. 镜像是 `linux/amd64`，目标服务器必须是 x86_64/amd64；ARM 服务器不能直接加载运行。
3. 当前 Compose 使用固定的容器内部路径 `/home/rs/wrs/onnxviewe/quantitize-platform`。
4. 当前 Compose 会绑定挂载 `apps/api`、`pipeline` 和 `apps/web`，所以除了镜像外仍必须恢复代码包。
5. 当前 Compose 还有一条指向旧项目的只读挂载：

   ```text
   /home/rs/wrs/onnxviewe/quantitize/shared_data:/home/rs/wrs/onnxviewe/quantitize/shared_data:ro
   ```

   NAS 共享数据包已经解引用为真实文件。目标服务器部署前应删除这条旧挂载，避免重新引入旧项目依赖。
6. `data/output_data` 和 `data/shared_data` 是宿主机绑定挂载；不能只加载镜像而不恢复或创建这两个目录。
7. `wrs` 无法连接 NAS 的 445 端口，但 Windows 本机可以。因此本次备份使用 Windows 中转。目标服务器若也无法访问 NAS，应使用本文档中的“Windows 中转下载”。
8. 不要把 NAS 密码写入本文档、Git、`.env`、Shell 历史或命令日志。

## 4. 阶段总览

| 阶段 | 状态 | 目标 | 完成条件 |
|---|---|---|---|
| B0 源端调查 | DONE | 确认镜像、代码、数据和权限 | 基线已记录 |
| B1 NAS 备份 | DONE | 上传全部迁移资产 | 51.782 GiB 文件通过完整性验证 |
| D0 目标信息 | BLOCKED | 获得目标服务器地址和账号 | 可通过 SSH 登录目标服务器 |
| D1 目标环境 | TODO | 安装 Docker、Compose、GPU Runtime | Docker 容器能识别 GPU |
| D2 获取备份 | TODO | 从 NAS 下载五个文件 | `sha256sum -c` 全部通过 |
| D3 恢复代码和数据 | TODO | 恢复平台目录并移除旧依赖 | 目录、权限、Compose 校验通过 |
| D4 加载镜像 | TODO | 导入 API 和 Web 镜像 | 镜像 ID 与基线一致 |
| D5 启动容器 | TODO | 启动 Web + API | 两个容器 healthy |
| D6 功能验收 | TODO | 验证 GPU、Runner、页面和任务 | 最小任务完整运行 |

## 5. D0：收集目标服务器信息

### D0-T1 配置并验证 SSH

```yaml
status: BLOCKED
depends_on: []
required_input:
  - target_host.address
  - target_host.ssh_user
```

Agent 获得目标地址和用户后，在控制机设置 SSH 别名，例如：

```ssh-config
Host quantitize-target
    HostName <TARGET_IP>
    User <TARGET_USER>
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

验证：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 quantitize-target "echo ok"
```

验收条件：

- [ ] 命令输出 `ok`。
- [ ] Agent 已把实际目标地址和用户写入本文档元数据。
- [ ] 目标服务器可以访问 NAS，或控制机可以同时访问 NAS 和目标 SSH。

进度记录：

```text
status: BLOCKED
started_at:
completed_at:
target_address:
target_user:
verification_output:
notes: waiting for target host address and SSH user
```

## 6. D1：准备目标服务器

### D1-T1 检查硬件、系统和磁盘

```yaml
status: TODO
depends_on: [D0-T1]
```

在目标服务器执行：

```bash
uname -m
sed -n '1,12p' /etc/os-release
nvidia-smi
df -h / /var/lib/docker /home 2>/dev/null || true
```

要求：

- `uname -m` 必须为 `x86_64`。
- 推荐 Ubuntu 24.04，与当前 API 基础镜像一致。
- NVIDIA 驱动必须支持目标 GPU及 CUDA 12.8 容器；优先使用与源机已验证的 580 系列相同或更新的兼容驱动。
- 只恢复必需资产时，建议至少 70 GiB 可用空间。
- 加载镜像并恢复全部历史数据时，建议至少 130 GiB 可用空间；还需为后续新任务预留空间。

验收条件：

- [ ] 架构为 x86_64。
- [ ] `nvidia-smi` 正常。
- [ ] 磁盘容量满足选择的恢复范围。
- [ ] 目标端口 8000 和 8088 未被占用，或已决定替代端口。

### D1-T2 安装 Docker Engine 和 Compose

```yaml
status: TODO
depends_on: [D1-T1]
```

安装前重新核对 Docker 官方 Ubuntu 指南：

```text
https://docs.docker.com/engine/install/ubuntu/
```

Ubuntu 官方仓库方式的核心包：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl zstd cifs-utils

# 按 Docker 官方文档配置 apt repository 后执行：
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

如果目标用户需要免 sudo 使用 Docker：

```bash
sudo usermod -aG docker <TARGET_USER>
```

修改组后必须重新登录 SSH。

验证：

```bash
docker version
docker compose version
docker buildx version
systemctl is-active docker
```

Compose 建议不低于源端已验证版本 `v2.32.4`，因为当前文件使用列表形式的 `gpus` 配置。

### D1-T3 安装并配置 NVIDIA Container Toolkit

```yaml
status: TODO
depends_on: [D1-T2]
```

安装前重新核对 NVIDIA 官方指南：

```text
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

按官方文档添加 NVIDIA Container Toolkit 软件源并安装后执行：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-ctk --version
```

此时尚未加载项目镜像，可先只验证宿主机：

```bash
nvidia-smi
docker info | sed -n '/Runtimes/,+2p'
```

验收条件：

- [ ] Docker 服务正常。
- [ ] `docker compose version` 正常。
- [ ] `nvidia-ctk` 正常。
- [ ] Docker 配置中存在 NVIDIA runtime 支持。

进度记录：

```text
status: TODO
started_at:
completed_at:
os:
architecture:
gpu:
driver:
docker_version:
compose_version:
nvidia_container_toolkit_version:
free_space:
verification_output:
notes:
```

## 7. D2：从 NAS 获取备份

### D2-T1 首选：目标服务器直接挂载 SMB

```yaml
status: TODO
depends_on: [D1-T2]
preferred: true
```

设置变量：

```bash
NAS_MOUNT=/mnt/quantitize-nas
NAS_BACKUP="$NAS_MOUNT/3-个人/16-王宇航/3-Project/2-DRO/模型量化/quantitize-platform-backup-20260901"
STAGING_ROOT=/srv/quantitize-migration
```

确认端口：

```bash
timeout 3 bash -c '</dev/tcp/10.2.26.26/445'
```

如果成功，创建凭据文件：

```bash
sudo install -d -m 0755 "$NAS_MOUNT"
sudo install -m 0600 /dev/null /root/.quantitize-nas-credentials
sudoedit /root/.quantitize-nas-credentials
```

凭据文件内容由执行者手工输入，不记录在本文档或 Agent 输出中：

```text
username=orange
password=<从用户安全获取>
```

只读挂载：

```bash
sudo mount -t cifs //10.2.26.26/902_data "$NAS_MOUNT" \
  -o credentials=/root/.quantitize-nas-credentials,vers=3.0,iocharset=utf8,ro

findmnt "$NAS_MOUNT"
ls -lah "$NAS_BACKUP"
```

复制到目标服务器本地磁盘：

```bash
sudo install -d -m 0755 "$STAGING_ROOT"
sudo cp --reflink=auto "$NAS_BACKUP"/docker-images.tar.zst "$STAGING_ROOT"/
sudo cp --reflink=auto "$NAS_BACKUP"/quantitize-platform-code.tar.zst "$STAGING_ROOT"/
sudo cp --reflink=auto "$NAS_BACKUP"/shared-data.tar.zst "$STAGING_ROOT"/
sudo cp --reflink=auto "$NAS_BACKUP"/output-data.tar.zst "$STAGING_ROOT"/
sudo cp --reflink=auto "$NAS_BACKUP"/yolov8_env.tar.gz "$STAGING_ROOT"/
sudo cp "$NAS_BACKUP"/SHA256SUMS "$STAGING_ROOT"/
sudo chown -R <TARGET_USER>:<TARGET_GROUP> "$STAGING_ROOT"
```

如果空间有限，可先只复制镜像、代码、共享数据和 `SHA256SUMS`；历史输出与环境包留在 NAS，后续再恢复。

### D2-T2 备用：通过 Windows 控制机中转

如果目标服务器无法访问 NAS 445，但控制机能访问 NAS，则在 Windows PowerShell 执行：

```powershell
$nasBackup = '\\10.2.26.26\902_data\3-个人\16-王宇航\3-Project\2-DRO\模型量化\quantitize-platform-backup-20260901'

ssh quantitize-target "sudo install -d -m 0755 /srv/quantitize-migration && sudo chown <TARGET_USER>:<TARGET_GROUP> /srv/quantitize-migration"

scp "$nasBackup\docker-images.tar.zst" quantitize-target:/srv/quantitize-migration/
scp "$nasBackup\quantitize-platform-code.tar.zst" quantitize-target:/srv/quantitize-migration/
scp "$nasBackup\shared-data.tar.zst" quantitize-target:/srv/quantitize-migration/
scp "$nasBackup\output-data.tar.zst" quantitize-target:/srv/quantitize-migration/
scp "$nasBackup\yolov8_env.tar.gz" quantitize-target:/srv/quantitize-migration/
scp "$nasBackup\SHA256SUMS" quantitize-target:/srv/quantitize-migration/
```

大文件传输中断时，不要直接认为目标文件有效。重新传输后必须重新执行 SHA-256 校验。

### D2-T3 校验目标端文件

```yaml
status: TODO
depends_on: [D2-T1 or D2-T2]
```

```bash
cd /srv/quantitize-migration
sha256sum -c SHA256SUMS
zstd -q -t docker-images.tar.zst
zstd -q -t quantitize-platform-code.tar.zst
zstd -q -t shared-data.tar.zst
zstd -q -t output-data.tar.zst
gzip -t yolov8_env.tar.gz
```

如果只复制必需文件，可从 `SHA256SUMS` 中筛选已复制文件后检查；不得把缺少文件误判为哈希失败。

验收条件：

- [ ] 所有已复制文件的 SHA-256 显示 `OK`。
- [ ] 所有压缩包完整性测试退出码为 0。
- [ ] 目标端文件大小与本文档表格一致。

进度记录：

```text
status: TODO
started_at:
completed_at:
transfer_mode: direct_smb | windows_relay
files_copied:
sha256_result:
archive_test_result:
notes:
```

## 8. D3：恢复代码和数据

### D3-T1 创建固定目录并恢复代码

```yaml
status: TODO
depends_on: [D2-T3]
```

为了与镜像和 Pipeline 中已经验证的绝对路径兼容，首轮迁移保持原路径：

```bash
TARGET_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform
sudo install -d -m 0755 "$TARGET_ROOT"
sudo zstd -dc /srv/quantitize-migration/quantitize-platform-code.tar.zst \
  | sudo tar -xf - -C "$TARGET_ROOT"
sudo install -d -m 0755 "$TARGET_ROOT/data/output_data" "$TARGET_ROOT/data/shared_data"
```

验证：

```bash
test -f "$TARGET_ROOT/deploy/docker-compose.yml"
test -f "$TARGET_ROOT/deploy/Dockerfile.api"
test -f "$TARGET_ROOT/apps/api/app.py"
test -f "$TARGET_ROOT/apps/web/app.py"
test -f "$TARGET_ROOT/pipeline/verify_runner.py"
find "$TARGET_ROOT" -maxdepth 3 -type f | sort
```

### D3-T2 恢复共享数据

```bash
sudo zstd -dc /srv/quantitize-migration/shared-data.tar.zst \
  | sudo tar -xf - -C "$TARGET_ROOT"
```

确认不再存在旧绝对符号链接：

```bash
find "$TARGET_ROOT/data/shared_data" -type l -print
find "$TARGET_ROOT/data/shared_data/cali/aituosha_moon_500" -type f | wc -l
find "$TARGET_ROOT/data/shared_data/test/aituosha_moon_840" -type f | wc -l
test -f "$TARGET_ROOT/data/shared_data/registry.json"
```

预期：第一条命令无输出；数据文件数量与 registry 中记录的 500 张校准图、840 张测试图相符。测试集还可能包含标签文件，因此总文件数可以大于 840。

### D3-T3 恢复历史任务

如果决定恢复历史任务：

```bash
sudo zstd -dc /srv/quantitize-migration/output-data.tar.zst \
  | sudo tar -xf - -C "$TARGET_ROOT"
find "$TARGET_ROOT/data/output_data" -mindepth 1 -maxdepth 1 -type d | wc -l
```

预期存在 10 个任务目录。历史 manifest 可能仍包含旧路径，这是历史数据现状；新任务不得产生旧路径。

### D3-T4 修正宿主机权限

```bash
sudo chown -R <TARGET_USER>:<TARGET_GROUP> /home/rs/wrs/onnxviewe/quantitize-platform
find "$TARGET_ROOT/data/output_data" -type f ! -readable | head
find "$TARGET_ROOT/data/shared_data" -type f ! -readable | head
```

两条 `find ... ! -readable` 应无输出。

### D3-T5 移除目标机上的旧项目挂载依赖

编辑：

```text
/home/rs/wrs/onnxviewe/quantitize-platform/deploy/docker-compose.yml
```

删除或注释以下 volume：

```yaml
- /home/rs/wrs/onnxviewe/quantitize/shared_data:/home/rs/wrs/onnxviewe/quantitize/shared_data:ro
```

保留以下挂载：

```yaml
- ../data/output_data:/home/rs/wrs/onnxviewe/quantitize-platform/data/output_data
- ../data/shared_data:/home/rs/wrs/onnxviewe/quantitize-platform/data/shared_data
- ../apps/api:/home/rs/wrs/onnxviewe/quantitize-platform/apps/api
- ../pipeline:/home/rs/wrs/onnxviewe/quantitize-platform/pipeline
- ../apps/web:/app
```

验证：

```bash
cd "$TARGET_ROOT"
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose-rendered.yml
rg -n '/home/rs/wrs/onnxviewe/quantitize/shared_data' deploy/docker-compose.yml /tmp/quantitize-compose-rendered.yml || true
```

验收条件：

- [ ] 代码文件完整。
- [ ] 共享数据不存在符号链接。
- [ ] 共享数据 registry 存在。
- [ ] 选择恢复历史数据时有 10 个任务目录。
- [ ] 目标运行用户可读取和写入数据目录。
- [ ] Compose 不再挂载旧项目 shared_data。
- [ ] `docker compose config` 退出码为 0。

进度记录：

```text
status: TODO
started_at:
completed_at:
project_owner:
shared_dataset_counts:
output_task_count:
legacy_mount_removed:
compose_config_result:
notes:
```

## 9. D4：加载 Docker 镜像

### D4-T1 从镜像包加载

```yaml
status: TODO
depends_on: [D3-T5]
```

```bash
cd /srv/quantitize-migration
zstd -dc docker-images.tar.zst | docker load
docker image ls --digests | grep -E 'quantitize-platform-(api|web)'
```

验证镜像 ID：

```bash
docker image inspect quantitize-platform-api:latest --format '{{.Id}} {{.Architecture}} {{.Os}} {{.Size}}'
docker image inspect quantitize-platform-web:latest --format '{{.Id}} {{.Architecture}} {{.Os}} {{.Size}}'
```

预期：

```text
API ID: sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
Web ID: sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
Architecture: amd64
OS: linux
```

### D4-T2 验证镜像中的 GPU 环境

```bash
docker run --rm --gpus all quantitize-platform-api:latest nvidia-smi -L

docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python -c \
  "import torch, onnxruntime; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('ort=', onnxruntime.__version__)"

docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py

docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_patches.py
```

验收条件：

- [ ] 两个镜像 ID 与基线一致。
- [ ] `nvidia-smi -L` 能看到目标 GPU。
- [ ] `torch.cuda.is_available()` 返回 `True`。
- [ ] PyTorch 和 ONNX Runtime 导入成功。
- [ ] Runner 与 patches 检查通过。

进度记录：

```text
status: TODO
started_at:
completed_at:
api_image_id:
web_image_id:
gpu_result:
torch_result:
runner_result:
patch_result:
notes:
```

## 10. D5：启动 Docker Compose

### D5-T1 启动服务

```yaml
status: TODO
depends_on: [D4-T2]
```

首次启动使用已加载镜像，不重新构建：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
docker compose -f deploy/docker-compose.yml config
docker compose -f deploy/docker-compose.yml up -d --no-build
```

如果 8000 或 8088 被占用：

```bash
PORT=8090 API_PORT=8001 docker compose -f deploy/docker-compose.yml up -d --no-build
```

检查：

```bash
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs --tail=150 quantitize-api
docker compose -f deploy/docker-compose.yml logs --tail=150 quantitize-web
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8088/health
```

预期 API：

```json
{"ok":true,"busy":false,"current_job":null}
```

Web health 应包含：

```json
{"web":"ok","demo":false}
```

检查挂载：

```bash
docker inspect quantitize-platform-api --format '{{json .Mounts}}'
docker inspect quantitize-platform-web --format '{{json .Mounts}}'
```

验收条件：

- [ ] API 和 Web 容器均为 `healthy`。
- [ ] `WEB_DEMO_MODE=0`。
- [ ] Web 的 `API_BASE_URL` 为 `http://quantitize-api:8000`。
- [ ] output 和 shared data 挂载到新平台目录。
- [ ] 不存在旧项目 shared_data 挂载。
- [ ] 宿主机和局域网均可访问 Web。

进度记录：

```text
status: TODO
started_at:
completed_at:
web_url:
api_url:
container_status:
health_result:
mount_result:
notes:
```

## 11. D6：迁移验收

### D6-T1 历史任务和共享数据验收

```bash
curl -fsS http://127.0.0.1:8000/api/datasets
curl -fsS http://127.0.0.1:8000/api/tasks
curl -f http://127.0.0.1:8088/datasets
curl -f http://127.0.0.1:8088/history
```

浏览器检查：

- 首页显示 Worker 空闲状态。
- 数据集页面显示 `aituosha_moon_500` 和 `aituosha_moon_840`。
- 恢复历史数据时，历史页显示已有任务。
- 任务详情、指标、图片、bundle inventory 和下载可访问。

### D6-T2 创建并运行最小真实任务

通过 Web 或 API 创建新任务并启动，观察八步：

```text
pt_eval
quantize
onnx_eval
fpga_test_pack
fpga_eval
generate_bin
merge_bin
bundle
```

持续检查：

```bash
curl -fsS http://127.0.0.1:8000/api/tasks/<NEW_TASK_ID>
docker compose -f deploy/docker-compose.yml logs -f --tail=100 quantitize-api
```

任务完成后：

```bash
TARGET_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform
/home/rs/miniconda3/envs/yolov8/bin/python -m json.tool "$TARGET_ROOT/data/output_data/<NEW_TASK_ID>/manifest.json" 2>/dev/null \
  || docker exec quantitize-platform-api \
     /home/rs/miniconda3/envs/yolov8/bin/python -m json.tool \
     "$TARGET_ROOT/data/output_data/<NEW_TASK_ID>/manifest.json"

rg -n '/home/rs/wrs/onnxviewe/quantitize/output_data' "$TARGET_ROOT/data/output_data/<NEW_TASK_ID>" || true
```

验收条件：

- [ ] 新任务八步全部 `completed`。
- [ ] 最终 bundle 存在并可下载、解压。
- [ ] 新 manifest 不包含旧项目路径。
- [ ] 指标和结果图片可通过 Web 查看。

### D6-T3 重启持久化验证

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
docker compose -f deploy/docker-compose.yml restart
curl -fsS http://127.0.0.1:8000/api/tasks

# 禁止添加 -v
docker compose -f deploy/docker-compose.yml down
docker compose -f deploy/docker-compose.yml up -d --no-build
curl -fsS http://127.0.0.1:8000/api/tasks
```

验收条件：

- [ ] restart 后任务仍存在。
- [ ] down/up 后任务仍存在。
- [ ] 两个容器重新进入 healthy。
- [ ] API 容器仍可访问 GPU。

进度记录：

```text
status: TODO
started_at:
completed_at:
new_task_id:
pipeline_steps:
bundle_result:
legacy_path_scan:
restart_result:
recreate_result:
user_acceptance:
notes:
```

## 12. 镜像损坏时的重建备用流程

正常迁移优先使用 `docker-images.tar.zst`，不要无故重新构建 21 GB API 镜像。仅在镜像包无法加载或代码需要重新固化时执行本节。

恢复环境包到 Dockerfile 预期位置：

```bash
sudo install -d -m 0755 /home/rs/wrs/onnxviewe
sudo cp /srv/quantitize-migration/yolov8_env.tar.gz /home/rs/wrs/onnxviewe/yolov8_env.tar.gz
sudo chown <TARGET_USER>:<TARGET_GROUP> /home/rs/wrs/onnxviewe/yolov8_env.tar.gz
```

构建 API：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
BUILD_ENV_CONTEXT=/tmp/quantitize_yolov8_env_context
mkdir -p "$BUILD_ENV_CONTEXT"
ln -sfn /home/rs/wrs/onnxviewe/yolov8_env.tar.gz "$BUILD_ENV_CONTEXT/yolov8_env.tar.gz"

DOCKER_BUILDKIT=1 docker build \
  --build-context yolov8env="$BUILD_ENV_CONTEXT" \
  -f deploy/Dockerfile.api \
  -t quantitize-platform-api:latest \
  .
```

构建 Web：

```bash
docker build -f deploy/Dockerfile.web -t quantitize-platform-web:latest .
```

重新执行 D4-T2、D5 和 D6 的全部验证。重新构建后的镜像 ID通常与备份镜像不同，因此必须记录新 ID和构建原因。

## 13. 停止和回退

停止服务：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
docker compose -f deploy/docker-compose.yml down
```

禁止：

```bash
docker compose -f deploy/docker-compose.yml down -v
```

回退原则：

- 容器启动失败时，保留目标端数据目录和 `/srv/quantitize-migration`。
- 不删除 NAS 备份。
- 不覆盖源服务器 `wrs` 上的项目、镜像或数据。
- 先收集 `docker compose ps`、日志、镜像 ID、挂载和 GPU 输出，再处理故障。
- 若新任务在目标端产生新数据，回退前先将对应任务目录增量备份到 NAS。

## 14. 后续增量备份注意事项

源端历史输出里出现过容器生成的 `root:root 0600` 文件。普通 `rs` 用户打包会遗漏文件，并出现 `tar: 无法 open: 权限不够`。

再次备份 output data 时，应通过正在运行的 API 容器读取：

```bash
docker exec quantitize-platform-api \
  tar -C /home/rs/wrs/onnxviewe/quantitize-platform \
  -cf - data/output_data \
  | zstd -T0 -3 -o output-data.tar.zst
```

必须满足：

- Shell 管道启用 `pipefail`，或分别检查 `docker exec` 和压缩命令退出码。
- 日志中不得出现 `Permission denied` 或 `权限不够`。
- 完成后执行 `zstd -t` 和 SHA-256。
- 目标文件在校验完成前标记为临时或未完成状态。

## 15. 变更日志

| 时间 | 执行者 | 任务 | 状态变化 | 结果 |
|---|---|---|---|---|
| 2026-09-01 | Codex | B0 | TODO → DONE | 确认双镜像、Compose、GPU 环境、45 GB output、11 GB 旧共享数据依赖 |
| 2026-09-01 | Codex | B1 | TODO → DONE | 通过 Windows 中转上传五个正式备份文件到 NAS，共 51.782 GiB |
| 2026-09-01 | Codex | B1 验证 | TODO → DONE | 四个 Zstandard 包通过完整流测试；环境包源/目标 SHA-256 一致 |
| 2026-09-01 | Codex | 权限修正 | IN_PROGRESS → DONE | 发现 840 个 root-only 文件后停止不完整归档，改由 API 容器读取并成功重传 |
| 2026-09-01 | Codex | 文档 | TODO → DONE | 创建本跨服务器迁移执行文档和 SHA256SUMS |

## 16. 当前恢复点

```yaml
completed:
  - source_inventory
  - docker_image_backup
  - code_backup
  - shared_data_backup_with_symlink_dereference
  - output_data_backup_including_root_only_files
  - environment_archive_backup
  - zstd_integrity_verification
  - environment_source_target_hash_comparison

current_phase: D0
next_task: D0-T1
required_next_input:
  - target server IP or hostname
  - target SSH username

first_command_after_input:
  - ssh -o BatchMode=yes -o ConnectTimeout=8 quantitize-target "echo ok"

do_not_repeat:
  - source backup unless source data changed
  - API image build unless docker image load fails or code must be baked again

do_not_delete:
  - NAS backup directory
  - source project on wrs
  - target staging files before D6 acceptance
```

目标服务器信息提供后，Agent 应从 `D0-T1` 继续，不应重新执行已经完成的 B0/B1 备份。
