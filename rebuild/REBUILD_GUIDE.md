# Quantitize Linux/NVIDIA 环境重建指南

> 面向 Agent 的可执行文档。目标宿主机可能是 Ubuntu 22.04、Ubuntu 24.04、CentOS/RHEL 系，GPU 可能是 NVIDIA H200。

## 1. 结论和边界

宿主机发行版不需要与容器内发行版相同。API 容器的用户态固定为 Ubuntu 24.04 + CUDA 12.8.1；宿主机主要提供 Linux 内核、NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。

当前资产的硬约束：

```yaml
os: Linux
cpu_architecture: amd64/x86_64
gpu_vendor: NVIDIA
container_cuda: 12.8.1
python: 3.9.25
torch: 2.8.0+cu128
torch_archs: [sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120]
h200_compute_capability: sm_90
onnxruntime: 1.19.2
required_ort_provider: CUDAExecutionProvider
```

H200 使用 Hopper compute capability 9.0。当前 PyTorch 二进制明确包含 `sm_90`，因此不需要为 H200 重新编译 PyTorch，但必须在目标 H200 上执行 GPU、ORT 和完整任务验收。

CUDA 12.8 GA 对应 Linux 驱动基线为 570.26。CUDA 12.x 的 minor-version compatibility 下限更低，但本项目为了减少 H200 和运行时差异，目标机验收门禁采用：

```text
NVIDIA driver >= 570.26
```

推荐使用目标服务器厂商为 H200 提供的当前数据中心驱动，不要在文档中长期写死某个最新驱动小版本。

## 2. 发行版支持策略

| 宿主机 | 策略 | 特别检查 |
|---|---|---|
| Ubuntu 24.04 | 首选 | Docker、驱动、Toolkit |
| Ubuntu 22.04 | 支持 | 不需要安装宿主机 CUDA Toolkit |
| RHEL 8/9 | 支持 | Docker 可用性、SELinux、NVIDIA dnf repo |
| CentOS 8 | 条件支持 | 已结束常规生命周期，先确认仓库和安全维护来源 |
| Rocky/Alma | 条件支持 | 按对应 RHEL 主版本验证 Toolkit |
| CentOS 7 | 不建议 | 工具链和生命周期过旧 |
| ARM64 Linux | 当前不支持 | 镜像、Conda 包和 `_x86_64-microarch-level` 均为 amd64 |

NVIDIA 当前平台矩阵明确覆盖 Ubuntu 22.04、Ubuntu 24.04、CentOS 8 和 RHEL 8/9 等版本。对于未在矩阵中的发行版，不应仅凭 `nvidia-smi` 成功就认定平台合格。

CentOS/RHEL 在 SELinux enforcing 模式下，Compose 绑定挂载可能出现 `Permission denied`。应对项目数据和代码挂载增加 `:Z`，或由管理员配置等价的持久 SELinux context。不要通过永久关闭 SELinux解决。

## 3. 重建级别选择

### Level 1：加载已备份镜像

适用：NAS 当前快照中的 `docker-images-api-web.tar.zst` 完整可用。

优点：最接近字节级复现，不依赖未来 apt、Conda 或 PyPI 仓库。

```bash
cd /srv/quantitize-migration
sha256sum -c SHA256SUMS
zstd -dc docker-images-api-web.tar.zst | docker load

docker image inspect quantitize-platform-api:latest --format '{{.Id}} {{.Architecture}} {{.Size}}'
docker image inspect quantitize-platform-web:latest --format '{{.Id}} {{.Architecture}} {{.Size}}'
```

预期镜像 ID：

```text
API sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
Web sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
```

### Level 2：从 Conda 环境包重建镜像

适用：镜像包不可用，`yolov8_env.tar.gz` 可用；目标为 Linux amd64。

这是当前最可靠的重新打包方式。环境包已经包含 Python、PyTorch、CUDA Python wheels、ONNX Runtime、Ultralytics 及其二进制依赖。

准备：

```bash
PROJECT_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform
ENV_ARCHIVE=/home/rs/wrs/onnxviewe/yolov8_env.tar.gz
BUILD_CONTEXT=/tmp/quantitize-yolov8-env-context

sha256sum "$ENV_ARCHIVE"
mkdir -p "$BUILD_CONTEXT"
ln -sfn "$ENV_ARCHIVE" "$BUILD_CONTEXT/yolov8_env.tar.gz"
```

环境包预期 SHA-256：

```text
bb66795409a56195830641d24620b37f7d3617bc4a424517d2885545d1ac7251
```

构建 API：

```bash
cd "$PROJECT_ROOT"
DOCKER_BUILDKIT=1 docker build \
  --build-context yolov8env="$BUILD_CONTEXT" \
  -f rebuild/snapshots/Dockerfile.api.pinned-amd64 \
  -t quantitize-platform-api:rebuilt \
  .
```

如果 `rebuild/` 不在项目构建上下文中，把 pinned Dockerfile 复制到 `deploy/` 后再构建。

构建 Web：

```bash
docker build \
  -f rebuild/snapshots/Dockerfile.web.pinned-amd64 \
  -t quantitize-platform-web:rebuilt \
  .
```

Pinned API 基础镜像：

```text
nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e
```

Pinned Web 基础镜像：

```text
python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
```

注意：基础镜像 digest 已固定，但 Dockerfile 中 `apt-get install` 仍访问动态 apt 仓库。因此重建结果功能上应一致，镜像 ID不保证一致。要获得字节级相同镜像，应使用 Level 1。

### Level 3：从 Conda/Pip 声明重建环境包

适用：镜像和 `yolov8_env.tar.gz` 都不可用，或者必须审计并更新依赖。

此路径需要联网访问 Anaconda、conda-forge 和 PyPI。它不能保证未来仓库仍保留所有旧包，生成后必须执行完整回归。

优先尝试完整环境文件：

```bash
conda env create \
  -p /opt/quantitize-env-build/yolov8 \
  -f manifests/environment-full-linux-64.yml

/opt/quantitize-env-build/yolov8/bin/python -m pip list --format=freeze
/opt/quantitize-env-build/yolov8/bin/python rebuild/verify_runtime.py
```

如果需要最大限度锁定 Conda 二进制包：

```bash
conda create -y \
  -p /opt/quantitize-env-build/yolov8 \
  --file manifests/conda-explicit-linux-64.txt

/opt/quantitize-env-build/yolov8/bin/python -m pip install \
  -r manifests/pip-list-linux-amd64.lock.txt
```

`pip-list-linux-amd64.lock.txt` 是当前最终可见版本清单；`pip-freeze-raw-linux-amd64.txt` 保留原始 provenance，但其中部分 `file:///...` 构建路径不可移植，不应直接作为主要安装输入。

当前源环境存在 NumPy 元数据重叠：`pip list` 报告 1.26.4，但实际 `import numpy` 为 2.0.1，并且 site-packages 同时存在两套 dist-info。Level 3 不应复制这个脏状态；应构建干净环境并确保实际导入 NumPy 2.0.1。真实导入版本以 `runtime-import-versions.json` 和 `verify_runtime.py` 为准。

完成后重新打包：

```bash
/opt/quantitize-env-build/yolov8/bin/python -m pip install conda-pack==0.8.1
conda-pack \
  -p /opt/quantitize-env-build/yolov8 \
  -o yolov8_env.rebuilt.tar.gz
sha256sum yolov8_env.rebuilt.tar.gz
```

然后使用 Level 2 的 Dockerfile 构建 API 镜像。

`environment-from-history.yml` 只描述最初人工选择的顶层 Conda 包，适合升级或跨平台重新求解，不适合精确复现。

## 4. 目标主机准备门禁

Agent 在任何镜像构建或启动前执行：

```bash
uname -m
sed -n '1,12p' /etc/os-release
nvidia-smi
docker version
docker compose version
nvidia-ctk --version
getenforce 2>/dev/null || true
df -h / /var/lib/docker
```

必须满足：

- `uname -m` 为 `x86_64`。
- NVIDIA 驱动不低于 570.26，或由服务器厂商明确验证可运行 CUDA 12.8/H200。
- Docker 和 NVIDIA Container Toolkit 正常。
- API 镜像展开约 21 GB，完整迁移建议至少保留 130 GiB 可用空间。
- 如果是 H200 MIG 实例，确认容器可见的是期望的 GPU/MIG device。

Ubuntu 使用 NVIDIA Toolkit 的 apt 安装路径；RHEL/CentOS 使用 dnf 路径。安装命令必须在执行时从 NVIDIA 官方文档重新确认，不复制过期仓库配置。

## 5. H200 专项验证

加载或构建 API 镜像后：

```bash
docker run --rm --gpus all quantitize-platform-api:latest nvidia-smi -L

docker run --rm --gpus all \
  -v "$PWD/rebuild:/rebuild:ro" \
  quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python \
  /rebuild/verify_runtime.py
```

成功标准：

- `gpu_names` 包含 NVIDIA H200 或预期 MIG 设备。
- `cuda_available=true`。
- `torch_arch_list` 包含 `sm_90`。
- ONNX Runtime providers 包含 `CUDAExecutionProvider`。
- NumPy、OpenCV、Ultralytics、FastAPI 和 Uvicorn 均可导入。

然后执行项目自身检查：

```bash
docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py

docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_patches.py
```

最后必须使用固定模型、固定校准集和固定测试集运行八步 golden task。仅通过 import 或 `nvidia-smi` 不能判定迁移成功。

## 6. CentOS/RHEL 的 SELinux 处理

先检查：

```bash
getenforce
```

若为 `Enforcing`，为 Compose 中项目绑定挂载增加 `:Z`，例如：

```yaml
volumes:
  - ../data/output_data:/home/rs/wrs/onnxviewe/quantitize-platform/data/output_data:Z
  - ../data/shared_data:/home/rs/wrs/onnxviewe/quantitize-platform/data/shared_data:Z
  - ../apps/api:/home/rs/wrs/onnxviewe/quantitize-platform/apps/api:Z
  - ../pipeline:/home/rs/wrs/onnxviewe/quantitize-platform/pipeline:Z
```

Web：

```yaml
volumes:
  - ../apps/web:/app:Z
```

修改后执行：

```bash
docker compose -f deploy/docker-compose.yml config
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml logs --tail=100
```

不要为了省事永久关闭 SELinux。

## 7. 依赖清单用途

| 文件 | 用途 | 精确度 |
|---|---|---|
| `conda-explicit-linux-64.txt` | 精确 Conda 包 URL和 build | 高，仅 linux-64 |
| `environment-full-linux-64.yml` | Conda + Pip 完整环境 | 高，仓库可用时 |
| `environment-from-history.yml` | 顶层意图、升级求解 | 低，不用于精确复现 |
| `pip-list-linux-amd64.lock.txt` | 最终 Python 版本锁 | 高，但未包含 wheel hash |
| `pip-freeze-raw-linux-amd64.txt` | provenance 审计 | 含不可移植 file URL |
| `runtime-import-versions.json` | 真实 import 版本及元数据重叠记录 | 行为验收基线 |
| `api-image-dpkg.txt` | API 镜像系统包审计 | 当前镜像精确快照 |
| `web-image-dpkg.txt` | Web 镜像系统包审计 | 当前镜像精确快照 |
| `pipeline-patches-sha256.txt` | 核心代码和补丁完整性 | SHA-256 |

`apps/api/requirements.txt` 与 `apps/web/requirements.txt` 中原先存在 `>=`。本资料增加了 `requirements.*.lock.txt`，锁定到当前镜像实际版本，但没有改动远端运行代码。

## 8. 已知异常

当前环境中的 `pip 26.0.1` 执行 `pip check` 时，在解析某个已安装 wheel tag 时抛出：

```text
ValueError: too many values to unpack (expected 3)
```

这表示 `pip check` 工具本身未完成检查，不等同于已发现依赖冲突。重建验收应以导入测试、Runner、patch 检查和 golden task 为准；如需恢复 `pip check`，应在隔离副本中测试兼容的 pip 版本，不能直接修改唯一生产环境。

此外，源环境同时存在 `numpy-1.26.4.dist-info` 和 `numpy-2.0.1.dist-info`，实际代码来自 NumPy 2.0.1。Conda/Pip 清单用于审计和灾难恢复，不能单独证明行为一致；重建后的干净环境必须通过 `verify_runtime.py` 和 golden task。

## 9. Agent 状态记录

```yaml
rebuild_materials:
  dockerfiles_original: DONE
  dockerfiles_pinned: DONE
  api_requirements_lock: DONE
  web_requirements_lock: DONE
  pip_final_versions: DONE
  pip_raw_freeze: DONE
  conda_explicit_lock: DONE
  conda_full_export: DONE
  conda_history_export: DONE
  api_os_package_audit: DONE
  web_os_package_audit: DONE
  source_runtime_metadata: DONE
  h200_sm90_check: DONE
  pipeline_patch_checksums: DONE
  offline_wheelhouse: NOT_CREATED
  offline_conda_mirror: NOT_CREATED

next_agent_action:
  - Select Level 1, 2, or 3 based on available NAS assets.
  - Record target distribution, driver, GPU and SELinux mode.
  - Run host gate, runtime verification, project checks and golden task.
```

## 10. 官方基线

- NVIDIA Container Toolkit 平台支持：`https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html`
- NVIDIA Container Toolkit 安装：`https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html`
- CUDA 12.8 release notes：`https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html`
- CUDA 基础镜像 manifest：`https://hub.docker.com/layers/nvidia/cuda/12.8.1-runtime-ubuntu24.04/images/sha256-828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e`

执行安装时应重新访问这些官方页面，避免使用文档生成日期之后已经废弃的软件源命令。
