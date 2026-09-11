# Quantitize 环境重建指南

本文以当前 gray1 容器为目标。主机准备涉及驱动、Docker 或 NVIDIA Toolkit 时必须由管理员授权；Agent 不在共享 H200 上自行修改全局软件。

## 1. 当前目标

```text
host: Linux x86_64 + NVIDIA H200
API Dockerfile: deploy/Dockerfile.api
base image: ywang/yolo-gray1-quant:ultralytics-8.3.98-v6
python: /opt/ywang/yolo-runtime/bin/python (当前容器观察 3.12.12)
torch: 2.8.0+cu128
onnx: 1.17.0
onnxruntime: 1.24.3
opencv-headless: 4.11.0.86
numpy: 1.26.4
input ABI: FLOAT16 [1,1,1280,1280]
```

版本风险：当前容器的 `pip show ultralytics` 为 8.4.21 editable，但基础镜像标签写 8.3.98，且 `PYTHONPATH=/workspace/src/Yolo_fine/ultralytics`。重建前必须确认实际 `import ultralytics` 的版本和文件路径，并将其固化；不能仅按镜像标签宣称可复现。

## 2. 重建路径

1. 首选：基础镜像仍可用时，用当前 `deploy/Dockerfile.api` 构建。
2. 次选：加载 2026-09-02 系统快照镜像，只作为 pre-gray1 底座，然后应用当前代码/环境切换并全量回归。
3. 最后：基础镜像和快照都不可用时，重新制作单通道运行时；必须先取得 gray1 基础镜像的构建来源并固定依赖和源代码 revision。仓库现有 manifests 不足以独立复现它。

旧 `yolov8_env.tar.gz`、`rebuild/manifests/` 与 `rebuild/snapshots/Dockerfile.api.pinned-amd64` 描述 Python 3.9/Ultralytics 8.4.6 的三通道环境，只用于历史恢复，不是当前 gray1 生产构建输入。

## 3. 主机门禁

```bash
uname -m
sed -n '1,12p' /etc/os-release
nvidia-smi
docker version
docker compose version
nvidia-ctk --version
df -h /data3 /data1
```

要求 x86_64、H200 驱动和容器 GPU runtime 可用、空间足够。不要因重建任务修改全局 Docker data-root、关闭 SELinux 或升级驱动。

## 4. 构建当前镜像

先记录基线：

```bash
cd /data3/ywang/quantitize-platform
git status --short
git rev-parse HEAD
docker image inspect ywang/yolo-gray1-quant:ultralytics-8.3.98-v6 --format '{{.Id}}'
```

得到授权后：

```bash
docker build -f deploy/Dockerfile.api -t quantitize-platform-api:latest .
docker build -f deploy/Dockerfile.web -t quantitize-platform-web:latest .
```

不要覆盖唯一可用镜像而不保留标签/ID。当前仓库记录的旧 API 镜像标签 `quantitize-platform-api:pre-gray1-20260910` 是回滚参考。

## 5. Runtime 验收

在获批 GPU 上运行隔离容器，核对：Python、实际 Ultralytics 文件和版本、PyTorch CUDA、`sm_90`、ONNX Runtime CUDA provider、OpenCV/NumPy、项目 patches 和 runner。

```bash
docker run --rm --gpus 'device=<approved-uuid>' quantitize-platform-api:latest nvidia-smi -L
docker run --rm --gpus 'device=<approved-uuid>' quantitize-platform-api:latest \
  /opt/ywang/yolo-runtime/bin/python pipeline/verify_runner.py
docker run --rm --gpus 'device=<approved-uuid>' quantitize-platform-api:latest \
  /opt/ywang/yolo-runtime/bin/python pipeline/verify_patches.py
```

还应执行 import 脚本并记录 `module.__version__` 与 `module.__file__`，不能只看 pip metadata。

## 6. 服务与功能验收

确认无运行任务后，按授权启动：

```bash
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose.resolved.yml
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8088/health
```

最终必须使用固定的单通道 21 类模型、标定集和测试集运行新任务，验证 PT、ONNX、输入回环、逐层 BIN、合并和 ZIP。当前最近一次单通道任务 ONNX F1=0 且 `generate_bin` 失败，所以仅通过 import、health 或 `nvidia-smi` 不能宣称重建成功。

## 7. 恢复旧快照时

快照 `snapshot-20260902-182606` 的哈希与流程见 `../docs/reference/RECOVERY.md`。加载旧镜像后先标记为 pre-gray1，不要直接覆盖当前服务。任何旧环境迁移到当前 ABI 都必须重新导出模型和完成全流程回归。
