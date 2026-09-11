# Web/API 部署与运行

工作目录：`/data3/ywang/quantitize-platform`。生产入口是 `deploy/docker-compose.yml`。

## 1. 当前组成

| 服务 | 容器/镜像 | 宿主机端口 | GPU |
|---|---|---:|---|
| API + worker + pipeline | `quantitize-platform-api` / `quantitize-platform-api:latest` | 8000 | Compose 只暴露 `GPU_DEVICE_ID` 指定的一张卡 |
| Web | `quantitize-platform-web` / `quantitize-platform-web:latest` | 8088 | 无 |

API 基于 `ywang/yolo-gray1-quant:ultralytics-8.3.98-v6` 构建，解释器 `/opt/ywang/yolo-runtime/bin/python`。`apps/api`、`pipeline` 和 `apps/web` 通过 bind mount 热更新；依赖或设备配置变化仍需重建镜像/容器。

2026-09-11 快照：两个容器 healthy、worker idle；API `/ready` 因容器 NVML 初始化失败返回 503。运行中 API 容器也未继承当前 Compose 新增的两个 `YOLO_INPUT_*` 环境变量，说明它与工作树配置有漂移。健康不代表配置同步或可运行量化任务。

## 2. 启动前门禁

```bash
cd /data3/ywang/quantitize-platform
pgrep -af '05_runner.py|pipeline/engine/quantitize.py' || true
sed -n '1,80p' deploy/.env
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
docker compose -f deploy/docker-compose.yml config >/tmp/quantitize-compose.resolved.yml
```

`deploy/.env` 必须使用经协调的 GPU/MIG UUID。不要依赖序号；不停止其他用户进程。当前门禁默认来自 `.env`：32768 MiB、20%、30 秒。

## 3. 构建

构建 API：

```bash
docker build -f deploy/Dockerfile.api -t quantitize-platform-api:latest .
```

构建 Web：

```bash
docker build -f deploy/Dockerfile.web -t quantitize-platform-web:latest .
```

构建会改变本地镜像；必须有授权。基础镜像标签与容器中 `pip show ultralytics` 的版本标识目前不一致，构建前记录基础镜像 ID，构建后记录实际 import 版本和来源文件。

## 4. 启动与网页访问

只有确认没有运行任务且授权覆盖服务操作后：

```bash
docker compose -f deploy/docker-compose.yml up -d --no-build
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -sS -i http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8088/health
```

浏览器：<http://10.2.29.180:8088>。API：<http://10.2.29.180:8000>。

`/health` 表示进程存活；`/ready` 表示 GPU 门禁。资源不足或 NVML 异常时，网页和 API 仍可访问，任务可创建但 worker 不应开始执行。

## 5. NVML 异常诊断

```bash
nvidia-smi -L
docker exec quantitize-platform-api nvidia-smi -L
curl -sS http://127.0.0.1:8000/api/gpu
docker inspect quantitize-platform-api --format '{{json .HostConfig.DeviceRequests}}'
```

若宿主机正常而容器失败，在没有运行任务且得到授权后可重建 API 容器，再检查 `/ready`。不要设置 `GPU_READINESS_REQUIRED=0` 规避生产门禁。

## 6. 日志与停止

```bash
docker compose -f deploy/docker-compose.yml logs --tail=100 quantitize-api
docker compose -f deploy/docker-compose.yml logs --tail=100 quantitize-web
```

确需停止时使用普通 `docker compose ... down` 或针对服务的操作；禁止 `docker compose down -v`。在共享机上不要修改全局 Docker 配置。
