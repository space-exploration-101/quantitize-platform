# Deploy — Web + API

工作目录：`quantitize-platform` 根目录（本文件所在目录的上一级）。

本机 `docker.io` 不含 Compose / Buildx。已在 `~/.docker/cli-plugins/` 安装 user-local `docker-compose`（v2.32.4）和 `docker-buildx`。

流水线 / 补丁 / 编排设计说明见：[`../pipeline/FLOW.md`](../pipeline/FLOW.md)（M2–M3）。

## API 环境包

当前生产 API 镜像基于 **`ywang/yolo-gray1-quant:ultralytics-8.3.98-v6`**（1 通道 gray1，Ultralytics 8.3.98）。不再解压仓库外的 `yolov8_env.tar.gz`（那是 8.4.6 / 3 通道旧环境）。

构建：

```bash
cd /data3/ywang/quantitize-platform
docker tag quantitize-platform-api:latest quantitize-platform-api:pre-gray1-20260910
docker build -f deploy/Dockerfile.api -t quantitize-platform-api:latest .
```

旧 21GB conda 打包方式（`yolov8_env.tar.gz` + extra build-context）仅作历史参考，见仓库内 `deploy/Dockerfile.api` 更早版本 / 镜像标签 `pre-gray1-20260910`。

检查：

```bash
docker run --rm quantitize-platform-api:latest \
  /opt/ywang/yolo-runtime/bin/python -c \
  "import ultralytics, fastapi; print(ultralytics.__version__); print(fastapi.__version__)"
docker run --rm quantitize-platform-api:latest \
  /opt/ywang/yolo-runtime/bin/python pipeline/verify_runner.py
```

## 打包 Web 镜像

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
docker build -f deploy/Dockerfile.web -t quantitize-platform-web:latest .
```

或用 compose：

```bash
docker compose -f deploy/docker-compose.web.yml build
```

## 启动服务（Web + API）

正式入口是 `deploy/docker-compose.yml`。API 用已构建的 `quantitize-platform-api:latest`（`--build` 只重建 Web，不会再解压 8.7G 环境包）。Web 通过 Compose 服务名 `http://quantitize-api:8000` 访问 API。

启动前必须把 `GPU_DEVICE_ID` 设置为管理员分配的单卡 UUID、卡号或 MIG UUID；不再向容器暴露全部 GPU。推荐使用 UUID，避免设备序号变化：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
export GPU_DEVICE_ID=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
docker compose -f deploy/docker-compose.yml up -d --build
```

GPU readiness 默认策略：空闲显存至少 24576 MiB、当前利用率不高于 90%，每 15 秒轮询一次。资源不足时 API 和 Web 仍保持健康，任务进入 `waiting_gpu`，达到门槛后才自动运行。按实际 golden task 峰值可覆盖：

```bash
export GPU_MIN_FREE_MIB=24576
export GPU_MAX_UTILIZATION_PERCENT=90
export GPU_READY_POLL_SECONDS=15
```

如使用 `1g.18gb` MIG，必须依据实测峰值调低 `GPU_MIN_FREE_MIB`。仅在诊断场景下才设置 `GPU_READINESS_REQUIRED=0` 绕过门禁。

端口：宿主机 **8088** → Web 8080，**8000** → API 8000。`WEB_DEMO_MODE=0`。

浏览器：http://127.0.0.1:8088/ （局域网 http://10.2.26.132:8088 ）。

禁止：`docker compose down -v`（会误伤命名卷；本编排用的是宿主机绑定挂载，停服务用普通 `down`）。

仅排查前端、不连后端时才设 `WEB_DEMO_MODE=1`，或用 `docker-compose.web.yml` 单启 Web。

换端口示例：`PORT=8090 API_PORT=8001 docker compose -f deploy/docker-compose.yml up -d`

## 查看日志

```bash
docker compose -f deploy/docker-compose.yml logs -f --tail=100
docker compose -f deploy/docker-compose.yml logs --tail=100 quantitize-api
docker compose -f deploy/docker-compose.yml logs --tail=100 quantitize-web
```

## 容器状态

```bash
docker compose -f deploy/docker-compose.yml ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8000/api/gpu
curl -fsS http://127.0.0.1:8088/health
```

`/health` 是服务存活检查，不会因 GPU 繁忙而失败；`/ready` 是 GPU 资源门禁，资源不足时返回 HTTP 503。

进入容器：

```bash
docker exec -it quantitize-platform-api bash
docker exec -it quantitize-platform-web bash
```

## 停止 / 删除

```bash
# 不要加 -v
docker compose -f deploy/docker-compose.yml down
```
