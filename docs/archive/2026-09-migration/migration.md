# Quantitize Platform 迁移执行与进度文档

> 本文档是 `/home/rs/wrs/onnxviewe/quantitize-platform` 的迁移工作台。
> 后续 Agent 应先阅读本文档，再检查远程实际状态；每完成一个任务，必须更新状态、验证结果和变更记录。

## 1. 任务元数据

```yaml
project: quantitize-platform
remote_host: wrs
remote_root: /home/rs/wrs/onnxviewe/quantitize-platform
legacy_root: /home/rs/wrs/onnxviewe/quantitize
repository_root: /home/rs/wrs/onnxviewe
runtime_python: /home/rs/miniconda3/envs/yolov8/bin/python3
target_web_port: 8088
target_api_port: 8000
target_architecture: web-bff + api-worker + pipeline + mounted-data
document_status: active
current_phase: P1
next_task: P1-T1
last_updated: 2026-09-01
```

状态取值：

- `TODO`：尚未开始。
- `IN_PROGRESS`：正在执行；同一时间只能有一个任务处于此状态。
- `BLOCKED`：存在明确阻塞，必须记录原因和解除条件。
- `DONE`：操作完成，且所有验收条件通过。
- `SKIPPED`：经确认无需执行，必须记录原因。

## 2. 最终目标

将旧项目中的核心能力和已验证的软件运行环境迁移到结构清晰、可测试、可通过 Docker 部署的新平台中：

```text
quantitize-platform/
├── apps/
│   ├── api/                 # JSON API、任务创建、Worker、文件服务
│   └── web/                 # 页面和表单，只通过 HTTP 调用 API
├── pipeline/
│   ├── engine/              # 核心算法脚本
│   ├── runner/              # 八步流程编排
│   ├── script_registry.py
│   └── verify_runner.py
├── patches/                 # 第三方依赖补丁
├── deploy/
│   ├── Dockerfile.api
│   ├── Dockerfile.web
│   └── docker-compose.yml
└── data/
    ├── output_data/         # 宿主机挂载，不进入 Git 和镜像
    └── shared_data/         # 宿主机挂载，不进入 Git 和镜像
```

最终必须满足：

1. Web、API、Worker、Pipeline 职责分离。
2. API 镜像自带已验证的 `yolov8` 软件运行环境并可使用 RTX 5090。
3. Web 镜像不需要 CUDA、GPU 或算法环境。
4. 任务数据通过宿主机目录挂载，容器重建后仍存在。
5. 八步 Pipeline 能在容器内完整运行。
6. 新任务和新日志不再依赖旧目录。
7. 旧系统在验收完成前始终可用于回退。

## 3. 已确认基线

以下结论来自 2026-09-01 的只读检查，不等于相关迁移任务已经完成。

```yaml
ssh_alias: wrs
ssh_passwordless: true
docker_available: true
gpu: NVIDIA GeForce RTX 5090
gpu_driver: 580.173.02
platform_size: 22G
legacy_project_size: 57G
runtime_archive: /home/rs/wrs/onnxviewe/yolov8_env.tar.gz
runtime_archive_size: 8.7G
apps_web_present: true
apps_api_implemented: false
web_demo_mode_default: true
runner_smoke_passed: true
pipeline_step_count: 8
platform_tracked_by_git: false
historical_manifest_has_legacy_paths: true
```

已通过的 Runner 检查：

```text
registered=17
PLATFORM_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform
OUTPUT_DATA_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform/data/output_data
SHARED_DATA_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform/data/shared_data
yolov8_python=/home/rs/miniconda3/envs/yolov8/bin/python3.9
PASS: runner smoke
```

当前已知缺口：

- `apps/api` 只有 `.gitkeep`，新平台后端尚未实现。
- `apps/web/app.py` 主要是页面 BFF；创建任务、指标和产物接口不完整。
- Web 默认使用 `WEB_DEMO_MODE=1`。
- 历史任务 manifest 中存在 `/home/rs/wrs/onnxviewe/quantitize/output_data` 旧路径。
- `quantitize-platform` 在父仓库中尚未被 Git 正式跟踪。
- 22 GB 任务数据必须在 `git add` 和 Docker build 前排除。

## 4. Agent 执行规则

后续 Agent 必须遵循以下流程：

1. 运行 `ssh wrs "echo ok"`，确认远程可达。
2. 阅读本文档中的 `current_phase`、`next_task` 和对应任务记录。
3. 检查远程实际文件及 `git status`，不得只相信文档。
4. 开始任务前将状态改为 `IN_PROGRESS`，记录开始时间。
5. 只修改当前任务需要的文件，保留用户已有及无关改动。
6. 运行任务列出的全部验证命令。
7. 只有验收条件全部满足，才能将状态改为 `DONE`。
8. 在“变更日志”中记录时间、文件、验证结果和 Git commit（如有）。
9. 更新 `next_task`；不要同时开始多个有依赖关系的任务。

安全约束：

- 不删除或覆盖 `/home/rs/wrs/onnxviewe/quantitize`。
- 不清理历史任务，不执行针对 `data/` 的递归删除。
- 不使用 `git reset --hard` 或 `git checkout --` 丢弃现有修改。
- 不提交模型、数据集、任务结果、日志、ZIP、tar 包或 conda 环境包。
- 执行 `docker compose down` 时不得添加 `-v`。
- 历史 manifest 的旧路径先保留；需要修复时必须编写可审计的独立迁移工具。
- 每个阶段结束前都要确认旧系统仍然可以作为回退入口。

## 5. 阶段总览

| 阶段 | 状态 | 目标 | 完成条件 |
|---|---|---|---|
| P0 基线调查 | DONE | 确认目录、环境和已有代码 | 调查结果记录在本文档 |
| P1 仓库与路径基线 | TODO | 隔离运行数据，统一新路径 | Git/Docker 不包含大数据，新任务无旧路径 |
| P2 最小 API | TODO | 打通任务查询和启动 | 核心 API 返回有效 JSON |
| P3 任务创建与产物 API | TODO | 迁移完整后端能力 | 两种建任务方式及指标/下载通过 |
| P4 Web BFF | TODO | Web 完全通过 API 工作 | 关闭 Demo 后页面完整可用 |
| P5 宿主机端到端 | TODO | 容器化前跑通八步流程 | 新任务八步全部 completed |
| P6 镜像与 Compose | TODO | 封装软件环境和服务 | 双容器启动、GPU 和挂载正常 |
| P7 回归与切换 | TODO | 新旧结果对比并上线 | 重启持久、结果一致、可回退 |

## 6. 详细操作清单

### P1：仓库与路径基线

#### P1-T1 隔离大文件和运行数据

```yaml
status: TODO
depends_on: []
files_expected:
  - quantitize-platform/.gitignore
  - quantitize-platform/.dockerignore
```

操作：

- 创建或完善 `.gitignore`，排除 `data/output_data/*`、`data/shared_data/*`、日志、缓存、模型、ZIP、tar 包和 `.env`。
- 为两个数据目录保留 `.gitkeep`。
- 创建或完善 `.dockerignore`，确保 Web 构建上下文不包含运行数据。
- 明确 API 镜像如何取得 `yolov8_env.tar.gz`，避免忽略规则导致 API 镜像无法构建。

验证：

```bash
cd /home/rs/wrs/onnxviewe
git status --short quantitize-platform
git check-ignore -v quantitize-platform/data/output_data/*
git check-ignore -v quantitize-platform/data/shared_data/*
```

验收条件：

- [ ] 任务输出和共享数据被 Git 忽略。
- [ ] `git add quantitize-platform` 不会暂存 GB 级文件。
- [ ] Docker 构建上下文不包含 22 GB 历史数据。
- [ ] API 环境包的构建来源有明确方案。

证据记录：

```text
started_at:
completed_at:
changed_files:
verification_output:
commit:
notes:
```

#### P1-T2 固定统一路径

```yaml
status: TODO
depends_on: [P1-T1]
```

标准路径：

```text
PLATFORM_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform
OUTPUT_DATA_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform/data/output_data
SHARED_DATA_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform/data/shared_data
YOLOV8_PYTHON=/home/rs/miniconda3/envs/yolov8/bin/python3
```

操作：

- 所有新代码从统一配置或环境变量获取路径。
- 禁止通过当前工作目录猜测数据目录。
- 搜索 `apps`、`pipeline`、`deploy`、`patches` 中残留的旧绝对路径。
- 不批量改写历史任务；先保证新任务只产生新路径。

验证：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
rg -n "/home/rs/wrs/onnxviewe/quantitize|quantitize/output_data|quantitize/shared_data" apps pipeline deploy patches
/home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py
```

验收条件：

- [ ] `verify_runner.py` 输出 `PASS: runner smoke`。
- [ ] 新代码不存在非兼容用途的旧目录引用。
- [ ] 新建 smoke job 的 manifest 不包含旧 `quantitize/output_data`。

证据记录：

```text
started_at:
completed_at:
changed_files:
verification_output:
commit:
notes:
```

### P2：最小 API

#### P2-T1 建立 API 应用和 Worker

```yaml
status: TODO
depends_on: [P1-T2]
source_references:
  - quantitize/docker/api_app.py
  - quantitize/platform/web/worker.py
target_files:
  - quantitize-platform/apps/api/app.py
  - quantitize-platform/apps/api/worker.py
```

第一批接口：

```text
GET  /health
GET  /api/worker
GET  /api/datasets
GET  /api/tasks
GET  /api/tasks/{task_id}
POST /api/tasks/{task_id}/start
GET  /api/tasks/{task_id}/download
```

操作：

- 将旧 API 和 Worker 的可复用逻辑迁移到 `apps/api`。
- API 使用新平台的 registry 和数据目录。
- Worker 负责启动 Runner，并防止同一任务重复运行。
- API 不导入旧 Web 应用，不依赖旧项目路径。

宿主机启动命令：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
export PYTHONPATH="$PWD/pipeline:$PWD/pipeline/runner/_lib:$PWD/pipeline/engine"
export OUTPUT_DATA_ROOT="$PWD/data/output_data"
export SHARED_DATA_ROOT="$PWD/data/shared_data"
export YOLOV8_PYTHON="/home/rs/miniconda3/envs/yolov8/bin/python3"
/home/rs/miniconda3/envs/yolov8/bin/python -m uvicorn apps.api.app:app --host 127.0.0.1 --port 8000
```

验证：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/api/worker
curl -fsS http://127.0.0.1:8000/api/datasets
curl -fsS http://127.0.0.1:8000/api/tasks
curl -fsS http://127.0.0.1:8000/api/tasks/20260727_151741_aituosh_moon_0727_2
```

验收条件：

- [ ] 全部接口返回预期 HTTP 状态和合法 JSON。
- [ ] API 能读取新平台现有任务。
- [ ] API 停止不会损坏任务数据。
- [ ] API 源码没有旧 Web 或旧项目运行时依赖。

证据记录：

```text
started_at:
completed_at:
changed_files:
verification_output:
commit:
notes:
```

### P3：任务创建和完整产物 API

#### P3-T1 迁移 ZIP 和共享数据建任务能力

```yaml
status: TODO
depends_on: [P2-T1]
source_reference: quantitize/platform/web/app.py
```

目标接口：

```text
POST /api/tasks
POST /api/tasks/shared
```

必须迁移：

- 安全解压和输入目录识别。
- ZIP 大小限制。
- 模型、校准集和测试集处理。
- `job_config.json` 解析和 `JobConfig` 校验。
- 数据集类别数 `nc` 检查。
- input manifest 生成。
- 防止 ZIP 路径穿越。

验证：

```bash
curl -f -F "archive=@/path/to/sample-task.zip" http://127.0.0.1:8000/api/tasks

curl -f \
  -F "model=@/path/to/model.pt" \
  -F "cali_dataset_id=<校准集ID>" \
  -F "test_dataset_id=<测试集ID>" \
  -F "imgsz=1280" \
  http://127.0.0.1:8000/api/tasks/shared
```

验收条件：

- [ ] 两种方式均能创建合法任务。
- [ ] API 返回新任务 ID 和初始状态。
- [ ] 缺文件或参数错误返回清晰的 4xx。
- [ ] 恶意 ZIP 被拒绝且任务目录外没有新文件。
- [ ] 新任务配置和 manifest 不包含旧路径。

证据记录：

```text
started_at:
completed_at:
created_test_task_ids:
changed_files:
verification_output:
commit:
notes:
```

#### P3-T2 迁移指标、图片、清单和下载接口

```yaml
status: TODO
depends_on: [P3-T1]
```

目标接口：

```text
GET /api/tasks/{task_id}/metrics
GET /api/tasks/{task_id}/bundle_inventory
GET /api/tasks/{task_id}/input_manifest
GET /api/tasks/{task_id}/results/{eval_name}/error_cases/{filename}
GET /api/tasks/{task_id}/results/{eval_name}/success_cases/{filename}
GET /api/tasks/{task_id}/download
```

验证：

```bash
TASK_ID=20260727_151741_aituosh_moon_0727_2
curl -fsS "http://127.0.0.1:8000/api/tasks/$TASK_ID/metrics"
curl -fsS "http://127.0.0.1:8000/api/tasks/$TASK_ID/bundle_inventory"
curl -fsS "http://127.0.0.1:8000/api/tasks/$TASK_ID/input_manifest"
curl -fOJ "http://127.0.0.1:8000/api/tasks/$TASK_ID/download"
curl -i "http://127.0.0.1:8000/api/tasks/$TASK_ID/results/pt_eval/error_cases/../../../../etc/passwd"
```

验收条件：

- [ ] PT、ONNX、FPGA 指标可以读取。
- [ ] bundle inventory 与压缩包内容一致。
- [ ] 下载文件可以正常解压。
- [ ] 图片响应 MIME 类型正确。
- [ ] 所有文件接口都阻止目录穿越。

证据记录：

```text
started_at:
completed_at:
changed_files:
verification_output:
commit:
notes:
```

### P4：Web BFF

#### P4-T1 让 Web 完全通过 API 工作

```yaml
status: TODO
depends_on: [P3-T2]
target_file: quantitize-platform/apps/web/app.py
```

目标页面和动作：

```text
GET  /
GET  /datasets
GET  /history
GET  /tasks/{task_id}
POST /tasks
POST /tasks/shared
POST /tasks/{task_id}/start
GET  /tasks/{task_id}/metrics
GET  /tasks/{task_id}/bundle_inventory
GET  /tasks/{task_id}/download
GET  /tasks/{task_id}/input_manifest
```

操作：

- Web 只负责模板、表单、跳转和后端错误展示。
- Web 不直接读取任务目录，不直接启动 Runner。
- 使用 `API_BASE_URL` 访问 API。
- 正式模式设置 `WEB_DEMO_MODE=0`。

验证：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
export WEB_DEMO_MODE=0
export API_BASE_URL=http://127.0.0.1:8000
/home/rs/miniconda3/envs/yolov8/bin/python -m uvicorn apps.web.app:app --host 0.0.0.0 --port 8088
```

```bash
curl -f http://127.0.0.1:8088/
curl -f http://127.0.0.1:8088/datasets
curl -f http://127.0.0.1:8088/history
```

浏览器验证地址：`http://10.2.26.132:8088`

验收条件：

- [ ] 关闭 Demo 后首页、数据集、历史和任务页面正常。
- [ ] 能通过 Web 创建并启动任务。
- [ ] 指标、图片、清单和下载可用。
- [ ] API 不可用时 Web 显示明确错误。
- [ ] Web 源码不直接调用 Pipeline。

证据记录：

```text
started_at:
completed_at:
changed_files:
verification_output:
manual_ui_checks:
commit:
notes:
```

### P5：宿主机完整流程

#### P5-T1 运行八步 Pipeline

```yaml
status: TODO
depends_on: [P4-T1]
pipeline_steps:
  - pt_eval
  - quantize
  - onnx_eval
  - fpga_test_pack
  - fpga_eval
  - generate_bin
  - merge_bin
  - bundle
```

操作与验证：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
/home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py
/home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_patches.py
curl -X POST "http://127.0.0.1:8000/api/tasks/<任务ID>/start"
watch -n 2 "curl -s http://127.0.0.1:8000/api/tasks/<任务ID>"
```

完成后检查：

```bash
/home/rs/miniconda3/envs/yolov8/bin/python -m json.tool data/output_data/<任务ID>/manifest.json
find data/output_data/<任务ID> -maxdepth 4 -type f | sort
rg -n "/home/rs/wrs/onnxviewe/quantitize/output_data" data/output_data/<任务ID>
```

验收条件：

- [ ] 两个验证脚本通过。
- [ ] 八个步骤全部为 `completed`。
- [ ] 总任务状态为 `completed`。
- [ ] 每一步有日志和预期产物。
- [ ] 最终 ZIP 存在并可解压。
- [ ] manifest 不包含旧路径。
- [ ] 重复启动和 Worker 忙碌情况得到明确响应。

证据记录：

```text
started_at:
completed_at:
test_task_id:
step_results:
artifact_paths:
verification_output:
commit:
notes:
```

### P6：镜像与 Compose

#### P6-T1 构建自带运行环境的 API 镜像

```yaml
status: TODO
depends_on: [P5-T1]
source_reference: quantitize/docker/Dockerfile.api
target_file: quantitize-platform/deploy/Dockerfile.api
base_image_candidate: nvidia/cuda:12.8.1-runtime-ubuntu24.04
```

镜像必须包含：

- `/home/rs/miniconda3/envs/yolov8` 或等价的可迁移环境。
- `apps/api`、`pipeline`、`patches` 和必要配置。

镜像不得包含：

- `data/output_data`、`data/shared_data`。
- 旧 `quantitize` 项目。
- 历史任务、模型、日志和压缩包。

构建和验证：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
docker build -f deploy/Dockerfile.api -t quantitize-platform-api:latest .

docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python -c \
  "import torch, onnxruntime; print(torch.__version__); print(torch.cuda.is_available()); print(onnxruntime.__version__)"

docker run --rm --gpus all quantitize-platform-api:latest nvidia-smi
```

验收条件：

- [ ] PyTorch 和 ONNX Runtime 可以导入。
- [ ] `torch.cuda.is_available()` 为 `True`。
- [ ] 容器识别 RTX 5090。
- [ ] Runner 和 patch 检查在容器内通过。
- [ ] 镜像不依赖宿主机 conda 环境。
- [ ] 镜像不包含运行数据。

证据记录：

```text
started_at:
completed_at:
image_id:
image_size:
changed_files:
verification_output:
commit:
notes:
```

#### P6-T2 完成 Web + API Compose

```yaml
status: TODO
depends_on: [P6-T1]
target_file: quantitize-platform/deploy/docker-compose.yml
```

目标服务：

- `quantitize-api`：GPU、Worker、Pipeline、数据卷、8000 端口。
- `quantitize-web`：无 GPU，通过 `http://quantitize-api:8000` 调用 API，8088 端口。

启动和验证：

```bash
cd /home/rs/wrs/onnxviewe/quantitize-platform
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs --tail=100 quantitize-api
docker compose -f deploy/docker-compose.yml logs --tail=100 quantitize-web
curl -fsS http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8088/
```

验收条件：

- [ ] 两个服务均正常运行并通过健康检查。
- [ ] Web 通过 Compose 服务名访问 API。
- [ ] API 可以使用 GPU。
- [ ] Web 未申请 GPU。
- [ ] 两个数据目录以宿主机卷挂载。
- [ ] 局域网可访问 `http://10.2.26.132:8088`。

证据记录：

```text
started_at:
completed_at:
container_images:
volume_mounts:
verification_output:
commit:
notes:
```

### P7：回归、持久化与切换

#### P7-T1 容器端到端和重启验证

```yaml
status: TODO
depends_on: [P6-T2]
```

操作：

1. 通过 Web 创建真实测试任务。
2. 启动并等待八步完成。
3. 查看指标、样例图片和 bundle inventory。
4. 下载并解压最终 ZIP。
5. 重启服务并确认历史任务仍存在。
6. 删除并重建容器，但保留宿主机数据目录。

验证：

```bash
docker compose -f deploy/docker-compose.yml restart
curl -fsS http://127.0.0.1:8000/api/tasks

docker compose -f deploy/docker-compose.yml down
docker compose -f deploy/docker-compose.yml up -d
curl -fsS http://127.0.0.1:8000/api/tasks
```

验收条件：

- [ ] 完整任务在容器内完成。
- [ ] 服务重启后历史任务仍存在。
- [ ] 容器重建后历史任务仍存在。
- [ ] 下载包可以解压且内容完整。
- [ ] 日志和新任务文件中没有旧路径。

证据记录：

```text
started_at:
completed_at:
test_task_id:
restart_result:
recreate_result:
verification_output:
commit:
notes:
```

#### P7-T2 新旧系统结果对比

```yaml
status: TODO
depends_on: [P7-T1]
```

使用相同模型、数据和参数，在旧系统与新系统各运行一次。对比：

- PT、ONNX、FPGA 指标。
- 量化模型是否可加载。
- BIN 文件数量、名称和尺寸。
- 合并产物和最终 bundle 文件清单。
- `job_config.json` 和八步日志。

校验和命令：

```bash
find data/output_data/<任务ID> -type f \
  \( -name '*.onnx' -o -name '*.bin' -o -name '*.zip' \) \
  -exec sha256sum {} \;
```

验收条件：

- [ ] 确定性产物校验和一致。
- [ ] 非确定性指标误差在业务允许范围内。
- [ ] bundle 结构和关键文件一致。
- [ ] 差异能够定位到具体 Pipeline 步骤。

证据记录：

```text
started_at:
completed_at:
legacy_task_id:
new_task_id:
comparison_summary:
accepted_differences:
verification_output:
commit:
notes:
```

#### P7-T3 切换入口并保留回退

```yaml
status: TODO
depends_on: [P7-T2]
```

操作：

- 新旧系统先使用不同端口并行运行。
- 新系统稳定通过验收后，将用户入口切换到新 Web。
- 旧系统保留为只读参考和紧急回退入口。
- 回退期内不删除旧项目、旧镜像、旧 conda 环境或旧任务数据。

验收条件：

- [ ] 新入口可被实际用户访问。
- [ ] 新系统连续运行期间没有阻塞性错误。
- [ ] 回退步骤已经实际验证并记录。
- [ ] 用户明确确认迁移完成后，才另行规划旧资产清理。

证据记录：

```text
started_at:
completed_at:
new_entrypoint:
rollback_procedure:
rollback_test_result:
user_acceptance:
commit:
notes:
```

## 7. 自动化验证矩阵

| 层级 | 是否需要 GPU | 执行时机 | 必须覆盖 |
|---|---:|---|---|
| Runner smoke | 否 | 每次 Pipeline 修改后 | 注册表、目录、Runner 基础行为 |
| Patch verification | 视实现而定 | 依赖和镜像修改后 | ORT/算子补丁存在且有效 |
| API smoke | 否 | 每次 API 修改后 | health、任务、数据集、错误码 |
| API security | 否 | 上传和文件接口修改后 | ZIP/文件路径穿越、非法 task ID |
| Web smoke | 否 | 每次 Web 修改后 | 首页、数据集、历史、任务页 |
| GPU integration | 是 | Pipeline 或镜像发布前 | PyTorch、ORT、CUDA、八步流程 |
| Persistence | 是 | Compose/卷修改后 | restart、down/up 后任务仍存在 |
| Legacy parity | 是 | 正式切换前 | 指标、BIN、bundle、配置一致性 |

## 8. Git 提交建议

建议按以下粒度提交，便于审查和回退：

```text
chore: define platform layout and ignore runtime data
refactor: normalize platform data paths
feat: add backend task and worker API
feat: add metrics and artifact API
refactor: make web frontend use backend API
build: add CUDA API image
build: add web-api compose deployment
test: add API and pipeline smoke tests
docs: add deployment and migration guide
```

每次提交前运行：

```bash
cd /home/rs/wrs/onnxviewe
git status --short
git diff --stat
git diff --check
git diff --cached --stat
git diff --cached --name-only
```

必须确认暂存区中没有：模型、数据集、任务结果、日志、ZIP、tar 包和运行环境。

## 9. 决策日志

| 日期 | 决策 | 原因 | 影响 |
|---|---|---|---|
| 2026-09-01 | 采用 Web + API/Worker 双服务 | Web 不需要 GPU，算法环境集中在 API | 可独立升级前端和算法环境 |
| 2026-09-01 | 运行数据使用宿主机挂载 | 平台目录含约 22 GB 数据 | 缩小 Git 仓库和镜像，支持持久化 |
| 2026-09-01 | 首版 API 镜像复用 yolov8_env.tar.gz | 宿主机环境已被实际验证 | 先保证运行一致性，后续再优化环境构建 |
| 2026-09-01 | 不直接批量修改历史 manifest | 历史产物可能依赖原路径 | 先保证新任务正确，再独立迁移历史数据 |
| 2026-09-01 | 先宿主机端到端，再构建镜像 | 当前最大缺口是后端边界和功能闭环 | 将代码问题与容器问题分开定位 |

新增决策时追加一行，不覆盖历史记录。

## 10. 变更日志

| 时间 | Agent/执行者 | 任务 | 状态变化 | 修改内容 | 验证结果 | Commit |
|---|---|---|---|---|---|---|
| 2026-09-01 | Codex | P0 | TODO → DONE | 完成目录、环境、旧代码和 Pipeline 基线调查 | `verify_runner.py` 输出 `PASS: runner smoke` | N/A |
| 2026-09-01 | Codex | 文档 | N/A | 创建本迁移执行与进度文档 | 文档结构和任务依赖已建立 | N/A |

## 11. 当前恢复点

```yaml
current_phase: P1
next_task: P1-T1
recommended_first_action: 检查并创建 quantitize-platform/.gitignore 与 .dockerignore
do_not_start_with: Dockerfile.api
reason: 必须先隔离 22G 运行数据并固定路径，否则 Git 提交和镜像构建存在大文件及旧路径风险
known_good_command: /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py
known_good_result: "PASS: runner smoke"
```

后续 Agent 接手时，应从 `P1-T1` 开始；如果远程状态已经变化，先核对验收条件，再决定补做、标记完成或记录阻塞。
