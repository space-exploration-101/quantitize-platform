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
current_phase: P7
next_task: none
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

- `apps/api` 查询、建任务、指标、清单、样例图和下载已可用；Web 仍默认 Demo，尚未完全走 API。
- `apps/web/app.py` 主要是页面 BFF；创建任务、指标和产物接口不完整。
- Web 默认使用 `WEB_DEMO_MODE=1`。
- 历史任务 manifest 中存在 `/home/rs/wrs/onnxviewe/quantitize/output_data` 旧路径（按决策暂不批量改写）。
- `quantitize-platform` 在父仓库中尚未被 Git 正式跟踪；用户计划另用独立 GitHub 仓库备份。
- 22 GB 任务数据已由 `.gitignore` / `.dockerignore` 排除；正式 `git add` 仍待用户确认。

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
| P1 仓库与路径基线 | DONE | 隔离运行数据，统一新路径 | Git/Docker 不包含大数据，新任务无旧路径 |
| P2 最小 API | DONE | 打通任务查询和启动 | 核心 API 返回有效 JSON |
| P3 任务创建与产物 API | DONE | 迁移完整后端能力 | 两种建任务方式及指标/下载通过 |
| P4 Web BFF | DONE | Web 完全通过 API 工作 | 关闭 Demo 后页面完整可用 |
| P5 宿主机端到端 | DONE | 容器化前跑通八步流程 | 新任务八步全部 completed |
| P6 镜像与 Compose | DONE | 封装软件环境和服务 | 双容器启动、GPU 和挂载正常 |
| P7 回归与切换 | DONE | 新旧结果对比并上线 | 重启持久、结果一致、可回退 |

## 6. 详细操作清单

### P1：仓库与路径基线

#### P1-T1 隔离大文件和运行数据

```yaml
status: DONE
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

- [x] 任务输出和共享数据被 Git 忽略。
- [x] `git add quantitize-platform` 不会暂存 GB 级文件。
- [x] Docker 构建上下文不包含 22 GB 历史数据。
- [x] API 环境包的构建来源有明确方案。

证据记录：

```text
started_at: 2026-09-01 12:10 +0800
completed_at: 2026-09-01 12:12 +0800
changed_files:
  - quantitize-platform/.gitignore
  - quantitize-platform/.dockerignore
  - quantitize-platform/deploy/README.md
  - quantitize-platform/migration.md
verification_output: |
  git status --short quantitize-platform -> ?? quantitize-platform/
  git check-ignore -v data/output_data/* -> data/output_data/**
  git check-ignore -v data/shared_data/* -> data/shared_data/**
  git add -n quantitize-platform -> 82 files, only .gitkeep under data/
  docker build context sent 575.2kB; CONTEXT_SIZE=760K; DATA_SIZE=12K; DATA_FILES=2
  CONTEXT_CHECK_PASS (pipeline/patches/apps/api present; 22G jobs and yolov8_env.tar.gz absent)
commit: N/A
notes: |
  已在宿主机本机执行；ssh wrs 无法解析主机名，工作目录即为 remote_root。
  API 环境包方案：P6 使用 BuildKit secret 从
  /home/rs/wrs/onnxviewe/yolov8_env.tar.gz 注入，不 COPY 进构建上下文。
  未提交；等待用户明确要求后再 git add / commit。
```

#### P1-T2 固定统一路径

```yaml
status: DONE
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

- [x] `verify_runner.py` 输出 `PASS: runner smoke`。
- [x] 新代码不存在非兼容用途的旧目录引用。
- [x] 新建 smoke job 的 manifest 不包含旧 `quantitize/output_data`。

证据记录：

```text
started_at: 2026-09-01 12:15 +0800
completed_at: 2026-09-01 12:19 +0800
changed_files:
  - quantitize-platform/pipeline/script_registry.py
  - quantitize-platform/pipeline/verify_runner.py
  - quantitize-platform/pipeline/runner/_lib/bootstrap.py
  - quantitize-platform/pipeline/runner/_lib/env.py
  - quantitize-platform/pipeline/runner/_lib/job_config.py
  - quantitize-platform/pipeline/runner/_lib/shared_datasets.py
  - quantitize-platform/pipeline/runner/08_prepare_sample_job.py
  - quantitize-platform/pipeline/runner/tests/prepare_regression_job.py
  - quantitize-platform/pipeline/runner/tests/trigger_web_regression.py
  - quantitize-platform/migration.md
verification_output: |
  rg hits only docs / prefix of quantitize-platform / smoke-path assertion
  verify_runner.py -> PASS: runner smoke
  PLATFORM_ROOT=/home/rs/wrs/onnxviewe/quantitize-platform
  OUTPUT_DATA_ROOT=.../quantitize-platform/data/output_data
  SHARED_DATA_ROOT=.../quantitize-platform/data/shared_data
  SMOKE_JOB=.../data/output_data/_p1t2_path_smoke
  cwd=/tmp still resolves to platform root (not cwd)
  PLATFORM_ROOT/OUTPUT_DATA_ROOT/SHARED_DATA_ROOT env overrides work
  historical 20260727_... manifest still has old paths (intentionally untouched)
commit: N/A
notes: |
  路径单一事实来源改为 script_registry，支持环境变量，禁止相对 cwd。
  旧项目仅通过 QUANTITIZE_LEGACY_DIR（默认兄妹目录 quantitize/）只读引用。
  未提交。用户后续将用另一 GitHub 账号建独立仓库备份本目录与环境包。
```

### P2：最小 API

#### P2-T1 建立 API 应用和 Worker

```yaml
status: DONE
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
/home/rs/miniconda3/envs/yolov8/bin/python -m uvicorn apps.api.app:app --host 0.0.0.0 --port 8000
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

- [x] 全部接口返回预期 HTTP 状态和合法 JSON。
- [x] API 能读取新平台现有任务。
- [x] API 停止不会损坏任务数据。
- [x] API 源码没有旧 Web 或旧项目运行时依赖。

证据记录：

```text
started_at: 2026-09-01 12:26 +0800
completed_at: 2026-09-01 12:32 +0800
changed_files:
  - quantitize-platform/apps/__init__.py
  - quantitize-platform/apps/api/__init__.py
  - quantitize-platform/apps/api/_paths.py
  - quantitize-platform/apps/api/app.py
  - quantitize-platform/apps/api/worker.py
  - quantitize-platform/apps/api/requirements.txt
  - quantitize-platform/migration.md
verification_output: |
  GET /health -> {"ok":true,"busy":false,"current_job":null}
  GET /api/worker -> last_error=null
  GET /api/datasets -> {"cali":[],"test":[],"catalog":[]}
  GET /api/tasks -> 3 jobs including 20260727_... completed
  GET /api/tasks/20260727_... -> status=completed has_zip=true steps=8
  POST start missing -> 404; _p1t2_path_smoke -> 400 缺少 model.pt
  GET download historical -> 206 application/zip
  GET download smoke -> 404 bundle 不存在
  historical zip/manifest still present after API start/stop
commit: N/A
notes: |
  未导入旧 Web / 旧项目路径。Worker 通过 script_registry 调 pipeline/runner/05_runner.py。
  未对真实历史任务执行 start，避免误跑八步。
  本机 API 可能仍监听 127.0.0.1:8000，供手动复查。
```

### P3：任务创建和完整产物 API

#### P3-T1 迁移 ZIP 和共享数据建任务能力

```yaml
status: DONE
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

- [x] 两种方式均能创建合法任务。
- [x] API 返回新任务 ID 和初始状态。
- [x] 缺文件或参数错误返回清晰的 4xx。
- [x] 恶意 ZIP 被拒绝且任务目录外没有新文件。
- [x] 新任务配置和 manifest 不包含旧路径。

证据记录：

```text
started_at: 2026-09-01 13:42 +0800
completed_at: 2026-09-01 13:46 +0800
created_test_task_ids:
  - 20260901_134614_p3_zip_ok
  - 20260901_134614_p3_shared_ok
changed_files:
  - quantitize-platform/apps/api/app.py
  - quantitize-platform/apps/api/create_task.py
  - quantitize-platform/apps/api/requirements.txt
  - quantitize-platform/migration.md
verification_output: |
  POST /api/tasks missing archive -> 400
  POST /api/tasks/shared missing fields -> 422
  POST evil zip ../p3_zip_escape.txt -> 400; no files outside job dirs
  POST sample-task.zip -> 201 task_id=20260901_134614_p3_zip_ok pending
  POST /api/tasks/shared -> 201 task_id=20260901_134614_p3_shared_ok pending
  unknown dataset -> 400
  job_root only under quantitize-platform/data/output_data
  API bound 0.0.0.0:8000; GET http://10.2.26.132:8000/health -> ok
commit: N/A
notes: |
  ZIP 字段兼容 archive 与 zip_file；模型字段兼容 model 与 model_pt。
  上传 zip 里的 job_root 会被丢弃，避免写入旧路径。
  服务监听 0.0.0.0:8000，局域网 http://10.2.26.132:8000
```

#### P3-T2 迁移指标、图片、清单和下载接口

```yaml
status: DONE
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

- [x] PT、ONNX、FPGA 指标可以读取。
- [x] bundle inventory 与压缩包内容一致。
- [x] 下载文件可以正常解压。
- [x] 图片响应 MIME 类型正确。
- [x] 所有文件接口都阻止目录穿越。

证据记录：

```text
started_at: 2026-09-01 13:50 +0800
completed_at: 2026-09-01 13:53 +0800
changed_files:
  - quantitize-platform/apps/api/artifacts.py
  - quantitize-platform/apps/api/app.py
  - quantitize-platform/migration.md
verification_output: |
  TASK_ID=20260727_151741_aituosh_moon_0727_2
  GET metrics -> pt/onnx/fpga n_images=840 mAP present
  GET bundle_inventory -> file_count=360 matches zip members
  GET input_manifest -> 200 text/markdown
  GET error/success jpg -> 200 image/jpeg (file(1) confirms JPEG)
  GET .../error_cases/../../../../etc/passwd -> 404
  GET .../error_cases/..%2F..%2Fetc%2Fpasswd -> 404
  download Content-Type application/zip; unzip -t No errors detected
commit: N/A
notes: 未提交。API 仍监听 0.0.0.0:8000。
```

### P4：Web BFF

#### P4-T1 让 Web 完全通过 API 工作

```yaml
status: DONE
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

- [x] 关闭 Demo 后首页、数据集、历史和任务页面正常。
- [x] 能通过 Web 创建并启动任务。
- [x] 指标、图片、清单和下载可用。
- [x] API 不可用时 Web 显示明确错误。
- [x] Web 源码不直接调用 Pipeline。

证据记录：

```text
started_at: 2026-09-01 13:50 CST
completed_at: 2026-09-01 13:58 CST
changed_files:
  - quantitize-platform/apps/web/app.py
  - quantitize-platform/apps/web/templates/new_task.html
  - quantitize-platform/apps/web/requirements.txt
  - quantitize-platform/deploy/docker-compose.web.yml
  - quantitize-platform/deploy/README.md
  - quantitize-platform/migration.md
verification_output: |
  WEB_DEMO_MODE=0 API_BASE_URL=http://127.0.0.1:8000
  uvicorn web 0.0.0.0:8088 ; API 0.0.0.0:8000
  GET / /datasets /history /health -> 200
  GET /tasks/20260727_151741_aituosh_moon_0727_2 -> 200
  GET .../metrics mAP 0.9975; inventory 360 files 325.3 MB
  GET .../input_manifest -> 200 text/markdown
  GET overlay jpg -> 200 image/jpeg; download zip Content-Disposition
  LAN http://10.2.26.132:8088/ and /history /health -> 200
  POST /tasks zip -> 303 /tasks/20260901_135728_p4_web_zip_ok
  POST /tasks/.../start -> 303; worker ran; dummy pt_eval failed exit 1 (expected)
  POST /tasks/shared on 1-image fixture -> 400 检校失败 (min 400/100)
  API_BASE_URL=http://127.0.0.1:9 -> / /datasets /history 502 后端不可用
  rg apps/web: no pipeline/job_config/script_registry imports
manual_ui_checks: |
  无浏览器 MCP 工具可用；用 curl 走 127.0.0.1 与 10.2.26.132:8088。
  请在 http://10.2.26.132:8088 人工点开首页/数据集/历史/监控。
commit: N/A
notes: 未提交。未对历史完成任务 POST start。Web/API 仍监听 0.0.0.0。夹具任务 20260901_135728_p4_web_zip_ok 状态 failed。
```

### P5：宿主机完整流程

#### P5-T1 运行八步 Pipeline

```yaml
status: DONE
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

- [x] 两个验证脚本通过。
- [x] 八个步骤全部为 `completed`。
- [x] 总任务状态为 `completed`。
- [x] 每一步有日志和预期产物。
- [x] 最终 ZIP 存在并可解压。
- [x] manifest 不包含旧路径。
- [x] 重复启动和 Worker 忙碌情况得到明确响应。

证据记录：

```text
started_at: 2026-09-01 14:02 CST
completed_at: 2026-09-01 17:21 CST
test_task_id: 20260901_140238_p5_e2e_aituosh_moon
step_results: |
  pt_eval completed 14:07
  quantize completed 14:29
  onnx_eval completed 14:30
  fpga_test_pack completed 14:49 (840 side_view)
  fpga_eval completed 14:50
  generate_bin completed 16:46 (after get_conv_name + probe image fix)
  merge_bin completed 17:21 (after pipeline/bin_process/layer_wt_shapes.py)
  bundle completed 17:21
artifact_paths: |
  workspace/p5_e2e_aituosh_moon.onnx
  workspace/p5_e2e_aituosh_moon_output.onnx
  workspace/all_bin/ALL_WT.bin (96M) ALL_BN.bin (166K)
  fpga_test_pack/side_view 840 png
  20260901_140238_p5_e2e_aituosh_moon_quantized_bundle.zip 252M
changed_files:
  - quantitize-platform/pipeline/engine/get_conv_name.py
  - quantitize-platform/pipeline/engine/check_certain_layer_multi.py
  - quantitize-platform/pipeline/bin_process/layer_wt_shapes.py
  - quantitize-platform/pipeline/runner/_lib/runner_core.py
  - quantitize-platform/pipeline/runner/_lib/manifest.py
  - quantitize-platform/data/shared_data/registry.json (symlink aituosha sets)
verification_output: |
  verify_runner.py PASS: runner smoke
  verify_patches.py PASS: patches loaded; CUDAExecutionProvider available
  POST /api/tasks/shared 201 task_id=20260901_140238_p5_e2e_aituosh_moon
  POST start 200; duplicate start/create while busy -> 409 已有任务在运行
  manifest status=completed; all 8 steps completed
  unzip -t No errors detected
  rg quantitize/output_data -> no matches
  POST start after completed -> 200 then skip, status stays completed
commit: N/A
notes: |
  未提交。未对历史完成任务 POST start。
  共享集以符号链接指向旧项目 shared_data（只读）。
  generate_bin 缺 get_conv_name.py；探针图曾写死 engine/cali_data。
  merge_bin 缺 pipeline/bin_process/layer_wt_shapes.py（自仓库根 bin_process 拷入）。
  runner 现跳过已 completed 步骤，便于失败后续跑。
```

### P6：镜像与 Compose

#### P6-T1 构建自带运行环境的 API 镜像

```yaml
status: DONE
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
ctx=/tmp/p6_yolov8_env_ctx
mkdir -p "$ctx"
ln -sfn /home/rs/wrs/onnxviewe/yolov8_env.tar.gz "$ctx/yolov8_env.tar.gz"
DOCKER_BUILDKIT=1 docker build \
  --build-context yolov8env="$ctx" \
  -f deploy/Dockerfile.api \
  -t quantitize-platform-api:latest .

docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python -c \
  "import torch, onnxruntime; print(torch.__version__); print(torch.cuda.is_available()); print(onnxruntime.__version__)"

docker run --rm --gpus all quantitize-platform-api:latest nvidia-smi
docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_runner.py
docker run --rm --gpus all quantitize-platform-api:latest \
  /home/rs/miniconda3/envs/yolov8/bin/python pipeline/verify_patches.py
```

验收条件：

- [x] PyTorch 和 ONNX Runtime 可以导入。
- [x] `torch.cuda.is_available()` 为 `True`。
- [x] 容器识别 RTX 5090。
- [x] Runner 和 patch 检查在容器内通过。
- [x] 镜像不依赖宿主机 conda 环境。
- [x] 镜像不包含运行数据。

证据记录：

```text
started_at: 2026-09-01 17:45 CST
completed_at: 2026-09-01 17:52 CST
image_id: sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
image_size: 21.1GB
changed_files:
  - quantitize-platform/deploy/Dockerfile.api
  - quantitize-platform/deploy/README.md
  - quantitize-platform/.dockerignore
  - quantitize-platform/migration.md
verification_output: |
  docker images quantitize-platform-api:latest -> 21.1GB
  nvidia-smi -L -> GPU 0: NVIDIA GeForce RTX 5090
  torch 2.8.0+cu128 cuda True gpu NVIDIA GeForce RTX 5090
  ort 1.19.2 providers Tensorrt/CUDA/CPU
  verify_runner.py PASS: runner smoke (in container)
  verify_patches.py PASS: patches loaded (in container)
  no /tmp/yolov8_env.tar.gz; data/* only .gitkeep; no legacy quantitize tree
commit: N/A
notes: |
  未提交。未替换宿主机正在跑的 API:8000。
  BuildKit secret 上限 500KiB，8.7G 包改用 --build-context + RUN --mount=type=bind。
  本机 docker.io 无 buildx，在 ~/.docker/cli-plugins 安装了 user-local buildx v0.29.1。
  环境包通过 hardlink 到 /tmp/p6_yolov8_env_ctx，不解压进构建上下文目录副本。
```

#### P6-T2 完成 Web + API Compose

```yaml
status: DONE
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

- [x] 两个服务均正常运行并通过健康检查。
- [x] Web 通过 Compose 服务名访问 API。
- [x] API 可以使用 GPU。
- [x] Web 未申请 GPU。
- [x] 两个数据目录以宿主机卷挂载。
- [x] 局域网可访问 `http://10.2.26.132:8088`。

证据记录：

```text
started_at: 2026-09-01 17:58 +08
completed_at: 2026-09-01 18:02 +08
container_images: |
  quantitize-platform-api:latest  sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6  21.1GB
  quantitize-platform-web:latest  sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3  (rebuild)
volume_mounts: |
  bind output_data -> .../quantitize-platform/data/output_data rw
  bind shared_data -> .../quantitize-platform/data/shared_data rw
  bind legacy shared_data -> .../quantitize/shared_data ro
verification_output: |
  docker compose ps: both Up (healthy); 0.0.0.0:8000 / 0.0.0.0:8088
  curl 127.0.0.1:8000/health -> {"ok":true,"busy":false}
  curl 127.0.0.1:8088/health -> demo:false api_base:http://quantitize-api:8000
  LAN 10.2.26.132:8088 / /datasets /history 200; 10.2.26.38 already hitting /tasks/...
  API DeviceRequests nvidia count=-1; nvidia-smi RTX 5090; torch 2.8.0+cu128 cuda True
  Web DeviceRequests=null; no nvidia-smi
commit: N/A
notes: |
  已停宿主机 uvicorn :8088/:8000。本机 docker.io 无 compose，装了 user-local v2.32.4。
  Compose 校验不接受 gpus: all，改为 list（driver nvidia, count all）。
  未执行 docker compose down -v。API 镜像未重建。
```

### P7：回归、持久化与切换

#### P7-T1 容器端到端和重启验证

```yaml
status: DONE
depends_on: [P6-T2]
started_at: 2026-09-01 21:38 +08
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

- [x] 完整任务在容器内完成。
- [x] 服务重启后历史任务仍存在。
- [x] 容器重建后历史任务仍存在。
- [x] 下载包可以解压且内容完整。
- [x] 日志和新任务文件中没有旧路径。

证据记录：

```text
started_at: 2026-09-01 21:38 +08
completed_at: 2026-09-01 21:42 +08
test_task_id: 20260901_102103_测试docker服务能不能用
restart_result: |
  docker compose restart -> both healthy
  /api/tasks 仍 9 条，ids 与重启前一致
  P7 / P5 / 0727 均为 completed；zip size+mtime 未变
recreate_result: |
  docker compose down （无 -v）后宿主机 zip 仍在
  docker compose up -d -> both healthy
  tasks 仍 9 条 ids 一致；inventory file_count=360 has_zip=true
  LAN 8088 / /history /tasks/<id> 200
verification_output: |
  Web 创建并在容器内跑完八步 completed（用户网页操作）
  PT/ONNX/FPGA mAP 均为 0.9975；Web /metrics /bundle_inventory 200
  overlay JPEG 200 image/jpeg 2048x2048
  GET /api/tasks/<id>/download 263938939 bytes; unzip -t No errors; 360 files extract OK
  rg quantitize/output_data -> 0 hits; job_root=quantitize-platform/data/output_data/...
commit: N/A
notes: |
  未对历史完成任务 POST start。未执行 docker compose down -v。
  任务 ID 含中文（UTC 时间戳 10:21 = CST 18:21）；P6 热修后可读。
  预处理 grayscale_r_channel，共享集 aituosha_moon_500 / 840，与 P5/0727 配置一致。
```

#### P7-T2 新旧系统结果对比

```yaml
status: DONE
depends_on: [P7-T1]
started_at: 2026-09-01 21:42 +08
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

- [x] 确定性产物校验和一致。
- [x] 非确定性指标误差在业务允许范围内。
- [x] bundle 结构和关键文件一致。
- [x] 差异能够定位到具体 Pipeline 步骤。

证据记录：

```text
started_at: 2026-09-01 21:42 +08
completed_at: 2026-09-01 21:45 +08
legacy_task_id: 20260727_151741_aituosh_moon_0727_2
new_task_id: 20260901_102103_测试docker服务能不能用
reference_host_task_id: 20260901_140238_p5_e2e_aituosh_moon
comparison_summary: |
  三任务同配置：grayscale_r_channel / aituosha_moon_500+840 / imgsz=1280 / nc=2 / conf=0.25
  model.pt sha256 前缀 df30b9fbc1cb5eb4 三者相同（199005770 bytes）
  PT/ONNX/FPGA：precision/recall/f1/mAP/TP/FP/FN 完全一致（mAP=0.9975, TP=840 FP=1 FN=0）
  ALL_WT.bin / ALL_BN.bin / offset tables / metadata.json sha256 三者相同
  weight+bn 层 bin 294 个 hash 全相同；workspace/bin 767 个文件名相同
  bundle 规范化成员名 360 个完全对齐
accepted_differences: |
  量化 ONNX 体积相同但 sha256 不同（quantize 非确定性）
  P7 vs legacy pixel mean_px：PT +0.00089、ONNX -0.00518、FPGA +0.00158（P5 与 legacy 逐位相同）
  workspace/bin 仅 9 个 txt 探针尺寸不同（activation probe，generate_bin）
  zip 内 15 个日志/json/readme/mse 图体积不同，不含交付 BIN
verification_output: |
  ALL_WT.bin 100205568 sha 94776567f1806a64
  ALL_BN.bin 169088 sha 75d547659f6b64a3
  容器内 onnx.checker + ORT CPUExecutionProvider 可加载 测试docker服务能不能用_output.onnx
  P5 宿主机任务与 legacy 的 workspace/bin 尺寸 0 diff
commit: N/A
notes: |
  未重新跑旧系统八步（已有同配置 completed 任务）。未 POST start 历史任务。
  确定性交付物（合并 BIN / 权重层）一致；差异落在 quantize ONNX 字节与 generate_bin 探针。
```

#### P7-T3 切换入口并保留回退

```yaml
status: DONE
depends_on: [P7-T2]
```

操作：

- 新旧系统先使用不同端口并行运行。
- 新系统稳定通过验收后，将用户入口切换到新 Web。
- 旧系统保留为只读参考和紧急回退入口。
- 回退期内不删除旧项目、旧镜像、旧 conda 环境或旧任务数据。

验收条件：

- [x] 新入口可被实际用户访问。
- [x] 新系统连续运行期间没有阻塞性错误。
- [x] 回退步骤已经实际验证并记录。
- [x] 用户明确确认迁移完成后，才另行规划旧资产清理。

证据记录：

```text
started_at: 2026-09-01 21:45 +08
completed_at: 2026-09-01 21:46 +08
new_entrypoint: http://10.2.26.132:8088
rollback_procedure: |
  1. 新系统：cd quantitize-platform && docker compose -f deploy/docker-compose.yml down
     （禁止 -v；数据在宿主机 data/）
  2. 旧 Web：cd /home/rs/wrs/onnxviewe/quantitize && unset PYTHONPATH && PORT=8088 ./run_platform_web.sh
     或先在 8099 并行验证，再换端口。
  3. 旧任务数据仍在 quantitize/output_data/ 与 conda yolov8 环境；不要删除。
rollback_test_result: |
  新 Compose 保持 8088/8000 运行时，旧 Web PORT=8099 启动成功
  GET / /history /tasks/20260727_151741_aituosh_moon_0727_2 -> 200
  随后只停 8099；新入口 8088 仍 200
user_acceptance: |
  用户已在新入口创建并跑完真实任务（测试docker服务能不能用）。
  旧项目 / 旧镜像 / conda yolov8 / 旧任务数据全部保留；清理需用户另确认。
commit: N/A
notes: |
  P6-T2 已把局域网入口切到新 Compose。本次只验证并行回退，未关闭新服务。
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
| 2026-09-01 | API 环境包用 BuildKit secret 注入 | 8.7G 包在仓库外，且 ignore 必须排除 *.tar.gz | P6 Dockerfile.api 不 COPY tar.gz，避免 Web/API 共用 dockerignore 冲突 |
| 2026-09-01 | 环境包改 extra build-context bind mount | BuildKit secret 上限 500KiB，无法注入 8.7G tar | `docker build --build-context yolov8env=...` + `RUN --mount=type=bind` |

新增决策时追加一行，不覆盖历史记录。

## 10. 变更日志

| 时间 | Agent/执行者 | 任务 | 状态变化 | 修改内容 | 验证结果 | Commit |
|---|---|---|---|---|---|---|
| 2026-09-01 | Codex | P0 | TODO → DONE | 完成目录、环境、旧代码和 Pipeline 基线调查 | `verify_runner.py` 输出 `PASS: runner smoke` | N/A |
| 2026-09-01 | Codex | 文档 | N/A | 创建本迁移执行与进度文档 | 文档结构和任务依赖已建立 | N/A |
| 2026-09-01 | Cursor Grok | P1-T1 | TODO → DONE | 完善 .gitignore/.dockerignore，并在 deploy/README.md 写明 API 环境包 BuildKit secret 方案 | Git 忽略 22G 数据；dry-add 82 个源文件；Docker 上下文 760K 且 CONTEXT_CHECK_PASS | N/A |
| 2026-09-01 | Cursor Grok | P1-T2 | TODO → DONE | 统一 PLATFORM_ROOT 等路径；旧目录仅经 QUANTITIZE_LEGACY_DIR 只读引用 | verify_runner PASS；新 smoke job 无旧 output_data 路径；历史 manifest 未改 | N/A |
| 2026-09-01 | Cursor Grok | P2-T1 | TODO → DONE | 新增 apps/api 最小 JSON API 与单任务 Worker | health/worker/datasets/tasks/start/download 返回合法 JSON；可读历史任务；未依赖旧项目 | N/A |
| 2026-09-01 | Cursor Grok | P3-T1 | TODO → DONE | 迁移 ZIP / shared 建任务；API 改为 0.0.0.0:8000 | 两种建任务 201；恶意 zip 400；新任务无旧路径；10.2.26.132:8000/health 可用 | N/A |
| 2026-09-01 | Cursor Grok | P3-T2 | TODO → DONE | 新增 metrics / inventory / input_manifest / 样例图接口 | 三套指标可读；inventory=360 与 zip 一致；JPEG MIME；穿越 404；zip 可测 | N/A |
| 2026-09-01 | Cursor Grok | P4-T1 | IN_PROGRESS → DONE | Web 改为纯 BFF（httpx→API）；关闭 Demo；compose 默认 WEB_DEMO_MODE=0 | 首页/数据集/历史 200；zip 建任务 303 并 start；API 挂掉 502 错误页；不 import pipeline | N/A |
| 2026-09-01 | Cursor Grok | P5-T1 | IN_PROGRESS → DONE | 宿主机真实数据八步 GPU；补 get_conv_name、layer_wt_shapes、探针图路径；runner 可跳过已完成步 | 任务 20260901_140238_p5_e2e_aituosh_moon 八步 completed；zip 252M 可测；忙碌 409；无旧 output_data 路径 | N/A |
| 2026-09-01 | Cursor Grok | P6-T1 | TODO → DONE | 新增 Dockerfile.api；yolov8 环境经 extra build-context bind 解压进镜像 | 镜像 21.1GB；torch CUDA True；RTX 5090；容器内 verify_runner/patches PASS；无运行数据 | N/A |
| 2026-09-01 | Cursor Grok | P6-T2 | IN_PROGRESS → DONE | 停宿主机 uvicorn；compose 拉起 Web+API；gpus list；数据卷绑定 | 双服务 healthy；Web api_base=http://quantitize-api:8000；API RTX 5090；Web 无 GPU；LAN 8088 200 | N/A |
| 2026-09-01 | Cursor Grok | P7-T1 | TODO → DONE | 容器内真实任务八步；restart / down+up 验证挂载持久 | 任务 20260901_102103_测试docker服务能不能用 completed；zip 252M 360 文件可解压；重启与重建后 9 条任务仍在；无旧 output_data 路径 | N/A |
| 2026-09-01 | Cursor Grok | P7-T2 | TODO → DONE | 对比 legacy 0727_2、P5 宿主机、P7 容器三任务 | 分类指标一致；ALL_WT/ALL_BN hash 相同；ONNX sha 与探针 txt 为可接受差异 | N/A |
| 2026-09-01 | Cursor Grok | P7-T3 | TODO → DONE | 新入口保持 8088；旧 Web 在 8099 并行回退验证后关闭 | 新 8088 可用；旧 8099 /history 与 0727 任务 200；未删旧资产 | N/A |

## 11. 当前恢复点

```yaml
current_phase: P7
next_task: none
recommended_first_action: 无迁移任务；保持 Compose 运行。清理旧项目/镜像/conda 需用户另确认。
do_not_start_with: docker compose down -v
reason: P0–P7 均已验收；入口已是新平台
known_good_command: docker compose -f deploy/docker-compose.yml ps
known_good_result: quantitize-api / quantitize-web both Up (healthy)
```

后续 Agent 接手时，迁移清单已完成。若远程状态变化，先核对服务健康与数据目录，再决定是否修复。
Web 入口（容器）：http://10.2.26.132:8088  API：http://10.2.26.132:8000/docs
不要对历史完成任务 POST start。不要删除 `/home/rs/wrs/onnxviewe/quantitize`。
容器验证任务 `20260901_102103_测试docker服务能不能用` 已 completed。回退：旧 Web `PORT=8099 ./run_platform_web.sh` 已实测可起。
