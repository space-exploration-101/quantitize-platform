# Agent Context — Quantitize Platform

```yaml
schema: quantitize-agent-context/v1
updated: 2026-09-02
authority:
  current_facts: this_file
  operations: RUNBOOK.md
  disaster_recovery: ../reference/RECOVERY.md
  environment_rebuild: ../../rebuild/REBUILD_GUIDE.md
  performance: ../reference/PERFORMANCE.md
  historical_only: ../archive/

control_machine:
  os: Windows
  workspace: E:\codex\quantitize
  ssh_config: C:\Users\admin\.ssh\config
  secrets_in_docs: prohibited
  routes:
    - destination: 10.2.29.0/24
      gateway: 192.168.31.1
      interface_index: 19
      persistent: true
      purpose: H200 bypasses Astrill
    - destination: 10.2.26.0/24
      gateway: 192.168.31.1
      interface_index: 19
      persistent: true
      purpose: wrs/NAS bypass Astrill

hosts:
  production:
    alias: H200
    address: 10.2.29.180
    user: ywang
    hostname: user-Rack-Server
    connection: direct
    proxy_jump: prohibited
    os_observed: Ubuntu 22.04.5 LTS
    kernel_observed: 6.8.0-90-generic
    architecture: x86_64
    cpu: 2x Intel Xeon Gold 6530
    logical_cpus: 128
    numa_nodes: 4
    memory: 1 TiB
    shared_server: true
  legacy:
    alias: wrs
    address: 10.2.26.132
    user: rs
    role: historical source only

gpu:
  count: 4
  model: NVIDIA H200 NVL
  memory_each_mib: 143771
  driver_observed: 580.82.07
  compute_capability: sm_90
  mig: disabled
  compose_assignment_uuid: GPU-1610284d-3afe-6e14-b8cb-bbbe40a1f28b
  container_visible_index: 0
  readiness:
    required: true
    min_free_mib: 32768
    max_utilization_percent: 20
    poll_seconds: 30
  allocation_is_not_reservation: true

project:
  host_root: /data3/ywang/quantitize-platform
  local_mirror: E:\codex\quantitize\h200-worktree
  compose: deploy/docker-compose.yml
  compose_env: deploy/.env
  services:
    api:
      container: quantitize-platform-api
      image: quantitize-platform-api:latest
      image_id: sha256:39131c33957dfe1d924047bc7de3236e57f1c33f31e0291886d6c14d746d0ef6
      url: http://10.2.29.180:8000
      health: /health
      readiness: /ready
    web:
      container: quantitize-platform-web
      image: quantitize-platform-web:latest
      image_id: sha256:af4e5f40d7f9948b516d4584b935a190210b208e3f7809080034801ac46534a3
      url: http://10.2.29.180:8088
      health: /health
  restart_policy: unless-stopped
  last_verified_state: healthy_idle

pipeline:
  orchestrator: pipeline/runner/05_runner.py
  calibration_provider_default: cuda
  calibration_provider_fallback: cpu
  performance_report: ../reference/PERFORMANCE_OPTIMIZATION_FINAL_20260903.md
  latest_cuda_validation:
    - id: 20260902_165033_perf_opt_h200_true_cuda_ab1
      elapsed_seconds: 790.513
      status: completed
    - id: 20260902_170544_perf_opt_h200_true_cuda_ab2
      elapsed_seconds: 725.451
      status: completed
  steps:
    - pt_eval
    - quantize
    - onnx_eval
    - fpga_test_pack
    - fpga_eval
    - generate_bin
    - merge_bin
    - bundle
  golden_task:
    id: 20260902_065807_test_h200
    status: completed
    calibration_dataset: aituosha_moon_500
    test_dataset: aituosha_moon_840
    preprocess: grayscale_r_channel
    all_steps_completed: true
    bundle_created: true
    elapsed_seconds: 6252.678

storage:
  project_disk:
    mount: /data3
    medium: SATA SSD
  docker_disk:
    path: /data1/docker
    medium: NVMe
  shared_inputs: /data3/ywang/quantitize-platform/data/shared_data
  task_outputs: /data3/ywang/quantitize-platform/data/output_data
  output_data_system_backup: excluded
  output_data_cleanup_requires_verified_archive: true
  nas:
    address: 10.2.26.26
    share: //10.2.26.26/902_data
    mount: /mnt/ywang-nas
    mount_persistent: false
    system_backup_root: /mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/H200
    agent_docs: /mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/agent_context
    latest_verified_snapshot: snapshot-20260902-182606

permissions_and_safety:
  ywang_in_docker_group: true
  docker_access_is_root_equivalent: true
  sudo_requires_password: true
  password_storage: prohibited
  no_stop_other_users: true
  no_global_host_change_without_admin: true
  forbidden_commands:
    - docker compose down -v
    - delete apt/dpkg lock files
    - broad recursive delete with globs
```
