# H200 Backup Status

```yaml
backup_status: VERIFIED
created_at: 2026-09-02
snapshot: snapshot-20260902-182606
nas_unc: \\10.2.26.26\902_data\0-项目\13-专项\4-代码\量化平台\H200\snapshot-20260902-182606
nas_linux: /mnt/ywang-nas/0-项目/13-专项/4-代码/量化平台/H200/snapshot-20260902-182606
source: /data3/ywang/quantitize-platform
output_data_included: false
assets:
  project:
    file: quantitize-platform-no-output.tar.zst
    sha256: 5d40756c8a3a5e333a58ddce881e56bd8c680b643bfde3ba1d538347c9e42d9c
    stream_test: PASS
  docker_images:
    file: docker-images-api-web.tar.zst
    sha256: 1ce12391b0b84abf4de369bd8f2a43f704cf9f131514c7e36fb25c049bad3262
    stream_test: PASS
  conda_environment:
    file: yolov8_env.tar.gz
    sha256: bb66795409a56195830641d24620b37f7d3617bc4a424517d2885545d1ac7251
    stream_test: PASS
validation:
  golden_task: 20260902_065807_test_h200
  golden_task_status: completed
  api_container: healthy
  web_container: healthy
```

Restore instructions are in `H200_RECOVERY.md`. `data/output_data` is intentionally outside this system backup and must use a separate retention/archive policy.

