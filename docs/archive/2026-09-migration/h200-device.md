# H200 Device Profile

> Local inventory for automation agents. Captured on 2026-09-01 and updated after production validation on 2026-09-02. Do not place plaintext passwords in this file.

```yaml
device:
  label: H200
  ssh_alias: H200
  connection_mode: direct
  proxy_jump: null
  address: 10.2.29.180
  user: ywang
  hostname: user-Rack-Server
  ssh_key: C:\\Users\\admin\\.ssh\\id_ed25519
  password_stored: false

local_route:
  destination: 10.2.29.0/24
  gateway: 192.168.31.1
  interface_index: 19
  metric: 1
  policy_store: PersistentStore
  purpose: bypass Astrill VPN for the H200 subnet

host:
  os: Ubuntu 22.04.5 LTS
  kernel: 6.8.0-90-generic
  architecture: x86_64
  cpu: INTEL XEON GOLD 6530
  sockets: 2
  cores_per_socket: 32
  logical_cpus: 128
  numa_nodes: 4
  memory_total: 1.0 TiB
  swap_total: 2.0 GiB
  swap_observed_used: 2.0 GiB

gpu:
  count: 4
  model: NVIDIA H200 NVL
  memory_each_mib: 143771
  compute_capability: "9.0"
  required_torch_arch: sm_90
  driver: 580.82.07
  mig_mode: Disabled

storage:
  root:
    mount: /
    size: 879 GiB
    free_observed: 184 GiB
    usage_observed: 78%
  data1:
    mount: /data1
    size: 3.5 TiB
    free_observed: 551 GiB
  data2:
    mount: /data2
    size: 3.5 TiB
    free_observed: 351 GiB
  data3:
    mount: /data3
    size: 3.5 TiB
    free_observed: 2.2 TiB
    writable_by_ywang: true
  recommended_user_root: /data3/ywang
  recommended_project_volume: /data3/ywang/quantitize-platform

containers:
  docker_service: active
  docker_client: 29.1.3
  compose: 2.37.1
  buildx: missing
  nvidia_container_toolkit: 1.17.8
  ywang_in_docker_group: true
  docker_access_as_ywang: allowed
  passwordless_sudo: false
  access_model_used_for_migration: docker_group
  docker_root: /data1/docker

quantitize_deployment:
  observed_at: 2026-09-02
  project_root: /data3/ywang/quantitize-platform
  staging_root: /data3/ywang/quantitize-migration
  api_container: quantitize-platform-api
  api_status: healthy
  api_url: http://10.2.29.180:8000
  web_container: quantitize-platform-web
  web_status: healthy
  web_url: http://10.2.29.180:8088
  observed_uptime: 10 hours
  docker_free_observed: 530 GiB
  data3_free_observed: 2.0 TiB
  golden_task_id: 20260902_065807_test_h200
  golden_task_status: completed

network:
  interface: ens24f0
  address: 10.2.29.180/24
  gateway: 10.2.29.1
  nas_address: 10.2.26.26
  nas_smb_port_445: reachable
  cifs_utils: installed
  nas_mount: /mnt/ywang-nas
  nas_share: //10.2.26.26/902_data
  nas_mount_persistent_in_fstab: false
  latest_verified_backup: snapshot-20260902-182606
  ports_8000_8088_observed: free
```

## Connection

From this Windows machine:

```powershell
ssh H200
```

Non-interactive verification:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 H200 "whoami; hostname"
```

Expected identity:

```text
ywang
user-Rack-Server
```

The SSH configuration intentionally contains no `ProxyJump`. If Astrill is enabled, Windows still sends `10.2.29.0/24` through the physical gateway because the persistent `/24` route is more specific than Astrill's broad routes.

## Operational Notes

- The host and existing images are both `amd64`; the H200 is `sm_90`, which is included in the captured PyTorch runtime.
- Driver `580.82.07` exceeds the project's CUDA 12.8 gate of `570.26`.
- Keep this user's migration assets below `/data3/ywang`: use `/data3/ywang/quantitize-migration` for staging and `/data3/ywang/quantitize-platform` for the restored project. Avoid filling `/`, which already had 78% usage when inspected.
- Docker is active and `ywang` is currently a member of the `docker` group. Treat Docker access as root-equivalent.
- `ywang` has sudo-group membership but sudo requires a password. An agent must not record that password in files, command history, logs, or Git.
- NAS SMB is mounted directly at `/mnt/ywang-nas`; `cifs-utils` is installed. Credentials remain in a root-only file and are not documented here.
- This profile is an observation, not a guarantee of reserved GPU, disk, port, or Docker capacity. Re-run the gates before deployment.
