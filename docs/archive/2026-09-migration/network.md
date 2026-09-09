# Network and SSH Notes

Date: 2026-08-31

This document records today's local network, Astrill VPN routing, and SSH passwordless login configuration for the `wrs` host.

## SSH Target

- SSH alias: `wrs`
- Remote host: `10.2.26.132`
- Remote user: `rs`
- Expected command:

```powershell
ssh wrs
```

## Local SSH Configuration

The local SSH config file is:

```text
C:\Users\admin\.ssh\config
```

It contains:

```ssh-config
Host wrs
    HostName 10.2.26.132
    User rs
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

The local SSH key pair is:

```text
C:\Users\admin\.ssh\id_ed25519
C:\Users\admin\.ssh\id_ed25519.pub
```

The public key was installed into the remote user's SSH authorization file:

```text
/home/rs/.ssh/authorized_keys
```

The install command used this logic on the remote host:

```sh
umask 077
mkdir -p ~/.ssh
touch ~/.ssh/authorized_keys
grep -qxF '<local-public-key>' ~/.ssh/authorized_keys || echo '<local-public-key>' >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

The plaintext login password is intentionally not recorded in this repository file.

## Verification

Passwordless login was verified with:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 wrs "whoami; hostname"
```

Result:

```text
rs
rs-System-Product-Name
```

This confirms that `ssh wrs` can log in as `rs` without password input.

## Astrill VPN Routing Issue

When Astrill VPN was enabled, traffic to `10.2.26.132` originally failed. The issue was not DNS-related for this host, because `10.2.26.132` is a literal IP address. The failure came from Windows routing.

Astrill created a Wintun tunnel interface:

```text
Interface: Wintun Userspace Tunnel
Alias: 本地连接
VPN local IP: 198.18.3.191
VPN gateway: 198.18.0.1
```

The physical LAN interface was:

```text
Interface: Realtek Gaming 2.5GbE Family Controller
Alias: 以太网
Local IP: 192.168.31.253
Local gateway: 192.168.31.1
Interface index: 19
```

Astrill installed split default routes similar to:

```text
0.0.0.0/1    -> 198.18.0.1
128.0.0.0/1  -> 198.18.0.1
```

These two routes together cover almost the entire IPv4 internet. Because `10.2.26.132` falls under `0.0.0.0/1`, Windows selected the Astrill tunnel route instead of the physical gateway. That is why `ssh wrs` timed out while Astrill was enabled.

## Persistent Route Fix

To force all traffic to `10.2.26.*` through the physical LAN gateway, this persistent route was added from an Administrator PowerShell:

```powershell
route -p add 10.2.26.0 mask 255.255.255.0 192.168.31.1 metric 1 if 19
```

Meaning:

```text
Destination: 10.2.26.0/24
Gateway:     192.168.31.1
Interface:   以太网, ifIndex 19
Metric:      1
Persistent:  yes
```

This covers:

```text
10.2.26.0 - 10.2.26.255
```

Including:

```text
10.2.26.132
```

The route is persistent because the command used `-p`, so Windows should keep it after reboot.

## Route Verification

Current route lookup for `10.2.26.132`:

```powershell
Find-NetRoute -RemoteIPAddress 10.2.26.132
```

Observed effective route:

```text
DestinationPrefix : 10.2.26.0/24
NextHop           : 192.168.31.1
InterfaceAlias    : 以太网
ifIndex           : 19
RouteMetric       : 1
InterfaceMetric   : 20
```

Persistent route store also contains:

```text
DestinationPrefix : 10.2.26.0/24
NextHop           : 192.168.31.1
InterfaceAlias    : 以太网
ifIndex           : 19
RouteMetric       : 1
```

`route print -4` shows:

```text
Persistent Routes:
  Network Address          Netmask  Gateway Address  Metric
        10.2.26.0    255.255.255.0     192.168.31.1       1
```

## Background: How Windows Chooses a Route

Windows chooses the route using longest-prefix match first, then route metric.

For example:

```text
10.2.26.0/24  is more specific than 0.0.0.0/1
0.0.0.0/1     is more specific than 0.0.0.0/0
```

Because `10.2.26.0/24` is the most specific matching route for `10.2.26.132`, Windows sends this traffic to `192.168.31.1` even when Astrill also has broad VPN routes installed.

Metric only matters when two matching routes have the same prefix length. In this setup, prefix length is the decisive part.

## Astrill Configuration Recommendation

For this workflow, prefer domain/site-based split tunneling over application-based split tunneling.

Reason: tools like `powershell.exe` and `ssh.exe` can need both local and foreign destinations.

Examples:

```text
ssh wrs            -> 10.2.26.132, should use local gateway
ssh git@github.com -> github.com, may need VPN
pip/npm downloads  -> may need VPN depending on registry/CDN
```

If the whole `powershell.exe` or `ssh.exe` application is forced through VPN, local SSH targets can break. Domain/site-based rules are safer for mixed development workflows.

Recommended Astrill approach:

```text
Protocol: OpenWeb
Mode: Smart Mode or Tunnel only selected sites
App Filter: do not force all powershell.exe or ssh.exe traffic through VPN
Keep local/LAN traffic outside VPN
```

Even if Astrill changes its broad default routes later, the persistent Windows route for `10.2.26.0/24` should keep this subnet on the physical gateway.

## Useful Commands

Check the final route selected for a target:

```powershell
Find-NetRoute -RemoteIPAddress 10.2.26.132
```

Check SSH alias expansion:

```powershell
ssh -G wrs
```

Check SSH passwordless login:

```powershell
ssh -o BatchMode=yes wrs "whoami; hostname"
```

Check TCP port 22:

```powershell
Test-NetConnection 10.2.26.132 -Port 22
```

Show persistent IPv4 routes:

```powershell
route print -4
```

Remove the persistent route if it is ever no longer needed:

```powershell
route delete 10.2.26.0
```

This delete command must also be run from an Administrator PowerShell.

## H200 Direct Route and SSH (2026-09-01)

The H200 host is configured for direct access. It does not use `wrs` as a jump host.

```text
SSH alias: H200
Address:   10.2.29.180
User:      ywang
```

The local SSH configuration contains:

```ssh-config
Host H200
    HostName 10.2.29.180
    User ywang
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

The local public key was added to `/home/ywang/.ssh/authorized_keys`. The login password is intentionally not recorded.

Astrill bypass is implemented by this persistent Windows route:

```text
DestinationPrefix : 10.2.29.0/24
NextHop           : 192.168.31.1
InterfaceIndex    : 19
RouteMetric       : 1
```

It was created from Administrator PowerShell with:

```powershell
route -p add 10.2.29.0 mask 255.255.255.0 192.168.31.1 metric 1 if 19
```

Verification completed on 2026-09-01:

```powershell
Get-NetRoute -PolicyStore PersistentStore -AddressFamily IPv4 -DestinationPrefix 10.2.29.0/24
ssh -o BatchMode=yes -o ConnectTimeout=8 H200 "whoami; hostname"
```

The route was present in `PersistentStore`, and direct public-key SSH reached the host without a jump configuration. To remove the route later, use Administrator PowerShell:

```powershell
route delete 10.2.29.0
```

## H200 Direct NAS Mount and Verified Backup (2026-09-02)

H200 has `cifs-utils` installed and can access the NAS directly without using the Windows machine as a relay:

```text
NAS:         //10.2.26.26/902_data
Mount point: /mnt/ywang-nas
SMB:         3.0
Status:      read-write
```

The credentials are stored in a root-only credentials file and are intentionally absent from Agent documents. The mount is currently manual and has no `/etc/fstab` entry. This does not affect Quantitize service startup because the project and runtime data are local under `/data3`; remount NAS before backup or restore operations.

The latest verified service snapshot is:

```text
//10.2.26.26/902_data/0-项目/13-专项/4-代码/量化平台/H200/snapshot-20260902-182606
```

It contains the project and shared input data, Docker API/Web images, original Conda environment archive, inventories, checksums, and `H200_RECOVERY.md`. It explicitly excludes `data/output_data`.
