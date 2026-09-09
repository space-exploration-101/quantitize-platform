#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享标定集 / 测试集（与任务目录分离，多任务复用）。"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from constants import IMAGE_EXTS

import sys

_PIPELINE = Path(__file__).resolve().parents[2]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
from script_registry import (  # noqa: E402
    DATA_DIR,
    OUTPUT_DATA_ROOT,
    PLATFORM_ROOT,
    QUANTITIZE_LEGACY_DIR,
    SHARED_DATA_ROOT,
)

REGISTRY_PATH = SHARED_DATA_ROOT / "registry.json"
SHARED_TEST_IMGSZ = 1280

# 一次性从旧工作区灌数据集时使用（只读参考，不修改）
_LEGACY = QUANTITIZE_LEGACY_DIR
SOURCE_CALI = _LEGACY / "cali_data"
SOURCE_TEST_IMG = _LEGACY / "test_data" / "images" / "test"
SOURCE_TEST_LAB = _LEGACY / "test_data" / "labels" / "test"
DEFAULT_TEST_DATASET_ID = "moon_earth_257"
LEGACY_TEST_DATASET_ID = "quantize_test_281"

# 兼容：部分代码仍引用 QUANTITIZE_DIR
QUANTITIZE_DIR = PLATFORM_ROOT


@dataclass
class DatasetEntry:
    id: str
    display_name: str
    kind: str  # cali | test | fpga_test
    rel_path: str
    image_count: int = 0
    note: str = ""

    @property
    def root(self) -> Path:
        return SHARED_DATA_ROOT / self.rel_path

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "path": self.rel_path,
            "image_count": self.image_count,
            "note": self.note,
        }


@dataclass
class CatalogRow:
    """Web 数据集目录行（测试集拆为原始图与 FPGA 包两行）。"""

    kind_label: str
    download_kind: str  # cali | test | fpga_test
    entry: DatasetEntry


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return len([p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def load_registry() -> Dict[str, Dict[str, dict]]:
    if not REGISTRY_PATH.is_file():
        return {"cali": {}, "test": {}}
    raw = REGISTRY_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return {"cali": {}, "test": {}}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"cali": {}, "test": {}}
    if not isinstance(data, dict):
        return {"cali": {}, "test": {}}
    return {"cali": data.get("cali", {}) or {}, "test": data.get("test", {}) or {}}


def test_dataset_label_wh_ref(dataset_id: str) -> Optional[int]:
    """pose 标签 w/h 归一化所用的原图边长（如 test_data 2048）。"""
    if not dataset_id:
        return None
    ref = load_registry().get("test", {}).get(dataset_id, {}).get("label_wh_ref")
    return int(ref) if ref else None


def save_registry(cali: Dict[str, dict], test: Dict[str, dict]) -> None:
    SHARED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps({"cali": cali, "test": test}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_cali_datasets() -> List[DatasetEntry]:
    reg = load_registry()
    out: List[DatasetEntry] = []
    for did, meta in reg["cali"].items():
        root = SHARED_DATA_ROOT / meta["path"]
        out.append(
            DatasetEntry(
                id=did,
                display_name=meta.get("display_name", did),
                kind="cali",
                rel_path=meta["path"],
                image_count=_count_images(root),
                note=meta.get("note", ""),
            )
        )
    return sorted(out, key=lambda d: d.id)


def _count_fpga_pack_images(pack_dir: Path) -> int:
    side = pack_dir / "side_view"
    if side.is_dir():
        n = _count_images(side)
        if n:
            return n
    bins = pack_dir / "bins"
    if bins.is_dir():
        return len([p for p in bins.iterdir() if p.suffix.lower() == ".bin"])
    return 0


def _count_fpga_from_jobs(test_dataset_id: str) -> int:
    """扫描 output_data 中使用该测试集的任务，取已生成侧视图张数。"""
    output_root = Path(OUTPUT_DATA_ROOT)
    best = 0
    if not output_root.is_dir():
        return 0
    for job in output_root.iterdir():
        if not job.is_dir():
            continue
        meta = job / "job_config.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(data.get("test_dataset_id", "")) != test_dataset_id:
            continue
        n = _count_fpga_pack_images(job / "fpga_test_pack")
        if n > best:
            best = n
    return best


def sync_registry_from_disk() -> None:
    """把 shared_data/cali|test 下已有目录补进 registry（不删除已有项）。"""
    cali_reg, test_reg = load_registry()["cali"], load_registry()["test"]
    cali_root_dir = SHARED_DATA_ROOT / "cali"
    test_root_dir = SHARED_DATA_ROOT / "test"
    if cali_root_dir.is_dir():
        for d in sorted(p for p in cali_root_dir.iterdir() if p.is_dir()):
            n = _count_images(d)
            if d.name not in cali_reg:
                cali_reg[d.name] = {
                    "display_name": f"{d.name} ({n} 张)",
                    "path": f"cali/{d.name}",
                    "image_count": n,
                }
            else:
                cali_reg[d.name]["image_count"] = n
                cali_reg[d.name]["path"] = f"cali/{d.name}"
    if test_root_dir.is_dir():
        for d in sorted(p for p in test_root_dir.iterdir() if p.is_dir()):
            n = _count_images(d / "images")
            if d.name not in test_reg:
                test_reg[d.name] = {
                    "display_name": f"{d.name} ({n} 张)",
                    "path": f"test/{d.name}",
                    "image_count": n,
                    "note": "PT/ONNX 评测用 images/ + labels/",
                }
            else:
                test_reg[d.name]["image_count"] = n
                test_reg[d.name]["path"] = f"test/{d.name}"
    save_registry(cali_reg, test_reg)


def list_dataset_catalog() -> List[CatalogRow]:
    """标定 + 原始测试 + FPGA 测试包（Web 列表用）。"""
    sync_registry_from_disk()
    rows: List[CatalogRow] = []
    for d in list_cali_datasets():
        rows.append(CatalogRow(kind_label="标定", download_kind="cali", entry=d))
    for d in list_test_datasets():
        rows.append(
            CatalogRow(
                kind_label="测试",
                download_kind="test",
                entry=DatasetEntry(
                    id=d.id,
                    display_name=d.display_name,
                    kind="test",
                    rel_path=f"{d.rel_path}/images, {d.rel_path}/labels",
                    image_count=d.image_count,
                    note=d.note or "PT/ONNX 评测：images/ + labels/",
                ),
            )
        )
        pack = d.root / "fpga_test_pack"
        fpga_n = _count_fpga_pack_images(pack) or _count_fpga_from_jobs(d.id)
        # 侧视包与测试集一一对应；尚无任务产物时仍展示测试集张数，避免页面显示 —
        shown_n = fpga_n or d.image_count
        short = d.display_name.split("(")[0].strip()
        short = short.replace("原始测试集", "").replace("测试集", "").strip() or d.id
        if fpga_n:
            fpga_note = (
                f"任务内已生成侧视 {fpga_n} 张（与测试集 {d.image_count} 张对应）；"
                "路径在 output_data/<task>/fpga_test_pack/"
            )
        else:
            fpga_note = (
                f"由绑定此测试集的任务在步骤④从 images/ 转换生成；"
                f"张数与测试集相同（{d.image_count}），路径在 output_data/<task>/fpga_test_pack/"
            )
        rows.append(
            CatalogRow(
                kind_label="FPGA测试",
                download_kind="fpga_test",
                entry=DatasetEntry(
                    id=d.id,
                    display_name=f"FPGA 测试包 — {short}",
                    kind="fpga_test",
                    rel_path=f"output_data/<task>/fpga_test_pack",
                    image_count=shown_n,
                    note=fpga_note,
                ),
            )
        )
    return rows


def list_test_datasets() -> List[DatasetEntry]:
    reg = load_registry()
    out: List[DatasetEntry] = []
    for did, meta in reg["test"].items():
        root = SHARED_DATA_ROOT / meta["path"]
        out.append(
            DatasetEntry(
                id=did,
                display_name=meta.get("display_name", did),
                kind="test",
                rel_path=meta["path"],
                image_count=_count_images(root / "images"),
                note=meta.get("note", ""),
            )
        )
    return sorted(out, key=lambda d: d.id)


def cali_root(dataset_id: str) -> Path:
    reg = load_registry()
    if dataset_id not in reg["cali"]:
        raise KeyError(f"未知标定集: {dataset_id}")
    return SHARED_DATA_ROOT / reg["cali"][dataset_id]["path"]


def test_root(dataset_id: str) -> Path:
    reg = load_registry()
    if dataset_id not in reg["test"]:
        raise KeyError(f"未知测试集: {dataset_id}")
    return SHARED_DATA_ROOT / reg["test"][dataset_id]["path"]


def get_dataset(kind: str, dataset_id: str) -> DatasetEntry:
    if kind == "fpga_test":
        for row in list_dataset_catalog():
            if row.download_kind == kind and row.entry.id == dataset_id:
                return row.entry
        raise KeyError(f"未知数据集: {kind}/{dataset_id}")
    if kind not in ("cali", "test"):
        raise KeyError(f"未知数据集类型: {kind}")
    for entry in list_cali_datasets() if kind == "cali" else list_test_datasets():
        if entry.id == dataset_id:
            return entry
    raise KeyError(f"未知数据集: {kind}/{dataset_id}")


def _dataset_zip_targets(kind: str, dataset_id: str) -> tuple[Path, str, Optional[List[str]]]:
    """返回 (root, zip_prefix, subdirs)。subdirs 为 None 时打包 root 下全部文件。"""
    shared = SHARED_DATA_ROOT.resolve()
    reg = load_registry()
    if kind == "cali":
        if dataset_id not in reg["cali"]:
            raise KeyError(f"未知标定集: {dataset_id}")
        root = cali_root(dataset_id).resolve()
        return root, dataset_id, None
    if kind == "test":
        if dataset_id not in reg["test"]:
            raise KeyError(f"未知测试集: {dataset_id}")
        root = test_root(dataset_id).resolve()
        return root, dataset_id, ["images", "labels"]
    if kind == "fpga_test":
        if dataset_id not in reg["test"]:
            raise KeyError(f"未知测试集: {dataset_id}")
        root = test_fpga_pack_dir(dataset_id).resolve()
        return root, f"{dataset_id}_fpga", None
    raise KeyError(f"未知数据集类型: {kind}")


def _iter_zip_files(root: Path, subdirs: Optional[List[str]]):
    if subdirs:
        for sub in subdirs:
            sub_root = root / sub
            if not sub_root.is_dir():
                continue
            for path in sorted(sub_root.rglob("*")):
                if path.is_dir():
                    continue
                real = path.resolve()
                if not real.is_file():
                    continue
                yield real, f"{sub}/{path.relative_to(sub_root)}"
        return
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        real = path.resolve()
        if not real.is_file():
            continue
        yield real, path.relative_to(root)


def _dataset_newest_mtime(root: Path) -> float:
    newest = root.stat().st_mtime if root.is_dir() else 0.0
    if not root.is_dir():
        return newest
    for path in root.rglob("*"):
        try:
            newest = max(newest, path.stat(follow_symlinks=False).st_mtime)
            if path.is_symlink() or path.is_file():
                newest = max(newest, path.resolve().stat().st_mtime)
        except OSError:
            continue
    reg_mtime = REGISTRY_PATH.stat().st_mtime if REGISTRY_PATH.is_file() else 0.0
    return max(newest, reg_mtime)


def build_dataset_zip(kind: str, dataset_id: str) -> Path:
    """打包共享数据集为 zip（符号链接写入真实文件内容，结果缓存在 .download_cache）。"""
    root, zip_prefix, subdirs = _dataset_zip_targets(kind, dataset_id)
    shared = SHARED_DATA_ROOT.resolve()
    if not str(root).startswith(str(shared)):
        raise FileNotFoundError(f"数据集目录非法: {root}")
    if subdirs is None and not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)

    cache_dir = shared / ".download_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{kind}_{dataset_id}.zip"
    newest = _dataset_newest_mtime(root)
    if zip_path.is_file() and zip_path.stat().st_mtime >= newest:
        return zip_path

    tmp_path = zip_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for real, rel in _iter_zip_files(root, subdirs):
            arcname = f"{zip_prefix}/{rel}"
            zf.write(real, arcname=str(arcname))
    tmp_path.replace(zip_path)
    return zip_path


def test_images_dir(dataset_id: str) -> Path:
    return test_root(dataset_id) / "images"


def test_labels_dir(dataset_id: str) -> Path:
    return test_root(dataset_id) / "labels"


def test_fpga_pack_dir(dataset_id: str) -> Path:
    """FPGA 测试包与测试集绑定（测试数据子集）。"""
    return test_root(dataset_id) / "fpga_test_pack"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not src.exists():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _populate_cali_dataset(dest: Path, source: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(source.glob("*.png")):
        _link_or_copy(img, dest / img.name)
        count += 1
    return count


def _write_resized_png(src: Path, dst: Path, imgsz: int = SHARED_TEST_IMGSZ) -> None:
    """从原图 resize 写入共享集（不修改 src）。"""
    import cv2
    import numpy as np

    with open(src, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {src}")
    resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if not cv2.imwrite(str(dst), resized):
        raise ValueError(f"无法写入图片: {dst}")


def _needs_test_image_materialize(img_dir: Path, imgsz: int = SHARED_TEST_IMGSZ) -> bool:
    """共享测试图仍为原图链接或非目标尺寸时需重建。"""
    if not img_dir.is_dir():
        return True
    pngs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not pngs:
        return True
    import cv2
    import numpy as np

    for p in pngs[:8]:
        if p.is_symlink():
            return True
        with open(p, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != imgsz or img.shape[1] != imgsz:
            return True
    return False


def _write_label_for_resized_image(
    src_lab: Path,
    dst_lab: Path,
    orig_hw: tuple[int, int],
    new_imgsz: int,
) -> None:
    """写入与 resize 后图像配套的 YOLO/pose 标签。

    源标签为相对宽高归一化坐标时，均匀缩放至正方形不改变数值；
    此处仍按新图尺寸写出副本，供共享集独立使用（不修改 test_data 原文件）。
    """
    orig_h, orig_w = orig_hw
    if orig_h == new_imgsz and orig_w == new_imgsz:
        shutil.copy2(src_lab, dst_lab)
        return
    lines_out: list[str] = []
    for line in src_lab.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
        rest = parts[5:]
        lines_out.append(
            " ".join([cls, f"{cx:.10g}", f"{cy:.10g}", f"{w:.10g}", f"{h:.10g}", *rest])
        )
    dst_lab.parent.mkdir(parents=True, exist_ok=True)
    dst_lab.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def materialize_shared_test_images(
    test_dest: Path,
    img_src: Path,
    lab_src: Path,
    *,
    imgsz: int = SHARED_TEST_IMGSZ,
) -> int:
    """将测试图物化为 imgsz×imgsz；标签写入共享集（不修改 test_data 原图/原标签）。"""
    img_out = test_dest / "images"
    lab_out = test_dest / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)
    (test_dest / "fpga_test_pack").mkdir(exist_ok=True)
    count = 0
    import cv2
    import numpy as np

    for img in sorted(img_src.glob("*.png")):
        with open(img, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"无法读取图片: {img}")
        orig_h, orig_w = raw.shape[:2]
        _write_resized_png(img, img_out / img.name, imgsz=imgsz)
        lab = lab_src / f"{img.stem}.txt"
        if lab.is_file():
            dst_lab = lab_out / lab.name
            if dst_lab.exists() or dst_lab.is_symlink():
                dst_lab.unlink()
            _write_label_for_resized_image(lab, dst_lab, (orig_h, orig_w), imgsz)
        count += 1
    return count


def remove_shared_test_dataset(dataset_id: str) -> None:
    """仅删除 shared_data 下某测试集目录及注册项，不触碰仓库原始数据。"""
    cali_reg, test_reg = load_registry()["cali"], load_registry()["test"]
    if dataset_id not in test_reg:
        return
    rel = test_reg[dataset_id]["path"]
    dest = (SHARED_DATA_ROOT / rel).resolve()
    root = SHARED_DATA_ROOT.resolve()
    if dest.is_dir() and str(dest).startswith(str(root)):
        shutil.rmtree(dest)
    del test_reg[dataset_id]
    save_registry(cali_reg, test_reg)


def bootstrap_default_datasets(*, force: bool = False) -> None:
    """从 cali_data / test_data 初始化默认共享数据集。"""
    cali_id = "default_496"
    test_id = DEFAULT_TEST_DATASET_ID
    cali_dest = SHARED_DATA_ROOT / "cali" / cali_id
    test_dest = SHARED_DATA_ROOT / "test" / test_id

    cali_count = 0
    test_count = 0
    if force or not cali_dest.is_dir() or not any(cali_dest.iterdir()):
        if SOURCE_CALI.is_dir():
            cali_count = _populate_cali_dataset(cali_dest, SOURCE_CALI)
    else:
        cali_count = _count_images(cali_dest)

    if LEGACY_TEST_DATASET_ID != test_id:
        remove_shared_test_dataset(LEGACY_TEST_DATASET_ID)

    if force or _needs_test_image_materialize(test_dest / "images"):
        if SOURCE_TEST_IMG.is_dir():
            test_count = materialize_shared_test_images(test_dest, SOURCE_TEST_IMG, SOURCE_TEST_LAB)
    else:
        test_count = _count_images(test_dest / "images")

    cali_reg, test_reg = load_registry()["cali"], load_registry()["test"]
    cali_reg[cali_id] = {
        "display_name": f"默认标定集 ({cali_count or 496} 张)",
        "path": f"cali/{cali_id}",
        "image_count": cali_count,
        "note": "来源 quantitize/cali_data",
    }
    # 合并写入默认测试集，禁止整表覆盖（否则会抹掉 aituosha_moon_840 等已有集）
    test_reg[test_id] = {
        "display_name": f"月亮/地球原始测试集 ({test_count or 257} 张, nc=2)",
        "path": f"test/{test_id}",
        "image_count": test_count,
        "nc": 2,
        "note": (
            f"images {SHARED_TEST_IMGSZ}×{SHARED_TEST_IMGSZ}；"
            "2 类 moon/earth；来源 quantitize/test_data；"
            "PT/ONNX 评测用 images/ + labels/"
        ),
    }
    save_registry(cali_reg, test_reg)
    sync_registry_from_disk()
    # 补全常用地标+月球测试集展示名
    cali_reg, test_reg = load_registry()["cali"], load_registry()["test"]
    if "aituosha_moon_840" in test_reg:
        n = test_reg["aituosha_moon_840"].get("image_count") or _count_images(
            SHARED_DATA_ROOT / "test" / "aituosha_moon_840" / "images"
        )
        test_reg["aituosha_moon_840"].update(
            {
                "display_name": f"一个地标+月球测试集 ({n} 张, nc=2)",
                "nc": 2,
                "class_names": {"0": "地标", "1": "月球"},
                "note": "混合分辨率原图；2 类；PT/ONNX 评测用 images/ + labels/",
            }
        )
        save_registry(cali_reg, test_reg)


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        if link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()
    os.symlink(target.resolve(), link)


def test_dataset_nc(dataset_id: str, default: int = 6) -> int:
    reg = load_registry()
    meta = reg["test"].get(dataset_id, {})
    return int(meta.get("nc", default))


def apply_test_dataset_nc(cfg) -> None:
    """根据共享测试集注册信息同步 job nc。"""
    if cfg.test_dataset_id:
        cfg.nc = test_dataset_nc(cfg.test_dataset_id, cfg.nc)


def attach_datasets_to_job(
    job_input_dir: Path,
    *,
    cali_dataset_id: Optional[str] = None,
    test_dataset_id: Optional[str] = None,
) -> None:
    """在任务 input/ 下建立指向共享数据集的符号链接。"""
    job_input_dir.mkdir(parents=True, exist_ok=True)
    if cali_dataset_id:
        ensure_symlink(job_input_dir / "cali", cali_root(cali_dataset_id))
    if test_dataset_id:
        ensure_symlink(job_input_dir / "test", test_root(test_dataset_id))
