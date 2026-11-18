# %%
# ============================================================================
# MSK X-ray SSL Phase 2 — DINOv3 ViT-B/16 Baseline
#
#   D3-E0:  DINOv3 core SSL (DINO + iBOT + KoLeo, Sinkhorn-Knopp centering)
#   D3-E1:  D3-E0 + InfoNCE Position         (Phase 1 E1c 이식; run_tag e1c / e1c_within)
#   D3-E2:  D3-E0 + Part-Whole (partwhole_method="adam": Adam-v2 2x2 독립 encoding part)
#   D3-E3:  D3-E0 + Position + Part-Whole     (미구현, 추후 추가)
#   D3-E0G: D3-E0 + Gram anchoring            (미구현, 추후 추가)
#
#   Source of truth:
#     - Architecture: facebookresearch/dinov3 → dinov3_vitb16() factory
#     - SSL framework: facebookresearch/dinov3 → dinov3/train/ssl_meta_arch.py
#     - Loss: facebookresearch/dinov3 → dinov3/loss/
#     - Dataset / augmentation / evaluation: msk_ssl_phase1.py (Phase 1)
#
#   실행:
#     ./venv/bin/python3 -u train_phase2.py --experiment D3-E0
# ============================================================================
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if "torch" in sys.modules:
    print(
        "[WARN] torch 가 이미 import 되어 있습니다. "
        "CUDA_VISIBLE_DEVICES 설정이 적용되지 않았을 수 있습니다."
    )
else:
    print("[OK] CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])

PYTHON_EXECUTABLE = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/venv/bin/python3"
WORK_DIR = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994"

# %%
# ============================================================================
# Imports
# ============================================================================
import copy
import csv
import glob
import importlib
import json
import math
import platform
import random
import time
import warnings
from dataclasses import dataclass, field, asdict, replace
from functools import partial
from typing import Dict, List, Optional, Tuple, Any, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_UMAP = importlib.util.find_spec("umap") is not None

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable")

DEVICE = torch.device("cuda:0")

ENV_INFO = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_total_memory_GB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
}
print("PHASE 2 — DINOv3 ViT-B/16 Baseline")
for k, v in ENV_INFO.items():
    print(f"  {k:22s} : {v}")


# %%
# ============================================================================
# Config
# ============================================================================
VALID_EXPERIMENTS = ("D3-E0", "D3-E1", "D3-E2", "D3-E3")
IMPLEMENTED_EXPERIMENTS = ("D3-E0", "D3-E1", "D3-E2")

@dataclass
class Config:
    # ---- paths ----
    python_executable: str = PYTHON_EXECUTABLE
    work_dir: str = WORK_DIR
    data_root: str = "/NHNHOME/WORKSPACE/26msit005_B/temp/MSK"
    output_root: str = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/outputs/msk_ssl_phase2"
    anatomical_annotation_csv: Optional[str] = None
    rad_dino_local_path: Optional[str] = None
    rad_dino_model_id: str = "microsoft/rad-dino"

    # ---- experiment ----
    experiment: str = "D3-E0"
    run_tag: str = ""
    seed: int = 42
    deterministic: bool = False

    # ---- dataset ----
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    exclude_dir_names: Tuple[str, ...] = ("keem_ckpt",)
    dataset_cache_list: Optional[str] = (
        "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/outputs/msk_ssl/file_list.txt"
    )
    scan_workers: int = 16
    max_train_images: Optional[int] = 300_000

    # ---- multi-crop views ----
    num_global_crops: int = 2
    num_local_crops: int = 6
    global_crop_size: int = 256        # DINOv3: 256/16 = 16×16 grid
    local_crop_size: int = 112         # DINOv3: 112/16 = 7×7 grid
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)
    crop_ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip_prob: float = 0.5
    foreground_aware_crop: bool = True
    foreground_map_size: int = 64
    foreground_threshold: float = 0.02
    min_foreground_global: float = 0.30
    min_foreground_local: float = 0.15
    foreground_max_attempts: int = 10
    pre_resize: Optional[int] = 512
    crop_max_attempts: int = 10
    brightness_jitter: float = 0.4
    contrast_jitter: float = 0.4
    global_blur_prob: Tuple[float, float] = (1.0, 0.1)
    local_blur_prob: float = 0.5
    solarize_prob: float = 0.2
    normalize_mean: Tuple[float, float, float] = (0.2495, 0.2495, 0.2495)
    normalize_std: Tuple[float, float, float] = (0.2609, 0.2609, 0.2609)

    # ---- DINOv3 ViT-B/16 backbone (dinov3_vitb16 factory) ----
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    in_chans: int = 3
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    mask_k_bias: bool = True
    drop_path_rate: float = 0.1       # MSK adaptation (factory default 0.0)
    layerscale_init: float = 1e-5
    ln_eps: float = 1e-5              # layernormbf16
    n_storage_tokens: int = 4
    rope_base: float = 100.0
    rope_normalize_coords: str = "separate"
    rope_rescale_coords: float = 2.0
    rope_dtype: str = "fp32"          # torch.float32

    # ---- projection heads ----
    K_cls: int = 65536
    K_patch: int = 65536
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_n_layers: int = 3
    head_norm_last_layer: bool = False   # DINOv3

    # ---- DINO CLS loss ----
    student_temp: float = 0.1
    teacher_temp_start: float = 0.04
    teacher_temp_final: float = 0.07
    teacher_temp_warmup_epochs: int = 30
    centering: str = "sinkhorn_knopp"    # "sinkhorn_knopp" | "centering"
    sk_n_iterations: int = 3
    center_momentum: float = 0.9        # fallback EMA center

    # ---- iBOT patch loss ----
    patch_student_temp: float = 0.1
    patch_teacher_temp_start: float = 0.04
    patch_teacher_temp_final: float = 0.07
    patch_center_momentum: float = 0.9
    mask_ratio_min: float = 0.10
    mask_ratio_max: float = 0.50
    mask_sample_prob: float = 0.5
    ibot_compute_on_masked_only: bool = True

    # ---- KoLeo ----
    koleo_eps: float = 1e-8

    # ---- position consistency (D3-E1/E3 전용, D3-E0 비활성) ----
    K_pos: int = 1024
    pos_overlap_threshold: float = 0.5
    position_topk: int = 1
    position_student_temp: float = 0.1
    position_teacher_temp: float = 0.04
    position_center_momentum: float = 0.9
    position_pair_chunk_size: int = 4096          # E1c 값 (메모리용, 수식 무관)
    # ---- position InfoNCE (Phase 1 E1c 값 그대로) ----
    position_loss_type: str = "infonce"           # "infonce" | "ce"
    position_infonce_temp: float = 0.2
    position_n_within_negatives: int = 32
    position_n_cross_negatives: int = 64          # D3-E1c-within: 0
    position_negative_ios_threshold: float = 0.05

    # ---- part-whole (D3-E2/E3 전용, D3-E0 비활성) ----
    part_grid_size: int = 4

    # ---- part-whole: Adam-v2 (D3-E2 전용. partwhole_method="none" 이면 전부 비활성) ----
    partwhole_method: str = "none"             # "none" | "adam"
    partwhole_start_epoch: int = 10            # 이 epoch 부터 comp/decomp 활성 (이전은 D3-E0 경로)
    partwhole_part_size: int = 128             # 128 = native 사분면(B안), 256 = 2배 확대(E안 ablation)
    partwhole_n_crops: int = 2                 # 적용할 global crop 수 (1 = crop 0 만)
    partwhole_batch_frac: float = 1.0          # 각 crop 에서 앞쪽 몇 % 샘플에 적용
    partwhole_embed: str = "patch_mean"        # "patch_mean" | "cls"  (whole/part 공통)
    partwhole_student_whole: str = "unmasked"  # "unmasked"(mask 0 샘플만, 추가비용 0) | "clean"(추가 forward)
    partwhole_loss: str = "nmse"               # "nmse" = ||n(a)-n(b)||^2 = 2(1-cos) | "cos" = 1-cos
    partwhole_head_hidden: int = 2048          # Adam-v2: 2-layer MLP hidden 2048
    partwhole_empty_fg_thr: float = 0.05       # 전경 비율 < 이 값이면 '거의 빈 조각' (로그 전용)

    # ---- loss weights ----
    lambda_dino: float = 1.0
    lambda_ibot: float = 1.0
    lambda_koleo: float = 0.1
    lambda_pos: float = 0.1
    lambda_comp: float = 0.1
    lambda_decomp: float = 0.1

    # ---- optimization ----
    batch_size: int = 32
    epochs: int = 100
    steps_per_epoch: Optional[int] = 2000
    lr: float = 5e-4
    lr_linear_scaling: bool = True
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    start_warmup_lr: float = 0.0
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1        # DINOv3 SSL default: last-layer LR=0 during first epoch
    teacher_momentum: float = 0.992        # DINOv3
    teacher_momentum_final: float = 1.0
    patch_embed_lr_mult: float = 0.2       # DINOv3
    layerwise_decay: float = 0.9           # DINOv3

    # ---- runtime ----
    use_bf16: bool = True
    use_compile: bool = False
    num_workers: int = 24
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True

    # ---- logging / checkpoint ----
    log_interval: int = 10
    progress_bar: bool = True
    save_iteration_log: bool = True
    monitor: str = "epoch_mean_total_loss"
    monitor_mode: str = "min"
    auto_resume: bool = True

    # ---- evaluation ----
    eval_use_teacher: bool = True
    eval_checkpoint_name: str = "best.pt"
    eval_num_images: int = 256
    eval_batch_size: int = 16
    eval_input_size: int = 256             # DINOv3: 256
    positive_threshold: float = 0.5
    negative_threshold: float = 0.05
    eval_seed: int = 12345
    probe_epochs: int = 50
    probe_lr: float = 1e-3
    probe_weight_decay: float = 0.0
    probe_batch_size: int = 64
    probe_train_ratio: float = 0.8
    knn_k: int = 5
    max_vis_samples: int = 2000

    def __post_init__(self):
        if self.experiment not in VALID_EXPERIMENTS:
            raise ValueError(f"experiment must be one of {VALID_EXPERIMENTS}, got {self.experiment!r}")
        if self.experiment not in IMPLEMENTED_EXPERIMENTS:
            raise NotImplementedError(
                f"{self.experiment} is not implemented yet. Currently supported: {IMPLEMENTED_EXPERIMENTS}"
            )
        if self.global_crop_size % self.patch_size != 0:
            raise ValueError(f"global_crop_size({self.global_crop_size}) % patch_size({self.patch_size}) != 0")
        if self.local_crop_size % self.patch_size != 0:
            raise ValueError(f"local_crop_size({self.local_crop_size}) % patch_size({self.patch_size}) != 0")
        if self.global_grid % self.part_grid_size != 0:
            raise ValueError(f"part_grid_size({self.part_grid_size})가 global grid({self.global_grid})를 나누지 못합니다.")
        if self.position_loss_type not in ("infonce", "ce"):
            raise ValueError(f"position_loss_type must be infonce|ce, got {self.position_loss_type!r}")
        if self.position_n_within_negatives < 0 or self.position_n_cross_negatives < 0:
            raise ValueError("position negative 수는 0 이상이어야 합니다.")
        if self.monitor not in ("epoch_mean_total_loss", "epoch_mean_base_loss"):
            raise ValueError(f"monitor must be epoch_mean_total_loss|epoch_mean_base_loss, got {self.monitor!r}")
        # ---- [D3-E2] part-whole ----
        if self.partwhole_method not in ("none", "adam"):
            raise ValueError(f"partwhole_method must be none|adam, got {self.partwhole_method!r}")
        if self.experiment == "D3-E2" and self.partwhole_method != "adam":
            raise ValueError("D3-E2 는 --partwhole-method adam 이 필요합니다.")
        if self.partwhole_method != "none" and self.experiment != "D3-E2":
            raise ValueError(f"partwhole_method={self.partwhole_method!r} 는 D3-E2 에서만 사용합니다.")
        if self.partwhole_method == "adam":
            if self.global_crop_size % 2 != 0 or self.partwhole_part_size % self.patch_size != 0:
                raise ValueError(f"partwhole_part_size({self.partwhole_part_size}) % patch_size != 0")
            if not (1 <= self.partwhole_n_crops <= self.num_global_crops):
                raise ValueError(f"partwhole_n_crops must be in [1, {self.num_global_crops}]")
            if not (0.0 < self.partwhole_batch_frac <= 1.0):
                raise ValueError("partwhole_batch_frac must be in (0, 1]")
            if self.partwhole_embed not in ("patch_mean", "cls"):
                raise ValueError(f"partwhole_embed must be patch_mean|cls, got {self.partwhole_embed!r}")
            if self.partwhole_student_whole not in ("unmasked", "clean"):
                raise ValueError("partwhole_student_whole must be unmasked|clean")
            if self.partwhole_loss not in ("nmse", "cos"):
                raise ValueError("partwhole_loss must be nmse|cos")

    @property
    def global_grid(self) -> int:
        return self.global_crop_size // self.patch_size  # 16

    @property
    def local_grid(self) -> int:
        return self.local_crop_size // self.patch_size    # 7

    @property
    def n_global_tokens(self) -> int:
        return self.global_grid ** 2                       # 256

    @property
    def n_local_tokens(self) -> int:
        return self.local_grid ** 2                        # 49

    @property
    def k_part(self) -> int:
        return self.part_grid_size ** 2                    # 16

    @property
    def use_position(self) -> bool:
        return self.experiment in ("D3-E1", "D3-E3")

    @property
    def use_partwhole(self) -> bool:
        return self.experiment in ("D3-E2", "D3-E3")

    @property
    def need_aux_clean_forward(self) -> bool:
        return self.use_position or self.use_partwhole

    @property
    def run_dir(self) -> str:
        name = self.experiment + (f"_{self.run_tag}" if self.run_tag else "")
        return os.path.join(self.output_root, name)

    def scaled_lr(self) -> float:
        return self.lr * self.batch_size / 256.0 if self.lr_linear_scaling else self.lr

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {"config": asdict(self), "environment": ENV_INFO}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)


cfg = Config()
print(f"experiment: {cfg.experiment}")
print(f"global grid: {cfg.global_grid}x{cfg.global_grid} = {cfg.n_global_tokens} patches")
print(f"local  grid: {cfg.local_grid}x{cfg.local_grid} = {cfg.n_local_tokens} patches")
print(f"storage tokens: {cfg.n_storage_tokens}")


# %%
# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2**31 - 1)
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % (2**31 - 1))


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


set_seed(cfg.seed, cfg.deterministic)


# %%
# ============================================================================
# Dataset Discovery (Phase 1 코드 재사용)
# ============================================================================
def _scan_subtree(task: Tuple[str, Tuple[str, ...]]) -> List[str]:
    root, exts = task
    found: List[str] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for e in entries:
                    name = e.name
                    if name.startswith(".") or name.startswith("._"):
                        continue
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            if os.path.splitext(name)[1].lower() in exts:
                                found.append(e.path)
                    except OSError:
                        continue
        except (PermissionError, OSError):
            continue
    return found


def scan_dataset(
    root: str, extensions: Sequence[str] = (".png",), exclude_dirs: Sequence[str] = (),
    workers: int = 16, verbose: bool = True,
) -> List[str]:
    from concurrent.futures import ProcessPoolExecutor
    exclude_set = set(exclude_dirs)
    try:
        top_dirs = sorted(
            e.path for e in os.scandir(root)
            if e.is_dir(follow_symlinks=False) and e.name not in exclude_set
        )
    except (PermissionError, FileNotFoundError) as exc:
        raise FileNotFoundError(f"data_root 접근 불가: {root}\n  {exc!r}") from exc
    if verbose:
        print(f"[scan] {root}  ({len(top_dirs)} top-level dirs, workers={workers})")
    tasks = [(d, tuple(extensions)) for d in top_dirs]
    files: List[str] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            for result in pool.map(_scan_subtree, tasks):
                files.extend(result)
    else:
        for t in tasks:
            files.extend(_scan_subtree(t))
    files.sort()
    if verbose:
        print(f"[scan] found {len(files):,} files")
    return files


def discover_images(
    root: str, extensions: Sequence[str], cache_list: Optional[str],
    max_images: Optional[int], verbose: bool = True,
    exclude_dirs: Sequence[str] = (), workers: int = 16,
    rebuild_cache: bool = False,
) -> List[str]:
    files = None
    if cache_list and os.path.isfile(cache_list) and not rebuild_cache:
        with open(cache_list) as f:
            files = [line.strip() for line in f if line.strip()]
        if verbose:
            print(f"[cache] loaded {len(files):,} paths from {cache_list}")
    if files is None:
        files = scan_dataset(root, extensions, exclude_dirs=exclude_dirs, workers=workers, verbose=verbose)
        if cache_list:
            os.makedirs(os.path.dirname(cache_list), exist_ok=True)
            tmp = cache_list + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(files))
            os.replace(tmp, cache_list)
    if max_images is not None and max_images < len(files):
        idx = np.linspace(0, len(files) - 1, num=max_images).round().astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]
    if verbose:
        print(f"[dataset] {len(files):,} images")
    return files


def get_image_paths(cfg: Config, max_images: Optional[int] = -1, verbose: bool = True,
                    rebuild_cache: bool = False) -> List[str]:
    n = cfg.max_train_images if max_images == -1 else max_images
    return discover_images(
        cfg.data_root, cfg.image_extensions, cfg.dataset_cache_list, n, verbose,
        exclude_dirs=cfg.exclude_dir_names, workers=cfg.scan_workers,
        rebuild_cache=rebuild_cache,
    )


# %%
# ============================================================================
# MSK X-ray Dataset (Phase 1 재사용)
# ============================================================================
class MSKXrayDataset(Dataset):
    def __init__(self, file_paths: Sequence[str], transform, pre_resize: Optional[int] = None):
        if len(file_paths) == 0:
            raise ValueError("빈 dataset")
        self.file_paths = list(file_paths)
        self.transform = transform
        self.pre_resize = pre_resize

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load(self, path: str) -> Image.Image:
        with Image.open(path) as im:
            im = im.convert("L")
            if self.pre_resize is not None and max(im.size) > self.pre_resize:
                scale = self.pre_resize / max(im.size)
                new_size = (max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale)))
                im = im.resize(new_size, Image.BILINEAR)
            im = im.convert("RGB")
            im.load()
        return im

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.file_paths[index]
        try:
            img = self._load(path)
        except Exception as exc:
            raise RuntimeError(f"이미지 로딩 실패 [{index}] {path}: {exc!r}") from exc
        out = self.transform(img)
        out["index"] = index
        return out


# %%
# ============================================================================
# Coordinate-aware Multi-Crop Augmentation (Phase 1 재사용)
# ============================================================================
@dataclass
class ViewMeta:
    crop_box: torch.Tensor   # [4] = [x0, y0, w, h] normalized
    hflip: bool
    output_height: int
    output_width: int
    patch_grid_height: int
    patch_grid_width: int


@dataclass
class BatchedViewMeta:
    crop_box: torch.Tensor   # [B, 4]
    hflip: torch.Tensor      # [B] bool
    output_height: int
    output_width: int
    patch_grid_height: int
    patch_grid_width: int

    def to(self, device: torch.device) -> "BatchedViewMeta":
        return BatchedViewMeta(
            crop_box=self.crop_box.to(device, non_blocking=True),
            hflip=self.hflip.to(device, non_blocking=True),
            output_height=self.output_height, output_width=self.output_width,
            patch_grid_height=self.patch_grid_height, patch_grid_width=self.patch_grid_width,
        )

    @property
    def batch_size(self) -> int:
        return self.crop_box.shape[0]

    @property
    def num_patches(self) -> int:
        return self.patch_grid_height * self.patch_grid_width


def sample_rrc_params(height, width, scale, ratio, rng, max_attempts=10):
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(max_attempts):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            i = rng.randint(0, height - h)
            j = rng.randint(0, width - w)
            return i, j, h, w
    in_ratio = width / height
    if in_ratio < ratio[0]:
        w, h = width, int(round(width / ratio[0]))
    elif in_ratio > ratio[1]:
        h, w = height, int(round(height * ratio[1]))
    else:
        w, h = width, height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


class CoordinateAwareCrop:
    def __init__(self, size, scale, ratio, patch_size, hflip_prob,
                 brightness, contrast, blur_prob, solarize_prob, mean, std,
                 min_foreground=0.0, max_attempts=1):
        if size % patch_size != 0:
            raise ValueError(f"crop size {size} % patch size {patch_size} != 0")
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.grid = size // patch_size
        self.hflip_prob = hflip_prob
        self.blur_prob = blur_prob
        self.solarize_prob = solarize_prob
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast)
        self.blur = T.GaussianBlur(kernel_size=max(3, (size // 20) * 2 + 1), sigma=(0.1, 2.0))
        self.normalize = T.Normalize(mean=mean, std=std)
        self.min_foreground = min_foreground
        self.max_attempts = max(1, max_attempts)

    def _sample_crop(self, H, W, rng, fg_map):
        if fg_map is None or self.min_foreground <= 0.0:
            return sample_rrc_params(H, W, self.scale, self.ratio, rng)
        S_h, S_w = fg_map.shape
        best, best_score = None, -1.0
        for _ in range(self.max_attempts):
            i, j, h, w = sample_rrc_params(H, W, self.scale, self.ratio, rng)
            r0 = int(i / H * S_h); r1 = max(r0 + 1, int(math.ceil((i + h) / H * S_h)))
            c0 = int(j / W * S_w); c1 = max(c0 + 1, int(math.ceil((j + w) / W * S_w)))
            score = float(fg_map[r0:r1, c0:c1].mean())
            if score >= self.min_foreground:
                return i, j, h, w
            if score > best_score:
                best_score, best = score, (i, j, h, w)
        return best

    def __call__(self, img, rng=None, fg_map=None):
        rng = rng or random
        W, H = img.size
        i, j, h, w = self._sample_crop(H, W, rng, fg_map)
        view = TF.resized_crop(img, i, j, h, w, [self.size, self.size],
                               interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        do_flip = rng.random() < self.hflip_prob
        if do_flip:
            view = TF.hflip(view)
        if rng.random() < 0.8:
            view = self.color_jitter(view)
        if rng.random() < self.blur_prob:
            view = self.blur(view)
        if rng.random() < self.solarize_prob:
            view = TF.solarize(view, threshold=128)
        x = TF.to_tensor(view)
        x = self.normalize(x)
        meta = ViewMeta(
            crop_box=torch.tensor([j / W, i / H, w / W, h / H], dtype=torch.float32),
            hflip=bool(do_flip),
            output_height=self.size, output_width=self.size,
            patch_grid_height=self.grid, patch_grid_width=self.grid,
        )
        return x, meta


class MultiCropTransform:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        gb = cfg.global_blur_prob
        self.global_crops = [
            CoordinateAwareCrop(
                size=cfg.global_crop_size, scale=cfg.global_crop_scale, ratio=cfg.crop_ratio,
                patch_size=cfg.patch_size, hflip_prob=cfg.hflip_prob,
                brightness=cfg.brightness_jitter, contrast=cfg.contrast_jitter,
                blur_prob=gb[k] if k < len(gb) else gb[-1],
                solarize_prob=0.0 if k == 0 else cfg.solarize_prob,
                mean=cfg.normalize_mean, std=cfg.normalize_std,
                min_foreground=cfg.min_foreground_global if cfg.foreground_aware_crop else 0.0,
                max_attempts=cfg.foreground_max_attempts,
            )
            for k in range(cfg.num_global_crops)
        ]
        self.local_crop = CoordinateAwareCrop(
            size=cfg.local_crop_size, scale=cfg.local_crop_scale, ratio=cfg.crop_ratio,
            patch_size=cfg.patch_size, hflip_prob=cfg.hflip_prob,
            brightness=cfg.brightness_jitter, contrast=cfg.contrast_jitter,
            blur_prob=cfg.local_blur_prob, solarize_prob=cfg.solarize_prob,
            mean=cfg.normalize_mean, std=cfg.normalize_std,
            min_foreground=cfg.min_foreground_local if cfg.foreground_aware_crop else 0.0,
            max_attempts=cfg.foreground_max_attempts,
        )

    def _make_foreground_map(self, img: Image.Image) -> Optional[np.ndarray]:
        if not self.cfg.foreground_aware_crop:
            return None
        sz = self.cfg.foreground_map_size
        small = img.resize((sz, sz), Image.BILINEAR)
        arr = np.array(small, dtype=np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return (arr > self.cfg.foreground_threshold).astype(np.float32)

    def __call__(self, img: Image.Image) -> Dict[str, Any]:
        rng = random.Random()
        rng.seed(random.randint(0, 2**31))
        fg_map = self._make_foreground_map(img)
        g_imgs, g_metas = [], []
        for gc in self.global_crops:
            x, m = gc(img, rng=rng, fg_map=fg_map)
            g_imgs.append(x); g_metas.append(m)
        l_imgs, l_metas = [], []
        for _ in range(self.cfg.num_local_crops):
            x, m = self.local_crop(img, rng=rng, fg_map=fg_map)
            l_imgs.append(x); l_metas.append(m)
        return {"global_images": g_imgs, "global_metas": g_metas,
                "local_images": l_imgs, "local_metas": l_metas}


def batch_view_metas(metas: List[ViewMeta]) -> BatchedViewMeta:
    return BatchedViewMeta(
        crop_box=torch.stack([m.crop_box for m in metas], 0),
        hflip=torch.tensor([m.hflip for m in metas]),
        output_height=metas[0].output_height, output_width=metas[0].output_width,
        patch_grid_height=metas[0].patch_grid_height, patch_grid_width=metas[0].patch_grid_width,
    )


def collate_multicrop(batch):
    n_g = len(batch[0]["global_images"])
    n_l = len(batch[0]["local_images"])
    return {
        "global_images": [torch.stack([s["global_images"][k] for s in batch], 0) for k in range(n_g)],
        "global_metas": [batch_view_metas([s["global_metas"][k] for s in batch]) for k in range(n_g)],
        "local_images": [torch.stack([s["local_images"][k] for s in batch], 0) for k in range(n_l)],
        "local_metas": [batch_view_metas([s["local_metas"][k] for s in batch]) for k in range(n_l)],
        "index": torch.tensor([s["index"] for s in batch], dtype=torch.long),
    }


# %%
# ============================================================================
# Coordinate Metadata & Patch Inverse Projection (Phase 1 재사용)
# ============================================================================
def patch_boxes_to_original(meta: BatchedViewMeta, validate: bool = True, eps: float = 1e-6) -> Tensor:
    crop_box = meta.crop_box
    hflip = meta.hflip
    device = crop_box.device
    B = crop_box.shape[0]
    gh, gw = meta.patch_grid_height, meta.patch_grid_width
    x0 = crop_box[:, 0].view(B, 1, 1)
    y0 = crop_box[:, 1].view(B, 1, 1)
    cw = crop_box[:, 2].view(B, 1, 1)
    ch = crop_box[:, 3].view(B, 1, 1)
    u = torch.arange(gw + 1, device=device, dtype=torch.float32) / gw
    v = torch.arange(gh + 1, device=device, dtype=torch.float32) / gh
    u1 = u[:-1].view(1, 1, gw); u2 = u[1:].view(1, 1, gw)
    v1 = v[:-1].view(1, gh, 1); v2 = v[1:].view(1, gh, 1)
    x1_noflip = x0 + u1 * cw; x2_noflip = x0 + u2 * cw
    x1_flip = x0 + (1.0 - u2) * cw; x2_flip = x0 + (1.0 - u1) * cw
    flip = hflip.view(B, 1, 1)
    x1 = torch.where(flip, x1_flip, x1_noflip).expand(B, gh, gw)
    x2 = torch.where(flip, x2_flip, x2_noflip).expand(B, gh, gw)
    y1 = (y0 + v1 * ch).expand(B, gh, gw)
    y2 = (y0 + v2 * ch).expand(B, gh, gw)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(B, gh * gw, 4)
    boxes = boxes.clamp(0.0, 1.0)
    if validate:
        assert torch.all(boxes >= -eps) and torch.all(boxes <= 1.0 + eps)
    return boxes


# %%
# ============================================================================
# DINOv3 ViT-B/16 Backbone
#   Source of truth: facebookresearch/dinov3 → vision_transformer.py + layers/
# ============================================================================

# ---- 2D Axial RoPE (공식 DINOv3 RopePositionEmbedding port) ----
class RoPE2D(nn.Module):
    """
    2D axial Rotary Position Embedding — 공식 DINOv3 구현 port.

    공식 구현 핵심 구조:
    - periods = base^(2i / (D_head/2)), shape [D_head/4]
    - coords: pixel-center (0.5/H, 1.5/H, ...) → [-1, +1]
    - rescale_coords: train 시 log-uniform 스케일, 고정값이면 상수 곱
    - angles = 2π * coords[:, :, None] / periods[None, None, :] → [HW, 2, D/4]
    - flatten → [HW, D/2] → tile(2) → [HW, D_head]
    - rotate_half: (-x2, x1) 방식 (GPT-NeoX style)
    - apply: x * cos + rotate_half(x) * sin
    """

    def __init__(self, dim: int, num_heads: int, base: float = 100.0,
                 normalize_coords: str = "separate", rescale_coords: float = 2.0,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.D_head = dim // num_heads
        assert self.D_head % 4 == 0, f"head_dim({self.D_head}) must be divisible by 4"
        self.base = base
        self.normalize_coords = normalize_coords
        self.rescale_coords = rescale_coords
        self.rope_dtype = dtype
        # 공식: periods = base^(2i / (D_head/2)), i = 0..D_head/4-1
        periods = base ** (2 * torch.arange(self.D_head // 4, dtype=dtype) / (self.D_head // 2))
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, H: int, W: int) -> Tuple[Tensor, Tensor]:
        """
        Returns (sin, cos), 각각 shape [H*W, D_head].
        공식 DINOv3와 동일한 구조.
        """
        device = self.periods.device
        dd = {"device": device, "dtype": self.rope_dtype}

        # 공식: pixel-center 좌표, normalize → [-1, +1]
        if self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unsupported normalize_coords: {self.normalize_coords}")

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )                                                 # [H, W, 2]
        coords = coords.flatten(0, 1)                     # [HW, 2]
        coords = 2.0 * coords - 1.0                       # [-1, +1]

        # 공식: rescale_coords (학습 시 log-uniform, 여기서는 고정 상수)
        if self.rescale_coords is not None:
            rescale_max = math.log(self.rescale_coords)
            if self.training:
                rescale_hw = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
            else:
                rescale_hw = torch.tensor([math.exp(0.0)], **dd)  # eval: no rescale
            coords = coords * rescale_hw

        # 공식: angles = 2π * coords / periods → [HW, 2, D/4]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)                      # [HW, D/2]
        angles = angles.tile(2)                            # [HW, D_head]

        return torch.sin(angles), torch.cos(angles)


def rope_rotate_half(x: Tensor) -> Tensor:
    """공식 DINOv3: (-x2, x1) 방식 rotation."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """공식 DINOv3: x * cos + rotate_half(x) * sin."""
    return (x * cos) + (rope_rotate_half(x) * sin)


# ---- LinearKMaskedBias (DINOv3: K의 bias를 0으로 마스킹) ----
class LinearKMaskedBias(nn.Linear):
    """
    QKV projection에서 K 부분의 bias를 0으로 마스킹한다.
    out_features = 3 * dim, bias 중 [dim:2*dim] 구간이 K에 해당.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0, f"out_features({o}) must be divisible by 3"
        if self.bias is not None:
            # bias_mask: Q=1, K=0, V=1
            mask = torch.ones(o)
            mask[o // 3: 2 * o // 3] = 0.0
            self.register_buffer("bias_mask", mask)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            masked_bias = self.bias * self.bias_mask.to(self.bias.dtype)
        else:
            masked_bias = None
        return F.linear(input, self.weight, masked_bias)


# ---- DropPath ----
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


# ---- LayerScale ----
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.init_value = init_value
        self.gamma = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


# ---- Mlp (GELU, DINOv3 ViT-B 는 standard MLP) ----
class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---- SelfAttention (DINOv3: RoPE + LinearKMaskedBias + SDPA) ----
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 proj_bias: bool = True, mask_k_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        if mask_k_bias and qkv_bias:
            self.qkv = LinearKMaskedBias(dim, dim * 3, bias=True)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def _apply_rope(self, q: Tensor, k: Tensor,
                    rope: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """공식 DINOv3 SelfAttention.apply_rope port.
        sin/cos: [HW, D_head], q/k: [B, heads, L, D_head].
        prefix(CLS+storage) 위치는 자동으로 건너뛴다.
        """
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        q = q.to(dtype=sin.dtype)
        k = k.to(dtype=sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        assert prefix >= 0, f"prefix={prefix}, q_len={q.shape[-2]}, sin_len={sin.shape[-2]}"
        q = torch.cat((q[:, :, :prefix, :], rope_apply(q[:, :, prefix:, :], sin, cos)), dim=-2)
        k = torch.cat((k[:, :, :prefix, :], rope_apply(k[:, :, prefix:, :], sin, cos)), dim=-2)
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]

        if rope is not None:
            q, k = self._apply_rope(q, k, rope)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# ---- SelfAttentionBlock (DINOv3: pre-norm, LayerScale, DropPath) ----
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, qkv_bias: bool,
                 proj_bias: bool, ffn_bias: bool, drop_path: float,
                 layerscale_init: Optional[float], mask_k_bias: bool,
                 ln_eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, proj_bias, mask_k_bias)
        hidden = int(dim * ffn_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = Mlp(dim, hidden, bias=ffn_bias)

        if layerscale_init is not None:
            self.ls1 = LayerScale(dim, layerscale_init)
            self.ls2 = LayerScale(dim, layerscale_init)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x), rope=rope)))
        x = x + self.dp2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---- PatchEmbed ----
class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def reset_parameters(self):
        # 공식 DINOv3 PatchEmbed.reset_parameters()
        k = 1.0 / (self.proj.in_channels * (self.patch_size ** 2))
        bound = math.sqrt(k)
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)                       # [B, D, gh, gw]
        gh, gw = x.shape[-2], x.shape[-1]
        x = x.permute(0, 2, 3, 1)              # [B, gh, gw, D] (공식 DINOv3: flatten_embedding=False)
        return x, gh, gw


# ---- DinoVisionTransformer (DINOv3 ViT-B/16) ----
class DinoVisionTransformer(nn.Module):
    """
    DINOv3 ViT-B/16 backbone.

    Token sequence: [CLS, storage_0, ..., storage_{n-1}, patch_0, ..., patch_{N-1}]
    n_prefix = 1 + n_storage_tokens

    forward 출력:
    {
        "x_norm_clstoken":     [B, D],
        "x_storage_tokens":    [B, n_storage, D],
        "x_norm_patchtokens":  [B, N_patch, D],
        "masks":               [B, N_patch] or None,
    }
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.n_storage_tokens = cfg.n_storage_tokens
        self.n_prefix = 1 + cfg.n_storage_tokens   # CLS + storage

        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, cfg.embed_dim))

        if cfg.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, cfg.n_storage_tokens, cfg.embed_dim))
        else:
            self.register_parameter("storage_tokens", None)

        rope_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.rope_dtype]
        self.rope = RoPE2D(
            dim=cfg.embed_dim, num_heads=cfg.num_heads,
            base=cfg.rope_base, normalize_coords=cfg.rope_normalize_coords,
            rescale_coords=cfg.rope_rescale_coords, dtype=rope_dtype,
        )

        dpr = [cfg.drop_path_rate] * cfg.depth  # 공식 DINOv3: 모든 block에 동일 drop_path_rate
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=cfg.embed_dim, num_heads=cfg.num_heads, ffn_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias, proj_bias=cfg.proj_bias, ffn_bias=cfg.ffn_bias,
                drop_path=dpr[i], layerscale_init=cfg.layerscale_init,
                mask_k_bias=cfg.mask_k_bias, ln_eps=cfg.ln_eps,
            )
            for i in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.ln_eps)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mask_token)  # DINOv3: zeros
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # LinearKMaskedBias: bias_mask 초기화
                if hasattr(m, "bias_mask") and m.bias is not None:
                    o = m.out_features
                    m.bias_mask.fill_(1)
                    m.bias_mask[o // 3: 2 * o // 3].fill_(0)
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, LayerScale):
                m.reset_parameters()
            elif isinstance(m, PatchEmbed):
                m.reset_parameters()
        self.apply(_init)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Optional[Tensor] = None):
        """
        Args:
            x: [B, 3, H, W]
            masks: [B, N_patch] bool (True = masked → iBOT)
        Returns:
            tokens: [B, 1+n_storage+N_patch, D]
            (gh, gw): patch grid 크기
        """
        x, gh, gw = self.patch_embed(x)  # [B, gh, gw, D]
        B = x.shape[0]
        x = x.flatten(1, 2)               # [B, N_patch, D]

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            # 공식: mask_token 이 graph에 포함되도록 dummy 연산
            cls_token = self.cls_token + 0 * self.mask_token

        if self.storage_tokens is not None:
            storage = self.storage_tokens.expand(B, -1, -1)
        else:
            storage = torch.empty(1, 0, self.embed_dim, dtype=x.dtype, device=x.device)

        x = torch.cat([cls_token.expand(B, -1, -1), storage, x], dim=1)
        return x, (gh, gw)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Args:
            x: [B, 3, H, W]
            mask: [B, N_patch] bool (True = masked)
        Returns:
            {"x_norm_clstoken": [B,D], "x_storage_tokens": [B,n_storage,D],
             "x_norm_patchtokens": [B,N_patch,D], "masks": mask or None}
        """
        tokens, (gh, gw) = self.prepare_tokens_with_masks(x, mask)

        # 공식 DINOv3는 training-time rescale_coords augmentation 때문에
        # 각 transformer block에서 RoPE를 다시 생성한다.
        for blk in self.blocks:
            rope = self.rope(gh, gw)
            tokens = blk(tokens, rope=rope)

        tokens = self.norm(tokens)

        return {
            "x_norm_clstoken": tokens[:, 0],                              # [B, D]
            "x_storage_tokens": tokens[:, 1:self.n_prefix],                # [B, n_storage, D]
            "x_norm_patchtokens": tokens[:, self.n_prefix:],               # [B, N_patch, D]
            "masks": mask,
        }


def build_backbone(cfg: Config) -> DinoVisionTransformer:
    return DinoVisionTransformer(cfg)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# %%
# ============================================================================
# DINOHead (공식 DINOv3: normalized bottleneck + plain bias-free Linear)
# ============================================================================
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256,
                 n_layers=3, norm_last_layer=False):
        super().__init__()
        # norm_last_layer는 Phase-1 호환 인자일 뿐 DINOv3 head에서는 사용하지 않는다.
        del norm_last_layer
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        return self.last_layer(x)


# ============================================================================
# [D3-E1] PositionHead — Phase 1 Cell 10 그대로
# ============================================================================
class PositionHead(nn.Module):
    """x: [B, N, 768] -> [B, N, K_pos]. position loss 전용 lightweight projection."""
    def __init__(self, in_dim: int = 768, out_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# ============================================================================
# [D3-E2 adam] Adam-v2 Part–Whole — composability / decomposability head & helpers
# ============================================================================
ADAM_N_PARTS = 4                                   # 2x2 고정 (Adam-v2 n=4)
_ADAM_PERM = ((0, 1, 2, 3), (1, 0, 3, 2))          # [no-flip, hflip]


class PartWholeMLP(nn.Module):
    """Adam-v2 h_C / h_D: 2-layer MLP (Linear-GELU-Linear). student 전용, EMA teacher 없음."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        for m in (self.fc1, self.fc2):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


def adam_extract_parts(x: Tensor) -> Tensor:
    """[R,C,H,W] -> [R*4,C,H/2,W/2]. sample 마다 연속 4개, view 기준 순서 TL,TR,BL,BR."""
    R, C, H, W = x.shape
    h, w = H // 2, W // 2
    return x.reshape(R, C, 2, h, 2, w).permute(0, 2, 4, 1, 3, 5).reshape(R * 4, C, h, w)


def adam_part_order_perm(hflip: Tensor) -> Tensor:
    """hflip [R] bool -> perm [R,4]. out[:,k] = view 사분면 index, k 는 '원본 좌표' TL,TR,BL,BR.
    hflip view 에서는 원본 왼쪽이 view 오른쪽이므로 좌우만 교환 (상하는 불변)."""
    table = torch.tensor(_ADAM_PERM, dtype=torch.long, device=hflip.device)
    return table[hflip.long()]


def adam_reorder(emb: Tensor, perm: Tensor) -> Tensor:
    """emb [R,4,D] (view 순서) -> [R,4,D] (원본 좌표 순서)."""
    return emb.gather(1, perm.unsqueeze(-1).expand(-1, -1, emb.shape[-1]))


# %%
# ============================================================================
# DINO Loss with Sinkhorn-Knopp Centering (공식 DINOv3 port)
# ============================================================================
class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, centering: str = "sinkhorn_knopp",
                 center_momentum: float = 0.9, sk_n_iterations: int = 3):
        super().__init__()
        self.centering = centering
        self.center_momentum = center_momentum
        self.sk_n_iterations = sk_n_iterations
        self.register_buffer("center", torch.zeros(1, out_dim))

    def sinkhorn_knopp_teacher(self, teacher_output: Tensor, teacher_temp: float) -> Tensor:
        """
        Sinkhorn-Knopp normalization on teacher softmax output.
        Args:
            teacher_output: [B, K] (raw logits, before temp/softmax)
            teacher_temp: temperature
        Returns:
            [B, K] probability (stop-gradient)
        """
        Q = torch.exp(teacher_output.detach().float() / teacher_temp).t()  # [K, B]
        Q /= Q.sum()  # normalize entire matrix
        K_dim, B = Q.shape
        for _ in range(self.sk_n_iterations):
            Q /= Q.sum(dim=1, keepdim=True)  # normalize rows (prototypes)
            Q /= K_dim
            Q /= Q.sum(dim=0, keepdim=True)  # normalize columns (samples)
            Q /= B
        Q = Q / Q.sum(dim=0, keepdim=True)   # final column normalization
        return Q.t()  # [B, K]

    def softmax_center_teacher(self, teacher_output: Tensor, teacher_temp: float) -> Tensor:
        """EMA center 방식 (fallback)."""
        return F.softmax((teacher_output.detach().float() - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        batch_center = teacher_output.float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    def forward(self, student_logits: List[Tensor], teacher_logits: List[Tensor],
                teacher_temp: float, student_temp: float) -> Tensor:
        # Teacher probability
        if self.centering == "sinkhorn_knopp":
            teacher_cat = torch.cat([t.detach() for t in teacher_logits], dim=0)
            teacher_probs_cat = self.sinkhorn_knopp_teacher(teacher_cat, teacher_temp)
            teacher_probs = list(teacher_probs_cat.chunk(len(teacher_logits), dim=0))
        else:
            teacher_probs = [
                self.softmax_center_teacher(t, teacher_temp) for t in teacher_logits
            ]
            self.update_center(torch.cat([t.detach() for t in teacher_logits], dim=0))

        n_teacher = len(teacher_probs)
        total, n_pairs = 0.0, 0
        for ti, q in enumerate(teacher_probs):
            for si, s in enumerate(student_logits):
                if si == ti and si < n_teacher:
                    continue
                log_p = F.log_softmax(s.float() / student_temp, dim=-1)
                total = total + (-(q * log_p).sum(dim=-1)).mean()
                n_pairs += 1
        if n_pairs == 0:
            return teacher_logits[0].new_zeros(())
        return total / n_pairs


# %%
# ============================================================================
# iBOT Patch Loss with Sinkhorn-Knopp (공식 DINOv3 port)
# ============================================================================
class MaskingGenerator:
    """공식 DINOv3 block-based MaskingGenerator port."""

    def __init__(self, input_size, num_masking_patches=None, min_num_patches=4,
                 max_num_patches=None, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top:top + h, left:left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return self.complete_mask_randomly(mask, num_masking_patches)

    @staticmethod
    def complete_mask_randomly(mask: np.ndarray, num_masking_patches: int) -> np.ndarray:
        flat = mask.flatten()
        n_add = int(num_masking_patches - flat.sum())
        if n_add > 0:
            candidates = np.where(~flat)[0]
            to_add = np.random.choice(candidates, size=n_add, replace=False)
            flat[to_add] = True
        return flat.reshape(mask.shape)


class IBOTPatchLoss(nn.Module):
    """
    iBOT patch loss — 공식 DINOv3 SSLMetaArch port.

    핵심 차이 (Phase 1 vs 공식):
    1. SK normalization: 모든 crop의 masked patches를 concat하여 한 번에 적용
    2. masks_weight: sample 별 masked patch 수에 기반한 가중치 적용
       (masked patch 수가 적은 sample이 과대 대표되지 않도록)
    """
    def __init__(self, out_dim: int, centering: str = "sinkhorn_knopp",
                 center_momentum: float = 0.9, sk_n_iterations: int = 3):
        super().__init__()
        self.centering = centering
        self.center_momentum = center_momentum
        self.sk_n_iterations = sk_n_iterations
        self.register_buffer("center", torch.zeros(1, out_dim))

    def sinkhorn_knopp_teacher(self, teacher_output: Tensor, teacher_temp: float) -> Tensor:
        Q = torch.exp(teacher_output.detach().float() / teacher_temp).t()
        Q /= Q.sum()
        K_dim, B = Q.shape
        for _ in range(self.sk_n_iterations):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= K_dim
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= B
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.t()

    def softmax_center_teacher(self, teacher_output: Tensor, teacher_temp: float) -> Tensor:
        return F.softmax((teacher_output.detach().float() - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        if teacher_output.shape[0] == 0:
            return
        batch_center = teacher_output.float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)

    def forward(self, student_patch_logits_cat: Tensor,
                teacher_patch_logits_cat: Tensor,
                masks_weight: Tensor, n_global_samples: int,
                teacher_temp: float, student_temp: float,
                device: torch.device) -> Tensor:
        """
        Args:
            student_patch_logits_cat: [total_masked, K]
            teacher_patch_logits_cat: [total_masked, K]
            masks_weight: [total_masked] — 공식: (1/n_masked_per_sample) per token
            n_global_samples: 전체 global sample 수(n_global_crops * B)
        Returns:
            scalar loss = -weighted_loss.sum() / n_global_samples (공식과 동일)
        """
        if student_patch_logits_cat.shape[0] == 0:
            return torch.zeros((), device=device)

        if self.centering == "sinkhorn_knopp":
            q = self.sinkhorn_knopp_teacher(teacher_patch_logits_cat, teacher_temp)
        else:
            q = self.softmax_center_teacher(teacher_patch_logits_cat, teacher_temp)
            self.update_center(teacher_patch_logits_cat)

        log_p = F.log_softmax(student_patch_logits_cat.float() / student_temp, dim=-1)
        per_token_loss = -(q * log_p).sum(dim=-1)  # [total_masked]
        # 공식: -loss.sum() / masks_flat.shape[0]
        return (per_token_loss * masks_weight).sum() / n_global_samples


# %%
# ============================================================================
# KoLeo Loss (Phase 1 동일)
# ============================================================================
class KoLeoLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        B = z.shape[0]
        if B < 2:
            return z.new_zeros(())
        z = F.normalize(z.float(), dim=-1, p=2, eps=1e-8)
        dist = torch.cdist(z, z, p=2)
        dist = dist + torch.eye(B, device=z.device, dtype=z.dtype) * 1e9
        min_dist = dist.min(dim=1).values
        return -torch.log(min_dist + self.eps).mean()


# %%
# ============================================================================
# [D3-E1] Position Consistency — Phase 1 E1c 이식 (Cell 14 / Cell 15)
#   [P2] 표시 줄만 Phase 1 원본과 다름 (통계/디버그 추가, loss 로직 동일)
# ============================================================================
def compute_ios_matrix(boxes_a: Tensor, boxes_b: Tensor, eps: float = 1e-8) -> Tensor:
    """boxes_a [B,Na,4], boxes_b [B,Nb,4] (x1,y1,x2,y2) -> IoS [B,Na,Nb]"""
    a = boxes_a.unsqueeze(2)
    b = boxes_b.unsqueeze(1)
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])).clamp(min=0)
    area_b = ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])).clamp(min=0)
    smaller = torch.minimum(area_a, area_b)
    return inter / (smaller + eps)


def find_correspondences(ios: Tensor, threshold: float, topk: int = 1):
    """각 teacher patch 에 대해 IoS top-k student patch (>= threshold) 선택."""
    B, Nt, Ns = ios.shape
    k = min(topk, Ns)
    vals, idx = ios.topk(k=k, dim=-1)
    valid = vals >= threshold
    if not bool(valid.any()):
        empty_l = torch.empty(0, dtype=torch.long, device=ios.device)
        empty_f = torch.empty(0, dtype=ios.dtype, device=ios.device)
        return empty_l, empty_l, empty_l, empty_f
    b_idx, t_idx, k_idx = valid.nonzero(as_tuple=True)
    s_idx = idx[b_idx, t_idx, k_idx]
    w = vals[b_idx, t_idx, k_idx]
    return b_idx, t_idx, s_idx, w


class PositionConsistencyLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.threshold = cfg.pos_overlap_threshold
        self.topk = cfg.position_topk
        self.teacher_temp = cfg.position_teacher_temp
        self.student_temp = cfg.position_student_temp
        self.chunk = cfg.position_pair_chunk_size
        self.center_momentum = cfg.position_center_momentum
        self.register_buffer("center", torch.zeros(1, cfg.K_pos))     # CE 분기 전용
        self.debug = False                                             # [P2] smoke test 전용 검사

    def forward(self, teacher_logits, teacher_boxes, student_logits, student_boxes,
                n_global_student: int, device: torch.device):
        weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
        weight_sum = torch.zeros((), device=device, dtype=torch.float32)
        n_pairs_total = 0
        within_valid_sum, within_total, n_cross_used = 0, 0, 0        # [P2]

        for ti, (t_logit, t_box) in enumerate(zip(teacher_logits, teacher_boxes)):
            B, Nt, K = t_logit.shape
            t_flat = t_logit.reshape(B * Nt, K)

            for si, (s_logit, s_box) in enumerate(zip(student_logits, student_boxes)):
                if si == ti and si < n_global_student:
                    continue
                Ns = s_logit.shape[1]

                # positive correspondence: IoS >= pos_overlap_threshold
                with torch.no_grad():
                    ios = compute_ios_matrix(t_box.float(), s_box.float())   # [B,Nt,Ns]
                    b_idx, t_idx, s_idx, w = find_correspondences(ios, self.threshold, self.topk)
                if b_idx.numel() == 0:
                    continue

                s_flat = s_logit.reshape(B * Ns, s_logit.shape[-1])
                t_gather = b_idx * Nt + t_idx
                s_gather = b_idx * Ns + s_idx
                P = b_idx.numel()

                # ============================ InfoNCE ============================
                if self.cfg.position_loss_type == "infonce":
                    t_feat = F.normalize(t_flat.float().detach(), dim=-1)
                    s_feat = F.normalize(s_flat.float(), dim=-1)
                    n_within = self.cfg.position_n_within_negatives
                    n_cross = self.cfg.position_n_cross_negatives
                    neg_thr = self.cfg.position_negative_ios_threshold
                    temp = self.cfg.position_infonce_temp

                    # strict cross-image bank: batch index != anchor 인 student patch 만
                    cross_bank = None
                    if n_cross > 0 and B > 1:
                        with torch.no_grad():
                            anchor_b = torch.arange(B, device=device)[:, None]
                            rand_b = torch.randint(0, B - 1, (B, n_cross), device=device)
                            rand_b = rand_b + (rand_b >= anchor_b).long()
                            rand_s = torch.randint(0, Ns, (B, n_cross), device=device)
                            cross_idx = rand_b * Ns + rand_s
                            if self.debug:                                                 # [P2]
                                assert bool((cross_idx // Ns != anchor_b).all()), \
                                    "cross-image negative 에 anchor 자신의 image 가 섞였습니다."
                        cross_bank = s_feat.index_select(0, cross_idx.reshape(-1)).reshape(B, n_cross, -1)
                        n_cross_used = n_cross                                             # [P2]

                    for start in range(0, P, self.chunk):
                        end = min(start + self.chunk, P)
                        C = end - start
                        b_c = b_idx[start:end]
                        t_c = t_idx[start:end]
                        s_pos_c = s_idx[start:end]

                        anchor = t_feat.index_select(0, t_gather[start:end])   # [C,K_pos]
                        pos = s_feat.index_select(0, s_gather[start:end])      # [C,K_pos]
                        pos_sim = (anchor * pos).sum(-1, keepdim=True)         # [C,1]
                        parts = [pos_sim]
                        within_valid = torch.zeros(C, dtype=torch.bool, device=device)

                        # 1) false-negative-safe within-image negatives (IoS <= neg_thr)
                        if n_within > 0:
                            with torch.no_grad():
                                ios_c = ios[b_c, t_c, :]
                                neg_mask = ios_c <= neg_thr
                                neg_mask.scatter_(1, s_pos_c[:, None], False)
                                k_neg = min(n_within, Ns)
                                rand = torch.rand((C, Ns), device=device)
                                rand.masked_fill_(~neg_mask, 2.0)
                                neg_idx = rand.topk(k=k_neg, dim=1, largest=False).indices
                                neg_selected_valid = neg_mask.gather(1, neg_idx)
                                within_valid = neg_selected_valid.any(dim=1)
                            flat_neg_idx = (b_c[:, None] * Ns + neg_idx).reshape(-1)
                            wneg = s_feat.index_select(0, flat_neg_idx).reshape(C, k_neg, -1)
                            within_sim = torch.bmm(wneg, anchor.unsqueeze(-1)).squeeze(-1)
                            within_sim = within_sim.masked_fill(~neg_selected_valid, float("-inf"))
                            parts.append(within_sim)
                            within_valid_sum += int(within_valid.sum().item())             # [P2]
                            within_total += C                                              # [P2]

                        # 2) strict cross-image negatives
                        if cross_bank is not None:
                            bank_c = cross_bank.index_select(0, b_c)
                            cross_sim = torch.bmm(bank_c, anchor.unsqueeze(-1)).squeeze(-1)
                            parts.append(cross_sim)
                            anchor_valid = torch.ones(C, dtype=torch.bool, device=device)
                        else:
                            anchor_valid = within_valid

                        logits = torch.cat(parts, dim=1) / temp
                        ce = -F.log_softmax(logits, dim=1)[:, 0]
                        wc = w[start:end].float() * anchor_valid.float()
                        weighted_sum = weighted_sum + (wc * ce).sum()
                        weight_sum = weight_sum + wc.sum()
                        n_pairs_total += int(anchor_valid.sum().item())

                # ============================ CE (기존) ============================
                else:
                    n_pairs_total += P
                    for start in range(0, P, self.chunk):
                        end = min(start + self.chunk, P)
                        tl = t_flat.index_select(0, t_gather[start:end]).float()
                        sl = s_flat.index_select(0, s_gather[start:end]).float()
                        q = F.softmax((tl.detach() - self.center) / self.teacher_temp, dim=-1)
                        log_p = F.log_softmax(sl / self.student_temp, dim=-1)
                        ce = -(q * log_p).sum(dim=-1)
                        wc = w[start:end].float()
                        weighted_sum = weighted_sum + (wc * ce).sum()
                        weight_sum = weight_sum + wc.sum()

        extra = {"within_valid_ratio": within_valid_sum / max(within_total, 1),   # [P2]
                 "n_cross_used": n_cross_used}
        if n_pairs_total == 0 or float(weight_sum) <= 0.0:
            return torch.zeros((), device=device), {"n_pairs": 0, "mean_ios": 0.0, **extra}
        loss = weighted_sum / (weight_sum + 1e-8)
        stats = {"n_pairs": int(n_pairs_total),
                 "mean_ios": float(weight_sum / max(n_pairs_total, 1)), **extra}
        return loss, stats

    @torch.no_grad()
    def update_center(self, teacher_logits) -> None:
        """CE 분기 전용 center. infonce 분기에서는 읽지 않음 (Phase 1 과 동일하게 갱신만)."""
        if not teacher_logits:
            return
        acc = torch.stack([t.float().reshape(-1, t.shape[-1]).mean(dim=0) for t in teacher_logits], 0)
        self.center.mul_(self.center_momentum).add_(acc.mean(dim=0, keepdim=True),
                                                    alpha=1 - self.center_momentum)


# %%
# ============================================================================
# Student / Teacher Model
# ============================================================================
class StudentTeacherModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.student_backbone = build_backbone(cfg)
        self.student_dino_head = DINOHead(
            cfg.embed_dim, cfg.K_cls, cfg.head_hidden_dim,
            cfg.head_bottleneck_dim, cfg.head_n_layers, cfg.head_norm_last_layer,
        )
        self.student_ibot_head = DINOHead(
            cfg.embed_dim, cfg.K_patch, cfg.head_hidden_dim,
            cfg.head_bottleneck_dim, cfg.head_n_layers, cfg.head_norm_last_layer,
        )
        # [D3-E1] position head. D3-E0 에서는 생성하지 않으므로 초기화 RNG/파라미터 불변
        self.student_position_head = (
            PositionHead(cfg.embed_dim, cfg.K_pos) if cfg.use_position else None
        )

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head)
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head)
        self.teacher_position_head = (
            copy.deepcopy(self.student_position_head) if cfg.use_position else None
        )

        self._ema_pairs: List[Tuple[nn.Module, nn.Module]] = [
            (self.student_backbone, self.teacher_backbone),
            (self.student_dino_head, self.teacher_dino_head),
            (self.student_ibot_head, self.teacher_ibot_head),
        ]
        if cfg.use_position:
            self._ema_pairs.append((self.student_position_head, self.teacher_position_head))

        for _, teacher_mod in self._ema_pairs:
            for p in teacher_mod.parameters():
                p.requires_grad = False
            teacher_mod.eval()

    def student_modules(self) -> Dict[str, nn.Module]:
        mods = {
            "student_backbone": self.student_backbone,
            "student_dino_head": self.student_dino_head,
            "student_ibot_head": self.student_ibot_head,
        }
        if self.student_position_head is not None:
            mods["student_position_head"] = self.student_position_head
        return mods

    def teacher_modules(self) -> Dict[str, nn.Module]:
        mods = {
            "teacher_backbone": self.teacher_backbone,
            "teacher_dino_head": self.teacher_dino_head,
            "teacher_ibot_head": self.teacher_ibot_head,
        }
        if self.teacher_position_head is not None:
            mods["teacher_position_head"] = self.teacher_position_head
        return mods

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student_mod, teacher_mod in self._ema_pairs:
            for ps, pt in zip(student_mod.parameters(), teacher_mod.parameters()):
                pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)
            for bs, bt in zip(student_mod.buffers(), teacher_mod.buffers()):
                bt.copy_(bs)

    def train(self, mode: bool = True):
        super().train(mode)
        for _, teacher_mod in self._ema_pairs:
            teacher_mod.eval()
        return self


# %%
# ============================================================================
# SSL Training Module — D3-E0 forward_loss
# ============================================================================
PW_LOG_KEYS = [   # [D3-E2] part-whole 로그 (use_partwhole 일 때만 CSV 에 기록)
    "pw_n_comp", "pw_n_decomp", "pw_empty_part_ratio", "pw_diag_cos",
    "pw_part_mem_GB", "pw_part_ms",
]
PW_RESUME_KEYS = (
    "experiment", "partwhole_method", "partwhole_start_epoch", "partwhole_part_size",
    "partwhole_n_crops", "partwhole_batch_frac", "partwhole_embed", "partwhole_student_whole",
    "partwhole_loss", "partwhole_head_hidden", "lambda_comp", "lambda_decomp",
)

LOSS_KEYS = [
    "L_DINO", "L_iBOT", "L_KoLeo", "L_base",
    "L_position", "L_composition", "L_decomposition", "L_total",
]

MODEL_DESCRIPTION = {
    "D3-E0": "DINOv3 ViT-B/16 core SSL baseline",
    "D3-E1": "D3-E0 + Position (coordinate-aware)",
    "D3-E2": "D3-E0 + Part-Whole (Adam-v2: 2x2 독립 encoding part, composition/decomposition)",
    "D3-E3": "D3-E0 + Position + Part-Whole",
}


class SSLTrainingModule(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = StudentTeacherModel(cfg)
        self.dino_loss = DINOLoss(cfg.K_cls, cfg.centering, cfg.center_momentum, cfg.sk_n_iterations)
        self.ibot_loss = IBOTPatchLoss(cfg.K_patch, cfg.centering, cfg.patch_center_momentum, cfg.sk_n_iterations)
        self.koleo_loss = KoLeoLoss(cfg.koleo_eps)
        self.position_loss = PositionConsistencyLoss(cfg) if cfg.use_position else None   # [D3-E1]
        self.mask_generator = MaskingGenerator(
            input_size=(cfg.global_grid, cfg.global_grid),
            max_num_patches=int(cfg.mask_ratio_max * cfg.n_global_tokens),
        )
        # [D3-E2 adam] 맨 마지막에 생성 → backbone/head 초기화 RNG 가 D3-E0 과 동일
        self.composition_predictor = None
        self.decomposition_predictor = None
        self.partwhole_active = True      # Trainer 가 epoch 마다 갱신 (smoke test 는 True)
        self.pw_timing = False            # Trainer 가 log step 에서만 True (cuda sync 발생)
        if cfg.use_partwhole and cfg.partwhole_method == "adam":
            D, H = cfg.embed_dim, cfg.partwhole_head_hidden
            self.composition_predictor = PartWholeMLP(ADAM_N_PARTS * D, H, D)     # 4D -> D
            self.decomposition_predictor = PartWholeMLP(D, H, ADAM_N_PARTS * D)   # D -> 4D

    def _generate_ibot_masks(self, total_global_samples: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """공식 DINOv3 collate_data_and_cast의 masking/masks_weight 동작을 single-GPU용으로 port."""
        cfg = self.cfg
        N = cfg.n_global_tokens
        n_samples_masked = int(total_global_samples * cfg.mask_sample_prob)
        masks_list: List[Tensor] = []

        if n_samples_masked > 0:
            probs = torch.linspace(cfg.mask_ratio_min, cfg.mask_ratio_max, n_samples_masked + 1)
            for i in range(n_samples_masked):
                prob_max = float(probs[i + 1])
                n_mask = int(N * prob_max)
                masks_list.append(torch.BoolTensor(self.mask_generator(n_mask)))

        for _ in range(n_samples_masked, total_global_samples):
            masks_list.append(torch.BoolTensor(self.mask_generator(0)))

        random.shuffle(masks_list)
        masks = torch.stack(masks_list).flatten(1).to(device=device, dtype=torch.bool)
        masks_weight = (
            (1.0 / masks.sum(-1).clamp(min=1.0))
            .unsqueeze(-1)
            .expand_as(masks)[masks]
        )
        return masks, masks_weight

    def trainable_modules(self) -> Dict[str, nn.Module]:
        mods = dict(self.model.student_modules())
        if self.composition_predictor is not None:          # [D3-E2]
            mods["composition_predictor"] = self.composition_predictor
            mods["decomposition_predictor"] = self.decomposition_predictor
        return mods

    # ---------------- [D3-E2 adam] Part–Whole ----------------
    def _adam_embed(self, out: Dict[str, Tensor]) -> Tensor:
        """whole/part 공통 embedding. storage(register) token 은 읽지 않는다."""
        if self.cfg.partwhole_embed == "cls":
            return out["x_norm_clstoken"]
        return out["x_norm_patchtokens"].mean(dim=1)

    def _pw_dist(self, pred: Tensor, target: Tensor) -> Tensor:
        a = F.normalize(pred.float(), dim=-1, eps=1e-6)
        b = F.normalize(target.float().detach(), dim=-1, eps=1e-6)
        cos = (a * b).sum(dim=-1)
        return 2.0 * (1.0 - cos) if self.cfg.partwhole_loss == "nmse" else 1.0 - cos

    def _adam_partwhole(self, g_cat: Tensor, global_metas, masks: Tensor,
                        t_out: Dict[str, Tensor], s_g: Dict[str, Tensor],
                        B: int, device: torch.device):
        """
        Whole = global crop (teacher: 기존 clean forward 재사용)
        Part  = global crop 2x2 사분면, 조각마다 독립 encoding (student/teacher)
        Comp  : h_C(concat s_part[TL,TR,BL,BR]) <-> t_whole           (전체 샘플)
        Decomp: h_D(s_whole) -> 4 parts <-> t_part (같은 slot)       (iBOT mask 0 샘플만)
        """
        cfg = self.cfg
        D = cfg.embed_dim
        n_use = max(1, int(math.ceil(B * cfg.partwhole_batch_frac)))
        rows = torch.cat([torch.arange(c * B, c * B + n_use, device=device)
                          for c in range(cfg.partwhole_n_crops)])                    # [R] (g_cat 행)
        R = rows.numel()
        hflip = torch.cat([m.hflip for m in global_metas], 0).to(device).index_select(0, rows)
        perm = adam_part_order_perm(hflip)                                           # [R,4]

        x = g_cat.index_select(0, rows)                    # [R,3,256,256]  iBOT mask 와 무관한 이미지
        parts = adam_extract_parts(x)                      # [R*4,3,128,128] (view 순서)
        if parts.shape[-1] != cfg.partwhole_part_size:
            parts = F.interpolate(parts, size=(cfg.partwhole_part_size,) * 2,
                                  mode="bicubic", align_corners=False)

        with torch.no_grad():                              # 거의 빈 조각 비율 (로그 전용)
            mean = torch.tensor(cfg.normalize_mean, device=device).view(1, -1, 1, 1)
            std = torch.tensor(cfg.normalize_std, device=device).view(1, -1, 1, 1)
            inten = (parts.float() * std + mean).mean(dim=1)
            fg = (inten > cfg.foreground_threshold).float().mean(dim=(1, 2))
            empty_ratio = (fg < cfg.partwhole_empty_fg_thr).float().mean()

        if self.pw_timing:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        mem0 = torch.cuda.memory_allocated()
        s_part = self._adam_embed(self.model.student_backbone(parts)).reshape(R, ADAM_N_PARTS, -1)
        mem_part = (torch.cuda.memory_allocated() - mem0) / 1024 ** 3   # backward 용으로 남은 activation
        with torch.no_grad():
            t_part = self._adam_embed(self.model.teacher_backbone(parts)).reshape(R, ADAM_N_PARTS, -1)
        part_ms = 0.0
        if self.pw_timing:                                 # student+teacher 조각 forward 시간 (backward 제외)
            ev1.record()
            torch.cuda.synchronize()
            part_ms = ev0.elapsed_time(ev1)

        s_part = adam_reorder(s_part, perm)                # 원본 좌표 TL,TR,BL,BR
        t_part = adam_reorder(t_part, perm).detach()
        t_whole = self._adam_embed(t_out).index_select(0, rows).detach()             # [R,D]

        # ---- Composition (part -> whole) ----
        y_c = self.composition_predictor(s_part.reshape(R, ADAM_N_PARTS * D))
        l_comp = self._pw_dist(y_c, t_whole).mean()

        # ---- Decomposition (whole -> parts) ----
        if cfg.partwhole_student_whole == "unmasked":
            keep = ~masks.index_select(0, rows).any(dim=1)                          # mask 전부 0 인 샘플
            d_idx = keep.nonzero(as_tuple=True)[0]
            s_whole = self._adam_embed(s_g).index_select(0, rows).index_select(0, d_idx)
        else:                                                                       # 옵션: 추가 clean forward
            d_idx = torch.arange(R, device=device)
            s_whole = self._adam_embed(self.model.student_backbone(x))
        n_decomp = int(d_idx.numel())
        if n_decomp > 0:
            y_d = self.decomposition_predictor(s_whole).reshape(n_decomp, ADAM_N_PARTS, D)
            l_decomp = self._pw_dist(y_d, t_part.index_select(0, d_idx)).mean()
        else:
            l_decomp = torch.zeros((), device=device)

        with torch.no_grad():                              # 진단: cos(teacher whole, mean(teacher parts))
            diag = F.cosine_similarity(t_whole.float(), t_part.float().mean(dim=1), dim=-1).mean()

        stats = {
            "pw_n_comp": float(R), "pw_n_decomp": float(n_decomp),
            "pw_empty_part_ratio": float(empty_ratio), "pw_diag_cos": float(diag),
            "pw_part_mem_GB": float(mem_part), "pw_part_ms": float(part_ms),
        }
        return l_comp, l_decomp, stats

    def forward_loss(self, batch: Dict[str, Any], teacher_temp: float,
                     device: torch.device) -> Tuple[Tensor, Dict[str, float]]:
        cfg = self.cfg

        # D3-E0 이외의 실험은 아직 미구현
        if cfg.experiment not in IMPLEMENTED_EXPERIMENTS:
            raise NotImplementedError(f"{cfg.experiment} 미구현. 지원: {IMPLEMENTED_EXPERIMENTS}")

        g_imgs = batch["global_images"]
        l_imgs = batch["local_images"]
        n_g, n_l = len(g_imgs), len(l_imgs)
        B = g_imgs[0].shape[0]
        N_g = cfg.n_global_tokens  # 256

        g_cat = torch.cat(g_imgs, dim=0)    # [n_g*B, 3, 256, 256]
        l_cat = torch.cat(l_imgs, dim=0)    # [n_l*B, 3, 112, 112]

        # 공식 DINOv3 collate 방식: 전체 global samples(n_g*B)에 대해 mask를 한 번 구성
        # mask는 patch token에만 적용되며 CLS/storage token은 포함하지 않는다.
        masks, masks_weight_cat = self._generate_ibot_masks(n_g * B, device)  # [n_g*B, 256]
        assert masks.shape == (n_g * B, N_g)
        masks_per_crop = list(masks.chunk(n_g, dim=0))                        # n_g x [B, 256]

        # ---- Teacher forward: global, unmasked ----
        with torch.no_grad():
            t_out = self.model.teacher_backbone(g_cat)
            t_cls_cat = t_out["x_norm_clstoken"]            # [n_g*B, D]
            t_patch_cat = t_out["x_norm_patchtokens"]        # [n_g*B, 256, D]

            # storage token은 loss에 사용하지 않음 — assert 검증
            assert t_out["x_storage_tokens"].shape[1] == cfg.n_storage_tokens, \
                f"storage tokens shape mismatch: {t_out['x_storage_tokens'].shape}"

            t_cls_logits_cat = self.model.teacher_dino_head(t_cls_cat)  # [n_g*B, K]
            t_cls_logits = list(t_cls_logits_cat.chunk(n_g, 0))
            t_patch_per_crop = list(t_patch_cat.chunk(n_g, 0))   # n_g x [B, 256, D]

            # iBOT: teacher는 unmasked global patch feature에서 student와 동일 mask 위치만 선택
            t_ibot_parts: List[Tensor] = []
            for c in range(n_g):
                m = masks_per_crop[c]                              # [B, 256]
                sel = t_patch_per_crop[c][m]                       # [M_c, D]
                t_ibot_parts.append(self.model.teacher_ibot_head(sel))

            t_ibot_cat = torch.cat(t_ibot_parts, dim=0) if t_ibot_parts else torch.empty(0, cfg.K_patch, device=device)

            # [D3-E1] 기존 unmasked teacher forward 의 patch token 재사용 (CLS/storage 제외)
            t_pos_logits = (
                [self.model.teacher_position_head(p) for p in t_patch_per_crop]   # n_g x [B,256,K_pos]
                if cfg.use_position else None
            )

        # ---- Student forward: masked global + unmasked local ----
        s_g = self.model.student_backbone(g_cat, mask=masks)
        s_l = self.model.student_backbone(l_cat)

        s_cls_g = s_g["x_norm_clstoken"]          # [n_g*B, D]
        s_cls_l = s_l["x_norm_clstoken"]           # [n_l*B, D]
        s_patch_g = s_g["x_norm_patchtokens"]      # [n_g*B, 256, D]

        # DINO CLS logits: global + local student
        s_cls_all = torch.cat([s_cls_g, s_cls_l], dim=0)
        s_cls_logits_cat = self.model.student_dino_head(s_cls_all)
        s_cls_logits = (
            list(s_cls_logits_cat[:n_g * B].chunk(n_g, 0))
            + list(s_cls_logits_cat[n_g * B:].chunk(n_l, 0))
        )

        # iBOT: masked 위치의 student patch logits — 모든 crop concat
        s_patch_per_crop = list(s_patch_g.chunk(n_g, 0))
        s_ibot_parts: List[Tensor] = []
        for c in range(n_g):
            m = masks_per_crop[c]
            sel = s_patch_per_crop[c][m]
            s_ibot_parts.append(self.model.student_ibot_head(sel))
        s_ibot_cat = torch.cat(s_ibot_parts, dim=0) if s_ibot_parts else torch.empty(0, cfg.K_patch, device=device)

        # ---- Losses ----
        assert s_ibot_cat.shape == t_ibot_cat.shape,             f"student/teacher iBOT shape mismatch: {s_ibot_cat.shape} vs {t_ibot_cat.shape}"
        assert s_ibot_cat.shape[0] == masks_weight_cat.numel(),             f"iBOT mask ordering/weight mismatch: {s_ibot_cat.shape[0]} vs {masks_weight_cat.numel()}"

        l_dino = self.dino_loss(s_cls_logits, t_cls_logits, teacher_temp, cfg.student_temp)
        # 공식 forward_masked의 분모는 student_masks_flat.shape[0] = 전체 global sample 수(n_g * B)
        l_ibot = self.ibot_loss(
            s_ibot_cat, t_ibot_cat, masks_weight_cat, n_g * B,
            teacher_temp, cfg.patch_student_temp, device,
        )

        # 공식 SSLMetaArch: mean over global crops 후 koleo_scale=n_global_crops.
        # 따라서 최종 contribution은 algebraically sum(KoLeo(crop_i))와 동일하다.
        s_cls_per_global = list(s_cls_g.chunk(n_g, 0))
        l_koleo_mean = torch.stack([self.koleo_loss(c) for c in s_cls_per_global]).mean()
        koleo_scale = n_g
        l_koleo = koleo_scale * l_koleo_mean

        l_base = cfg.lambda_dino * l_dino + cfg.lambda_ibot * l_ibot + cfg.lambda_koleo * l_koleo
        l_total = l_base

        # ---- [D3-E1] Position InfoNCE (Phase 1 Cell 19 position 부분 이식) ----
        l_position = torch.zeros((), device=device)
        pos_stats = {"n_pairs": 0, "mean_ios": 0.0, "within_valid_ratio": 0.0, "n_cross_used": 0}
        if cfg.use_position:
            # student global: masked forward 가 아닌 clean(unmasked) forward 사용
            s_aux = self.model.student_backbone(g_cat)
            s_aux_patch = list(s_aux["x_norm_patchtokens"].chunk(n_g, 0))   # n_g x [B,256,D] (CLS/storage 제외)
            s_loc_patch = list(s_l["x_norm_patchtokens"].chunk(n_l, 0))     # n_l x [B, 49,D]
            assert s_aux_patch[0].shape[1] == cfg.n_global_tokens, s_aux_patch[0].shape
            assert s_loc_patch[0].shape[1] == cfg.n_local_tokens, s_loc_patch[0].shape
            with torch.no_grad():
                t_boxes = [patch_boxes_to_original(m.to(device), validate=False)
                           for m in batch["global_metas"]]
                # student global crop == teacher global crop -> box 재사용 (Phase 1 동일)
                s_boxes = t_boxes + [patch_boxes_to_original(m.to(device), validate=False)
                                     for m in batch["local_metas"]]
            s_pos_logits = [self.model.student_position_head(p) for p in s_aux_patch + s_loc_patch]
            l_position, pos_stats = self.position_loss(
                teacher_logits=t_pos_logits, teacher_boxes=t_boxes,
                student_logits=s_pos_logits, student_boxes=s_boxes,
                n_global_student=n_g, device=device,
            )
            l_total = l_total + cfg.lambda_pos * l_position
            with torch.no_grad():
                self.position_loss.update_center(t_pos_logits)

        # ---- [D3-E2 adam] Part–Whole (Adam-v2) ----
        l_comp = torch.zeros((), device=device)
        l_decomp = torch.zeros((), device=device)
        pw_stats = {k: 0.0 for k in PW_LOG_KEYS}
        if cfg.use_partwhole and self.partwhole_active:
            l_comp, l_decomp, pw_stats = self._adam_partwhole(
                g_cat, batch["global_metas"], masks, t_out, s_g, B, device)
            l_total = l_total + cfg.lambda_comp * l_comp + cfg.lambda_decomp * l_decomp

        logs = {
            "L_DINO": float(l_dino.detach()),
            "L_iBOT": float(l_ibot.detach()),
            "L_KoLeo": float(l_koleo.detach()),
            "L_base": float(l_base.detach()),
            "L_position": float(l_position.detach()) if cfg.use_position else 0.0,
            "L_composition": float(l_comp.detach()) if cfg.use_partwhole else 0.0,
            "L_decomposition": float(l_decomp.detach()) if cfg.use_partwhole else 0.0,
            "L_total": float(l_total.detach()),
            "mask_ratio": float(masks.float().mean()),
            "position_pairs": float(pos_stats["n_pairs"]),
            "position_mean_ios": float(pos_stats["mean_ios"]),
            "position_within_valid_ratio": float(pos_stats["within_valid_ratio"]),
            "position_n_cross_used": float(pos_stats["n_cross_used"]),
            **{k: float(pw_stats[k]) for k in PW_LOG_KEYS},
        }
        return l_total, logs


# %%
# ============================================================================
# Schedules
# ============================================================================
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0.0):
    warmup_iters = int(warmup_epochs * niter_per_ep)
    total_iters = int(epochs * niter_per_ep)
    warmup = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_iters > 0 else np.array([])
    iters = np.arange(total_iters - warmup_iters)
    cos = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / max(len(iters), 1)))
    schedule = np.concatenate([warmup, cos])
    assert len(schedule) == total_iters
    return schedule


def teacher_temp_scheduler(start, final, warmup_epochs, epochs, niter_per_ep):
    warmup_iters = int(min(warmup_epochs, epochs) * niter_per_ep)
    total_iters = int(epochs * niter_per_ep)
    warm = np.linspace(start, final, warmup_iters) if warmup_iters > 0 else np.array([])
    rest = np.full(total_iters - warmup_iters, final)
    return np.concatenate([warm, rest])


def momentum_scheduler(base, final, epochs, niter_per_ep):
    total_iters = int(epochs * niter_per_ep)
    iters = np.arange(total_iters)
    return final - (final - base) * (np.cos(np.pi * iters / max(total_iters, 1)) + 1) / 2


# %%
# ============================================================================
# Optimizer with layerwise decay + patch_embed LR multiplier
# ============================================================================
def _get_layer_id(name: str, depth: int) -> int:
    """공식 DINOv3 get_vit_lr_decay_rate와 같은 layer id 규칙."""
    if "patch_embed" in name or "cls_token" in name or "storage_tokens" in name or "mask_token" in name:
        return 0
    if "blocks." in name:
        block_id = int(name.split("blocks.")[1].split(".")[0])
        return block_id + 1
    # final norm 등 backbone의 마지막 stage
    return depth + 1


def build_param_groups(cfg: Config, module: SSLTrainingModule) -> List[Dict[str, Any]]:
    """
    DINOv3-style parameter groups:
    - patch_embed: lr * patch_embed_lr_mult
    - layerwise_decay: layer i gets lr_scale = layerwise_decay^(depth + 1 - i)
    - bias/norm/token: no weight decay
    """
    depth = cfg.depth
    lr_base = cfg.scaled_lr()
    seen = set()
    param_groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    trainable_mods = module.trainable_modules()
    # teacher params 확인
    teacher_ids = {id(p) for m in module.model.teacher_modules().values() for p in m.parameters()}

    for mod_name, mod in trainable_mods.items():
        for pname, p in mod.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            assert id(p) not in teacher_ids, f"teacher param in optimizer: {mod_name}.{pname}"
            seen.add(id(p))

            full_name = f"{mod_name}.{pname}"
            # weight decay 여부
            is_no_decay = (
                p.ndim <= 1
                or full_name.endswith(("cls_token", "storage_tokens", "mask_token"))
            )

            # layer id & lr scale (backbone 내부만)
            if "student_backbone" in mod_name:
                layer_id = _get_layer_id(pname, depth)
                lr_scale = cfg.layerwise_decay ** (depth + 1 - layer_id)
                if "patch_embed" in pname:
                    lr_scale *= cfg.patch_embed_lr_mult
            else:
                layer_id = depth + 1  # head (lr scale 1.0)
                lr_scale = 1.0

            wd = 0.0 if is_no_decay else cfg.weight_decay
            is_last_layer = "last_layer" in full_name
            # 공식 DINOv3 fuse key처럼 lr/wd/last-layer 상태가 다른 파라미터는 분리한다.
            group_key = (layer_id, round(float(lr_scale), 12), bool(not is_no_decay), bool(is_last_layer))

            if group_key not in param_groups:
                param_groups[group_key] = {
                    "params": [],
                    "lr_scale": lr_scale,
                    "weight_decay": wd,
                    "apply_wd": not is_no_decay,
                    "layer_id": layer_id,
                    "is_last_layer": is_last_layer,
                }
            param_groups[group_key]["params"].append(p)

    groups = list(param_groups.values())
    for g in groups:
        g["lr"] = lr_base * g["lr_scale"]

    n_params = sum(p.numel() for g in groups for p in g["params"])
    print(f"[optimizer] {len(groups)} param groups, {n_params/1e6:.2f}M total params")
    return groups


def build_optimizer(cfg: Config, module: SSLTrainingModule) -> torch.optim.Optimizer:
    groups = build_param_groups(cfg, module)
    return torch.optim.AdamW(groups, lr=cfg.scaled_lr(), betas=(0.9, 0.999))


def apply_schedules(optimizer, lr_schedule_val, wd_val, last_layer_lr_val=None):
    if last_layer_lr_val is None:
        last_layer_lr_val = lr_schedule_val
    for group in optimizer.param_groups:
        base_lr = last_layer_lr_val if group.get("is_last_layer", False) else lr_schedule_val
        group["lr"] = base_lr * group.get("lr_scale", 1.0)
        if group.get("apply_wd", True):
            group["weight_decay"] = wd_val


# %%
# ============================================================================
# Checkpoint Manager
# ============================================================================
class CheckpointManager:
    def __init__(self, cfg: Config, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.latest_path = os.path.join(run_dir, "latest.pt")
        self.best_path = os.path.join(run_dir, "best.pt")

    def state_dict(self, module, optimizer, epoch, global_step, best_metric):
        m = module.model
        state = {
            "epoch": epoch, "global_step": global_step, "best_metric": best_metric,
            "config": asdict(self.cfg), "environment": ENV_INFO,
            "student_backbone": m.student_backbone.state_dict(),
            "student_dino_head": m.student_dino_head.state_dict(),
            "student_ibot_head": m.student_ibot_head.state_dict(),
            "teacher_backbone": m.teacher_backbone.state_dict(),
            "teacher_dino_head": m.teacher_dino_head.state_dict(),
            "teacher_ibot_head": m.teacher_ibot_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dino_center": module.dino_loss.center.detach().cpu(),
            "ibot_center": module.ibot_loss.center.detach().cpu(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if m.student_position_head is not None:   # [D3-E1] D3-E0 checkpoint 키는 그대로
            state["student_position_head"] = m.student_position_head.state_dict()
            state["teacher_position_head"] = m.teacher_position_head.state_dict()
            state["position_center"] = module.position_loss.center.detach().cpu()
        if module.composition_predictor is not None:   # [D3-E2] D3-E0/E1 checkpoint 키는 그대로
            state["composition_predictor"] = module.composition_predictor.state_dict()
            state["decomposition_predictor"] = module.decomposition_predictor.state_dict()
        return state

    def save(self, state, is_best=False):
        tmp = self.latest_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.latest_path)
        if is_best:
            torch.save(state, self.best_path + ".tmp")
            os.replace(self.best_path + ".tmp", self.best_path)

    def load(self, path, module, optimizer=None, map_location="cpu", strict=True):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint 없음: {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = module.model
        m.student_backbone.load_state_dict(ckpt["student_backbone"], strict=strict)
        m.student_dino_head.load_state_dict(ckpt["student_dino_head"], strict=strict)
        m.student_ibot_head.load_state_dict(ckpt["student_ibot_head"], strict=strict)
        m.teacher_backbone.load_state_dict(ckpt["teacher_backbone"], strict=strict)
        m.teacher_dino_head.load_state_dict(ckpt["teacher_dino_head"], strict=strict)
        m.teacher_ibot_head.load_state_dict(ckpt["teacher_ibot_head"], strict=strict)
        if m.student_position_head is not None:   # [D3-E1]
            if ckpt.get("student_position_head") is None or ckpt.get("teacher_position_head") is None:
                raise RuntimeError(
                    f"checkpoint architecture mismatch ({path}): 현재 실험은 position head 를 요구하지만 "
                    "checkpoint 에 student/teacher_position_head 가 없습니다."
                )
            m.student_position_head.load_state_dict(ckpt["student_position_head"], strict=strict)
            m.teacher_position_head.load_state_dict(ckpt["teacher_position_head"], strict=strict)
            if ckpt.get("position_center") is not None:
                module.position_loss.center.copy_(ckpt["position_center"].to(module.position_loss.center.device))
        elif ckpt.get("student_position_head") is not None:
            print(f"[WARN] {path} 에 position head 가 있으나 현재 실험({self.cfg.experiment})은 사용하지 않음")
        if module.composition_predictor is not None:   # [D3-E2]
            if ckpt.get("composition_predictor") is None or ckpt.get("decomposition_predictor") is None:
                raise RuntimeError(
                    f"checkpoint architecture mismatch ({path}): 현재 실험은 part-whole predictor 를 "
                    "요구하지만 checkpoint 에 없습니다.")
            module.composition_predictor.load_state_dict(ckpt["composition_predictor"], strict=strict)
            module.decomposition_predictor.load_state_dict(ckpt["decomposition_predictor"], strict=strict)
        elif ckpt.get("composition_predictor") is not None:
            print(f"[WARN] {path} 에 part-whole predictor 가 있으나 현재 실험({self.cfg.experiment})은 사용하지 않음")
        module.dino_loss.center.copy_(ckpt["dino_center"].to(module.dino_loss.center.device))
        module.ibot_loss.center.copy_(ckpt["ibot_center"].to(module.ibot_loss.center.device))
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt


# %%
# ============================================================================
# Virtual Epoch Sampler & Trainer
# ============================================================================
class VirtualEpochSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, samples_per_epoch, seed=42):
        self.n = int(dataset_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = seed
        self._perm = None
        self._pos = 0
        self._round = 0

    def __len__(self):
        return self.samples_per_epoch

    def _new_permutation(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._round)
        self._perm = torch.randperm(self.n, generator=g)
        self._pos = 0
        self._round += 1

    def __iter__(self):
        chunks, need = [], self.samples_per_epoch
        while need > 0:
            if self._perm is None or self._pos >= self.n:
                self._new_permutation()
            take = min(need, self.n - self._pos)
            chunks.append(self._perm[self._pos:self._pos + take])
            self._pos += take
            need -= take
        return iter(torch.cat(chunks).tolist())

    def state_dict(self):
        return {"pos": self._pos, "round": self._round}

    def load_state_dict(self, state):
        self._round = int(state["round"])
        self._pos = int(state["pos"])
        if self._round > 0:
            g = torch.Generator()
            g.manual_seed(self.seed + self._round - 1)
            self._perm = torch.randperm(self.n, generator=g)


class Trainer:
    def __init__(self, cfg: Config, file_paths=None):
        self.cfg = cfg
        self.device = DEVICE
        self.run_dir = cfg.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        cfg.to_json(os.path.join(self.run_dir, "config.json"))
        set_seed(cfg.seed, cfg.deterministic)

        if file_paths is None:
            file_paths = get_image_paths(cfg)
        self.file_paths = list(file_paths)
        self.transform = MultiCropTransform(cfg)
        self.dataset = MSKXrayDataset(self.file_paths, self.transform, pre_resize=cfg.pre_resize)

        if cfg.steps_per_epoch is not None:
            self.sampler = VirtualEpochSampler(len(self.dataset), cfg.steps_per_epoch * cfg.batch_size, cfg.seed)
        else:
            self.sampler = None
        self.loader = DataLoader(
            self.dataset, batch_size=cfg.batch_size,
            shuffle=(self.sampler is None), sampler=self.sampler,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last, collate_fn=collate_multicrop,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            generator=make_dataloader_generator(cfg.seed),
        )
        self.niter_per_ep = len(self.loader)
        assert self.niter_per_ep > 0

        self.module = SSLTrainingModule(cfg).to(self.device)
        n_student = count_parameters(self.module.model.student_backbone)
        print(f"[model] student backbone params: {n_student/1e6:.1f}M")

        self.optimizer = build_optimizer(cfg, self.module)
        self.lr_schedule = cosine_scheduler(
            cfg.scaled_lr(), cfg.min_lr, cfg.epochs, self.niter_per_ep,
            cfg.warmup_epochs, cfg.start_warmup_lr)
        self.last_layer_lr_schedule = self.lr_schedule.copy()
        freeze_steps = min(len(self.last_layer_lr_schedule), cfg.freeze_last_layer_epochs * self.niter_per_ep)
        self.last_layer_lr_schedule[:freeze_steps] = 0.0
        self.wd_schedule = cosine_scheduler(
            cfg.weight_decay, cfg.weight_decay_end, cfg.epochs, self.niter_per_ep)
        self.momentum_schedule = momentum_scheduler(
            cfg.teacher_momentum, cfg.teacher_momentum_final, cfg.epochs, self.niter_per_ep)
        self.teacher_temp_schedule = teacher_temp_scheduler(
            cfg.teacher_temp_start, cfg.teacher_temp_final,
            cfg.teacher_temp_warmup_epochs, cfg.epochs, self.niter_per_ep)

        self.ckpt = CheckpointManager(cfg, self.run_dir)
        self.train_log_path = os.path.join(self.run_dir, "training_log.csv")
        self.iter_log_path = os.path.join(self.run_dir, "iteration_log.csv")
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = math.inf

        if cfg.auto_resume and os.path.isfile(self.ckpt.latest_path):
            self._resume(self.ckpt.latest_path)

        print(f"[trainer] {cfg.experiment}  images={len(self.dataset):,}  "
              f"iters/epoch={self.niter_per_ep}  run_dir={self.run_dir}")

    def _resume(self, path):
        ck = self.ckpt.load(path, self.module, self.optimizer, map_location="cpu")
        ck_cfg = ck.get("config", {})                  # [D3-E2] part-whole 설정 호환성 검사
        if self.cfg.use_partwhole or ck_cfg.get("partwhole_method", "none") != "none":
            for key in PW_RESUME_KEYS:
                if key in ck_cfg and ck_cfg[key] != getattr(self.cfg, key):
                    raise RuntimeError(
                        f"resume 거부: checkpoint 의 {key}={ck_cfg[key]!r} 가 현재 Config 의 "
                        f"{getattr(self.cfg, key)!r} 와 다릅니다 ({path}).\n"
                        f"  --no-resume 를 쓰거나 run_tag 를 분리하세요.")
        self.start_epoch = int(ck["epoch"]) + 1
        self.global_step = int(ck["global_step"])
        self.best_metric = float(ck["best_metric"])
        if self.sampler is not None and ck.get("sampler_state"):
            self.sampler.load_state_dict(ck["sampler_state"])
        try:
            torch.set_rng_state(ck["torch_rng_state"])
            torch.cuda.set_rng_state_all(ck["cuda_rng_state"])
            np.random.set_state(ck["numpy_rng_state"])
            random.setstate(ck["python_rng_state"])
        except Exception as exc:
            print(f"[WARN] RNG state 복원 실패: {exc}")
        print(f"[resume] epoch {self.start_epoch}, step {self.global_step}, best {self.best_metric:.6f}")

    def _gpu_mem(self):
        return {
            "gpu_alloc_GB": torch.cuda.memory_allocated() / 1024**3,
            "gpu_reserved_GB": torch.cuda.memory_reserved() / 1024**3,
        }

    def _batch_to_device(self, batch):
        return {
            "global_images": [x.to(self.device, non_blocking=True) for x in batch["global_images"]],
            "local_images": [x.to(self.device, non_blocking=True) for x in batch["local_images"]],
            "global_metas": batch["global_metas"],
            "local_metas": batch["local_metas"],
            "index": batch["index"],
        }

    def _append_csv(self, path, row):
        exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.module.train()
        self.module.partwhole_active = epoch >= cfg.partwhole_start_epoch   # [D3-E2]
        accum = {k: 0.0 for k in LOSS_KEYS}
        accum.update({"mask_ratio": 0.0, "position_pairs": 0.0,
                      "position_mean_ios": 0.0, "position_within_valid_ratio": 0.0})
        if cfg.use_partwhole:
            accum.update({k: 0.0 for k in PW_LOG_KEYS if k != "pw_part_ms"})
        n_iter = 0
        epoch_start = time.time()
        last_log_t, last_log_n = epoch_start, 0

        pbar = tqdm(self.loader, desc=f"[{cfg.experiment}] epoch {epoch:03d}",
                    disable=not cfg.progress_bar, leave=False)
        for it, batch in enumerate(pbar):
            step = self.global_step
            if step >= len(self.lr_schedule):
                break
            lr = float(self.lr_schedule[step])
            last_layer_lr = float(self.last_layer_lr_schedule[step])
            wd = float(self.wd_schedule[step])
            momentum = float(self.momentum_schedule[step])
            teacher_temp = float(self.teacher_temp_schedule[step])
            apply_schedules(self.optimizer, lr, wd, last_layer_lr)

            batch = self._batch_to_device(batch)
            self.module.pw_timing = cfg.use_partwhole and (it % cfg.log_interval == 0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                loss, logs = self.module.forward_loss(batch, teacher_temp, self.device)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss at epoch {epoch} iter {it}")

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float("nan")
            if cfg.clip_grad > 0:
                params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad))
            self.optimizer.step()
            self.module.model.update_teacher(momentum)

            for k in accum:
                accum[k] += logs.get(k, 0.0)
            n_iter += 1
            self.global_step += 1

            if it % cfg.log_interval == 0:
                if cfg.save_iteration_log:
                    torch.cuda.synchronize()        # log step 에서만 sync (구간 평균 s/it)
                    now = time.time()
                    sec_per_it = (now - last_log_t) / max(n_iter - last_log_n, 1)
                    last_log_t, last_log_n = now, n_iter
                    mem = self._gpu_mem()
                    row = {
                        "epoch": epoch, "iteration": it, "global_step": self.global_step,
                        **{k: round(logs.get(k, 0.0), 6) for k in LOSS_KEYS},
                        "learning_rate": lr, "last_layer_lr": last_layer_lr, "weight_decay": wd,
                        "teacher_momentum": momentum, "teacher_temperature": teacher_temp,
                        "grad_norm": grad_norm,
                        "mask_ratio": round(logs["mask_ratio"], 4),
                        "position_pairs": int(logs["position_pairs"]),
                        "position_mean_ios": round(logs["position_mean_ios"], 4),
                        "position_within_valid_ratio": round(logs["position_within_valid_ratio"], 4),
                        "sec_per_it": round(sec_per_it, 4),
                        **{k: round(v, 3) for k, v in mem.items()},
                        "gpu_max_alloc_GB": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                    }
                    if cfg.use_partwhole:
                        row.update({k: round(logs[k], 4) for k in PW_LOG_KEYS})
                    self._append_csv(self.iter_log_path, row)
                postfix = {"total": f"{logs['L_total']:.3f}",
                           "dino": f"{logs['L_DINO']:.3f}",
                           "ibot": f"{logs['L_iBOT']:.3f}"}
                if cfg.use_position:
                    postfix["pos"] = f"{logs['L_position']:.3f}"
                if cfg.use_partwhole and self.module.partwhole_active:
                    postfix["comp"] = f"{logs['L_composition']:.3f}"
                    postfix["dec"] = f"{logs['L_decomposition']:.3f}"
                pbar.set_postfix(postfix)

        mean_logs = {k: v / max(n_iter, 1) for k, v in accum.items()}
        mean_logs["epoch_time_sec"] = time.time() - epoch_start
        return mean_logs

    def fit(self):
        cfg = self.cfg
        print(f"=== training start: {cfg.experiment} (epochs {self.start_epoch} -> {cfg.epochs - 1}) ===")
        for epoch in range(self.start_epoch, cfg.epochs):
            stats = self.train_one_epoch(epoch)
            monitor_value = stats["L_base"] if cfg.monitor == "epoch_mean_base_loss" else stats["L_total"]

            row = {"epoch": epoch, "global_step": self.global_step,
                   **{k: round(stats[k], 6) for k in LOSS_KEYS},
                   "epoch_time_sec": round(stats["epoch_time_sec"], 2)}
            if cfg.use_position:   # [D3-E1] D3-E0 training_log.csv 스키마는 그대로 유지
                row.update({
                    "position_pairs": round(stats["position_pairs"], 1),
                    "position_mean_ios": round(stats["position_mean_ios"], 4),
                    "position_within_valid_ratio": round(stats["position_within_valid_ratio"], 4),
                    "gpu_max_alloc_GB": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                })
            if cfg.use_partwhole:   # [D3-E2] D3-E0/E1 training_log.csv 스키마는 그대로
                row.update({k: round(stats[k], 4) for k in PW_LOG_KEYS if k in stats})
                row["gpu_max_alloc_GB"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            self._append_csv(self.train_log_path, row)

            is_best = monitor_value < self.best_metric
            if is_best:
                self.best_metric = monitor_value
            state = self.ckpt.state_dict(self.module, self.optimizer, epoch, self.global_step, self.best_metric)
            if self.sampler is not None:
                state["sampler_state"] = self.sampler.state_dict()
            self.ckpt.save(state, is_best=is_best)

            pos_str = (f" pos {stats['L_position']:.4f} (pairs {stats['position_pairs']:.0f}, "
                       f"within_valid {stats['position_within_valid_ratio']:.3f})") if cfg.use_position else ""
            pos_str += (f" comp {stats['L_composition']:.4f} dec {stats['L_decomposition']:.4f}"
                        f" (n_decomp {stats['pw_n_decomp']:.0f}, empty {stats['pw_empty_part_ratio']:.3f},"
                        f" diag_cos {stats['pw_diag_cos']:.3f})") if cfg.use_partwhole else ""
            print(f"[{cfg.experiment}] epoch {epoch:03d} | total {stats['L_total']:.4f} "
                  f"| dino {stats['L_DINO']:.4f} ibot {stats['L_iBOT']:.4f} koleo {stats['L_KoLeo']:.4f}"
                  f"{pos_str} | maxmem {torch.cuda.max_memory_allocated()/1024**3:.1f}GB"
                  f" | best {self.best_metric:.4f}{'  *' if is_best else ''}", flush=True)
        print(f"=== training finished: {cfg.experiment} ===")


# %%
# ============================================================================
# Evaluation Feature Extraction
# ============================================================================
class FeatureExtractor:
    def __init__(self, name, forward_fn, mean, std, patch_size, embed_dim,
                 device=DEVICE, use_bf16=True):
        self.name = name
        self.forward_fn = forward_fn
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        self.use_bf16 = use_bf16

    @torch.no_grad()
    def __call__(self, images_unnorm):
        x = images_unnorm.to(self.device, non_blocking=True)
        x = (x - self.mean) / self.std
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.forward_fn(x)
        return {"cls": out["x_norm_clstoken"].float(), "patch": out["x_norm_patchtokens"].float()}

    def grid_size(self, input_size):
        return input_size // self.patch_size


def experiment_run_dir(cfg, experiment):
    name = experiment + (f"_{cfg.run_tag}" if cfg.run_tag else "")
    return os.path.join(cfg.output_root, name)


def load_ssl_backbone(cfg, experiment, checkpoint_name=None, use_teacher=None, device=DEVICE):
    checkpoint_name = checkpoint_name or cfg.eval_checkpoint_name
    use_teacher = cfg.eval_use_teacher if use_teacher is None else use_teacher
    run_dir = experiment_run_dir(cfg, experiment)
    path = os.path.join(run_dir, checkpoint_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 없음: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Config 복원 시 Phase 2 Config 사용
    ck_cfg_dict = ckpt["config"]
    valid_keys = set(Config.__dataclass_fields__.keys())
    tuple_fields = {"image_extensions", "global_crop_scale", "local_crop_scale", "crop_ratio",
                    "global_blur_prob", "normalize_mean", "normalize_std"}
    kwargs = {}
    for k, v in ck_cfg_dict.items():
        if k not in valid_keys:
            continue
        kwargs[k] = tuple(v) if k in tuple_fields and isinstance(v, list) else v
    ck_cfg = Config(**kwargs)
    backbone = build_backbone(ck_cfg)
    key = "teacher_backbone" if use_teacher else "student_backbone"
    backbone.load_state_dict(ckpt[key], strict=True)
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    tag = "teacher" if use_teacher else "student"
    return FeatureExtractor(
        name=experiment, forward_fn=lambda x: backbone(x),
        mean=ck_cfg.normalize_mean, std=ck_cfg.normalize_std,
        patch_size=ck_cfg.patch_size, embed_dim=ck_cfg.embed_dim,
        device=device, use_bf16=cfg.use_bf16,
    ), tag


# %%
# ============================================================================
# Smoke Test
# ============================================================================
def make_synthetic_meta(batch_size, grid, crop_size, device, scale_range=(0.4, 1.0), seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    side = torch.empty(batch_size).uniform_(scale_range[0], scale_range[1], generator=g).sqrt()
    x0 = torch.rand(batch_size, generator=g) * (1 - side)
    y0 = torch.rand(batch_size, generator=g) * (1 - side)
    crop_box = torch.stack([x0, y0, side, side], dim=1).to(device)
    hflip = (torch.rand(batch_size, generator=g) < 0.5).to(device)
    return BatchedViewMeta(crop_box, hflip, crop_size, crop_size, grid, grid)


def make_synthetic_batch(cfg, batch_size, device):
    g_imgs = [torch.randn(batch_size, 3, cfg.global_crop_size, cfg.global_crop_size, device=device)
              for _ in range(cfg.num_global_crops)]
    l_imgs = [torch.randn(batch_size, 3, cfg.local_crop_size, cfg.local_crop_size, device=device)
              for _ in range(cfg.num_local_crops)]
    g_metas = [make_synthetic_meta(batch_size, cfg.global_grid, cfg.global_crop_size, device,
                                   cfg.global_crop_scale, seed=100 + k) for k in range(cfg.num_global_crops)]
    l_metas = [make_synthetic_meta(batch_size, cfg.local_grid, cfg.local_crop_size, device,
                                   cfg.local_crop_scale, seed=200 + k) for k in range(cfg.num_local_crops)]
    return {"global_images": g_imgs, "global_metas": g_metas,
            "local_images": l_imgs, "local_metas": l_metas, "index": torch.arange(batch_size)}


def smoke_test_position(cfg: Config, batch_size: int = 4, device: torch.device = DEVICE) -> None:
    """[D3-E1] IoS/correspondence + n_cross 64/0 loss·backward·grad·EMA 검사."""
    assert (cfg.global_grid, cfg.local_grid) == (16, 7), (cfg.global_grid, cfg.local_grid)

    print("--- [position: IoS / correspondence] ---")
    def _meta(cb, size, grid):
        return BatchedViewMeta(torch.tensor([cb], device=device), torch.tensor([False], device=device),
                               size, size, grid, grid)
    ba = patch_boxes_to_original(_meta([0.100, 0.100, 0.60, 0.60], cfg.global_crop_size, cfg.global_grid))
    bb = patch_boxes_to_original(_meta([0.175, 0.175, 0.60, 0.60], cfg.global_crop_size, cfg.global_grid))
    b_idx, _, _, w = find_correspondences(compute_ios_matrix(ba, bb), cfg.pos_overlap_threshold, cfg.position_topk)
    assert b_idx.numel() > 0 and torch.all(w >= cfg.pos_overlap_threshold)
    bf = patch_boxes_to_original(_meta([0.80, 0.80, 0.19, 0.19], cfg.local_crop_size, cfg.local_grid))
    b2, _, _, _ = find_correspondences(compute_ios_matrix(ba, bf), cfg.pos_overlap_threshold, cfg.position_topk)
    assert b2.numel() == 0
    print(f"  OK  overlap pairs={b_idx.numel()}, non-overlap pairs=0")

    for n_cross in (64, 0):
        print(f"--- [position loss/grad: n_cross={n_cross}] ---")
        pcfg = replace(cfg, position_n_cross_negatives=n_cross)
        module = SSLTrainingModule(pcfg).to(device)
        module.train()
        module.position_loss.debug = True
        seen_tokens: List[int] = []
        hook = module.model.student_position_head.register_forward_hook(
            lambda m, inp, out: seen_tokens.append(int(inp[0].shape[-2])))
        batch = make_synthetic_batch(pcfg, batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=pcfg.use_bf16):
            loss, logs = module.forward_loss(batch, teacher_temp=pcfg.teacher_temp_start, device=device)
        hook.remove()

        # position head 입력 = patch token 만 (global 256, local 49). CLS/storage(+5) 미포함
        expected = [pcfg.n_global_tokens] * pcfg.num_global_crops + [pcfg.n_local_tokens] * pcfg.num_local_crops
        assert seen_tokens == expected, f"position head 입력 token 수 {seen_tokens} != {expected}"
        assert torch.isfinite(loss)
        assert logs["position_pairs"] > 0 and logs["L_position"] > 0, logs
        assert int(logs["position_n_cross_used"]) == n_cross, logs["position_n_cross_used"]
        assert math.isclose(logs["L_total"], logs["L_base"] + pcfg.lambda_pos * logs["L_position"],
                            rel_tol=1e-4, abs_tol=1e-4)
        print(f"  L_position={logs['L_position']:.4f} pairs={logs['position_pairs']:.0f} "
              f"within_valid={logs['position_within_valid_ratio']:.3f}")

        module.zero_grad(set_to_none=True)
        loss.backward()
        g = module.model.student_position_head.proj.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "position head grad 없음"
        for p in module.model.teacher_position_head.parameters():
            assert not p.requires_grad and p.grad is None
        before = module.model.teacher_position_head.proj.weight.detach().clone()
        with torch.no_grad():
            module.model.student_position_head.proj.weight.add_(1e-3)
        module.model.update_teacher(0.9)
        assert not torch.equal(before, module.model.teacher_position_head.proj.weight), "position head EMA 미갱신"
        print("  OK  backward / position-head grad / teacher no-grad / EMA")
        del module, batch, loss
        torch.cuda.empty_cache()


def smoke_test_partwhole(cfg: Config, batch_size: int = 4, device: torch.device = DEVICE) -> None:
    """[D3-E2 adam] 조각 추출 순서 / hflip 재배열 / storage 차단 / loss·grad / mask 0 샘플 / 비활성 구간."""
    G, h = cfg.global_crop_size, cfg.global_crop_size // 2

    print("--- [part-whole: 조각 추출 순서] ---")
    x = torch.zeros(2, 3, G, G, device=device)
    for k, (r, c) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        x[:, :, r * h:(r + 1) * h, c * h:(c + 1) * h] = float(k)
    p = adam_extract_parts(x).reshape(2, ADAM_N_PARTS, -1)
    for k in range(ADAM_N_PARTS):
        assert torch.all(p[:, k] == float(k)), f"part {k} 추출 순서 오류"
    print("  OK  view 순서 TL,TR,BL,BR")

    print("--- [part-whole: hflip -> 원본 좌표 순서] ---")
    gg, n = cfg.global_grid, 8
    hh = gg // 2
    meta = make_synthetic_meta(n, gg, G, device, cfg.global_crop_scale, seed=11)
    meta.hflip = torch.tensor([False, True] * (n // 2), device=device)
    boxes = patch_boxes_to_original(meta)                                  # [n,256,4]

    def _quad_center(i0, i1):                                              # view 사분면 중심 (원본 좌표)
        v = 0.5 * (boxes[..., i0] + boxes[..., i1])
        return v.reshape(n, 2, hh, 2, hh).mean(dim=(2, 4)).reshape(n, 4)

    cx, cy = _quad_center(0, 2), _quad_center(1, 3)
    perm = adam_part_order_perm(meta.hflip)
    cx_o, cy_o = cx.gather(1, perm), cy.gather(1, perm)
    assert torch.all(cx_o[:, 0] < cx_o[:, 1]) and torch.all(cx_o[:, 2] < cx_o[:, 3]), "재배열 후 좌우 순서 오류"
    assert torch.all(cy_o[:, 0] < cy_o[:, 2]) and torch.all(cy_o[:, 1] < cy_o[:, 3]), "상하 순서 오류"
    assert torch.all(cx[1::2, 0] > cx[1::2, 1]), "negative control: flip 샘플은 재배열 전 좌우가 뒤집혀 있어야 함"
    emb = torch.arange(4, device=device, dtype=torch.float32).view(1, 4, 1).expand(n, 4, 3)
    assert torch.equal(adam_reorder(emb, perm)[:, :, 0], perm.float()), "adam_reorder 오류"
    print("  OK  hflip 샘플 좌우 교환, slot 0 = 원본 왼쪽 위")

    print("--- [part-whole: loss / grad / decomposition 샘플 선택] ---")
    module = SSLTrainingModule(cfg).to(device)
    module.train()
    fake = {"x_norm_clstoken": torch.randn(2, cfg.embed_dim, device=device),
            "x_norm_patchtokens": torch.randn(2, 5, cfg.embed_dim, device=device),
            "x_storage_tokens": torch.full((2, cfg.n_storage_tokens, cfg.embed_dim), float("nan"), device=device)}
    assert torch.isfinite(module._adam_embed(fake)).all(), "storage(register) token 이 embedding 에 섞였습니다."

    batch = make_synthetic_batch(cfg, batch_size, device)
    n_rows = cfg.partwhole_n_crops * max(1, math.ceil(batch_size * cfg.partwhole_batch_frac))
    n_tot = cfg.num_global_crops * batch_size
    module.pw_timing = True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        loss, logs = module.forward_loss(batch, teacher_temp=cfg.teacher_temp_start, device=device)
    assert torch.isfinite(loss)
    assert int(logs["pw_n_comp"]) == n_rows, logs["pw_n_comp"]
    if n_rows == n_tot and cfg.partwhole_student_whole == "unmasked":
        expected = n_tot - int(n_tot * cfg.mask_sample_prob)               # mask 0 샘플 수
        assert int(logs["pw_n_decomp"]) == expected, (logs["pw_n_decomp"], expected)
    assert logs["L_composition"] > 0 and logs["L_decomposition"] > 0, logs
    assert math.isclose(logs["L_total"],
                        logs["L_base"] + cfg.lambda_comp * logs["L_composition"]
                        + cfg.lambda_decomp * logs["L_decomposition"], rel_tol=1e-4, abs_tol=1e-4)
    print(f"  L_comp={logs['L_composition']:.4f} L_decomp={logs['L_decomposition']:.4f} "
          f"n_comp={logs['pw_n_comp']:.0f} n_decomp={logs['pw_n_decomp']:.0f} "
          f"diag_cos={logs['pw_diag_cos']:.3f} part_mem={logs['pw_part_mem_GB']:.2f}GB part_ms={logs['pw_part_ms']:.1f}")

    module.zero_grad(set_to_none=True)
    loss.backward()
    for name, head in (("composition", module.composition_predictor),
                       ("decomposition", module.decomposition_predictor)):
        g = head.fc2.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{name} head grad 없음"
    print("  OK  backward / composition·decomposition head grad")

    module.partwhole_active = False                                        # epoch < start_epoch
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
        _, logs0 = module.forward_loss(batch, teacher_temp=cfg.teacher_temp_start, device=device)
    assert logs0["L_composition"] == 0.0 and logs0["pw_n_comp"] == 0.0
    assert math.isclose(logs0["L_total"], logs0["L_base"], rel_tol=1e-5, abs_tol=1e-5)
    print("  OK  비활성 구간: L_total == L_base (D3-E0 경로)")
    del module, batch, loss
    torch.cuda.empty_cache()

    ccfg = replace(cfg, partwhole_student_whole="clean")                   # 옵션 경로
    m2 = SSLTrainingModule(ccfg).to(device)
    m2.train()
    b2 = make_synthetic_batch(ccfg, batch_size, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ccfg.use_bf16):
        _, logs_c = m2.forward_loss(b2, teacher_temp=ccfg.teacher_temp_start, device=device)
    assert int(logs_c["pw_n_decomp"]) == n_rows
    print(f"  OK  student_whole=clean 옵션: n_decomp={logs_c['pw_n_decomp']:.0f}")
    del m2, b2
    torch.cuda.empty_cache()


def run_smoke_test(cfg: Config, experiment: Optional[str] = None, batch_size: int = 4) -> None:
    experiment = experiment or cfg.experiment
    print("=" * 74)
    print(f"SMOKE TEST — {experiment} (DINOv3 ViT-B/16)")
    print("=" * 74)
    test_cfg = replace(cfg, experiment=experiment)
    device = DEVICE

    # 1) Backbone shape test
    print("--- [backbone shape test] ---")
    backbone = build_backbone(test_cfg).to(device).eval()
    with torch.no_grad():
        x_g = torch.randn(2, 3, test_cfg.global_crop_size, test_cfg.global_crop_size, device=device)
        out_g = backbone(x_g)
        assert out_g["x_norm_clstoken"].shape == (2, test_cfg.embed_dim), \
            f"CLS shape: {out_g['x_norm_clstoken'].shape}"
        assert out_g["x_storage_tokens"].shape == (2, test_cfg.n_storage_tokens, test_cfg.embed_dim), \
            f"storage shape: {out_g['x_storage_tokens'].shape}"
        assert out_g["x_norm_patchtokens"].shape == (2, test_cfg.n_global_tokens, test_cfg.embed_dim), \
            f"patch shape: {out_g['x_norm_patchtokens'].shape}"
        print(f"  global: CLS {out_g['x_norm_clstoken'].shape}, "
              f"storage {out_g['x_storage_tokens'].shape}, "
              f"patch {out_g['x_norm_patchtokens'].shape}")

        x_l = torch.randn(2, 3, test_cfg.local_crop_size, test_cfg.local_crop_size, device=device)
        out_l = backbone(x_l)
        assert out_l["x_norm_patchtokens"].shape == (2, test_cfg.n_local_tokens, test_cfg.embed_dim), \
            f"local patch shape: {out_l['x_norm_patchtokens'].shape}"
        print(f"  local:  CLS {out_l['x_norm_clstoken'].shape}, "
              f"storage {out_l['x_storage_tokens'].shape}, "
              f"patch {out_l['x_norm_patchtokens'].shape}")
    del backbone
    torch.cuda.empty_cache()
    print("  OK")

    # 2) Loss + gradient test
    print(f"--- [loss + gradient test: {experiment}] ---")
    module = SSLTrainingModule(test_cfg).to(device)
    module.train()

    # DINOv3 head: plain bias-free Linear (weight_norm 없음)
    for head_name, head in (
        ("DINO", module.model.student_dino_head),
        ("iBOT", module.model.student_ibot_head),
    ):
        assert isinstance(head.last_layer, nn.Linear), f"{head_name} last_layer is not nn.Linear"
        assert head.last_layer.bias is None, f"{head_name} last_layer bias must be None"
        assert not hasattr(head.last_layer, "weight_g"), f"{head_name} still uses weight_norm"

    optimizer = build_optimizer(test_cfg, module)
    batch = make_synthetic_batch(test_cfg, batch_size, device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=test_cfg.use_bf16):
        loss, logs = module.forward_loss(batch, teacher_temp=test_cfg.teacher_temp_start, device=device)
    for k in LOSS_KEYS:
        assert math.isfinite(logs[k]), f"{k} is not finite: {logs[k]}"
    assert torch.isfinite(loss)
    print("  losses:", "  ".join(f"{k}={logs[k]:.4f}" for k in LOSS_KEYS))
    if not test_cfg.use_position:   # [D3-E1] D3-E0 회귀 검사
        assert module.model.student_position_head is None and module.position_loss is None
        assert logs["L_position"] == 0.0
    if not test_cfg.use_partwhole:  # [D3-E2] D3-E0/E1 회귀 검사
        assert module.composition_predictor is None and module.decomposition_predictor is None
        assert logs["L_composition"] == 0.0 and logs["L_decomposition"] == 0.0

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Student gradient 확인
    n_grad = sum(1 for p in module.model.student_backbone.parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    n_param = sum(1 for _ in module.model.student_backbone.parameters())
    assert n_grad > 0, "student backbone gradient 없음"
    print(f"  student backbone grads: {n_grad}/{n_param}")

    for head_name, head in (
        ("DINO", module.model.student_dino_head),
        ("iBOT", module.model.student_ibot_head),
    ):
        head_grads = [p.grad for p in head.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in head_grads), \
            f"student {head_name} head gradient 없음"
    print("  student DINO/iBOT head grads: OK")

    # Teacher gradient 없음 확인
    for name, mod in module.model.teacher_modules().items():
        for p in mod.parameters():
            assert not p.requires_grad, f"{name}: requires_grad=True"
            assert p.grad is None, f"{name}: grad 존재"
    print("  teacher grads: None (OK)")

    # EMA 확인
    before = module.model.teacher_backbone.cls_token.detach().clone()
    optimizer.step()
    module.model.update_teacher(0.9)
    after = module.model.teacher_backbone.cls_token.detach()
    assert not torch.equal(before, after), "teacher EMA 미갱신"
    print("  EMA update: OK")

    # 3) Checkpoint round-trip
    print("--- [checkpoint round-trip] ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_mgr = CheckpointManager(test_cfg, tmpdir)
        state = ckpt_mgr.state_dict(module, optimizer, 0, 1, 999.0)
        ckpt_mgr.save(state, is_best=True)

        # 새 모듈 로딩
        module2 = SSLTrainingModule(test_cfg).to(device)
        ckpt_mgr.load(os.path.join(tmpdir, "best.pt"), module2)

        # teacher feature 일치 확인
        module.eval(); module2.eval()
        with torch.no_grad():
            x_test = torch.randn(2, 3, test_cfg.global_crop_size, test_cfg.global_crop_size, device=device)
            out1 = module.model.teacher_backbone(x_test)
            out2 = module2.model.teacher_backbone(x_test)
            o1, o2 = out1["x_norm_clstoken"], out2["x_norm_clstoken"]
            p1, p2 = out1["x_norm_patchtokens"], out2["x_norm_patchtokens"]
            assert torch.allclose(o1, o2, atol=1e-5), f"teacher CLS mismatch: max diff {(o1-o2).abs().max()}"
            assert torch.allclose(p1, p2, atol=1e-5), f"teacher patch mismatch: max diff {(p1-p2).abs().max()}"
            if test_cfg.use_position:   # [D3-E1]
                for a, b in zip(module.model.teacher_position_head.parameters(),
                                module2.model.teacher_position_head.parameters()):
                    assert torch.equal(a, b), "teacher position head round-trip 불일치"
                assert torch.equal(module.position_loss.center, module2.position_loss.center)
            if test_cfg.use_partwhole:   # [D3-E2]
                for a, b in zip(list(module.composition_predictor.parameters()) + list(module.decomposition_predictor.parameters()),
                                list(module2.composition_predictor.parameters()) + list(module2.decomposition_predictor.parameters())):
                    assert torch.equal(a, b), "part-whole predictor round-trip 불일치"
        print("  save/load round-trip: OK (teacher CLS/patch features match)")
    del module, module2, optimizer
    torch.cuda.empty_cache()

    if test_cfg.use_position:
        smoke_test_position(test_cfg, batch_size, device)
    if test_cfg.use_partwhole:
        smoke_test_partwhole(test_cfg, batch_size, device)

    print("=" * 74)
    print(f"ALL {experiment} SMOKE TESTS PASSED")
    print("=" * 74)


# %%
# ============================================================================
# Usage
# ============================================================================
USAGE_TEXT = """
사용 순서 (Phase 2 — DINOv3)
------------------------------
1) smoke test
   run_smoke_test(cfg, experiment="D3-E0")

2) 학습
   cfg.experiment = "D3-E0"
   trainer = Trainer(cfg)
   trainer.fit()

3) CLI 학습 (train_phase2.py 사용)
   ./venv/bin/python3 -u train_phase2.py --experiment D3-E0 --batch-size 256 --max-train-images 0 2>&1 | tee outputs/msk_ssl_phase2/D3-E0/console.log
   ./venv/bin/python3 -u train_phase2.py --experiment D3-E1 --run-tag e1c        --pos-n-cross 64 --batch-size 256 --max-train-images 0
   ./venv/bin/python3 -u train_phase2.py --experiment D3-E1 --run-tag e1c_within --pos-n-cross 0  --batch-size 256 --max-train-images 0
   ./venv/bin/python3 -u train_phase2.py --experiment D3-E2 --partwhole-method adam --run-tag adam --batch-size 256 --max-train-images 0 --monitor epoch_mean_base_loss
"""

if __name__ == "__main__" or "ipykernel" in sys.modules:
    print(USAGE_TEXT)
