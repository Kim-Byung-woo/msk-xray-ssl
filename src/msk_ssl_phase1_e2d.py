# %%
# ============================================================================
# Cell 00. CUDA / Environment Setup
#   반드시 torch import 이전에 실행되어야 한다.
#   Physical GPU 7 -> 프로세스 내부에서는 cuda:0 으로 보인다.
# ============================================================================
import os
import sys

# 기본값은 physical GPU 7. train.py 등에서 미리 지정한 값이 있으면 그것을 존중한다.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# multi-crop 으로 인한 fragmentation 완화 (B200 180GB 에서도 안전한 기본값)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if "torch" in sys.modules:
    print(
        "[WARN] torch 가 이미 import 되어 있습니다. "
        "CUDA_VISIBLE_DEVICES 설정이 적용되지 않았을 수 있습니다.\n"
        "       Kernel > Restart Kernel 후 Cell 00 부터 다시 실행하십시오."
    )
else:
    print("[OK] CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])

PYTHON_EXECUTABLE = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/venv/bin/python3"
WORK_DIR = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994"

print("[INFO] python executable (expected):", PYTHON_EXECUTABLE)
print("[INFO] python executable (current) :", sys.executable)
print("[INFO] working directory           :", os.getcwd())


# %%
# ============================================================================
# Cell 01. Imports & Package Availability Check
#   누락 패키지는 자동 설치하지 않고 설치 명령만 출력한다.
# ============================================================================
import importlib
import platform
from typing import Dict, List, Optional, Tuple, Any, Sequence

# (pip name, import name, required?)
_PACKAGE_SPEC = [
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("Pillow", "PIL", True),
    ("tqdm", "tqdm", True),
    ("scikit-learn", "sklearn", True),
    ("matplotlib", "matplotlib", True),
    ("timm", "timm", False),
    ("transformers", "transformers", False),
    ("umap-learn", "umap", False),
]

PACKAGE_STATUS: Dict[str, Dict[str, Any]] = {}
_missing_required, _missing_optional = [], []

for pip_name, import_name, required in _PACKAGE_SPEC:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        PACKAGE_STATUS[pip_name] = {"available": True, "version": version, "required": required}
    except Exception as exc:  # noqa: BLE001
        PACKAGE_STATUS[pip_name] = {
            "available": False, "version": None, "required": required, "error": repr(exc),
        }
        (_missing_required if required else _missing_optional).append(pip_name)

print("=" * 74)
print("PACKAGE AVAILABILITY")
print("=" * 74)
for name, info in PACKAGE_STATUS.items():
    tag = "REQUIRED" if info["required"] else "OPTIONAL"
    state = f"OK   {info['version']}" if info["available"] else "MISSING"
    print(f"  [{tag:8s}] {name:16s} : {state}")

if _missing_optional:
    print("\n[NOTE] optional package 누락 — 관련 기능만 비활성화됩니다:", ", ".join(_missing_optional))
    print(f"  {PYTHON_EXECUTABLE} -m pip install " + " ".join(_missing_optional))

if _missing_required:
    print("\n[ERROR] 필수 package 누락:", ", ".join(_missing_required))
    print("  아래 명령으로 설치한 뒤 커널을 재시작하십시오:")
    print(f"  {PYTHON_EXECUTABLE} -m pip install " + " ".join(_missing_required))
    raise ImportError(f"required packages missing: {_missing_required}")

# --- 실제 import (Cell 00 이후에만 torch 를 import 한다) -----------------------
import copy
import csv
import glob
import json
import math
import random
import time
import warnings
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

HAS_TRANSFORMERS = PACKAGE_STATUS["transformers"]["available"]
HAS_UMAP = PACKAGE_STATUS["umap-learn"]["available"]
HAS_TIMM = PACKAGE_STATUS["timm"]["available"]

# B200: TF32 matmul 허용 (bf16 autocast 밖의 fp32 연산 가속)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU unavailable: torch.cuda.is_available() == False.\n"
        "  - CUDA_VISIBLE_DEVICES=7 로 노출된 GPU 가 실제로 존재하는지 확인\n"
        "  - nvidia-smi 로 GPU 7 상태 확인\n"
        "  - venv 의 torch 가 CUDA build 인지 확인"
    )

DEVICE = torch.device("cuda:0")  # physical GPU 7 == 논리 cuda:0

ENV_INFO = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "gpu_total_memory_GB": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "transformers": PACKAGE_STATUS["transformers"]["version"],
    "timm": PACKAGE_STATUS["timm"]["version"],
    "umap-learn": PACKAGE_STATUS["umap-learn"]["version"],
}
print("\n" + "=" * 74)
print("ENVIRONMENT")
print("=" * 74)
for k, v in ENV_INFO.items():
    print(f"  {k:22s} : {v}")


# %%
# ============================================================================
# Cell 02. Global Config Dataclass
#   모든 경로/hyperparameter 는 이 Config 에서만 변경한다.
#   E0~E3 는 cfg.experiment 만 바꾸고 나머지는 전부 동일하게 유지한다 (§42).
# ============================================================================
@dataclass
class Config:
    # ---------------- paths ----------------
    python_executable: str = PYTHON_EXECUTABLE
    work_dir: str = WORK_DIR
    data_root: str = "/NHNHOME/WORKSPACE/26msit005_B/temp/MSK"
    output_root: str = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/outputs/msk_ssl"
    anatomical_annotation_csv: Optional[str] = None
    rad_dino_local_path: Optional[str] = None
    rad_dino_model_id: str = "microsoft/rad-dino"

    # ---------------- experiment ----------------
    experiment: str = "E0"                    # "E0" | "E1" | "E2" | "E3"
    run_tag: str = ""                         # 출력 디렉토리 suffix (선택)
    seed: int = 42
    deterministic: bool = False

    # ---------------- dataset ----------------
    image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    # 데이터 루트 바로 아래에서 제외할 디렉토리 (다른 사용자 산출물 등)
    exclude_dir_names: Tuple[str, ...] = ("keem_ckpt",)
    # 전체 PNG 3,490,701장 / 디렉토리 약 260만개 -> 매 실행 재스캔은 90분 이상 소요.
    # 반드시 캐시를 사용한다 (없으면 최초 1회만 스캔 후 저장).
    dataset_cache_list: Optional[str] = (
        "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/outputs/msk_ssl/file_list.txt"
    )
    scan_workers: int = 16                    # 파일 스캔 병렬 프로세스 수
    max_train_images: Optional[int] = 300_000  # None = 전체 3.49M 사용

    # ---------------- multi-crop views (§5) ----------------
    num_global_crops: int = 2
    num_local_crops: int = 6
    global_crop_size: int = 224               # 224 / 14 = 16 -> N_global = 256
    local_crop_size: int = 98                 # 98  / 14 = 7  -> N_local  = 49
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)
    crop_ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    hflip_prob: float = 0.5                   # geometry aug 는 RRC + HFlip 만 (§6)
    # --- foreground-aware crop ---------------------------------------------
    # EDA 결과 border_black_ratio 평균 0.63, black_ratio 평균 0.34.
    # local scale (0.05, 0.4) 에서는 순수 배경 crop 이 자주 발생하므로 rejection sampling 으로 거른다.
    # crop 좌표계 자체는 바꾸지 않으므로 coordinate tracking 에는 영향이 없다 (§6 준수).
    foreground_aware_crop: bool = True
    foreground_map_size: int = 64             # 전경 판정용 축소 해상도
    foreground_threshold: float = 0.02        # normalized intensity 기준 전경 판정
    min_foreground_global: float = 0.30       # global crop 최소 전경 비율
    min_foreground_local: float = 0.15        # local crop 최소 전경 비율
    foreground_max_attempts: int = 10         # 재추첨 횟수. 초과 시 전경 비율이 가장 큰 crop 사용
    # 1024px PNG 을 매번 8회 resized_crop 하는 비용을 줄이기 위한 사전 축소.
    # local crop 최소 변 = sqrt(0.05)*512 = 114px >= 98px 이므로 출력 해상도 손실이 없다.
    pre_resize: Optional[int] = 512           # None = 원본 1024px 그대로 사용
    crop_max_attempts: int = 10               # 실패 시 전경 비율이 가장 높은 후보 사용
    # photometric (coordinate metadata 에 영향 없음)
    brightness_jitter: float = 0.4
    contrast_jitter: float = 0.4
    global_blur_prob: Tuple[float, float] = (1.0, 0.1)   # (G1, G2)
    local_blur_prob: float = 0.5
    solarize_prob: float = 0.2
    # EDA pixel-level QC (3,000장) 기준 dataset 통계. ImageNet 통계 대신 사용한다.
    normalize_mean: Tuple[float, float, float] = (0.2495, 0.2495, 0.2495)
    normalize_std: Tuple[float, float, float] = (0.2609, 0.2609, 0.2609)

    # ---------------- backbone (§4, 전 실험 동일) ----------------
    patch_size: int = 14
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    in_chans: int = 3
    qkv_bias: bool = True
    drop_path_rate: float = 0.1
    layerscale_init: float = 1e-5
    pos_embed_grid: int = 16                  # 224/14 기준 pos-embed 해상도

    # ---------------- projection heads ----------------
    K_cls: int = 65536
    K_patch: int = 65536
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_n_layers: int = 3
    head_norm_last_layer: bool = True

    # ---------------- DINO CLS loss (§11) ----------------
    student_temp: float = 0.1
    teacher_temp_start: float = 0.04
    teacher_temp_final: float = 0.07
    teacher_temp_warmup_epochs: int = 30
    center_momentum: float = 0.9

    # ---------------- iBOT patch loss (§12) ----------------
    patch_student_temp: float = 0.1
    patch_teacher_temp_start: float = 0.04
    patch_teacher_temp_final: float = 0.07
    patch_center_momentum: float = 0.9
    mask_ratio_min: float = 0.10
    mask_ratio_max: float = 0.50
    mask_sample_prob: float = 0.5             # 배치 내 masking 적용 sample 비율
    ibot_compute_on_masked_only: bool = True  # True: masked token 만 head 통과 (수식 동일, 메모리 절약)

    # ---------------- KoLeo (§13) ----------------
    koleo_eps: float = 1e-8

    # ---------------- position consistency (§14~17) ----------------
    K_pos: int = 1024
    pos_overlap_threshold: float = 0.5
    position_topk: int = 1
    position_student_temp: float = 0.1
    position_teacher_temp: float = 0.04
    position_center_momentum: float = 0.9
    position_pair_chunk_size: int = 4096

# # --- InfoNCE 변형: cross-image negative 로 image-signature shortcut 차단 ---
#     position_loss_type: str = "infonce"           # "ce"(기존) | "infonce"
#     position_infonce_temp: float = 0.2
#     position_n_within_negatives: int = 32         # 같은 영상 내 다른 위치 (anchor 별)
#     position_n_cross_negatives: int = 1024        # 다른 영상 patch (chunk 공유 bank)

    # ---------------- position consistency ----------------
    position_loss_type: str = "infonce"
    position_infonce_temp: float = 0.2
    position_n_within_negatives: int = 32
    position_n_cross_negatives: int = 64
    position_negative_ios_threshold: float = 0.05
    

    # ---------------- part-whole (§18~21) ----------------
    part_grid_size: int = 4                   # 16x16 patch grid -> 4x4 regions, K_part = 16
    # E2b: teacher part-whole target 에 batch centering 적용
    partwhole_center_whole: bool = True
    # --- E2c: independent local-crop Part ---------------------------------
    #   "global_grid" : E2/E2b. global view 의 final patch token 을 4x4 pooling (globally
    #                   contextualized token -> structural shortcut 가능)
    #   "local_crop"  : E2c. 독립 인코딩된 local crop 을 Part 로 사용
    partwhole_part_source: str = "global_grid"      # "global_grid" | "local_crop"
    #   local Part feature pooling. E2 의 avg-pool 연산자를 그대로 유지하려면 "patch_mean".
    partwhole_local_part_feature: str = "patch_mean"  # "patch_mean" | "cls"
    #   local crop 이 global crop 안에 이 비율 이상 포함될 때만 Part-Whole pair 로 사용
    partwhole_containment_threshold: float = 0.8
    #   relative geometry encoding 차원 (아래 local_in_global_geometry 와 반드시 일치)
    partwhole_geom_dim: int = 8

    # ---------------- E2d: local preservation (patch Gram anchoring) ----------
    #   "none" : 기존 동작 (E0~E2c)
    #   "gram" : student clean global patch token 의 patch-patch cosine 구조(Gram)를
    #            frozen Gram teacher (= 학습 완료된 E0 teacher backbone) 의 구조에 고정한다.
    #            feature 자체가 아니라 patch 간 상대적 유사도만 제약하므로
    #            Part-Whole 이 feature 를 바꾸는 것은 허용하고, 서로 다른 위치의 patch 가
    #            비슷해지는 방향(local specificity 손실)만 억제한다.
    local_preservation: str = "none"          # "none" | "gram"
    gram_teacher_ckpt: Optional[str] = None   # 예: .../outputs/msk_ssl/E0/best.pt
    gram_teacher_key: str = "teacher_backbone"
    lambda_gram: float = 1.0

    # ---------------- loss weights (§10, §23) ----------------
    lambda_dino: float = 1.0
    lambda_ibot: float = 1.0
    lambda_koleo: float = 0.1
    lambda_pos: float = 0.1
    lambda_comp: float = 0.1
    lambda_decomp: float = 0.1

    # ---------------- optimization (§24) ----------------
    batch_size: int = 32                      # 보수적 기본값. B200 memory 보며 사용자가 증가
    epochs: int = 100
    # steps_per_epoch 를 지정하면 "가상 epoch" 으로 동작한다.
    # (전체 3.49M 장은 batch 32 기준 1 epoch = 109k iteration 이라 checkpoint 간격이 너무 길다)
    steps_per_epoch: Optional[int] = 2000
    lr: float = 5e-4                          # base_lr (batch 기준 linear scaling 은 아래 flag)
    lr_linear_scaling: bool = True            # lr * batch_size / 256
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    start_warmup_lr: float = 0.0
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    teacher_momentum: float = 0.996
    teacher_momentum_final: float = 1.0

    # ---------------- runtime (§25, §26) ----------------
    use_bf16: bool = True
    use_compile: bool = False                 # dynamic multi-crop shape 로 인해 default False
    num_workers: int = 24                     # 1024px PNG decode 가 병목. 72코어 서버 기준.
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True

    # ---------------- logging / checkpoint (§27, §28) ----------------
    log_interval: int = 10
    progress_bar: bool = True                 # nohup 실행 시 False 권장 (tqdm 로그 폭증 방지)
    save_iteration_log: bool = True
    monitor: str = "epoch_mean_total_loss"
    monitor_mode: str = "min"                 # "min" | "max"
    auto_resume: bool = True

    # ---------------- evaluation (§30~34) ----------------
    eval_use_teacher: bool = True             # representation 평가는 teacher backbone 사용
    eval_checkpoint_name: str = "best.pt"
    eval_num_images: int = 256
    eval_batch_size: int = 16
    eval_input_size: int = 224
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

    # ---------------- derived (post_init) ----------------
    def __post_init__(self):
        if self.experiment not in ("E0", "E1", "E2", "E3"):
            raise ValueError(f"experiment must be one of E0/E1/E2/E3, got {self.experiment!r}")
        if self.global_crop_size % self.patch_size != 0:
            raise ValueError(
                f"ViT patch size mismatch: global_crop_size({self.global_crop_size}) % "
                f"patch_size({self.patch_size}) != 0"
            )
        if self.local_crop_size % self.patch_size != 0:
            raise ValueError(
                f"ViT patch size mismatch: local_crop_size({self.local_crop_size}) % "
                f"patch_size({self.patch_size}) != 0"
            )
        if self.partwhole_part_source not in ("global_grid", "local_crop"):
            raise ValueError(
                f"partwhole_part_source must be 'global_grid' | 'local_crop', "
                f"got {self.partwhole_part_source!r}"
            )
        if self.partwhole_local_part_feature not in ("patch_mean", "cls"):
            raise ValueError(
                f"partwhole_local_part_feature must be 'patch_mean' | 'cls', "
                f"got {self.partwhole_local_part_feature!r}"
            )
        if not (0.0 < self.partwhole_containment_threshold <= 1.0):
            raise ValueError("partwhole_containment_threshold must be in (0, 1].")
        if self.local_preservation not in ("none", "gram"):
            raise ValueError(
                f"local_preservation must be 'none' | 'gram', got {self.local_preservation!r}"
            )
        if self.local_preservation == "gram" and not self.gram_teacher_ckpt:
            raise ValueError("local_preservation='gram' 에는 gram_teacher_ckpt 경로가 필요합니다.")
        if self.global_grid % self.part_grid_size != 0:
            raise ValueError(
                f"part_grid_size({self.part_grid_size}) 가 global patch grid"
                f"({self.global_grid}x{self.global_grid}) 를 나누지 못합니다."
            )

    # --- shape helpers ---
    @property
    def global_grid(self) -> int:
        return self.global_crop_size // self.patch_size      # 16

    @property
    def local_grid(self) -> int:
        return self.local_crop_size // self.patch_size       # 7

    @property
    def n_global_tokens(self) -> int:
        return self.global_grid ** 2                          # 256

    @property
    def n_local_tokens(self) -> int:
        return self.local_grid ** 2                           # 49

    @property
    def k_part(self) -> int:
        return self.part_grid_size ** 2                       # 16

    # --- experiment flags (§23) ---
    @property
    def use_position(self) -> bool:
        return self.experiment in ("E1", "E3")

    @property
    def use_partwhole(self) -> bool:
        return self.experiment in ("E2", "E3")

    @property
    def need_aux_clean_forward(self) -> bool:
        """E1/E2/E3 에서만 unmasked student global forward 를 추가로 수행한다 (§9)."""
        return self.use_position or self.use_partwhole or self.local_preservation != "none"

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
print("experiment          :", cfg.experiment)
print("global grid / tokens:", f"{cfg.global_grid}x{cfg.global_grid}", "->", cfg.n_global_tokens)
print("local  grid / tokens:", f"{cfg.local_grid}x{cfg.local_grid}", "->", cfg.n_local_tokens)
print("K_part              :", cfg.k_part)
print("part source         :", cfg.partwhole_part_source)
print("local preservation  :", cfg.local_preservation)
print("use_position        :", cfg.use_position)
print("use_partwhole       :", cfg.use_partwhole)
print("effective lr        :", cfg.scaled_lr())
print("run_dir             :", cfg.run_dir)


# %%
# ============================================================================
# Cell 03. Reproducibility & Seed
# ============================================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Python / NumPy / torch(CPU, CUDA) seed 를 모두 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # noqa: BLE001
            print("[WARN] deterministic algorithms 활성화 실패:", exc)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 별 독립적이면서 재현 가능한 seed."""
    base = torch.initial_seed() % (2 ** 31 - 1)
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % (2 ** 31 - 1))


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


set_seed(cfg.seed, cfg.deterministic)
print(f"[OK] seed={cfg.seed}, deterministic={cfg.deterministic}")


# %%
# ============================================================================
# Cell 04. Dataset Discovery & File Validation
# ============================================================================
def _scan_subtree(task: Tuple[str, Tuple[str, ...]]) -> List[str]:
    """단일 subtree 를 os.scandir 로 순회하며 이미지 경로를 수집한다 (worker 함수)."""
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
    root: str,
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
    exclude_dirs: Sequence[str] = (),
    workers: int = 16,
    verbose: bool = True,
) -> List[str]:
    """
    데이터 루트를 병렬로 스캔한다.

    구조 예시
    ---------
    /NHNHOME/WORKSPACE/26msit005_B/temp/MSK/
        R-ID리스트(2408-113-1562)_1_1 영상/<patient>_<study>/xxx.png
        ...
        keem_ckpt/            <- exclude_dirs 로 제외

    단일 프로세스 os.walk 는 약 260만 디렉토리에서 90분 이상 걸리므로
    2-depth 까지 펼친 뒤 process pool 로 분산 처리한다.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset path 없음: {root!r}\n  cfg.data_root 를 확인하십시오."
        )
    exts = tuple(e.lower() for e in extensions)
    exclude = set(exclude_dirs)

    t0 = time.time()
    # --- depth 1: 최상위 항목 --------------------------------------------
    top_dirs, direct_files = [], []
    with os.scandir(root) as entries:
        for e in entries:
            if e.name.startswith(".") or e.name in exclude:
                continue
            if e.is_dir(follow_symlinks=False):
                top_dirs.append(e.path)
            elif e.is_file(follow_symlinks=False) and os.path.splitext(e.name)[1].lower() in exts:
                direct_files.append(e.path)
    if verbose:
        print(f"[scan] top-level dirs: {len(top_dirs)} "
              f"(excluded: {sorted(exclude)})")

    # --- depth 2: task 분할 ----------------------------------------------
    tasks: List[str] = []
    for d in top_dirs:
        children = []
        try:
            with os.scandir(d) as entries:
                for e in entries:
                    if e.name.startswith("."):
                        continue
                    if e.is_dir(follow_symlinks=False):
                        children.append(e.path)
                    elif e.is_file(follow_symlinks=False) and os.path.splitext(e.name)[1].lower() in exts:
                        direct_files.append(e.path)
        except (PermissionError, OSError) as exc:
            print(f"[scan][WARN] {d}: {exc}")
        tasks.extend(children if children else [d])
    if verbose:
        print(f"[scan] subtree tasks: {len(tasks):,}  workers: {workers}")

    files = list(direct_files)
    payload = [(t, exts) for t in tasks]
    if workers and workers > 1 and len(payload) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_scan_subtree, payload, chunksize=64),
                total=len(payload), desc="[scan] subtrees", disable=not verbose,
            ):
                files.extend(res)
    else:
        for t in tqdm(payload, desc="[scan] subtrees", disable=not verbose):
            files.extend(_scan_subtree(t))

    files.sort()   # deterministic ordering (§42)
    if verbose:
        print(f"[scan] {len(files):,} images in {time.time() - t0:.1f}s")
    if not files:
        raise FileNotFoundError(
            f"PNG 없음: {root!r} 아래에서 {exts} 파일을 찾지 못했습니다."
        )
    return files


def discover_images(
    root: str,
    extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
    cache_list: Optional[str] = None,
    max_images: Optional[int] = None,
    verbose: bool = True,
    exclude_dirs: Sequence[str] = (),
    workers: int = 16,
    rebuild_cache: bool = False,
) -> List[str]:
    """
    Recursive image discovery (캐시 우선).

    Returns
    -------
    List[str] : 정렬된 절대 경로 목록. max_images 지정 시 균등 간격 deterministic subsample.
    """
    if cache_list is not None and os.path.isfile(cache_list) and not rebuild_cache:
        with open(cache_list) as f:
            files = [ln.strip() for ln in f if ln.strip()]
        if verbose:
            print(f"[cache] {len(files):,} files loaded from {cache_list}")
    else:
        files = scan_dataset(root, extensions, exclude_dirs, workers, verbose)
        if cache_list is not None:
            os.makedirs(os.path.dirname(cache_list), exist_ok=True)
            tmp = cache_list + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(files))
            os.replace(tmp, cache_list)
            if verbose:
                print(f"[cache] wrote {len(files):,} paths -> {cache_list}")

    if max_images is not None and max_images < len(files):
        # 균등 간격 subsample: 정렬 순서를 보존하므로 12개 상위 폴더에 고르게 분포한다
        idx = np.linspace(0, len(files) - 1, num=max_images).round().astype(int)
        files = [files[i] for i in sorted(set(idx.tolist()))]

    if verbose:
        print(f"[dataset] {len(files):,} images")
        for p in files[:3]:
            print("   e.g.", p)
    return files


def get_image_paths(
    cfg: Config, max_images: Optional[int] = -1, verbose: bool = True,
    rebuild_cache: bool = False,
) -> List[str]:
    """Config 기반 통일 진입점. max_images=-1 이면 cfg.max_train_images 를 사용한다."""
    n = cfg.max_train_images if max_images == -1 else max_images
    return discover_images(
        cfg.data_root, cfg.image_extensions, cfg.dataset_cache_list, n, verbose,
        exclude_dirs=cfg.exclude_dir_names, workers=cfg.scan_workers,
        rebuild_cache=rebuild_cache,
    )


def validate_images(paths: Sequence[str], n_check: int = 8) -> None:
    """상위 n_check 개 파일을 실제로 열어 손상 여부/모드/해상도를 확인한다."""
    n_check = min(n_check, len(paths))
    print(f"[validate] checking {n_check} sample(s)")
    for p in paths[:n_check]:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                print(f"   OK  mode={im.mode:5s} size={im.size} {os.path.basename(p)}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"손상되었거나 읽을 수 없는 이미지: {p}\n  {exc!r}") from exc


# %%
# ============================================================================
# Cell 05. MSK X-ray Dataset
# ============================================================================
class MSKXrayDataset(Dataset):
    """
    MSK X-ray SSL dataset.

    __getitem__ 반환
    ----------------
    {
      "global_images": List[Tensor[3, Hg, Wg]]  (len = num_global_crops)
      "global_metas" : List[ViewMeta]
      "local_images" : List[Tensor[3, Hl, Wl]]  (len = num_local_crops)
      "local_metas"  : List[ViewMeta]
      "index"        : int
    }
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        transform: "MultiCropTransform",
        pre_resize: Optional[int] = None,
    ):
        if len(file_paths) == 0:
            raise ValueError("빈 dataset: file_paths 가 비어 있습니다.")
        self.file_paths = list(file_paths)
        self.transform = transform
        self.pre_resize = pre_resize

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load(self, path: str) -> Image.Image:
        # X-ray 는 grayscale -> L 로 읽고 3채널로 replicate (§4)
        with Image.open(path) as im:
            im = im.convert("L")
            # 1024px 원본을 8회 resized_crop 하는 비용을 줄이기 위한 사전 축소 (aspect 유지)
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
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"이미지 로딩 실패 [{index}] {path}: {exc!r}") from exc
        out = self.transform(img)
        out["index"] = index
        return out


# %%
# ============================================================================
# Cell 06. Coordinate-aware Multi-Crop Augmentation
#   geometry 를 바꾸는 augmentation 은 RandomResizedCrop + HorizontalFlip 만 허용 (§6).
#   photometric augmentation 은 coordinate metadata 에 영향을 주지 않는다.
# ============================================================================
@dataclass
class ViewMeta:
    """
    단일 sample 단일 view 의 coordinate metadata.

    crop_box: Tensor[4] = [x0, y0, width, height]
        원본 영상 기준 normalized coordinates ([0,1] x [0,1])
    hflip: horizontal flip 적용 여부
    """
    crop_box: torch.Tensor
    hflip: bool
    output_height: int
    output_width: int
    patch_grid_height: int
    patch_grid_width: int


@dataclass
class BatchedViewMeta:
    """
    collate 후 batch 단위 metadata.

    crop_box: Tensor[B, 4]  = [x0, y0, w, h]  (normalized, original image frame)
    hflip   : Tensor[B]     (bool)
    """
    crop_box: torch.Tensor
    hflip: torch.Tensor
    output_height: int
    output_width: int
    patch_grid_height: int
    patch_grid_width: int

    def to(self, device: torch.device) -> "BatchedViewMeta":
        return BatchedViewMeta(
            crop_box=self.crop_box.to(device, non_blocking=True),
            hflip=self.hflip.to(device, non_blocking=True),
            output_height=self.output_height,
            output_width=self.output_width,
            patch_grid_height=self.patch_grid_height,
            patch_grid_width=self.patch_grid_width,
        )

    @property
    def batch_size(self) -> int:
        return self.crop_box.shape[0]

    @property
    def num_patches(self) -> int:
        return self.patch_grid_height * self.patch_grid_width


def sample_rrc_params(
    height: int,
    width: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    rng: random.Random,
    max_attempts: int = 10,
) -> Tuple[int, int, int, int]:
    """
    torchvision RandomResizedCrop.get_params 와 동일한 규칙의 crop parameter sampling.
    외부 rng 를 받아 evaluation 에서 deterministic 재현이 가능하도록 한다.

    Returns
    -------
    (i, j, h, w) : top, left, height, width  (pixel 단위)
    """
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
    # fallback: center crop (aspect 제한 적용)
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
    """
    한 view 를 생성하고 coordinate metadata 를 함께 반환한다.

    Returns
    -------
    (image: Tensor[3, size, size] normalized, meta: ViewMeta)
    """

    def __init__(
        self,
        size: int,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        patch_size: int,
        hflip_prob: float,
        brightness: float,
        contrast: float,
        blur_prob: float,
        solarize_prob: float,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        min_foreground: float = 0.0,
        max_attempts: int = 1,
    ):
        if size % patch_size != 0:
            raise ValueError(f"crop size {size} 가 patch size {patch_size} 로 나누어지지 않습니다.")
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

    def _sample_crop(
        self, H: int, W: int, rng: random.Random, fg_map: Optional[np.ndarray]
    ) -> Tuple[int, int, int, int]:
        """
        RandomResizedCrop parameter 를 뽑되, foreground map 이 주어지면 전경 비율이
        min_foreground 이상인 crop 이 나올 때까지 재추첨한다.
        crop 자체는 여전히 RandomResizedCrop 이므로 coordinate tracking 은 정확히 유지된다.
        (EDA: border_black_ratio 평균 0.63 -> 배경만 담긴 crop 이 자주 나온다)
        """
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

    def __call__(
        self,
        img: Image.Image,
        rng: Optional[random.Random] = None,
        fg_map: Optional[np.ndarray] = None,
    ):
        rng = rng or random
        W, H = img.size

        # ---- geometry 1: RandomResizedCrop (선택적으로 foreground-aware) -------
        i, j, h, w = self._sample_crop(H, W, rng, fg_map)
        view = TF.resized_crop(
            img, i, j, h, w, [self.size, self.size],
            interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
        )
        # ---- geometry 2: HorizontalFlip --------------------------------------
        do_flip = rng.random() < self.hflip_prob
        if do_flip:
            view = TF.hflip(view)

        # ---- photometric (coordinate 무관) -----------------------------------
        if rng.random() < 0.8:
            view = self.color_jitter(view)
        if rng.random() < self.blur_prob:
            view = self.blur(view)
        if rng.random() < self.solarize_prob:
            view = TF.solarize(view, threshold=128)

        x = TF.to_tensor(view)                 # [3, size, size], [0,1]
        x = self.normalize(x)

        meta = ViewMeta(
            crop_box=torch.tensor([j / W, i / H, w / W, h / H], dtype=torch.float32),
            hflip=bool(do_flip),
            output_height=self.size,
            output_width=self.size,
            patch_grid_height=self.grid,
            patch_grid_width=self.grid,
        )
        return x, meta


class MultiCropTransform:
    """
    DINOv2-style multi-crop + coordinate metadata.

    Returns (per sample)
    --------------------
    {
      "global_images": List[Tensor[3,224,224]] x num_global_crops
      "global_metas" : List[ViewMeta]
      "local_images" : List[Tensor[3,98,98]]   x num_local_crops
      "local_metas"  : List[ViewMeta]
    }
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        gb = cfg.global_blur_prob
        self.global_crops = [
            CoordinateAwareCrop(
                size=cfg.global_crop_size,
                scale=cfg.global_crop_scale,
                ratio=cfg.crop_ratio,
                patch_size=cfg.patch_size,
                hflip_prob=cfg.hflip_prob,
                brightness=cfg.brightness_jitter,
                contrast=cfg.contrast_jitter,
                blur_prob=gb[k] if k < len(gb) else gb[-1],
                solarize_prob=0.0 if k == 0 else cfg.solarize_prob,
                mean=cfg.normalize_mean,
                std=cfg.normalize_std,
                min_foreground=cfg.min_foreground_global if cfg.foreground_aware_crop else 0.0,
                max_attempts=cfg.foreground_max_attempts,
            )
            for k in range(cfg.num_global_crops)
        ]
        self.local_crop = CoordinateAwareCrop(
            size=cfg.local_crop_size,
            scale=cfg.local_crop_scale,
            ratio=cfg.crop_ratio,
            patch_size=cfg.patch_size,
            hflip_prob=cfg.hflip_prob,
            brightness=cfg.brightness_jitter,
            contrast=cfg.contrast_jitter,
            blur_prob=cfg.local_blur_prob,
            solarize_prob=cfg.solarize_prob,
            mean=cfg.normalize_mean,
            std=cfg.normalize_std,
            min_foreground=cfg.min_foreground_local if cfg.foreground_aware_crop else 0.0,
            max_attempts=cfg.foreground_max_attempts,
        )
        self.num_local_crops = cfg.num_local_crops

    def _foreground_map(self, img: Image.Image) -> Optional[np.ndarray]:
        """이미지당 1회만 계산하는 축소 전경 마스크 [S, S] (bool)."""
        if not self.cfg.foreground_aware_crop:
            return None
        s = self.cfg.foreground_map_size
        small = img.convert("L").resize((s, s), Image.BILINEAR)
        return (np.asarray(small, dtype=np.float32) / 255.0) > self.cfg.foreground_threshold

    def __call__(self, img: Image.Image) -> Dict[str, Any]:
        g_imgs, g_metas, l_imgs, l_metas = [], [], [], []
        fg_map = self._foreground_map(img)
        for crop in self.global_crops:
            x, m = crop(img, fg_map=fg_map)
            g_imgs.append(x)
            g_metas.append(m)
        for _ in range(self.num_local_crops):
            x, m = self.local_crop(img, fg_map=fg_map)
            l_imgs.append(x)
            l_metas.append(m)
        return {
            "global_images": g_imgs,
            "global_metas": g_metas,
            "local_images": l_imgs,
            "local_metas": l_metas,
        }


def batch_view_metas(metas: Sequence[ViewMeta]) -> BatchedViewMeta:
    """List[ViewMeta] (per-sample) -> BatchedViewMeta (crop_box [B,4], hflip [B])."""
    return BatchedViewMeta(
        crop_box=torch.stack([m.crop_box for m in metas], dim=0),       # [B, 4]
        hflip=torch.tensor([m.hflip for m in metas], dtype=torch.bool), # [B]
        output_height=metas[0].output_height,
        output_width=metas[0].output_width,
        patch_grid_height=metas[0].patch_grid_height,
        patch_grid_width=metas[0].patch_grid_width,
    )


def collate_multicrop(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    custom collate: view 별로 stack 하고 metadata 를 batch 화한다.

    Returns
    -------
    {
      "global_images": List[Tensor[B,3,224,224]]  (len = num_global_crops)
      "global_metas" : List[BatchedViewMeta]      (crop_box [B,4])
      "local_images" : List[Tensor[B,3,98,98]]
      "local_metas"  : List[BatchedViewMeta]
      "index"        : Tensor[B]
    }
    """
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
# Cell 07. Coordinate Metadata & Patch Inverse Projection
#   augmented view 의 각 patch token 이 원본 영상에서 차지하는 bbox 를 계산한다 (§7).
# ============================================================================
def patch_boxes_to_original(
    meta: BatchedViewMeta,
    validate: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args
    ----
    meta: BatchedViewMeta
        crop_box [B, 4] = [x0, y0, w, h] (original-normalized), hflip [B]

    Returns
    -------
    patch_boxes: [B, N, 4]
        각 patch 의 [x1, y1, x2, y2] (original-normalized, [0,1])
        patch 순서는 row-major (row i, col j) -> index i * grid_w + j
    """
    crop_box = meta.crop_box                                   # [B, 4]
    hflip = meta.hflip                                         # [B]
    device = crop_box.device
    B = crop_box.shape[0]
    gh, gw = meta.patch_grid_height, meta.patch_grid_width

    x0 = crop_box[:, 0].view(B, 1, 1)                          # [B,1,1]
    y0 = crop_box[:, 1].view(B, 1, 1)
    cw = crop_box[:, 2].view(B, 1, 1)
    ch = crop_box[:, 3].view(B, 1, 1)

    # view 내부 normalized coordinate (patch 경계)
    u = torch.arange(gw + 1, device=device, dtype=torch.float32) / gw   # [gw+1]
    v = torch.arange(gh + 1, device=device, dtype=torch.float32) / gh   # [gh+1]
    u1 = u[:-1].view(1, 1, gw)                                  # [1,1,gw]
    u2 = u[1:].view(1, 1, gw)
    v1 = v[:-1].view(1, gh, 1)                                  # [1,gh,1]
    v2 = v[1:].view(1, gh, 1)

    # no-flip:  x_orig = x0 + u * crop_width
    x1_noflip = x0 + u1 * cw
    x2_noflip = x0 + u2 * cw
    # hflip:    view interval [u1,u2] -> original [1-u2, 1-u1]
    x1_flip = x0 + (1.0 - u2) * cw
    x2_flip = x0 + (1.0 - u1) * cw

    flip = hflip.view(B, 1, 1)
    x1 = torch.where(flip, x1_flip, x1_noflip).expand(B, gh, gw)
    x2 = torch.where(flip, x2_flip, x2_noflip).expand(B, gh, gw)
    y1 = (y0 + v1 * ch).expand(B, gh, gw)
    y2 = (y0 + v2 * ch).expand(B, gh, gw)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(B, gh * gw, 4)  # [B, N, 4]
    boxes = boxes.clamp(0.0, 1.0)   # floating point 오차 보정

    if validate:
        assert torch.all(boxes >= -eps) and torch.all(boxes <= 1.0 + eps), \
            "patch box 가 [0,1] 범위를 벗어났습니다."
        assert torch.all(boxes[..., 2] > boxes[..., 0] - eps), "x1 < x2 위반"
        assert torch.all(boxes[..., 3] > boxes[..., 1] - eps), "y1 < y2 위반"
    return boxes


def describe_patch_boxes(boxes: torch.Tensor, name: str = "patch_boxes") -> None:
    """디버그용 요약 출력."""
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    print(
        f"[{name}] shape={tuple(boxes.shape)} "
        f"x∈[{boxes[...,0].min():.4f},{boxes[...,2].max():.4f}] "
        f"y∈[{boxes[...,1].min():.4f},{boxes[...,3].max():.4f}] "
        f"mean_area={area.mean():.6f}"
    )


# %%
# ============================================================================
# Cell 08. ViT-B/14 Backbone
#   patch 14, dim 768, depth 12, heads 12, mlp_ratio 4  (E0~E3 공통, §4)
#   forward 출력은 항상 {"cls": [B,768], "patch": [B,N,768]} (§5)
# ============================================================================
class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention (SDPA 사용)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim({dim}) 가 num_heads({num_heads}) 로 나누어지지 않습니다.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # each [B, heads, L, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)    # [B, heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop_path, ls_init):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.ls1 = LayerScale(dim, ls_init)
        self.dp1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, ls_init)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.dp2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    """[B,3,H,W] -> ([B,N,D], grid_h, grid_w), row-major token ordering."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"ViT patch size 불일치: input {H}x{W} 가 patch_size {self.patch_size} 로 "
                f"나누어지지 않습니다."
            )
        x = self.proj(x)                       # [B, D, gh, gw]
        gh, gw = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)       # [B, gh*gw, D]  (row-major)
        return x, gh, gw


class VisionTransformer(nn.Module):
    """
    ViT-B/14.

    forward(x, mask=None)
    ---------------------
    x   : [B, 3, H, W]
    mask: [B, N] bool  (True = masked -> mask_token 으로 치환, iBOT)

    Returns
    -------
    {"cls": [B, D], "patch": [B, N, D]}
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.base_grid = cfg.pos_embed_grid
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.base_grid ** 2, cfg.embed_dim))

        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList([
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias, dpr[i], cfg.layerscale_init)
            for i in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_init)

    def interpolate_pos_encoding(self, gh: int, gw: int, dtype: torch.dtype) -> torch.Tensor:
        """local crop(7x7) 등 다른 grid 에 맞춰 pos-embed 를 bicubic interpolation."""
        cls_pos = self.pos_embed[:, :1]                                   # [1,1,D]
        patch_pos = self.pos_embed[:, 1:]                                 # [1, s*s, D]
        s = self.base_grid
        if (gh, gw) != (s, s):
            patch_pos = patch_pos.reshape(1, s, s, self.embed_dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, self.embed_dim)
        return torch.cat([cls_pos, patch_pos], dim=1).to(dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        tokens, gh, gw = self.patch_embed(x)                              # [B, N, D]
        if mask is not None:
            if mask.shape != tokens.shape[:2]:
                raise ValueError(f"mask shape {tuple(mask.shape)} != token shape {tuple(tokens.shape[:2])}")
            tokens = torch.where(
                mask.unsqueeze(-1), self.mask_token.to(tokens.dtype).expand_as(tokens), tokens
            )
        cls = self.cls_token.expand(B, -1, -1).to(tokens.dtype)           # [B,1,D]
        x = torch.cat([cls, tokens], dim=1)                               # [B, 1+N, D]
        x = x + self.interpolate_pos_encoding(gh, gw, x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"cls": x[:, 0], "patch": x[:, 1:]}                        # [B,D], [B,N,D]


def build_backbone(cfg: Config) -> VisionTransformer:
    return VisionTransformer(cfg)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
# %%
# ============================================================================
# Cell 09. Student / Teacher Networks
#   Teacher 는 gradient 를 받지 않고 Student 의 EMA 로만 갱신된다 (§8).
#   NOTE: DINOHead / PositionHead 는 Cell 10 에서 정의된다.
#         본 셀은 class 정의만 수행하므로 Cell 10 실행 이후에 instantiate 하면 된다.
# ============================================================================
class StudentTeacherModel(nn.Module):
    """
    Student / Teacher backbone + projection heads 묶음.

    modules
    -------
    student_backbone / teacher_backbone : ViT-B/14
    student_cls_head  / teacher_cls_head   : DINOHead -> [*, K_cls]
    student_patch_head/ teacher_patch_head : DINOHead -> [*, K_patch]
    student_position_head / teacher_position_head : Linear(768 -> K_pos)  (E1/E3 만 생성)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ---- student (trainable) ------------------------------------------
        self.student_backbone = build_backbone(cfg)
        self.student_cls_head = DINOHead(
            cfg.embed_dim, cfg.K_cls, cfg.head_hidden_dim,
            cfg.head_bottleneck_dim, cfg.head_n_layers, cfg.head_norm_last_layer,
        )
        self.student_patch_head = DINOHead(
            cfg.embed_dim, cfg.K_patch, cfg.head_hidden_dim,
            cfg.head_bottleneck_dim, cfg.head_n_layers, cfg.head_norm_last_layer,
        )
        self.student_position_head = PositionHead(cfg.embed_dim, cfg.K_pos) if cfg.use_position else None

        # ---- teacher (EMA, frozen) ----------------------------------------
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_cls_head = copy.deepcopy(self.student_cls_head)
        self.teacher_patch_head = copy.deepcopy(self.student_patch_head)
        self.teacher_position_head = (
            copy.deepcopy(self.student_position_head) if cfg.use_position else None
        )

        self._ema_pairs: List[Tuple[nn.Module, nn.Module]] = [
            (self.student_backbone, self.teacher_backbone),
            (self.student_cls_head, self.teacher_cls_head),
            (self.student_patch_head, self.teacher_patch_head),
        ]
        if cfg.use_position:
            self._ema_pairs.append((self.student_position_head, self.teacher_position_head))

        # teacher 는 gradient 를 받지 않는다 (§8)
        for _, teacher_mod in self._ema_pairs:
            for p in teacher_mod.parameters():
                p.requires_grad = False
            teacher_mod.eval()

    # ---------------- trainable module 접근자 ----------------
    def student_modules(self) -> Dict[str, nn.Module]:
        mods = {
            "student_backbone": self.student_backbone,
            "student_cls_head": self.student_cls_head,
            "student_patch_head": self.student_patch_head,
        }
        if self.student_position_head is not None:
            mods["student_position_head"] = self.student_position_head
        return mods

    def teacher_modules(self) -> Dict[str, nn.Module]:
        mods = {
            "teacher_backbone": self.teacher_backbone,
            "teacher_cls_head": self.teacher_cls_head,
            "teacher_patch_head": self.teacher_patch_head,
        }
        if self.teacher_position_head is not None:
            mods["teacher_position_head"] = self.teacher_position_head
        return mods

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """θ_teacher <- m * θ_teacher + (1-m) * θ_student  (§8)."""
        for student_mod, teacher_mod in self._ema_pairs:
            for ps, pt in zip(student_mod.parameters(), teacher_mod.parameters()):
                pt.mul_(momentum).add_(ps.detach(), alpha=1.0 - momentum)
            for bs, bt in zip(student_mod.buffers(), teacher_mod.buffers()):
                bt.copy_(bs)

    def train(self, mode: bool = True):
        super().train(mode)
        # teacher 는 항상 eval (drop-path/stochastic depth 비활성화)
        for _, teacher_mod in self._ema_pairs:
            teacher_mod.eval()
        return self


# %%
# ============================================================================
# Cell 10. DINO CLS Head / Patch Head / Position Head
# ============================================================================
class WeightNormalizedLinear(nn.Module):
    """
    DINO prototype layer: weight-normalized linear (bias 없음).
    norm_last_layer=True 이면 magnitude(g)를 1로 고정하여 학습 초기 불안정을 막는다.
    """

    def __init__(self, in_dim: int, out_dim: int, norm_last_layer: bool = True):
        super().__init__()
        self.weight_v = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.trunc_normal_(self.weight_v, std=0.02)
        self.weight_g = nn.Parameter(torch.ones(out_dim, 1))
        if norm_last_layer:
            self.weight_g.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, in_dim] -> [*, out_dim]
        w = self.weight_v / (self.weight_v.norm(dim=1, keepdim=True) + 1e-6) * self.weight_g
        return F.linear(x, w)


class DINOHead(nn.Module):
    """
    DINO/iBOT projection head.

    forward
    -------
    x: [*, D]        (마지막 차원만 D 이면 임의 rank 허용: [B,D] 또는 [B,N,D])
    out: [*, K]
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, bottleneck_dim)]
            self.mlp = nn.Sequential(*layers)
        self.last_layer = WeightNormalizedLinear(bottleneck_dim, out_dim, norm_last_layer)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)                                  # [*, bottleneck]
        x = F.normalize(x, dim=-1, p=2)                  # bottleneck L2 normalize
        return self.last_layer(x)                        # [*, K]


class PositionHead(nn.Module):
    """
    Position loss 전용 lightweight projection (§16).
    65K prototype 을 pairwise position matching 에 쓰지 않기 위한 별도 head.

    forward
    -------
    x  : [B, N, 768]
    out: [B, N, K_pos]
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)   # Linear(768 -> 1024, bias=False) 만 사용
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# %%
# ============================================================================
# Cell 11. DINO CLS Loss  (§11)
#   Teacher: 2 global views (stop-gradient)
#   Student: 2 global + 6 local
#   동일 augmentation index 의 (teacher G_i, student G_i) pair 는 제외
# ============================================================================
class DINOLoss(nn.Module):
    """
    Args (forward)
    --------------
    student_logits: List[Tensor[B, K]]   길이 = n_global + n_local (앞 n_global 은 global)
    teacher_logits: List[Tensor[B, K]]   길이 = n_global
    teacher_temp  : float
    student_temp  : float

    Returns
    -------
    scalar loss  (모든 valid pair CE 의 평균)
    """

    def __init__(self, out_dim: int, center_momentum: float = 0.9):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        teacher_temp: float,
        student_temp: float,
    ) -> torch.Tensor:
        # teacher probability (centering + sharpening) -> stop-gradient
        teacher_probs = [
            F.softmax((t.float().detach() - self.center) / teacher_temp, dim=-1)
            for t in teacher_logits
        ]
        n_teacher = len(teacher_probs)
        total, n_pairs = 0.0, 0
        for ti, q in enumerate(teacher_probs):
            for si, s in enumerate(student_logits):
                if si == ti and si < n_teacher:
                    continue  # 동일 augmentation view pair 제외
                log_p = F.log_softmax(s.float() / student_temp, dim=-1)
                total = total + (-(q * log_p).sum(dim=-1)).mean()
                n_pairs += 1
        if n_pairs == 0:
            return teacher_logits[0].new_zeros(())
        return total / n_pairs

    @torch.no_grad()
    def update_center(self, teacher_logits: List[torch.Tensor]) -> None:
        """teacher CLS center EMA 업데이트."""
        cat = torch.cat([t.float() for t in teacher_logits], dim=0)    # [n*B, K]
        batch_center = cat.mean(dim=0, keepdim=True)                   # [1, K]
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


# %%
# ============================================================================
# Cell 12. iBOT Patch Mask Generator & Patch Loss (§12)
# ============================================================================
class PatchMaskGenerator:
    """
    Random patch masking.

    __call__
    --------
    batch_size, num_tokens -> Tensor[B, N] (bool, True = masked)
    """

    def __init__(self, cfg: Config):
        self.ratio_min = cfg.mask_ratio_min
        self.ratio_max = cfg.mask_ratio_max
        self.sample_prob = cfg.mask_sample_prob

    def __call__(self, batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
        ratios = torch.empty(batch_size, device=device).uniform_(self.ratio_min, self.ratio_max)
        apply = torch.rand(batch_size, device=device) < self.sample_prob
        n_mask = (ratios * num_tokens).long().clamp(min=1, max=num_tokens - 1)
        n_mask = torch.where(apply, n_mask, torch.zeros_like(n_mask))       # [B]
        noise = torch.rand(batch_size, num_tokens, device=device)
        rank = noise.argsort(dim=1).argsort(dim=1)                          # [B, N]
        mask = rank < n_mask.unsqueeze(1)                                   # [B, N] bool
        return mask


class IBOTPatchLoss(nn.Module):
    """
    Masked image modeling loss.

    forward
    -------
    student_logits_per_crop: List[Tensor[M_c, K]]   masked token 위치의 student logits
    teacher_logits_per_crop: List[Tensor[M_c, K]]   같은 위치의 teacher(unmasked input) logits
    두 리스트는 crop 순서(G1, G2)로 정렬되어 있고 M_c 는 crop 별 masked token 수.

    Returns
    -------
    scalar loss = mean_c( mean_masked_token CE )
    """

    def __init__(self, out_dim: int, center_momentum: float = 0.9):
        super().__init__()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_logits_per_crop: List[torch.Tensor],
        teacher_logits_per_crop: List[torch.Tensor],
        teacher_temp: float,
        student_temp: float,
        device: torch.device,
    ) -> torch.Tensor:
        losses, n = [], 0
        for s_logits, t_logits in zip(student_logits_per_crop, teacher_logits_per_crop):
            if s_logits.numel() == 0 or s_logits.shape[0] == 0:
                continue  # 해당 crop 에서 masked token 이 없으면 skip
            q = F.softmax((t_logits.float().detach() - self.center) / teacher_temp, dim=-1)
            log_p = F.log_softmax(s_logits.float() / student_temp, dim=-1)
            losses.append((-(q * log_p).sum(dim=-1)).mean())
            n += 1
        if n == 0:
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_center(self, teacher_logits_per_crop: List[torch.Tensor]) -> None:
        valid = [t.float() for t in teacher_logits_per_crop if t.shape[0] > 0]
        if not valid:
            return
        cat = torch.cat(valid, dim=0)                                   # [M_total, K]
        batch_center = cat.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


# %%
# ============================================================================
# Cell 13. KoLeo Loss (§13)
# ============================================================================
class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko differential entropy regularizer.

    forward
    -------
    z: [B, D]   (student global CLS embedding)

    Returns
    -------
    scalar  L_KoLeo = -mean(log(d_i + eps)),  d_i = min_{j!=i} ||z_i - z_j||_2
    B == 1 인 경우 device-safe zero 를 반환한다.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        if B < 2:
            return z.new_zeros(())
        z = F.normalize(z.float(), dim=-1, p=2, eps=1e-8)               # [B, D]
        dist = torch.cdist(z, z, p=2)                                    # [B, B]
        dist = dist + torch.eye(B, device=z.device, dtype=z.dtype) * 1e9  # self 제외
        min_dist = dist.min(dim=1).values                                # [B]
        return -torch.log(min_dist + self.eps).mean()


# %%
# ============================================================================
# Cell 14. Coordinate-aware Position Correspondence (§15)
#   crop scale 이 다르므로 IoU 대신 Intersection-over-Smaller(IoS) 를 사용한다.
# ============================================================================
def compute_ios_matrix(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args
    ----
    boxes_a: [B, Na, 4]  (x1,y1,x2,y2, original-normalized)
    boxes_b: [B, Nb, 4]

    Returns
    -------
    ios: [B, Na, Nb]  = intersection / min(area_a, area_b)
    """
    a = boxes_a.unsqueeze(2)                    # [B, Na, 1, 4]
    b = boxes_b.unsqueeze(1)                    # [B, 1, Nb, 4]
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)          # [B, Na, Nb]
    area_a = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])).clamp(min=0)
    area_b = ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])).clamp(min=0)
    smaller = torch.minimum(area_a, area_b)
    return inter / (smaller + eps)


def find_correspondences(
    ios: torch.Tensor,
    threshold: float,
    topk: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    각 teacher patch 에 대해 IoS 가 가장 높은 student patch top-k 를 선택한다.

    Args
    ----
    ios: [B, Nt, Ns]

    Returns
    -------
    batch_idx  : [P]  matched pair 의 batch index
    teacher_idx: [P]
    student_idx: [P]
    weights    : [P]  = IoS (spatial weight w_ij)
    valid match 가 없는 teacher patch 는 제외된다.
    """
    B, Nt, Ns = ios.shape
    k = min(topk, Ns)
    vals, idx = ios.topk(k=k, dim=-1)                                # [B, Nt, k]
    valid = vals >= threshold                                        # [B, Nt, k]
    if not bool(valid.any()):
        empty_l = torch.empty(0, dtype=torch.long, device=ios.device)
        empty_f = torch.empty(0, dtype=ios.dtype, device=ios.device)
        return empty_l, empty_l, empty_l, empty_f
    b_idx, t_idx, k_idx = valid.nonzero(as_tuple=True)               # each [P]
    s_idx = idx[b_idx, t_idx, k_idx]
    w = vals[b_idx, t_idx, k_idx]
    return b_idx, t_idx, s_idx, w


# %%
# ============================================================================
# Cell 15. Position Consistency Loss  (L_position, §14 / §17)
#   dense [B, Nt, Ns, K_pos] tensor 를 절대 만들지 않는다.
#   IoS 로 top-k correspondence index 를 먼저 얻고 matched logits 만 gather 한다.
# ============================================================================
class PositionConsistencyLoss(nn.Module):
    """
    Proposed coordinate-aware spatial consistency.
    (AFiRe 에서 제기된 fine-grained anatomical consistency 문제의식에 동기를 받은 objective이며
     AFiRe 원 논문의 직접 재현이 아니다.)

    forward
    -------
    teacher_logits: List[Tensor[B, Nt, K_pos]]   (global views only, detached)
    teacher_boxes : List[Tensor[B, Nt, 4]]
    student_logits: List[Tensor[B, Ns, K_pos]]   (index 0..n_global-1 = global, 이후 local)
    student_boxes : List[Tensor[B, Ns, 4]]

    Returns
    -------
    (loss, stats): scalar, {"n_pairs": int, "mean_ios": float}
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.threshold = cfg.pos_overlap_threshold
        self.topk = cfg.position_topk
        self.teacher_temp = cfg.position_teacher_temp
        self.student_temp = cfg.position_student_temp
        self.chunk = cfg.position_pair_chunk_size
        self.center_momentum = cfg.position_center_momentum
        self.register_buffer("center", torch.zeros(1, cfg.K_pos))     # [1, K_pos]

    def forward(
        self,
        teacher_logits: List[torch.Tensor],
        teacher_boxes: List[torch.Tensor],
        student_logits: List[torch.Tensor],
        student_boxes: List[torch.Tensor],
        n_global_student: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        weighted_sum = torch.zeros((), device=device, dtype=torch.float32)
        weight_sum = torch.zeros((), device=device, dtype=torch.float32)
        n_pairs_total = 0
    
        for ti, (t_logit, t_box) in enumerate(zip(teacher_logits, teacher_boxes)):
            B, Nt, K = t_logit.shape
            t_flat = t_logit.reshape(B * Nt, K)
    
            for si, (s_logit, s_box) in enumerate(zip(student_logits, student_boxes)):
                if si == ti and si < n_global_student:
                    continue
    
                Ns = s_logit.shape[1]
    
                # --------------------------------------------------------------
                # Positive correspondence
                # teacher patch -> 가장 잘 대응되는 student patch
                # positive 조건: IoS >= pos_overlap_threshold (= 0.5)
                # --------------------------------------------------------------
                with torch.no_grad():
                    ios = compute_ios_matrix(t_box.float(), s_box.float())  # [B,Nt,Ns]
                    b_idx, t_idx, s_idx, w = find_correspondences(
                        ios, self.threshold, self.topk
                    )
    
                if b_idx.numel() == 0:
                    continue
    
                s_flat = s_logit.reshape(B * Ns, s_logit.shape[-1])
                t_gather = b_idx * Nt + t_idx
                s_gather = b_idx * Ns + s_idx
                P = b_idx.numel()
    
                # ==============================================================
                # InfoNCE
                # ==============================================================
                if self.cfg.position_loss_type == "infonce":
                    # teacher: stop-gradient / student: gradient 유지
                    t_feat = F.normalize(t_flat.float().detach(), dim=-1)
                    s_feat = F.normalize(s_flat.float(), dim=-1)
    
                    n_within = self.cfg.position_n_within_negatives
                    n_cross = self.cfg.position_n_cross_negatives
                    neg_thr = self.cfg.position_negative_ios_threshold
                    temp = self.cfg.position_infonce_temp
    
                    # ----------------------------------------------------------
                    # Strict cross-image negative bank
                    #
                    # cross_bank[b]: image b의 anchor가 사용할 negative bank.
                    # 반드시 batch index != b 인 student patch만 포함한다.
                    #
                    # shape: [B, n_cross, K_pos]
                    # ----------------------------------------------------------
                    cross_bank = None
                    if n_cross > 0 and B > 1:
                        with torch.no_grad():
                            anchor_b = torch.arange(B, device=device)[:, None]       # [B,1]
                            rand_b = torch.randint(0, B - 1, (B, n_cross), device=device)
                            rand_b = rand_b + (rand_b >= anchor_b).long()            # != anchor_b
                            rand_s = torch.randint(0, Ns, (B, n_cross), device=device)
                            cross_idx = rand_b * Ns + rand_s                         # [B,n_cross]
    
                        # gather는 no_grad 밖 -> student negative에도 gradient 전달
                        cross_bank = s_feat.index_select(
                            0, cross_idx.reshape(-1)
                        ).reshape(B, n_cross, -1)
    
                    # ----------------------------------------------------------
                    # Pair chunk
                    # ----------------------------------------------------------
                    for start in range(0, P, self.chunk):
                        end = min(start + self.chunk, P)
                        C = end - start
    
                        b_c = b_idx[start:end]
                        t_c = t_idx[start:end]
                        s_pos_c = s_idx[start:end]
    
                        anchor = t_feat.index_select(0, t_gather[start:end])  # [C,K_pos]
                        pos = s_feat.index_select(0, s_gather[start:end])    # [C,K_pos]
                        pos_sim = (anchor * pos).sum(-1, keepdim=True)       # [C,1]
    
                        parts = [pos_sim]
                        within_valid = torch.zeros(C, dtype=torch.bool, device=device)
    
                        # ------------------------------------------------------
                        # 1) False-negative-safe within-image negatives
                        #    같은 image + 같은 student view
                        #    IoS(anchor, candidate) <= 0.05 만 허용
                        # ------------------------------------------------------
                        if n_within > 0:
                            with torch.no_grad():
                                ios_c = ios[b_c, t_c, :]                      # [C,Ns]
                                neg_mask = ios_c <= neg_thr                   # [C,Ns]
    
                                # positive patch는 명시적으로 제외
                                neg_mask.scatter_(1, s_pos_c[:, None], False)
    
                                k_neg = min(n_within, Ns)
                                rand = torch.rand((C, Ns), device=device)
                                rand.masked_fill_(~neg_mask, 2.0)
    
                                neg_idx = rand.topk(
                                    k=k_neg, dim=1, largest=False
                                ).indices                                     # [C,k_neg]
    
                                neg_selected_valid = neg_mask.gather(1, neg_idx)
                                within_valid = neg_selected_valid.any(dim=1)
    
                            flat_neg_idx = (
                                b_c[:, None] * Ns + neg_idx
                            ).reshape(-1)
    
                            wneg = s_feat.index_select(
                                0, flat_neg_idx
                            ).reshape(C, k_neg, -1)                           # [C,k_neg,K]
    
                            within_sim = torch.bmm(
                                wneg, anchor.unsqueeze(-1)
                            ).squeeze(-1)                                     # [C,k_neg]
    
                            # 후보가 부족해 채워진 invalid slot은 denominator에서 제거
                            within_sim = within_sim.masked_fill(
                                ~neg_selected_valid, float("-inf")
                            )
                            parts.append(within_sim)
                        # ------------------------------------------------------
                        # 2) Strict cross-image negatives
                        #    각 anchor image b에 대해 cross_bank[b] 사용
                        #    vectorized batched matmul
                        # ------------------------------------------------------
                        if cross_bank is not None:
                            bank_c = cross_bank.index_select(0, b_c)          # [C,n_cross,K]
                            cross_sim = torch.bmm(
                                bank_c, anchor.unsqueeze(-1)
                            ).squeeze(-1)                                     # [C,n_cross]
                        
                            parts.append(cross_sim)
                        
                            # cross negative가 존재하므로 모든 anchor가 valid
                            anchor_valid = torch.ones(C, dtype=torch.bool, device=device)
                        else:
                            # cross negative가 없으면 safe within negative가 있어야 valid
                            anchor_valid = within_valid
    
                        # ------------------------------------------------------
                        # InfoNCE
                        # [positive | safe-within | strict-cross]
                        # ------------------------------------------------------
                        logits = torch.cat(parts, dim=1) / temp
                        ce = -F.log_softmax(logits, dim=1)[:, 0]
    
                        wc = w[start:end].float() * anchor_valid.float()
    
                        weighted_sum = weighted_sum + (wc * ce).sum()
                        weight_sum = weight_sum + wc.sum()
                        n_pairs_total += int(anchor_valid.sum().item())
    
                # ==============================================================
                # 기존 CE position loss
                # ==============================================================
                else:
                    n_pairs_total += P
    
                    for start in range(0, P, self.chunk):
                        end = min(start + self.chunk, P)
    
                        tl = t_flat.index_select(0, t_gather[start:end]).float()
                        sl = s_flat.index_select(0, s_gather[start:end]).float()
    
                        q = F.softmax(
                            (tl.detach() - self.center) / self.teacher_temp, dim=-1
                        )
                        log_p = F.log_softmax(sl / self.student_temp, dim=-1)
                        ce = -(q * log_p).sum(dim=-1)
    
                        wc = w[start:end].float()
                        weighted_sum = weighted_sum + (wc * ce).sum()
                        weight_sum = weight_sum + wc.sum()
    
        if n_pairs_total == 0 or float(weight_sum) <= 0.0:
            return torch.zeros((), device=device), {
                "n_pairs": 0,
                "mean_ios": 0.0,
            }
    
        loss = weighted_sum / (weight_sum + 1e-8)
    
        stats = {
            "n_pairs": int(n_pairs_total),
            "mean_ios": float(weight_sum / max(n_pairs_total, 1)),
        }
        return loss, stats

    @torch.no_grad()
    def update_center(self, teacher_logits: List[torch.Tensor]) -> None:
        """teacher position center EMA ([1, K_pos])."""
        if not teacher_logits:
            return
        acc = torch.stack([t.float().reshape(-1, t.shape[-1]).mean(dim=0) for t in teacher_logits], 0)
        batch_center = acc.mean(dim=0, keepdim=True)                     # [1, K_pos]
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
# %%
# ============================================================================
# Cell 16. Part Grid Extraction (§18)
#   16x16 patch grid -> 4x4 spatial regions (region 당 4x4 token) -> average pooling
#   part 순서는 항상 row-major (top-left -> top-right -> ... -> bottom-right)
# ============================================================================
def extract_part_features(
    patch_tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    part_grid_size: int,
    l2_normalize: bool = True,
) -> torch.Tensor:
    """
    Args
    ----
    patch_tokens: [B, N, D]        (N == grid_h * grid_w, row-major)
    grid_h, grid_w: patch grid 크기 (global crop: 16, 16)
    part_grid_size: spatial decomposition 격자 (4 -> K_part = 16)

    Returns
    -------
    part_features: [B, K_part, D]  (row-major part ordering, 선택적 L2 normalize)
    """
    B, N, D = patch_tokens.shape
    if N != grid_h * grid_w:
        raise ValueError(f"token 수 불일치: N={N}, grid={grid_h}x{grid_w}")
    if grid_h % part_grid_size != 0 or grid_w % part_grid_size != 0:
        raise ValueError(
            f"part_grid_size({part_grid_size}) 가 patch grid({grid_h}x{grid_w}) 를 나누지 못합니다."
        )
    ph, pw = grid_h // part_grid_size, grid_w // part_grid_size
    x = patch_tokens.reshape(B, grid_h, grid_w, D).permute(0, 3, 1, 2)      # [B, D, gh, gw]
    x = F.avg_pool2d(x, kernel_size=(ph, pw), stride=(ph, pw))              # [B, D, pg, pg]
    parts = x.flatten(2).transpose(1, 2).contiguous()                       # [B, K_part, D] row-major
    if l2_normalize:
        parts = F.normalize(parts.float(), dim=-1, p=2, eps=1e-8)
    return parts


# %%
# ============================================================================
# Cell 17. Composition Predictor & Loss  (L_composition, Part -> Whole, §20)
#   정확히 하나의 linear layer 만 사용한다 (attention / multi-layer MLP 금지, §22).
# ============================================================================
class CompositionPredictor(nn.Module):
    """
    Args (forward)
    --------------
    parts: [B, K_part, D]

    Returns
    -------
    w_hat: [B, D]     (part 들을 flatten 하여 whole representation 을 예측)
    """

    def __init__(self, k_part: int, dim: int):
        super().__init__()
        self.k_part = k_part
        self.dim = dim
        self.proj = nn.Linear(k_part * dim, dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, parts: torch.Tensor) -> torch.Tensor:
        B, K, D = parts.shape
        if K != self.k_part or D != self.dim:
            raise ValueError(f"expected [B,{self.k_part},{self.dim}], got {tuple(parts.shape)}")
        return self.proj(parts.reshape(B, K * D))                          # [B, D]


def composition_loss(w_hat_student: torch.Tensor, w_teacher: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    w_hat_student: [B, D]   composition predictor 출력 (gradient 흐름)
    w_teacher    : [B, D]   teacher CLS (stop-gradient)

    Returns
    -------
    scalar  mean(1 - cos(normalize(w_hat), normalize(w_t)))
    """
    a = F.normalize(w_hat_student.float(), dim=-1, p=2, eps=1e-8)
    b = F.normalize(w_teacher.float().detach(), dim=-1, p=2, eps=1e-8)
    return (1.0 - (a * b).sum(dim=-1)).mean()


# %%
# ============================================================================
# Cell 18. Decomposition Predictor & Loss (L_decomposition, Whole -> Parts, §21)
# ============================================================================
class DecompositionPredictor(nn.Module):
    """
    Args (forward)
    --------------
    whole: [B, D]

    Returns
    -------
    parts_hat: [B, K_part, D]  (각 part vector 는 L2 normalize)
    """

    def __init__(self, k_part: int, dim: int):
        super().__init__()
        self.k_part = k_part
        self.dim = dim
        self.proj = nn.Linear(dim, k_part * dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, whole: torch.Tensor) -> torch.Tensor:
        B, D = whole.shape
        if D != self.dim:
            raise ValueError(f"expected [B,{self.dim}], got {tuple(whole.shape)}")
        out = self.proj(whole).reshape(B, self.k_part, self.dim)           # [B, K_part, D]
        return F.normalize(out.float(), dim=-1, p=2, eps=1e-8)


def decomposition_loss(parts_hat: torch.Tensor, parts_teacher: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    parts_hat    : [B, K_part, D]  (L2 normalized, gradient 흐름)
    parts_teacher: [B, K_part, D]  (L2 normalized, stop-gradient)

    같은 row-major part index 끼리만 비교한다.

    Returns
    -------
    scalar  mean_{b,k}(1 - cos)
    """
    a = F.normalize(parts_hat.float(), dim=-1, p=2, eps=1e-8)
    b = F.normalize(parts_teacher.float().detach(), dim=-1, p=2, eps=1e-8)
    return (1.0 - (a * b).sum(dim=-1)).mean()


# %%
# ============================================================================
# Cell 18b. E2c — Independent Local-Crop Part-Whole
#   Part  = 독립 인코딩된 local crop (global self-attention context 없음)
#   Whole = global crop
#   Part <-> Whole 관계는 원본 좌표계의 crop_box 로부터 계산한 relative geometry 로 준다.
# ============================================================================
PARTWHOLE_GEOM_DIM = 8


def local_in_global_geometry(
    global_metas: Sequence[BatchedViewMeta],
    local_metas: Sequence[BatchedViewMeta],
    containment_threshold: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    local crop 이 global crop 안에서 차지하는 위치/크기를, *global view 자신의 좌표계* 에서
    계산한다. global view 는 224x224 로 resize 되므로 원본 aspect ratio 보정은 필요 없다.

    Args
    ----
    global_metas: G x BatchedViewMeta  (crop_box [B,4] = [x0,y0,w,h] original-normalized)
    local_metas : L x BatchedViewMeta

    Returns
    -------
    geom : [B, G, L, 8]  finite. (cx, cy, w, h, log w, log h, rel_hflip, containment)
           cx/cy/rel_hflip/containment 는 [-1,1] 로 scaling.
    valid: [B, G, L] bool   containment >= threshold
    cont : [B, G, L] float  containment (= intersection / local area), 로깅용
    """
    device = global_metas[0].crop_box.device
    gb = torch.stack([m.crop_box for m in global_metas], dim=1).float()      # [B,G,4]
    lb = torch.stack([m.crop_box for m in local_metas], dim=1).float()       # [B,L,4]
    gf = torch.stack([m.hflip for m in global_metas], dim=1).bool()          # [B,G]
    lf = torch.stack([m.hflip for m in local_metas], dim=1).bool()           # [B,L]

    X0 = gb[..., 0].unsqueeze(2)                                            # [B,G,1]
    Y0 = gb[..., 1].unsqueeze(2)
    W = gb[..., 2].unsqueeze(2).clamp(min=eps)
    H = gb[..., 3].unsqueeze(2).clamp(min=eps)

    x0 = lb[..., 0].unsqueeze(1)                                            # [B,1,L]
    y0 = lb[..., 1].unsqueeze(1)
    w = lb[..., 2].unsqueeze(1).clamp(min=eps)
    h = lb[..., 3].unsqueeze(1).clamp(min=eps)

    # --- 원본 좌표계 교집합 ------------------------------------------------
    ix1 = torch.maximum(X0, x0)
    iy1 = torch.maximum(Y0, y0)
    ix2 = torch.minimum(X0 + W, x0 + w)
    iy2 = torch.minimum(Y0 + H, y0 + h)
    iw = (ix2 - ix1).clamp(min=0.0)
    ih = (iy2 - iy1).clamp(min=0.0)
    cont = (iw * ih) / (w * h)                                              # [B,G,L]
    cont = cont.clamp(0.0, 1.0)
    valid = cont >= containment_threshold

    # --- global view 좌표계로 사상 (교집합 box 를 사용해 항상 [0,1] 유지) ---
    a1 = ((ix1 - X0) / W).clamp(0.0, 1.0)
    a2 = ((ix2 - X0) / W).clamp(0.0, 1.0)
    a2 = torch.maximum(a2, a1)
    flip_g = gf.unsqueeze(2)                                                # [B,G,1]
    u1 = torch.where(flip_g, 1.0 - a2, a1)
    u2 = torch.where(flip_g, 1.0 - a1, a2)
    v1 = ((iy1 - Y0) / H).clamp(0.0, 1.0)
    v2 = ((iy2 - Y0) / H).clamp(0.0, 1.0)
    v2 = torch.maximum(v2, v1)

    w_rel = (u2 - u1).clamp(min=0.0)
    h_rel = (v2 - v1).clamp(min=0.0)
    rel_flip = torch.where(
        lf.unsqueeze(1) ^ flip_g,
        torch.ones_like(cont),
        -torch.ones_like(cont),
    )                                                                       # [B,G,L]

    geom = torch.stack(
        [
            2.0 * (0.5 * (u1 + u2)) - 1.0,
            2.0 * (0.5 * (v1 + v2)) - 1.0,
            w_rel,
            h_rel,
            torch.log(w_rel.clamp(min=1e-3)),
            torch.log(h_rel.clamp(min=1e-3)),
            rel_flip,
            2.0 * cont - 1.0,
        ],
        dim=-1,
    )                                                                       # [B,G,L,8]
    return geom, valid, cont


def pool_local_part(out: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    """
    local crop backbone 출력 -> Part feature (L2 normalized).

    mode
    ----
    "patch_mean": patch token 평균. E2 의 4x4 average pooling 과 동일한 연산자.
    "cls"       : CLS token. DINO loss 가 teacher global CLS 로 끌어당기는 대상이므로
                  representation 수준에서 이미 global 로 밀려 있다 (덜 clean 한 ablation).
    """
    if mode == "patch_mean":
        f = out["patch"].mean(dim=1)
    elif mode == "cls":
        f = out["cls"]
    else:
        raise ValueError(f"unknown local part feature mode: {mode!r}")
    return F.normalize(f.float(), dim=-1, p=2, eps=1e-8)


class LocalPartComposer(nn.Module):
    """
    Composition (Part -> Whole), E2c.

      token_{b,l}   = W_c [ p^s_{b,l} ; g_{b,l} ]            (linear 1개, §22)
      w_hat_{b}     = sum_l m_{b,l} token_{b,l} / sum_l m_{b,l}

    valid local 개수가 sample 마다 다르므로 E2 의 고정 K flatten Linear 대신
    masked mean (permutation invariant) 을 사용한다. 순서 정보는 geometry 가 대신한다.
    """

    def __init__(self, dim: int, geom_dim: int = PARTWHOLE_GEOM_DIM):
        super().__init__()
        self.dim = dim
        self.geom_dim = geom_dim
        self.proj = nn.Linear(dim + geom_dim, dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(
        self, parts: torch.Tensor, geom: torch.Tensor, valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        parts: [B, L, D]   (student, gradient O)
        geom : [B, L, Fg]
        valid: [B, L] bool

        Returns
        -------
        w_hat    : [B, D]
        has_valid: [B] bool   (valid local 이 1개 이상인 sample)
        """
        B, L, D = parts.shape
        if D != self.dim or geom.shape[-1] != self.geom_dim:
            raise ValueError(
                f"expected parts [B,L,{self.dim}] / geom [B,L,{self.geom_dim}], "
                f"got {tuple(parts.shape)} / {tuple(geom.shape)}"
            )
        m = valid.to(parts.dtype).unsqueeze(-1)                             # [B,L,1]
        tok = self.proj(torch.cat([parts, geom.to(parts.dtype)], dim=-1))   # [B,L,D]
        num = (tok * m).sum(dim=1)                                          # [B,D]
        den = m.sum(dim=1).clamp(min=1.0)                                   # [B,1]
        return num / den, valid.any(dim=1)


class GeometryDecomposer(nn.Module):
    """
    Decomposition (Whole -> Part), E2c.

      p_hat_{b,l} = normalize( W_d [ w^s_b ; g_{b,l} ] )       (linear 1개)
    """

    def __init__(self, dim: int, geom_dim: int = PARTWHOLE_GEOM_DIM):
        super().__init__()
        self.dim = dim
        self.geom_dim = geom_dim
        self.proj = nn.Linear(dim + geom_dim, dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, whole: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """
        whole: [B, D]
        geom : [B, L, Fg]
        Returns parts_hat: [B, L, D] (L2 normalized)
        """
        B, L, _ = geom.shape
        w = whole.unsqueeze(1).expand(B, L, self.dim)
        out = self.proj(torch.cat([w, geom.to(whole.dtype)], dim=-1))       # [B,L,D]
        return F.normalize(out.float(), dim=-1, p=2, eps=1e-8)


def masked_cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mean_{masked} (1 - cos(pred, target)).  target 은 stop-gradient.

    Returns (loss_sum, n_valid) — caller 가 여러 global view 를 합산한 뒤 나눈다.
    zero-valid 일 때 loss_sum = 0, n_valid = 0 이므로 NaN 이 생기지 않는다.
    """
    a = F.normalize(pred.float(), dim=-1, p=2, eps=1e-8)
    b = F.normalize(target.float().detach(), dim=-1, p=2, eps=1e-8)
    d = 1.0 - (a * b).sum(dim=-1)                                           # [...]
    m = mask.to(d.dtype)
    return (d * m).sum(), m.sum()


# %%
# ============================================================================
# Cell 18c. E2d — Local Preservation via Patch Gram Anchoring
#   L_gram = mean_{b} mean_{i,j} ( <x^s_i, x^s_j> - <x^G_i, x^G_j> )^2
#   x^s : student clean global patch token (L2 normalized, gradient O)
#   x^G : frozen Gram teacher patch token  (L2 normalized, gradient X)
#   feature 의 회전/재배치는 허용하고 patch 간 상대 유사도 구조만 고정한다.
#   (DINOv3 의 Gram anchoring 과 같은 형태. 여기서는 Gram teacher = 학습 완료된 E0)
# ============================================================================
GRAM_COMPAT_FIELDS = (
    "patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio",
    "pos_embed_grid", "in_chans", "qkv_bias", "normalize_mean", "normalize_std",
)


def load_gram_teacher(cfg: Config) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    frozen Gram teacher backbone 로딩. 학습/EMA/optimizer/checkpoint 어디에도 포함되지 않는다.

    Returns
    -------
    (backbone [eval, requires_grad=False], info dict)
    """
    path = cfg.gram_teacher_ckpt
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Gram teacher checkpoint 없음: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ck_cfg = config_from_dict(ckpt["config"])
    for f in GRAM_COMPAT_FIELDS:
        a, b = getattr(cfg, f), getattr(ck_cfg, f)
        na = tuple(a) if isinstance(a, (list, tuple)) else a
        nb = tuple(b) if isinstance(b, (list, tuple)) else b
        if na != nb:
            raise RuntimeError(f"Gram teacher 비호환: {f} cfg={a!r} ckpt={b!r} ({path})")
    if cfg.gram_teacher_key not in ckpt:
        raise KeyError(f"checkpoint 에 {cfg.gram_teacher_key!r} 가 없습니다 ({path}).")
    backbone = build_backbone(ck_cfg)
    backbone.load_state_dict(ckpt[cfg.gram_teacher_key], strict=True)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    info = {
        "path": path,
        "experiment": ck_cfg.experiment,
        "run_tag": ck_cfg.run_tag,
        "epoch": int(ckpt.get("epoch", -1)),
        "key": cfg.gram_teacher_key,
    }
    del ckpt
    return backbone, info


def gram_anchor_loss(s_patch: torch.Tensor, g_patch: torch.Tensor) -> torch.Tensor:
    """
    s_patch: [B, N, D]  student clean global patch token (gradient O)
    g_patch: [B, N, D]  Gram teacher patch token (stop-gradient)

    Returns scalar  mean_{b,i,j} (G^s - G^G)^2
    bf16 autocast 에서는 cosine 차이가 수치 오차에 묻히므로 fp32 로 강제 계산한다.
    """
    with torch.autocast(device_type="cuda", enabled=False):
        xs = F.normalize(s_patch.float(), dim=-1, p=2, eps=1e-8)
        xg = F.normalize(g_patch.float().detach(), dim=-1, p=2, eps=1e-8)
        gs = torch.bmm(xs, xs.transpose(1, 2))                  # [B, N, N]
        gg = torch.bmm(xg, xg.transpose(1, 2))                  # [B, N, N]
        return (gs - gg).pow(2).mean()


# %%
# ============================================================================
# Cell 19. Integrated SSL Loss for E0/E1/E2/E3 (§9, §10, §23)
#   experiment flag 에 따라 필요 없는 auxiliary forward 는 수행하지 않는다.
# ============================================================================
LOSS_KEYS = [
    "L_DINO", "L_iBOT", "L_KoLeo", "L_base",
    "L_position", "L_composition", "L_decomposition", "L_gram", "L_total",
]


class SSLTrainingModule(nn.Module):
    """
    Student/Teacher + 모든 loss module + part-whole predictor 를 묶은 학습 단위.

    forward_loss(batch, teacher_temp, device) -> (total_loss, logs: Dict[str, float])
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.model = StudentTeacherModel(cfg)
        self.dino_loss = DINOLoss(cfg.K_cls, cfg.center_momentum)
        self.ibot_loss = IBOTPatchLoss(cfg.K_patch, cfg.patch_center_momentum)
        self.koleo_loss = KoLeoLoss(cfg.koleo_eps)
        self.position_loss = PositionConsistencyLoss(cfg) if cfg.use_position else None
        if not cfg.use_partwhole:
            self.composition_predictor = None
            self.decomposition_predictor = None
        elif cfg.partwhole_part_source == "local_crop":          # E2c
            self.composition_predictor = LocalPartComposer(cfg.embed_dim, cfg.partwhole_geom_dim)
            self.decomposition_predictor = GeometryDecomposer(cfg.embed_dim, cfg.partwhole_geom_dim)
        else:                                                    # E2 / E2b
            self.composition_predictor = CompositionPredictor(cfg.k_part, cfg.embed_dim)
            self.decomposition_predictor = DecompositionPredictor(cfg.k_part, cfg.embed_dim)
        self.mask_generator = PatchMaskGenerator(cfg)

        # E2d: frozen Gram teacher (optimizer / EMA / checkpoint 대상 아님)
        self.gram_teacher = None
        self.gram_teacher_info: Dict[str, Any] = {}
        if cfg.local_preservation == "gram":
            self.gram_teacher, self.gram_teacher_info = load_gram_teacher(cfg)
            print(f"[gram] teacher loaded: {self.gram_teacher_info}")

    def train(self, mode: bool = True):
        super().train(mode)
        # Gram teacher 는 항상 eval (drop-path 비활성화)
        if self.gram_teacher is not None:
            self.gram_teacher.eval()
        return self

    # ---------------- optimizer 대상 모듈 (teacher 제외, §24) ----------------
    def trainable_modules(self) -> Dict[str, nn.Module]:
        mods = dict(self.model.student_modules())
        if self.composition_predictor is not None:
            mods["composition_predictor"] = self.composition_predictor
        if self.decomposition_predictor is not None:
            mods["decomposition_predictor"] = self.decomposition_predictor
        return mods

    # ---------------- 학습 1-step forward ----------------
    def forward_loss(
        self,
        batch: Dict[str, Any],
        teacher_temp: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        g_imgs: List[torch.Tensor] = batch["global_images"]        # each [B,3,224,224]
        l_imgs: List[torch.Tensor] = batch["local_images"]         # each [B,3,98,98]
        g_metas: List[BatchedViewMeta] = batch["global_metas"]
        l_metas: List[BatchedViewMeta] = batch["local_metas"]
        n_g, n_l = len(g_imgs), len(l_imgs)
        B = g_imgs[0].shape[0]
        N_g = cfg.n_global_tokens
        gh = gw = cfg.global_grid

        g_cat = torch.cat(g_imgs, dim=0)                            # [n_g*B, 3, 224, 224]
        l_cat = torch.cat(l_imgs, dim=0)                            # [n_l*B, 3, 98, 98]
        masks = self.mask_generator(n_g * B, N_g, device)           # [n_g*B, 256] bool
        masks_per_crop = list(masks.chunk(n_g, dim=0))              # n_g x [B, 256]

        # ---------------- Teacher forward: 2 global, unmasked, no grad (§9) ----
        with torch.no_grad():
            t_out = self.model.teacher_backbone(g_cat)
            t_cls_cat = t_out["cls"]                                # [n_g*B, D]
            t_patch_cat = t_out["patch"]                            # [n_g*B, 256, D]
            t_cls_logits = list(self.model.teacher_cls_head(t_cls_cat).chunk(n_g, 0))
            t_cls_per_crop = list(t_cls_cat.chunk(n_g, 0))          # n_g x [B, D]
            t_patch_per_crop = list(t_patch_cat.chunk(n_g, 0))      # n_g x [B, 256, D]

            t_patch_logits_masked: List[torch.Tensor] = []
            for c in range(n_g):
                m = masks_per_crop[c]                               # [B, 256]
                if cfg.ibot_compute_on_masked_only:
                    sel = t_patch_per_crop[c][m]                    # [M, D]
                    t_patch_logits_masked.append(self.model.teacher_patch_head(sel))   # [M, K]
                else:
                    full = self.model.teacher_patch_head(t_patch_per_crop[c])          # [B,256,K]
                    t_patch_logits_masked.append(full[m])
            t_pos_logits = (
                [self.model.teacher_position_head(p) for p in t_patch_per_crop]        # [B,256,K_pos]
                if cfg.use_position else None
            )
            # E2c: teacher local Part target. no_grad block 안이므로 gradient 가 없다.
            t_local_parts = None
            if cfg.use_partwhole and cfg.partwhole_part_source == "local_crop":
                t_l_out = self.model.teacher_backbone(l_cat)                           # [n_l*B, 49, D]
                t_local_parts = torch.stack(
                    [
                        pool_local_part(
                            {"cls": c_cls, "patch": c_patch}, cfg.partwhole_local_part_feature
                        )
                        for c_cls, c_patch in zip(
                            t_l_out["cls"].chunk(n_l, 0), t_l_out["patch"].chunk(n_l, 0)
                        )
                    ],
                    dim=1,
                ).detach()                                                             # [B, L, D]

        # ---------------- Student base SSL forward (§9) ------------------------
        s_g = self.model.student_backbone(g_cat, mask=masks)        # masked global
        s_l = self.model.student_backbone(l_cat)                    # unmasked local
        s_cls_all = torch.cat([s_g["cls"], s_l["cls"]], dim=0)      # [(n_g+n_l)*B, D]
        s_cls_logits_cat = self.model.student_cls_head(s_cls_all)   # [(n_g+n_l)*B, K_cls]
        s_cls_logits = (
            list(s_cls_logits_cat[: n_g * B].chunk(n_g, 0))
            + list(s_cls_logits_cat[n_g * B:].chunk(n_l, 0))
        )                                                           # (n_g+n_l) x [B, K_cls]
        s_patch_per_crop = list(s_g["patch"].chunk(n_g, 0))         # n_g x [B, 256, D]
        s_patch_logits_masked: List[torch.Tensor] = []
        for c in range(n_g):
            m = masks_per_crop[c]
            if cfg.ibot_compute_on_masked_only:
                sel = s_patch_per_crop[c][m]                        # [M, D]
                s_patch_logits_masked.append(self.model.student_patch_head(sel))       # [M, K]
            else:
                full = self.model.student_patch_head(s_patch_per_crop[c])              # [B,256,K]
                s_patch_logits_masked.append(full[m])

        # ---------------- L_base --------------------------------------------
        l_dino = self.dino_loss(s_cls_logits, t_cls_logits, teacher_temp, cfg.student_temp)
        l_ibot = self.ibot_loss(
            s_patch_logits_masked, t_patch_logits_masked,
            teacher_temp, cfg.patch_student_temp, device,
        )
        s_cls_per_global = list(s_g["cls"].chunk(n_g, 0))           # n_g x [B, D]
        l_koleo = torch.stack([self.koleo_loss(c) for c in s_cls_per_global]).mean()
        l_base = cfg.lambda_dino * l_dino + cfg.lambda_ibot * l_ibot + cfg.lambda_koleo * l_koleo

        zero = torch.zeros((), device=device)
        l_position, l_comp, l_decomp = zero, zero, zero
        pos_stats = {"n_pairs": 0, "mean_ios": 0.0}
        pw_stats = {
            "n_valid_pairs": 0.0, "valid_ratio": 0.0, "mean_valid_locals": 0.0,
            "zero_valid_wholes": 0.0, "mean_containment": 0.0,
        }

        # ---------------- clean (unmasked) student global forward -------------
        # structural loss 에는 masked patch token 을 사용하지 않는다 (§9).
        if cfg.need_aux_clean_forward:
            student_aux_global_outputs = self.model.student_backbone(g_cat)
            s_aux_patch = list(student_aux_global_outputs["patch"].chunk(n_g, 0))   # n_g x [B,256,D]
            s_aux_cls = list(student_aux_global_outputs["cls"].chunk(n_g, 0))       # n_g x [B,D]
        else:
            s_aux_patch, s_aux_cls = None, None

        # ---------------- L_position (E1/E3) ---------------------------------
        if cfg.use_position:
            with torch.no_grad():
                t_boxes = [patch_boxes_to_original(m.to(device), validate=False) for m in g_metas]
                s_boxes = t_boxes + [
                    patch_boxes_to_original(m.to(device), validate=False) for m in l_metas
                ]   # student global crop 은 teacher 와 동일한 crop 이므로 box 를 재사용
            s_pos_logits = [self.model.student_position_head(p) for p in s_aux_patch]      # [B,256,K_pos]
            s_pos_logits += [
                self.model.student_position_head(p)
                for p in s_l["patch"].chunk(n_l, 0)
            ]                                                                             # [B,49,K_pos]
            l_position, pos_stats = self.position_loss(
                teacher_logits=t_pos_logits,
                teacher_boxes=t_boxes,
                student_logits=s_pos_logits,
                student_boxes=s_boxes,
                n_global_student=n_g,
                device=device,
            )

        # ---------------- L_composition / L_decomposition (E2/E2b/E2c/E3) -----
        if cfg.use_partwhole and cfg.partwhole_part_source == "local_crop":
            # ===================== E2c =======================================
            # Part  = 독립 인코딩된 local crop  (student: s_l, teacher: t_local_parts)
            # Whole = global crop              (student: s_aux_cls, teacher: t_cls_per_crop)
            with torch.no_grad():
                geom_all, valid_all, cont_all = local_in_global_geometry(
                    [m.to(device) for m in g_metas],
                    [m.to(device) for m in l_metas],
                    cfg.partwhole_containment_threshold,
                )                                                 # [B,G,L,Fg], [B,G,L], [B,G,L]
            s_local_parts = torch.stack(
                [
                    pool_local_part(
                        {"cls": c_cls, "patch": c_patch}, cfg.partwhole_local_part_feature
                    )
                    for c_cls, c_patch in zip(
                        s_l["cls"].chunk(n_l, 0), s_l["patch"].chunk(n_l, 0)
                    )
                ],
                dim=1,
            )                                                     # [B, L, D]  gradient O

            comp_sum = torch.zeros((), device=device, dtype=torch.float32)
            comp_n = torch.zeros((), device=device, dtype=torch.float32)
            decomp_sum = torch.zeros((), device=device, dtype=torch.float32)
            decomp_n = torch.zeros((), device=device, dtype=torch.float32)

            for c in range(n_g):
                geom_c = geom_all[:, c]                           # [B,L,Fg]
                valid_c = valid_all[:, c]                         # [B,L]

                # ---- Part -> Whole -------------------------------------
                w_hat_s, has_valid = self.composition_predictor(
                    s_local_parts.to(s_l["patch"].dtype), geom_c, valid_c
                )                                                 # [B,D], [B]
                w_t = t_cls_per_crop[c].float()
                if cfg.partwhole_center_whole:
                    w_t = w_t - w_t.mean(dim=0, keepdim=True)
                cs, cn = masked_cosine_loss(w_hat_s, w_t, has_valid)
                comp_sum = comp_sum + cs
                comp_n = comp_n + cn

                # ---- Whole -> Part -------------------------------------
                p_hat_s = self.decomposition_predictor(s_aux_cls[c], geom_c)   # [B,L,D]
                p_t = t_local_parts.float()
                if cfg.partwhole_center_whole:
                    # local crop 번호에 고정된 해부학적 의미가 없으므로
                    # batch × local 전체 평균으로 centering (E2b grid Part 와 달리)
                    p_t = p_t - p_t.mean(dim=(0, 1), keepdim=True)
                    p_t = F.normalize(p_t, dim=-1, p=2, eps=1e-8)
                ds, dn = masked_cosine_loss(p_hat_s, p_t, valid_c)
                decomp_sum = decomp_sum + ds
                decomp_n = decomp_n + dn

            l_comp = comp_sum / comp_n.clamp(min=1.0)
            l_decomp = decomp_sum / decomp_n.clamp(min=1.0)
            pw_stats = {
                "n_valid_pairs": float(valid_all.sum().item()),
                "valid_ratio": float(valid_all.float().mean().item()),
                "mean_valid_locals": float(valid_all.float().sum(dim=2).mean().item()),
                "zero_valid_wholes": float((~valid_all.any(dim=2)).float().sum().item()),
                "mean_containment": float(cont_all.mean().item()),
            }
        elif cfg.use_partwhole:
            # ===================== E2 / E2b (기존) ============================
            comp_terms, decomp_terms = [], []
            for c in range(n_g):
                # Part -> Whole
                p_s = extract_part_features(s_aux_patch[c], gh, gw, cfg.part_grid_size)   # [B,16,768]
                w_hat_s = self.composition_predictor(p_s.to(s_aux_patch[c].dtype))        # [B,768]
                w_t = t_cls_per_crop[c].float()
                if cfg.partwhole_center_whole:
                    w_t = w_t - w_t.mean(dim=0, keepdim=True)
                comp_terms.append(composition_loss(w_hat_s, w_t))
                # Whole -> Part
                p_hat_s = self.decomposition_predictor(s_aux_cls[c])                      # [B,16,768]
                p_t = extract_part_features(
                    t_patch_per_crop[c].detach(), gh, gw, cfg.part_grid_size
                )                                                                          # [B,16,768]
                if cfg.partwhole_center_whole:
                    p_t = p_t.float() - p_t.float().mean(dim=0, keepdim=True)
                    p_t = F.normalize(p_t, dim=-1, p=2, eps=1e-8)
                decomp_terms.append(decomposition_loss(p_hat_s, p_t))
            l_comp = torch.stack(comp_terms).mean()
            l_decomp = torch.stack(decomp_terms).mean()

        # ---------------- L_gram (E2d local preservation) --------------------
        l_gram = torch.zeros((), device=device)
        if cfg.local_preservation == "gram":
            with torch.no_grad():
                g_patch_cat = self.gram_teacher(g_cat)["patch"]                        # [n_g*B,256,D]
            g_patch_per_crop = list(g_patch_cat.chunk(n_g, 0))                         # n_g x [B,256,D]
            l_gram = torch.stack([
                gram_anchor_loss(s_aux_patch[c], g_patch_per_crop[c]) for c in range(n_g)
            ]).mean()

        # ---------------- total ----------------------------------------------
        l_total = l_base
        if cfg.use_position:
            l_total = l_total + cfg.lambda_pos * l_position
        if cfg.use_partwhole:
            l_total = l_total + cfg.lambda_comp * l_comp + cfg.lambda_decomp * l_decomp
        if cfg.local_preservation == "gram":
            l_total = l_total + cfg.lambda_gram * l_gram

        # ---------------- center EMA 업데이트 ---------------------------------
        with torch.no_grad():
            self.dino_loss.update_center(t_cls_logits)
            self.ibot_loss.update_center(t_patch_logits_masked)
            if cfg.use_position:
                self.position_loss.update_center(t_pos_logits)

        logs = {
            "L_DINO": float(l_dino.detach()),
            "L_iBOT": float(l_ibot.detach()),
            "L_KoLeo": float(l_koleo.detach()),
            "L_base": float(l_base.detach()),
            "L_position": float(l_position.detach()) if cfg.use_position else 0.0,
            "L_composition": float(l_comp.detach()) if cfg.use_partwhole else 0.0,
            "L_decomposition": float(l_decomp.detach()) if cfg.use_partwhole else 0.0,
            "L_gram": float(l_gram.detach()),
            "L_total": float(l_total.detach()),
            "mask_ratio": float(masks.float().mean()),
            "position_pairs": float(pos_stats["n_pairs"]),
            "position_mean_ios": float(pos_stats["mean_ios"]),
            "partwhole_valid_ratio": pw_stats["valid_ratio"],
            "partwhole_mean_valid_locals": pw_stats["mean_valid_locals"],
            "partwhole_zero_valid_wholes": pw_stats["zero_valid_wholes"],
            "partwhole_mean_containment": pw_stats["mean_containment"],
        }
        return l_total, logs


# %%
# ============================================================================
# Cell 20. EMA / Center / Temperature / Momentum Schedules (§8, §11, §24)
# ============================================================================
def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray[epochs * niter_per_ep]  (linear warmup + cosine decay)
    """
    warmup_iters = int(warmup_epochs * niter_per_ep)
    total_iters = int(epochs * niter_per_ep)
    warmup = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_iters > 0 else np.array([])
    iters = np.arange(total_iters - warmup_iters)
    cos = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / max(len(iters), 1))
    )
    schedule = np.concatenate([warmup, cos])
    assert len(schedule) == total_iters
    return schedule


def teacher_temp_scheduler(
    start: float, final: float, warmup_epochs: int, epochs: int, niter_per_ep: int
) -> np.ndarray:
    """teacher temperature: warmup_epochs 동안 선형 증가 후 final 로 고정 (§11)."""
    warmup_iters = int(min(warmup_epochs, epochs) * niter_per_ep)
    total_iters = int(epochs * niter_per_ep)
    warm = np.linspace(start, final, warmup_iters) if warmup_iters > 0 else np.array([])
    rest = np.full(total_iters - warmup_iters, final)
    return np.concatenate([warm, rest])


def momentum_scheduler(base: float, final: float, epochs: int, niter_per_ep: int) -> np.ndarray:
    """teacher EMA momentum: cosine 으로 base -> final(1.0) 증가 (§8)."""
    total_iters = int(epochs * niter_per_ep)
    iters = np.arange(total_iters)
    return final - (final - base) * (np.cos(np.pi * iters / max(total_iters, 1)) + 1) / 2


# %%
# ============================================================================
# Cell 21. Optimizer / LR / Weight Decay Scheduler (§24)
#   Teacher parameter 는 optimizer 에 절대 포함하지 않는다.
# ============================================================================
def build_param_groups(trainable_modules: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
    """
    weight decay 를 적용할 파라미터와 제외할 파라미터(bias / norm / token / gain)를 분리한다.
    """
    regularized, not_regularized = [], []
    seen = set()
    for mod_name, mod in trainable_modules.items():
        for pname, p in mod.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            full = f"{mod_name}.{pname}"
            if p.ndim <= 1 or full.endswith(("cls_token", "mask_token", "pos_embed", "weight_g")):
                not_regularized.append(p)
            else:
                regularized.append(p)
    return [
        {"params": regularized, "apply_wd": True},
        {"params": not_regularized, "apply_wd": False, "weight_decay": 0.0},
    ]


def build_optimizer(cfg: Config, module: "SSLTrainingModule") -> torch.optim.Optimizer:
    groups = build_param_groups(module.trainable_modules())
    n_reg = sum(p.numel() for p in groups[0]["params"])
    n_nonreg = sum(p.numel() for p in groups[1]["params"])
    print(f"[optimizer] regularized={n_reg/1e6:.2f}M  not_regularized={n_nonreg/1e6:.2f}M")
    # teacher parameter 가 섞여 있지 않은지 방어적으로 확인
    teacher_ids = {id(p) for m in module.model.teacher_modules().values() for p in m.parameters()}
    for g in groups:
        for p in g["params"]:
            if id(p) in teacher_ids:
                raise RuntimeError("teacher parameter 가 optimizer 에 포함되었습니다.")
    return torch.optim.AdamW(groups, lr=cfg.scaled_lr(), betas=(0.9, 0.999), weight_decay=cfg.weight_decay)


def apply_schedules(
    optimizer: torch.optim.Optimizer, lr: float, wd: float
) -> None:
    """iteration 단위로 lr / weight decay 를 갱신한다."""
    for group in optimizer.param_groups:
        group["lr"] = lr
        if group.get("apply_wd", True):
            group["weight_decay"] = wd


def cancel_last_layer_gradients(module: "SSLTrainingModule") -> None:
    """학습 초기 prototype layer 를 고정하여 붕괴를 방지 (DINO 관행)."""
    for head in (module.model.student_cls_head, module.model.student_patch_head):
        if head.last_layer.weight_v.grad is not None:
            head.last_layer.weight_v.grad = None


# %%
# ============================================================================
# Cell 22. Checkpoint Manager (§28)
# ============================================================================
class CheckpointManager:
    """latest.pt / best.pt 저장 및 정확한 resume (optimizer, EMA, centers, step)."""

    def __init__(self, cfg: Config, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.latest_path = os.path.join(run_dir, "latest.pt")
        self.best_path = os.path.join(run_dir, "best.pt")

    def state_dict(
        self,
        module: "SSLTrainingModule",
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        best_metric: float,
    ) -> Dict[str, Any]:
        m = module.model
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "config": asdict(self.cfg),
            "environment": ENV_INFO,
            "student_backbone": m.student_backbone.state_dict(),
            "student_cls_head": m.student_cls_head.state_dict(),
            "student_patch_head": m.student_patch_head.state_dict(),
            "student_position_head": (
                m.student_position_head.state_dict() if m.student_position_head is not None else None
            ),
            "teacher_backbone": m.teacher_backbone.state_dict(),
            "teacher_cls_head": m.teacher_cls_head.state_dict(),
            "teacher_patch_head": m.teacher_patch_head.state_dict(),
            "teacher_position_head": (
                m.teacher_position_head.state_dict() if m.teacher_position_head is not None else None
            ),
            "composition_predictor": (
                module.composition_predictor.state_dict()
                if module.composition_predictor is not None else None
            ),
            "decomposition_predictor": (
                module.decomposition_predictor.state_dict()
                if module.decomposition_predictor is not None else None
            ),
            "optimizer": optimizer.state_dict(),
            "dino_center": module.dino_loss.center.detach().cpu(),
            "ibot_center": module.ibot_loss.center.detach().cpu(),
            "position_center": (
                module.position_loss.center.detach().cpu()
                if module.position_loss is not None else None
            ),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        return state

    def save(self, state: Dict[str, Any], is_best: bool = False) -> None:
        tmp = self.latest_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.latest_path)
        if is_best:
            torch.save(state, self.best_path + ".tmp")
            os.replace(self.best_path + ".tmp", self.best_path)

    def load(
        self,
        path: str,
        module: "SSLTrainingModule",
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint 없음: {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = module.model
        try:
            m.student_backbone.load_state_dict(ckpt["student_backbone"], strict=strict)
            m.student_cls_head.load_state_dict(ckpt["student_cls_head"], strict=strict)
            m.student_patch_head.load_state_dict(ckpt["student_patch_head"], strict=strict)
            m.teacher_backbone.load_state_dict(ckpt["teacher_backbone"], strict=strict)
            m.teacher_cls_head.load_state_dict(ckpt["teacher_cls_head"], strict=strict)
            m.teacher_patch_head.load_state_dict(ckpt["teacher_patch_head"], strict=strict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint architecture mismatch ({path}).\n"
                f"  현재 Config 의 backbone/head 설정이 checkpoint 와 다릅니다.\n  원인: {exc}"
            ) from exc

        if m.student_position_head is not None:
            if ckpt.get("student_position_head") is None:
                raise RuntimeError(
                    "checkpoint architecture mismatch: 현재 experiment 는 position head 를 요구하지만 "
                    "checkpoint 에는 position head 가 없습니다."
                )
            m.student_position_head.load_state_dict(ckpt["student_position_head"], strict=strict)
            m.teacher_position_head.load_state_dict(ckpt["teacher_position_head"], strict=strict)
        if module.composition_predictor is not None:
            if ckpt.get("composition_predictor") is None:
                raise RuntimeError(
                    "checkpoint architecture mismatch: composition predictor 가 checkpoint 에 없습니다."
                )
            module.composition_predictor.load_state_dict(ckpt["composition_predictor"], strict=strict)
            module.decomposition_predictor.load_state_dict(ckpt["decomposition_predictor"], strict=strict)

        module.dino_loss.center.copy_(ckpt["dino_center"].to(module.dino_loss.center.device))
        module.ibot_loss.center.copy_(ckpt["ibot_center"].to(module.ibot_loss.center.device))
        if module.position_loss is not None and ckpt.get("position_center") is not None:
            module.position_loss.center.copy_(
                ckpt["position_center"].to(module.position_loss.center.device)
            )
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt


# %%
# ============================================================================
# Cell 23. Complete Trainer (§23, §27, §42, §43)
# ============================================================================
class VirtualEpochSampler(torch.utils.data.Sampler):
    """
    전체 dataset 을 순회하는 permutation 을 유지하면서 epoch 당 samples_per_epoch 개만 반환한다.

    3.49M 장 전체를 1 epoch 으로 두면 batch 32 기준 109k iteration 이 되어
    checkpoint / logging 간격이 지나치게 길어진다. 가상 epoch 으로 끊되,
    permutation 위치를 이어가므로 모든 이미지가 순차적으로 소진된다.
    """

    def __init__(self, dataset_size: int, samples_per_epoch: int, seed: int = 42):
        self.n = int(dataset_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = seed
        self._perm: Optional[torch.Tensor] = None
        self._pos = 0
        self._round = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _new_permutation(self) -> None:
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

    def state_dict(self) -> Dict[str, int]:
        return {"pos": self._pos, "round": self._round}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self._round = int(state["round"])
        self._pos = int(state["pos"])
        if self._round > 0:
            g = torch.Generator()
            g.manual_seed(self.seed + self._round - 1)
            self._perm = torch.randperm(self.n, generator=g)


class Trainer:
    """
    사용법
    ------
    cfg.experiment = "E1"
    trainer = Trainer(cfg)
    trainer.fit()      # <- 사용자가 명시적으로 호출할 때만 학습이 시작된다
    """

    def __init__(self, cfg: Config, file_paths: Optional[Sequence[str]] = None):
        self.cfg = cfg
        self.device = DEVICE
        self.run_dir = cfg.run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        # 실험 조건 기록 (§42)
        cfg.to_json(os.path.join(self.run_dir, "config.json"))

        # E0~E3 동일 조건 보장을 위해 매 run 시작 시 동일 seed 로 초기화
        set_seed(cfg.seed, cfg.deterministic)

        # ---------------- data ----------------
        if file_paths is None:
            file_paths = get_image_paths(cfg)
        self.file_paths = list(file_paths)
        self.transform = MultiCropTransform(cfg)
        self.dataset = MSKXrayDataset(self.file_paths, self.transform, pre_resize=cfg.pre_resize)

        # steps_per_epoch 지정 시 가상 epoch 사용 (전체 순회 위치는 sampler 가 유지)
        if cfg.steps_per_epoch is not None:
            self.sampler = VirtualEpochSampler(
                len(self.dataset), cfg.steps_per_epoch * cfg.batch_size, cfg.seed
            )
        else:
            self.sampler = None
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            collate_fn=collate_multicrop,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
            generator=make_dataloader_generator(cfg.seed),
        )
        self.niter_per_ep = len(self.loader)
        if self.niter_per_ep == 0:
            raise RuntimeError(
                f"DataLoader 가 비었습니다 (dataset {len(self.dataset)}, batch {cfg.batch_size}, "
                f"drop_last={cfg.drop_last})."
            )

        # ---------------- model ----------------
        self.module = SSLTrainingModule(cfg).to(self.device)
        if cfg.use_compile:
            # coordinate/correspondence 코드는 compile 대상에서 제외한다 (§25)
            print("[compile] student/teacher backbone 만 torch.compile 적용")
            self.module.model.student_backbone = torch.compile(self.module.model.student_backbone)
            self.module.model.teacher_backbone = torch.compile(self.module.model.teacher_backbone)

        n_student = count_parameters(self.module.model.student_backbone)
        print(f"[model] student backbone params: {n_student/1e6:.1f}M")

        # ---------------- optimizer / schedules ----------------
        self.optimizer = build_optimizer(cfg, self.module)
        self.lr_schedule = cosine_scheduler(
            cfg.scaled_lr(), cfg.min_lr, cfg.epochs, self.niter_per_ep,
            cfg.warmup_epochs, cfg.start_warmup_lr,
        )
        self.wd_schedule = cosine_scheduler(
            cfg.weight_decay, cfg.weight_decay_end, cfg.epochs, self.niter_per_ep
        )
        self.momentum_schedule = momentum_scheduler(
            cfg.teacher_momentum, cfg.teacher_momentum_final, cfg.epochs, self.niter_per_ep
        )
        self.teacher_temp_schedule = teacher_temp_scheduler(
            cfg.teacher_temp_start, cfg.teacher_temp_final,
            cfg.teacher_temp_warmup_epochs, cfg.epochs, self.niter_per_ep,
        )

        # ---------------- checkpoint / logging ----------------
        self.ckpt = CheckpointManager(cfg, self.run_dir)
        self.train_log_path = os.path.join(self.run_dir, "training_log.csv")
        self.iter_log_path = os.path.join(self.run_dir, "iteration_log.csv")
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = math.inf if cfg.monitor_mode == "min" else -math.inf

        if cfg.auto_resume and os.path.isfile(self.ckpt.latest_path):
            self._resume(self.ckpt.latest_path)

        images_per_epoch = self.niter_per_ep * cfg.batch_size
        print(f"[trainer] experiment={cfg.experiment}  images={len(self.dataset):,}  "
              f"iters/epoch={self.niter_per_ep}  images/epoch={images_per_epoch:,}  "
              f"total_iters={self.niter_per_ep * cfg.epochs:,}  run_dir={self.run_dir}")
        print(f"[trainer] 전체 dataset 1회 순회에 필요한 epoch 수: "
              f"{len(self.dataset) / max(images_per_epoch, 1):.1f}")

    # ---------------------------------------------------------------- resume
    def _resume(self, path: str) -> None:
        # run_dir 는 experiment+run_tag 로 분리되지만, part 정의가 다른 checkpoint 를
        # 잘못 읽으면 predictor shape 가 조용히 어긋날 수 있으므로 미리 막는다.
        head = torch.load(path, map_location="cpu", weights_only=False)
        ck_cfg = head.get("config", {})
        for key in ("experiment", "run_tag", "partwhole_part_source",
                    "partwhole_local_part_feature", "partwhole_center_whole",
                    "local_preservation", "gram_teacher_ckpt", "lambda_gram"):
            if key in ck_cfg and getattr(self.cfg, key) != ck_cfg[key]:
                raise RuntimeError(
                    f"resume 거부: checkpoint 의 {key}={ck_cfg[key]!r} 가 현재 "
                    f"Config 의 {getattr(self.cfg, key)!r} 와 다릅니다 ({path}).\n"
                    f"  --no-resume 를 쓰거나 run_tag 를 분리하세요."
                )
        del head
        ck = self.ckpt.load(path, self.module, self.optimizer, map_location="cpu")
        self.start_epoch = int(ck["epoch"]) + 1
        self.global_step = int(ck["global_step"])
        self.best_metric = float(ck["best_metric"])
        if self.sampler is not None and ck.get("sampler_state") is not None:
            self.sampler.load_state_dict(ck["sampler_state"])
        try:
            torch.set_rng_state(ck["torch_rng_state"])
            torch.cuda.set_rng_state_all(ck["cuda_rng_state"])
            np.random.set_state(ck["numpy_rng_state"])
            random.setstate(ck["python_rng_state"])
        except Exception as exc:  # noqa: BLE001
            print("[WARN] RNG state 복원 실패 (학습은 계속 진행):", exc)
        print(f"[resume] {path}  -> epoch {self.start_epoch}, step {self.global_step}, "
              f"best {self.best_metric:.6f}")

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _gpu_mem() -> Dict[str, float]:
        return {
            "gpu_alloc_GB": torch.cuda.memory_allocated() / 1024 ** 3,
            "gpu_reserved_GB": torch.cuda.memory_reserved() / 1024 ** 3,
            "gpu_max_alloc_GB": torch.cuda.max_memory_allocated() / 1024 ** 3,
        }

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "global_images": [x.to(self.device, non_blocking=True) for x in batch["global_images"]],
            "local_images": [x.to(self.device, non_blocking=True) for x in batch["local_images"]],
            "global_metas": batch["global_metas"],   # meta 는 loss 내부에서 필요 시 이동
            "local_metas": batch["local_metas"],
            "index": batch["index"],
        }

    def _append_csv(self, path: str, row: Dict[str, Any]) -> None:
        exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------ train loop
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        cfg = self.cfg
        self.module.train()
        accum: Dict[str, float] = {k: 0.0 for k in LOSS_KEYS}
        accum.update({"mask_ratio": 0.0, "position_pairs": 0.0, "position_mean_ios": 0.0})
        n_iter = 0
        epoch_start = time.time()

        pbar = tqdm(self.loader, desc=f"[{cfg.experiment}] epoch {epoch}", leave=False)
        for it, batch in enumerate(pbar):
            step = self.global_step
            if step >= len(self.lr_schedule):
                step = len(self.lr_schedule) - 1
            lr = float(self.lr_schedule[step])
            wd = float(self.wd_schedule[step])
            momentum = float(self.momentum_schedule[step])
            teacher_temp = float(self.teacher_temp_schedule[step])
            apply_schedules(self.optimizer, lr, wd)

            t0 = time.time()
            batch = self._batch_to_device(batch)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_bf16):
                loss, logs = self.module.forward_loss(batch, teacher_temp, self.device)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NaN/Inf loss at epoch {epoch} iter {it} (global_step {self.global_step}).\n"
                    f"  losses={logs}\n"
                    f"  lr={lr:.3e}, wd={wd:.4f}, teacher_temp={teacher_temp:.4f}\n"
                    f"  대응: lr 감소, warmup_epochs 증가, clip_grad 강화, batch_size 조정을 검토하십시오."
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()   # BF16 autocast 에서는 GradScaler 불필요 (§25)
            if epoch < cfg.freeze_last_layer_epochs:
                cancel_last_layer_gradients(self.module)
            grad_norm = float("nan")
            if cfg.clip_grad is not None and cfg.clip_grad > 0:
                params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad))
            self.optimizer.step()

            # teacher EMA 업데이트 (student step 이후)
            self.module.model.update_teacher(momentum)

            torch.cuda.synchronize()
            iter_time = time.time() - t0
            ips = cfg.batch_size / max(iter_time, 1e-8)

            for k in accum:
                accum[k] += logs.get(k, 0.0)
            n_iter += 1
            self.global_step += 1

            if it % cfg.log_interval == 0:
                mem = self._gpu_mem()
                row = {
                    "epoch": epoch, "iteration": it, "global_step": self.global_step,
                    **{k: round(logs.get(k, 0.0), 6) for k in LOSS_KEYS},
                    "learning_rate": lr, "weight_decay": wd,
                    "teacher_momentum": momentum, "teacher_temperature": teacher_temp,
                    "grad_norm": grad_norm,
                    "mask_ratio": round(logs["mask_ratio"], 4),
                    "position_pairs": int(logs["position_pairs"]),
                    "position_mean_ios": round(logs["position_mean_ios"], 4),
                    "pw_valid_ratio": round(logs.get("partwhole_valid_ratio", 0.0), 4),
                    "pw_mean_valid_locals": round(logs.get("partwhole_mean_valid_locals", 0.0), 3),
                    "pw_zero_valid_wholes": int(logs.get("partwhole_zero_valid_wholes", 0.0)),
                    "iteration_time": round(iter_time, 4),
                    "images_per_second": round(ips, 2),
                    **{k: round(v, 3) for k, v in mem.items()},
                }
                if cfg.save_iteration_log:
                    self._append_csv(self.iter_log_path, row)
                pbar.set_postfix({
                    "total": f"{logs['L_total']:.3f}",
                    "dino": f"{logs['L_DINO']:.3f}",
                    "ibot": f"{logs['L_iBOT']:.3f}",
                    "pos": f"{logs['L_position']:.3f}",
                    "comp": f"{logs['L_composition']:.3f}",
                    "dec": f"{logs['L_decomposition']:.3f}",
                    "ips": f"{ips:.1f}",
                })

        mean_logs = {k: v / max(n_iter, 1) for k, v in accum.items()}
        mean_logs["epoch_time_sec"] = time.time() - epoch_start
        return mean_logs

    # -------------------------------------------------------------------- fit
    def fit(self) -> None:
        cfg = self.cfg
        print(f"=== training start: {cfg.experiment} "
              f"(epochs {self.start_epoch} -> {cfg.epochs - 1}) ===")
        for epoch in range(self.start_epoch, cfg.epochs):
            stats = self.train_one_epoch(epoch)
            monitor_value = stats["L_total"]     # epoch_mean_total_loss

            mem = self._gpu_mem()
            row = {
                "epoch": epoch,
                "global_step": self.global_step,
                **{k: round(stats[k], 6) for k in LOSS_KEYS},
                "epoch_mean_total_loss": round(monitor_value, 6),
                "mask_ratio": round(stats["mask_ratio"], 4),
                "position_pairs": round(stats["position_pairs"], 1),
                "position_mean_ios": round(stats["position_mean_ios"], 4),
                "learning_rate": float(self.lr_schedule[min(self.global_step - 1, len(self.lr_schedule) - 1)]),
                "weight_decay": float(self.wd_schedule[min(self.global_step - 1, len(self.wd_schedule) - 1)]),
                "teacher_momentum": float(self.momentum_schedule[min(self.global_step - 1, len(self.momentum_schedule) - 1)]),
                "teacher_temperature": float(self.teacher_temp_schedule[min(self.global_step - 1, len(self.teacher_temp_schedule) - 1)]),
                "epoch_time_sec": round(stats["epoch_time_sec"], 2),
                "images_per_second": round(
                    len(self.dataset) / max(stats["epoch_time_sec"], 1e-8), 2
                ),
                **{k: round(v, 3) for k, v in mem.items()},
            }
            self._append_csv(self.train_log_path, row)

            is_best = (
                monitor_value < self.best_metric if cfg.monitor_mode == "min"
                else monitor_value > self.best_metric
            )
            if is_best:
                self.best_metric = monitor_value
            state = self.ckpt.state_dict(
                self.module, self.optimizer, epoch, self.global_step, self.best_metric
            )
            if self.sampler is not None:
                state["sampler_state"] = self.sampler.state_dict()
            self.ckpt.save(state, is_best=is_best)

            print(
                f"[{cfg.experiment}] epoch {epoch:03d} | total {stats['L_total']:.4f} "
                f"| dino {stats['L_DINO']:.4f} ibot {stats['L_iBOT']:.4f} koleo {stats['L_KoLeo']:.4f} "
                f"| pos {stats['L_position']:.4f} comp {stats['L_composition']:.4f} "
                f"dec {stats['L_decomposition']:.4f} | best {self.best_metric:.4f}"
                f"{'  *' if is_best else ''}"
            )
        print(f"=== training finished: {cfg.experiment} ===")
# %%
# ============================================================================
# Cell 24. Smoke Test Suite (§35)
#   Notebook 실행만으로 자동 수행되지 않는다.
#   사용자가 run_smoke_test(cfg, experiment="E1") 처럼 명시적으로 호출한다.
# ============================================================================
def make_synthetic_meta(
    batch_size: int,
    grid: int,
    crop_size: int,
    device: torch.device,
    scale_range: Tuple[float, float] = (0.4, 1.0),
    seed: int = 0,
) -> BatchedViewMeta:
    """무작위 crop box / flip 을 가진 BatchedViewMeta 생성 (test 전용)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    side = torch.empty(batch_size).uniform_(scale_range[0], scale_range[1], generator=g).sqrt()   # [B]
    x0 = torch.rand(batch_size, generator=g) * (1 - side)
    y0 = torch.rand(batch_size, generator=g) * (1 - side)
    crop_box = torch.stack([x0, y0, side, side], dim=1).to(device)              # [B,4]
    hflip = (torch.rand(batch_size, generator=g) < 0.5).to(device)
    return BatchedViewMeta(crop_box, hflip, crop_size, crop_size, grid, grid)


def make_synthetic_batch(cfg: Config, batch_size: int, device: torch.device) -> Dict[str, Any]:
    """dataset 없이 loss/gradient test 를 수행하기 위한 random batch."""
    g_imgs = [
        torch.randn(batch_size, 3, cfg.global_crop_size, cfg.global_crop_size, device=device)
        for _ in range(cfg.num_global_crops)
    ]
    l_imgs = [
        torch.randn(batch_size, 3, cfg.local_crop_size, cfg.local_crop_size, device=device)
        for _ in range(cfg.num_local_crops)
    ]
    g_metas = [
        make_synthetic_meta(batch_size, cfg.global_grid, cfg.global_crop_size, device,
                            cfg.global_crop_scale, seed=100 + k)
        for k in range(cfg.num_global_crops)
    ]
    l_metas = [
        make_synthetic_meta(batch_size, cfg.local_grid, cfg.local_crop_size, device,
                            cfg.local_crop_scale, seed=200 + k)
        for k in range(cfg.num_local_crops)
    ]
    return {
        "global_images": g_imgs, "global_metas": g_metas,
        "local_images": l_imgs, "local_metas": l_metas,
        "index": torch.arange(batch_size),
    }


def smoke_test_dataset(cfg: Config, n_files: int = 4) -> None:
    """Dataset test: 한 sample 에서 2 global + 6 local view 와 metadata shape 확인."""
    print("--- [dataset test] ---")
    files = discover_images(cfg.data_root, cfg.image_extensions, cfg.dataset_cache_list,
                            max_images=n_files, verbose=False)
    validate_images(files, n_check=min(2, len(files)))
    ds = MSKXrayDataset(files, MultiCropTransform(cfg), pre_resize=cfg.pre_resize)
    sample = ds[0]
    assert len(sample["global_images"]) == cfg.num_global_crops
    assert len(sample["local_images"]) == cfg.num_local_crops
    for x in sample["global_images"]:
        assert tuple(x.shape) == (3, cfg.global_crop_size, cfg.global_crop_size), x.shape
    for x in sample["local_images"]:
        assert tuple(x.shape) == (3, cfg.local_crop_size, cfg.local_crop_size), x.shape
    for m in sample["global_metas"] + sample["local_metas"]:
        assert m.crop_box.shape == (4,)
        assert 0.0 <= float(m.crop_box[0]) <= 1.0 and 0.0 <= float(m.crop_box[1]) <= 1.0
    batch = collate_multicrop([ds[i] for i in range(min(2, len(ds)))])
    assert batch["global_images"][0].shape[1:] == (3, cfg.global_crop_size, cfg.global_crop_size)
    assert batch["global_metas"][0].crop_box.shape[1] == 4
    print(f"  OK  global {tuple(batch['global_images'][0].shape)}  "
          f"local {tuple(batch['local_images'][0].shape)}  "
          f"crop_box {tuple(batch['global_metas'][0].crop_box.shape)}")


def smoke_test_coordinates(cfg: Config, device: torch.device = DEVICE) -> None:
    """Coordinate test: patch box 범위 확인 + flip 전후 mapping unit test."""
    print("--- [coordinate test] ---")
    meta = make_synthetic_meta(8, cfg.global_grid, cfg.global_crop_size, device, seed=7)
    boxes = patch_boxes_to_original(meta, validate=True)
    assert boxes.shape == (8, cfg.n_global_tokens, 4), boxes.shape
    describe_patch_boxes(boxes, "global")

    meta_l = make_synthetic_meta(8, cfg.local_grid, cfg.local_crop_size, device,
                                 cfg.local_crop_scale, seed=8)
    boxes_l = patch_boxes_to_original(meta_l, validate=True)
    assert boxes_l.shape == (8, cfg.n_local_tokens, 4), boxes_l.shape
    describe_patch_boxes(boxes_l, "local")

    # --- flip unit test: 2x2 grid, crop_box = [0.2, 0.3, 0.4, 0.5] ------------
    cb = torch.tensor([[0.2, 0.3, 0.4, 0.5]], device=device)
    m_no = BatchedViewMeta(cb, torch.tensor([False], device=device), 28, 28, 2, 2)
    m_fl = BatchedViewMeta(cb, torch.tensor([True], device=device), 28, 28, 2, 2)
    b_no = patch_boxes_to_original(m_no)[0]      # [4,4] row-major: (0,0),(0,1),(1,0),(1,1)
    b_fl = patch_boxes_to_original(m_fl)[0]
    # no-flip: col0 x∈[0.2,0.4], col1 x∈[0.4,0.6]
    assert torch.allclose(b_no[0, [0, 2]], torch.tensor([0.2, 0.4], device=device), atol=1e-6), b_no[0]
    assert torch.allclose(b_no[1, [0, 2]], torch.tensor([0.4, 0.6], device=device), atol=1e-6), b_no[1]
    # flip: view col0 는 원본 오른쪽 절반에 대응해야 한다
    assert torch.allclose(b_fl[0, [0, 2]], torch.tensor([0.4, 0.6], device=device), atol=1e-6), b_fl[0]
    assert torch.allclose(b_fl[1, [0, 2]], torch.tensor([0.2, 0.4], device=device), atol=1e-6), b_fl[1]
    # y 좌표는 flip 영향 없음
    assert torch.allclose(b_no[:, [1, 3]], b_fl[:, [1, 3]], atol=1e-6)
    print("  OK  flip mapping (view col0 <-> original right half) 검증 완료")


def smoke_test_correspondence(cfg: Config, device: torch.device = DEVICE) -> None:
    """Correspondence test: 겹치는 두 crop 에서 IoS / valid count / top-k 확인."""
    print("--- [correspondence test] ---")
    # 의도적으로 크게 겹치는 두 global crop
    # patch 폭 = 0.60/16 = 0.0375 이므로 offset 0.075 = 정확히 2 patch (grid 정렬 -> IoS ~ 1.0)
    cb_a = torch.tensor([[0.100, 0.100, 0.60, 0.60]], device=device)
    cb_b = torch.tensor([[0.175, 0.175, 0.60, 0.60]], device=device)
    meta_a = BatchedViewMeta(cb_a, torch.tensor([False], device=device),
                             cfg.global_crop_size, cfg.global_crop_size, cfg.global_grid, cfg.global_grid)
    meta_b = BatchedViewMeta(cb_b, torch.tensor([False], device=device),
                             cfg.global_crop_size, cfg.global_crop_size, cfg.global_grid, cfg.global_grid)
    boxes_a = patch_boxes_to_original(meta_a)
    boxes_b = patch_boxes_to_original(meta_b)
    ios = compute_ios_matrix(boxes_a, boxes_b)
    assert ios.shape == (1, cfg.n_global_tokens, cfg.n_global_tokens), ios.shape
    assert float(ios.min()) >= 0.0 and float(ios.max()) <= 1.0 + 1e-5

    b_idx, t_idx, s_idx, w = find_correspondences(ios, cfg.pos_overlap_threshold, cfg.position_topk)
    print(f"  IoS shape={tuple(ios.shape)} max={float(ios.max()):.3f} "
          f"valid_pairs={b_idx.numel()} (threshold={cfg.pos_overlap_threshold})")
    assert b_idx.numel() > 0, "겹치는 crop 인데 valid correspondence 가 없습니다."
    assert torch.all(w >= cfg.pos_overlap_threshold)
    assert b_idx.shape == t_idx.shape == s_idx.shape == w.shape

    # 겹치지 않는 crop -> valid pair 0 이어야 한다
    cb_far = torch.tensor([[0.80, 0.80, 0.19, 0.19]], device=device)
    meta_far = BatchedViewMeta(cb_far, torch.tensor([False], device=device),
                               cfg.local_crop_size, cfg.local_crop_size, cfg.local_grid, cfg.local_grid)
    ios_far = compute_ios_matrix(boxes_a, patch_boxes_to_original(meta_far))
    b2, _, _, _ = find_correspondences(ios_far, cfg.pos_overlap_threshold, cfg.position_topk)
    print(f"  non-overlapping pair -> valid_pairs={b2.numel()}")
    assert b2.numel() == 0
    print("  OK")


def smoke_test_losses(cfg: Config, experiment: str, batch_size: int = 4,
                      device: torch.device = DEVICE) -> Dict[str, float]:
    """Loss test: random mini batch 로 모든 loss 가 finite 인지 확인."""
    print(f"--- [loss test: {experiment}] ---")
    test_cfg = replace(cfg, experiment=experiment)
    module = SSLTrainingModule(test_cfg).to(device)
    module.train()
    batch = make_synthetic_batch(test_cfg, batch_size, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=test_cfg.use_bf16):
        loss, logs = module.forward_loss(batch, teacher_temp=test_cfg.teacher_temp_start, device=device)
    for k in LOSS_KEYS:
        v = logs[k]
        assert math.isfinite(v), f"{k} is not finite: {v}"
    assert torch.isfinite(loss), "total loss is not finite"
    print("  " + "  ".join(f"{k}={logs[k]:.4f}" for k in LOSS_KEYS))
    del module, batch, loss
    torch.cuda.empty_cache()
    return logs


def smoke_test_gradients(cfg: Config, experiment: str, batch_size: int = 2,
                         device: torch.device = DEVICE) -> None:
    """Gradient test: student gradient 존재 / teacher gradient 부재 / loss finite."""
    print(f"--- [gradient test: {experiment}] ---")
    test_cfg = replace(cfg, experiment=experiment)
    module = SSLTrainingModule(test_cfg).to(device)
    module.train()
    optimizer = build_optimizer(test_cfg, module)
    batch = make_synthetic_batch(test_cfg, batch_size, device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=test_cfg.use_bf16):
        loss, logs = module.forward_loss(batch, teacher_temp=test_cfg.teacher_temp_start, device=device)
    assert torch.isfinite(loss), "loss is not finite"
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # student backbone gradient 존재 확인
    n_grad = sum(
        1 for p in module.model.student_backbone.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    )
    n_param = sum(1 for _ in module.model.student_backbone.parameters())
    assert n_grad > 0, "student backbone gradient 가 없습니다."
    # teacher gradient 부재 확인
    for name, mod in module.model.teacher_modules().items():
        for p in mod.parameters():
            assert p.requires_grad is False, f"{name}: teacher parameter 가 requires_grad=True 입니다."
            assert p.grad is None, f"{name}: teacher parameter 에 gradient 가 생성되었습니다."
    # Gram teacher: gradient 없음 + optimizer/EMA 대상 아님
    gram_before = None
    if module.gram_teacher is not None:
        for p in module.gram_teacher.parameters():
            assert p.requires_grad is False, "Gram teacher parameter 가 requires_grad=True 입니다."
            assert p.grad is None, "Gram teacher parameter 에 gradient 가 생성되었습니다."
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert not any(id(p) in opt_ids for p in module.gram_teacher.parameters()), \
            "Gram teacher parameter 가 optimizer 에 포함되어 있습니다."
        assert not module.gram_teacher.training, "Gram teacher 가 train mode 입니다."
        gram_before = module.gram_teacher.cls_token.detach().clone()
        print(f"  Gram teacher OK  L_gram={logs['L_gram']:.5f}")
    # auxiliary predictor gradient (E2/E3)
    if test_cfg.use_partwhole:
        assert module.composition_predictor.proj.weight.grad is not None, "composition predictor grad 없음"
        assert module.decomposition_predictor.proj.weight.grad is not None, "decomposition predictor grad 없음"
    if test_cfg.use_position:
        assert module.model.student_position_head.proj.weight.grad is not None, "position head grad 없음"

    # EMA 동작 확인
    before = module.model.teacher_backbone.cls_token.detach().clone()
    optimizer.step()
    module.model.update_teacher(0.9)
    after = module.model.teacher_backbone.cls_token.detach()
    assert not torch.equal(before, after), "teacher EMA 업데이트가 반영되지 않았습니다."
    if gram_before is not None:
        assert torch.equal(gram_before, module.gram_teacher.cls_token.detach()), \
            "Gram teacher 가 optimizer/EMA 에 의해 변경되었습니다."

    print(f"  OK  student grads {n_grad}/{n_param} params, teacher grads = None, "
          f"L_total={logs['L_total']:.4f}")
    del module, optimizer, batch, loss
    torch.cuda.empty_cache()


def smoke_test_partwhole_geometry(cfg: Config, batch_size: int = 8,
                                  device: torch.device = DEVICE) -> None:
    """E2c geometry / valid mask / teacher-student 분리 검증."""
    if cfg.partwhole_part_source != "local_crop":
        return
    print("--- [E2c geometry test] ---")
    batch = make_synthetic_batch(cfg, batch_size, device)
    g_metas = [m.to(device) for m in batch["global_metas"]]
    l_metas = [m.to(device) for m in batch["local_metas"]]
    geom, valid, cont = local_in_global_geometry(
        g_metas, l_metas, cfg.partwhole_containment_threshold
    )
    B, G, L, F_ = geom.shape
    assert (B, G, L) == (batch_size, cfg.num_global_crops, cfg.num_local_crops), \
        f"geometry shape 불일치: {(B, G, L)}"
    assert F_ == cfg.partwhole_geom_dim, f"geom dim {F_} != cfg {cfg.partwhole_geom_dim}"
    assert torch.isfinite(geom).all(), "geometry 에 NaN/Inf 가 있습니다."
    assert float(cont.min()) >= 0.0 and float(cont.max()) <= 1.0, "containment 범위 위반"
    print(f"  geom={tuple(geom.shape)}  valid={tuple(valid.shape)}  "
          f"valid_ratio={valid.float().mean():.3f}  "
          f"mean_valid_locals={valid.float().sum(2).mean():.2f}  "
          f"zero_valid_wholes={(~valid.any(2)).sum().item()}/{B*G}  "
          f"mean_containment={cont.mean():.3f}")

    # 극단 케이스: local 이 global 밖에 있으면 valid=False 이고 loss 가 유한해야 한다.
    zero_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    comp = LocalPartComposer(cfg.embed_dim, cfg.partwhole_geom_dim).to(device)
    parts = F.normalize(torch.randn(B, L, cfg.embed_dim, device=device), dim=-1)
    w_hat, has_valid = comp(parts, geom[:, 0], zero_mask)
    ls, ln = masked_cosine_loss(w_hat, torch.randn(B, cfg.embed_dim, device=device), has_valid)
    assert torch.isfinite(ls) and float(ln) == 0.0, "zero-valid 처리 실패"
    print("  zero-valid batch -> loss_sum=0, n=0 (OK)")


def run_smoke_test(
    cfg: Config,
    experiment: Optional[str] = None,
    include_dataset: bool = True,
    batch_size: int = 4,
) -> None:
    """
    사용 예
    -------
    run_smoke_test(cfg, experiment="E1")
    run_smoke_test(cfg, experiment="E3", include_dataset=False)
    """
    experiment = experiment or cfg.experiment
    print("=" * 74)
    print(f"SMOKE TEST — experiment {experiment}")
    print("=" * 74)
    if include_dataset:
        try:
            smoke_test_dataset(cfg)
        except FileNotFoundError as exc:
            print(f"  [SKIPPED] dataset test: {exc}")
    smoke_test_coordinates(cfg)
    smoke_test_correspondence(cfg)
    smoke_test_partwhole_geometry(replace(cfg, experiment=experiment))
    smoke_test_losses(cfg, experiment, batch_size=batch_size)
    smoke_test_gradients(cfg, experiment, batch_size=max(2, batch_size // 2))
    print("=" * 74)
    print(f"SMOKE TEST PASSED — {experiment}")
    print("=" * 74)


# %%
# ============================================================================
# Cell 25. RAD-DINO + Feature Extraction + Representation Metrics
#   (§29, §30, §31, §32, §33)
# ============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def config_from_dict(d: Dict[str, Any]) -> Config:
    """checkpoint 에 저장된 config dict 로부터 Config 를 복원한다."""
    valid = set(Config.__dataclass_fields__.keys())
    tuple_fields = {
        "image_extensions", "global_crop_scale", "local_crop_scale", "crop_ratio",
        "global_blur_prob", "normalize_mean", "normalize_std",
    }
    kwargs = {}
    for k, v in d.items():
        if k not in valid:
            continue
        kwargs[k] = tuple(v) if k in tuple_fields and isinstance(v, list) else v
    return Config(**kwargs)


def experiment_run_dir(cfg: Config, experiment: str) -> str:
    """Trainer 와 동일한 규칙(run_tag 포함)으로 실험 출력 디렉토리를 만든다."""
    name = experiment + (f"_{cfg.run_tag}" if cfg.run_tag else "")
    return os.path.join(cfg.output_root, name)


class FeatureExtractor:
    """
    통일된 feature 추출 인터페이스.

    __call__
    --------
    images_unnorm: [B, 3, H, W]  (값 범위 [0,1], normalize 이전)

    Returns
    -------
    {"cls": [B, D], "patch": [B, N, D]}  (fp32)
    """

    def __init__(
        self,
        name: str,
        forward_fn,
        mean: Sequence[float],
        std: Sequence[float],
        patch_size: int,
        embed_dim: int,
        device: torch.device = DEVICE,
        use_bf16: bool = True,
    ):
        self.name = name
        self.forward_fn = forward_fn
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        self.use_bf16 = use_bf16

    @torch.no_grad()
    def __call__(self, images_unnorm: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = images_unnorm.to(self.device, non_blocking=True)
        x = (x - self.mean) / self.std
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.forward_fn(x)
        return {"cls": out["cls"].float(), "patch": out["patch"].float()}

    def grid_size(self, input_size: int) -> int:
        return input_size // self.patch_size


def load_ssl_backbone(
    cfg: Config,
    experiment: str,
    checkpoint_name: Optional[str] = None,
    use_teacher: Optional[bool] = None,
    device: torch.device = DEVICE,
) -> Tuple[FeatureExtractor, str]:
    """
    E0~E3 checkpoint 로부터 backbone 을 로드하여 (FeatureExtractor, "teacher"|"student") 를 반환한다.
    """
    checkpoint_name = checkpoint_name or cfg.eval_checkpoint_name
    use_teacher = cfg.eval_use_teacher if use_teacher is None else use_teacher
    run_dir = experiment_run_dir(cfg, experiment)
    path = os.path.join(run_dir, checkpoint_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint 없음: {path} (experiment {experiment} 미학습?)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ck_cfg = config_from_dict(ckpt["config"])
    backbone = build_backbone(ck_cfg)
    key = "teacher_backbone" if use_teacher else "student_backbone"
    try:
        backbone.load_state_dict(ckpt[key], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint architecture mismatch ({path}, {key}): {exc}") from exc
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    tag = "teacher" if use_teacher else "student"
    return FeatureExtractor(
        name=f"{experiment}", forward_fn=lambda x: backbone(x),
        mean=ck_cfg.normalize_mean, std=ck_cfg.normalize_std,
        patch_size=ck_cfg.patch_size, embed_dim=ck_cfg.embed_dim,
        device=device, use_bf16=cfg.use_bf16,
    ), tag


class RadDinoWrapper(nn.Module):
    """RAD-DINO (external pretrained reference) 를 공통 출력 형식으로 감싼다."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(pixel_values=x, interpolate_pos_encoding=True)
        h = out.last_hidden_state                    # [B, 1+N, D]
        return {"cls": h[:, 0], "patch": h[:, 1:]}


def load_rad_dino(cfg: Config, device: torch.device = DEVICE) -> FeatureExtractor:
    """
    Loading 우선순위 (§29)
    1) cfg.rad_dino_local_path 가 존재하면 local (local_files_only=True)
    2) HuggingFace hub (microsoft/rad-dino)
    3) 둘 다 불가능하면 informative error
    """
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "RAD-DINO 로딩에 transformers 가 필요합니다.\n"
            f"  {cfg.python_executable} -m pip install transformers"
        )
    from transformers import AutoModel, AutoImageProcessor

    local = cfg.rad_dino_local_path
    if local and os.path.isdir(local):
        source, kwargs = local, {"local_files_only": True}
        print(f"[rad-dino] local model 사용: {local}")
    elif local and not os.path.isdir(local):
        raise FileNotFoundError(
            f"RAD-DINO local model 없음: {local!r}\n"
            f"  cfg.rad_dino_local_path 를 수정하거나 None 으로 두고 HuggingFace 를 사용하십시오."
        )
    else:
        source, kwargs = cfg.rad_dino_model_id, {}
        print(f"[rad-dino] HuggingFace 사용: {source}")

    try:
        model = AutoModel.from_pretrained(source, **kwargs)
        processor = AutoImageProcessor.from_pretrained(source, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"RAD-DINO 로딩 실패 (source={source}).\n"
            f"  - 오프라인 환경이면 모델을 미리 내려받아 cfg.rad_dino_local_path 에 지정하십시오.\n"
            f"  원인: {exc!r}"
        ) from exc

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    wrapper = RadDinoWrapper(model)
    mean = tuple(getattr(processor, "image_mean", IMAGENET_MEAN))
    std = tuple(getattr(processor, "image_std", IMAGENET_STD))
    patch_size = getattr(model.config, "patch_size", 14)
    dim = getattr(model.config, "hidden_size", 768)
    print(f"[rad-dino] patch={patch_size} dim={dim} mean={mean} std={std}")
    return FeatureExtractor(
        name="RAD-DINO", forward_fn=lambda x: wrapper(x), mean=mean, std=std,
        patch_size=patch_size, embed_dim=dim, device=device, use_bf16=cfg.use_bf16,
    )


# ---------------------------------------------------------------------------
# Evaluation view 생성 (학습과 동일한 coordinate mapping 사용, §31)
# ---------------------------------------------------------------------------
@dataclass
class EvalCropParams:
    top: int
    left: int
    height: int
    width: int
    flip: bool
    img_w: int
    img_h: int


class EvalCropSampler:
    """seed 고정된 deterministic crop parameter sampler."""

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = random.Random(cfg.eval_seed if seed is None else seed)

    def sample(self, img_w: int, img_h: int) -> EvalCropParams:
        i, j, h, w = sample_rrc_params(
            img_h, img_w, self.cfg.global_crop_scale, self.cfg.crop_ratio, self.rng
        )
        flip = self.rng.random() < self.cfg.hflip_prob
        return EvalCropParams(i, j, h, w, flip, img_w, img_h)

    def full_view(self, img_w: int, img_h: int) -> EvalCropParams:
        """probe / annotation 용 결정적 full-image view (crop 없음, flip 없음)."""
        return EvalCropParams(0, 0, img_h, img_w, False, img_w, img_h)


def render_view(
    img: Image.Image, params: EvalCropParams, out_size: int, patch_size: int
) -> Tuple[torch.Tensor, ViewMeta]:
    """
    Returns
    -------
    tensor: [3, out_size, out_size]  값 범위 [0,1] (모델별 normalize 는 FeatureExtractor 가 수행)
    meta  : ViewMeta (원본 normalized coordinate)
    """
    view = TF.resized_crop(
        img, params.top, params.left, params.height, params.width, [out_size, out_size],
        interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
    )
    if params.flip:
        view = TF.hflip(view)
    x = TF.to_tensor(view)
    grid = out_size // patch_size
    meta = ViewMeta(
        crop_box=torch.tensor(
            [params.left / params.img_w, params.top / params.img_h,
             params.width / params.img_w, params.height / params.img_h],
            dtype=torch.float32,
        ),
        hflip=params.flip, output_height=out_size, output_width=out_size,
        patch_grid_height=grid, patch_grid_width=grid,
    )
    return x, meta


def load_eval_images(cfg: Config, n: Optional[int] = None) -> List[str]:
    n = n or cfg.eval_num_images
    return discover_images(cfg.data_root, cfg.image_extensions, cfg.dataset_cache_list,
                           max_images=n, verbose=False)


# ---------------------------------------------------------------------------
# §31 Spatial Consistency Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_spatial_consistency(
    extractor: FeatureExtractor,
    image_paths: Sequence[str],
    cfg: Config,
) -> Dict[str, float]:
    """
    Returns
    -------
    {"positive_spatial_similarity", "negative_spatial_similarity", "spatial_similarity_gap",
     "n_positive_pairs", "n_negative_pairs"}
    """
    size = cfg.eval_input_size
    grid = extractor.grid_size(size)
    sampler = EvalCropSampler(cfg)
    neg_gen = torch.Generator(device="cpu").manual_seed(cfg.eval_seed)

    pos_sum, pos_cnt, neg_sum, neg_cnt = 0.0, 0, 0.0, 0
    bs = cfg.eval_batch_size

    for start in tqdm(range(0, len(image_paths), bs), desc=f"[spatial] {extractor.name}", leave=False):
        chunk = image_paths[start:start + bs]
        xa, xb, ma, mb = [], [], [], []
        for p in chunk:
            with Image.open(p) as im:
                im = im.convert("L").convert("RGB")
                im.load()
            pa = sampler.sample(*im.size)
            pb = sampler.sample(*im.size)
            ta, meta_a = render_view(im, pa, size, extractor.patch_size)
            tb, meta_b = render_view(im, pb, size, extractor.patch_size)
            xa.append(ta); xb.append(tb); ma.append(meta_a); mb.append(meta_b)

        xa = torch.stack(xa); xb = torch.stack(xb)                       # [B,3,S,S]
        meta_a = batch_view_metas(ma).to(DEVICE)
        meta_b = batch_view_metas(mb).to(DEVICE)

        fa = extractor(xa)["patch"]                                      # [B, N, D]
        fb = extractor(xb)["patch"]
        fa = F.normalize(fa, dim=-1, p=2)                                # backbone feature 자체를 L2 normalize
        fb = F.normalize(fb, dim=-1, p=2)

        boxes_a = patch_boxes_to_original(meta_a, validate=False)        # [B, N, 4]
        boxes_b = patch_boxes_to_original(meta_b, validate=False)
        ios = compute_ios_matrix(boxes_a, boxes_b)                       # [B, Na, Nb]

        pos_idx = (ios >= cfg.positive_threshold).nonzero(as_tuple=False)   # [P, 3]
        neg_idx = (ios <= cfg.negative_threshold).nonzero(as_tuple=False)   # [Q, 3]
        if pos_idx.shape[0] == 0:
            continue
        # negative 는 positive 와 비슷한 개수로 seed 고정 random sampling
        n_take = min(pos_idx.shape[0], neg_idx.shape[0])
        if n_take > 0:
            perm = torch.randperm(neg_idx.shape[0], generator=neg_gen)[:n_take]
            neg_sel = neg_idx[perm.to(neg_idx.device)]
        else:
            neg_sel = neg_idx[:0]

        def _cos(pairs: torch.Tensor) -> torch.Tensor:
            if pairs.shape[0] == 0:
                return torch.zeros(0, device=fa.device)
            b, i, j = pairs[:, 0], pairs[:, 1], pairs[:, 2]
            return (fa[b, i] * fb[b, j]).sum(dim=-1)

        pc = _cos(pos_idx)
        nc = _cos(neg_sel)
        pos_sum += float(pc.sum()); pos_cnt += pc.numel()
        neg_sum += float(nc.sum()); neg_cnt += nc.numel()

    pos_mean = pos_sum / pos_cnt if pos_cnt else float("nan")
    neg_mean = neg_sum / neg_cnt if neg_cnt else float("nan")
    return {
        "positive_spatial_similarity": pos_mean,
        "negative_spatial_similarity": neg_mean,
        "spatial_similarity_gap": pos_mean - neg_mean,
        "n_positive_pairs": pos_cnt,
        "n_negative_pairs": neg_cnt,
    }


# ---------------------------------------------------------------------------
# §32 Fair Part-Whole Evaluation (frozen backbone + linear probe)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_frozen_partwhole_features(
    extractor: FeatureExtractor, image_paths: Sequence[str], cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    parts: [N_img, K_part, D]  (L2 normalized)
    whole: [N_img, D]          (CLS, L2 normalized)
    """
    size = cfg.eval_input_size
    grid = extractor.grid_size(size)
    if grid % cfg.part_grid_size != 0:
        raise ValueError(
            f"part_grid_size({cfg.part_grid_size}) 가 patch grid({grid}x{grid}) 를 나누지 못합니다."
        )
    sampler = EvalCropSampler(cfg)
    parts_all, whole_all = [], []
    bs = cfg.eval_batch_size
    for start in tqdm(range(0, len(image_paths), bs), desc=f"[probe-feat] {extractor.name}", leave=False):
        xs = []
        for p in image_paths[start:start + bs]:
            with Image.open(p) as im:
                im = im.convert("L").convert("RGB")
                im.load()
            params = sampler.full_view(*im.size)
            t, _ = render_view(im, params, size, extractor.patch_size)
            xs.append(t)
        out = extractor(torch.stack(xs))
        parts = extract_part_features(out["patch"], grid, grid, cfg.part_grid_size)   # [B,K,D]
        whole = F.normalize(out["cls"], dim=-1, p=2)
        parts_all.append(parts.cpu()); whole_all.append(whole.cpu())
    return torch.cat(parts_all, 0), torch.cat(whole_all, 0)


def _train_linear_probe(
    x: torch.Tensor, y: torch.Tensor, cfg: Config, out_shape: Tuple[int, ...],
    device: torch.device = DEVICE,
) -> float:
    """
    cosine-error 를 최소화하는 linear probe 학습 후 evaluation split 의 cosine error 반환.
    모든 모델에서 동일한 optimizer / epoch / split 을 사용한다.
    """
    set_seed(cfg.eval_seed, deterministic=False)
    n = x.shape[0]
    n_train = max(1, int(n * cfg.probe_train_ratio))
    g = torch.Generator().manual_seed(cfg.eval_seed)
    perm = torch.randperm(n, generator=g)
    tr, ev = perm[:n_train], perm[n_train:]
    if ev.numel() == 0:
        ev = tr[-max(1, n // 10):]

    x_tr, y_tr = x[tr].to(device), y[tr].to(device)
    x_ev, y_ev = x[ev].to(device), y[ev].to(device)
    probe = nn.Linear(x.shape[1], int(np.prod(out_shape)), bias=False).to(device)
    nn.init.trunc_normal_(probe.weight, std=0.02)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay)

    def cos_err(pred_flat: torch.Tensor, target_flat: torch.Tensor) -> torch.Tensor:
        pred = pred_flat.reshape(-1, *out_shape)
        target = target_flat.reshape(-1, *out_shape)
        pred = F.normalize(pred, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)
        return (1.0 - (pred * target).sum(dim=-1)).mean()

    for _ in range(cfg.probe_epochs):
        idx = torch.randperm(x_tr.shape[0], device=device)
        for s in range(0, x_tr.shape[0], cfg.probe_batch_size):
            sel = idx[s:s + cfg.probe_batch_size]
            loss = cos_err(probe(x_tr[sel]), y_tr[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float(cos_err(probe(x_ev), y_ev))


def evaluate_partwhole_linear_probe(
    extractor: FeatureExtractor, image_paths: Sequence[str], cfg: Config
) -> Dict[str, float]:
    """
    frozen backbone feature 에서 part-whole 관계가 얼마나 linearly recoverable 한지 측정 (§32).
    """
    parts, whole = extract_frozen_partwhole_features(extractor, image_paths, cfg)
    n, k, d = parts.shape
    comp_err = _train_linear_probe(parts.reshape(n, k * d), whole, cfg, out_shape=(d,))
    decomp_err = _train_linear_probe(whole, parts.reshape(n, k * d), cfg, out_shape=(k, d))
    return {
        "linear_probe_composition_error": comp_err,
        "linear_probe_decomposition_error": decomp_err,
    }


# ---------------------------------------------------------------------------
# §33 Anatomical Annotation-Based Metrics (optional)
# ---------------------------------------------------------------------------
def load_anatomical_annotations(cfg: Config) -> Optional[pd.DataFrame]:
    """
    기대 형식 (CSV)
    ---------------
    image_path,label,x,y      x,y 는 원본 영상 기준 normalized [0,1] 좌표
    """
    if not cfg.anatomical_annotation_csv:
        return None
    if not os.path.isfile(cfg.anatomical_annotation_csv):
        raise FileNotFoundError(
            f"anatomical annotation CSV 없음: {cfg.anatomical_annotation_csv}"
        )
    df = pd.read_csv(cfg.anatomical_annotation_csv)
    required = {"image_path", "label", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"annotation CSV 컬럼 누락: {sorted(missing)} (필요: image_path,label,x,y / x,y 는 [0,1])"
        )
    if (df["x"].max() > 1.0) or (df["y"].max() > 1.0):
        raise ValueError("annotation 의 x,y 는 원본 기준 normalized [0,1] 좌표여야 합니다.")
    return df


@torch.no_grad()
def extract_annotated_patch_features(
    extractor: FeatureExtractor, df: pd.DataFrame, cfg: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    features: [N, D] (L2 normalized)
    labels  : [N]    (str)
    """
    size = cfg.eval_input_size
    grid = extractor.grid_size(size)
    feats, labels = [], []
    for path, group in tqdm(df.groupby("image_path"), desc=f"[anat] {extractor.name}", leave=False):
        if not os.path.isfile(path):
            continue
        with Image.open(path) as im:
            im = im.convert("L").convert("RGB")
            im.load()
        params = EvalCropParams(0, 0, im.size[1], im.size[0], False, im.size[0], im.size[1])
        t, _ = render_view(im, params, size, extractor.patch_size)
        patch = extractor(t.unsqueeze(0))["patch"][0]                    # [N, D]
        patch = F.normalize(patch, dim=-1, p=2)
        for _, row in group.iterrows():
            col = int(min(max(row["x"], 0.0), 0.999999) * grid)
            r = int(min(max(row["y"], 0.0), 0.999999) * grid)
            feats.append(patch[r * grid + col].cpu().numpy())
            labels.append(str(row["label"]))
    if not feats:
        return np.zeros((0, extractor.embed_dim), dtype=np.float32), np.array([])
    return np.stack(feats), np.array(labels)


def evaluate_anatomical_metrics(
    extractor: FeatureExtractor, cfg: Config
) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns
    -------
    (metrics, features, labels) — annotation 미제공 시 metrics 는 NaN, features/labels 는 None
    """
    nan_metrics = {
        "silhouette": float("nan"), "davies_bouldin": float("nan"),
        "knn_accuracy": float("nan"),
        "intra_structure_similarity": float("nan"),
        "inter_structure_similarity": float("nan"),
    }
    df = load_anatomical_annotations(cfg)
    if df is None:
        print("SKIPPED: anatomical annotation not provided")
        return nan_metrics, None, None

    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    feats, labels = extract_annotated_patch_features(extractor, df, cfg)
    if feats.shape[0] < 10 or len(np.unique(labels)) < 2:
        print("SKIPPED: anatomical annotation 이 너무 적거나 label 종류가 1개입니다.")
        return nan_metrics, None, None

    metrics = dict(nan_metrics)
    try:
        metrics["silhouette"] = float(silhouette_score(feats, labels, metric="cosine"))
        metrics["davies_bouldin"] = float(davies_bouldin_score(feats, labels))
    except Exception as exc:  # noqa: BLE001
        print("[WARN] clustering metric 계산 실패:", exc)

    try:
        strat = labels if min(pd.Series(labels).value_counts()) >= 2 else None
        x_tr, x_te, y_tr, y_te = train_test_split(
            feats, labels, test_size=0.3, random_state=cfg.eval_seed, stratify=strat
        )
        knn = KNeighborsClassifier(n_neighbors=min(cfg.knn_k, len(x_tr)), metric="cosine")
        knn.fit(x_tr, y_tr)
        metrics["knn_accuracy"] = float(knn.score(x_te, y_te))
    except Exception as exc:  # noqa: BLE001
        print("[WARN] kNN 평가 실패:", exc)

    # intra / inter structure cosine similarity
    f = torch.from_numpy(feats)
    f = F.normalize(f, dim=-1, p=2)
    sim = (f @ f.t()).numpy()
    same = labels[:, None] == labels[None, :]
    eye = np.eye(len(labels), dtype=bool)
    intra = sim[same & ~eye]
    inter = sim[~same]
    metrics["intra_structure_similarity"] = float(intra.mean()) if intra.size else float("nan")
    metrics["inter_structure_similarity"] = float(inter.mean()) if inter.size else float("nan")
    return metrics, feats, labels


# ---------------------------------------------------------------------------
# 모델 단위 representation 평가 + representation_metrics.csv 저장
# ---------------------------------------------------------------------------
def evaluate_representation(
    extractor: FeatureExtractor,
    cfg: Config,
    image_paths: Optional[Sequence[str]] = None,
    save_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """모든 representation metric 을 계산하고 representation_metrics.csv 로 저장한다."""
    model_name = model_name or extractor.name
    image_paths = list(image_paths) if image_paths is not None else load_eval_images(cfg)
    print(f"[evaluate] {model_name}  (images={len(image_paths)})")

    row: Dict[str, Any] = {"Model": model_name, "feature_dim": extractor.embed_dim,
                           "patch_size": extractor.patch_size, "n_eval_images": len(image_paths)}
    row.update(evaluate_spatial_consistency(extractor, image_paths, cfg))
    row.update(evaluate_partwhole_linear_probe(extractor, image_paths, cfg))
    anat, feats, labels = evaluate_anatomical_metrics(extractor, cfg)
    row.update(anat)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame([row]).to_csv(os.path.join(save_dir, "representation_metrics.csv"), index=False)
        print(f"  -> {os.path.join(save_dir, 'representation_metrics.csv')}")
    return row, feats, labels


# %%
# ============================================================================
# Cell 26. Representation Comparison & Visualization (§34, §44)
# ============================================================================
def visualize_embeddings(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    cfg: Config = None,
    methods: Sequence[str] = ("pca", "tsne", "umap"),
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Args
    ----
    features: [N, D]
    labels  : [N] or None

    PCA / t-SNE / UMAP 은 동일 random seed 를 사용하고,
    sample 이 많으면 cfg.max_vis_samples 만큼 deterministic subsampling 한다.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    cfg = cfg or globals()["cfg"]
    seed = cfg.eval_seed
    x = np.asarray(features)
    y = None if labels is None else np.asarray(labels)
    if x.shape[0] > cfg.max_vis_samples:
        idx = np.linspace(0, x.shape[0] - 1, cfg.max_vis_samples).round().astype(int)
        idx = np.unique(idx)
        x, y = x[idx], (None if y is None else y[idx])

    methods = [m for m in methods if m != "umap" or HAS_UMAP]
    if "umap" in methods and not HAS_UMAP:
        print(f"[SKIPPED] UMAP 미설치: {cfg.python_executable} -m pip install umap-learn")

    embeddings = {}
    if "pca" in methods:
        embeddings["PCA"] = PCA(n_components=2, random_state=seed).fit_transform(x)
    if "tsne" in methods:
        perp = max(5, min(30, x.shape[0] // 4))
        embeddings["t-SNE"] = TSNE(
            n_components=2, random_state=seed, perplexity=perp, init="pca"
        ).fit_transform(x)
    if "umap" in methods and HAS_UMAP:
        import umap
        embeddings["UMAP"] = umap.UMAP(n_components=2, random_state=seed).fit_transform(x)

    if not embeddings:
        print("[WARN] 사용 가능한 visualization method 가 없습니다.")
        return None

    fig, axes = plt.subplots(1, len(embeddings), figsize=(5.2 * len(embeddings), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, emb) in zip(axes, embeddings.items()):
        if y is None:
            ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6)
        else:
            for lab in np.unique(y):
                m = y == lab
                ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.7, label=str(lab))
            if len(np.unique(y)) <= 12:
                ax.legend(fontsize=7, markerscale=1.5)
        ax.set_title(f"{name} — {title}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viz] saved -> {save_path}")
    return fig


COMPARISON_COLUMNS = [
    "Model",
    "positive_spatial_similarity",
    "negative_spatial_similarity",
    "spatial_similarity_gap",
    "linear_probe_composition_error",
    "linear_probe_decomposition_error",
    "silhouette",
    "davies_bouldin",
    "knn_accuracy",
    "intra_structure_similarity",
    "inter_structure_similarity",
]

MODEL_DESCRIPTION = {
    "E0": "Base DINOv2-style SSL",
    "E1": "Base + Position (coordinate-aware spatial consistency)",
    "E2": "Base + Part-Whole (ACE-inspired composition/decomposition). "
          "partwhole_part_source=local_crop 이면 E2c (independent local-crop Part).",
    "E3": "Full (Position + Part-Whole)",
    "RAD-DINO": "External pretrained reference",
}


def run_phase1_comparison(
    cfg: Config,
    experiments: Sequence[str] = ("E0", "E1", "E2", "E3"),
    include_rad_dino: bool = True,
    image_paths: Optional[Sequence[str]] = None,
    visualize: bool = True,
) -> pd.DataFrame:
    """
    E0~E3 + RAD-DINO representation 비교표를 만들고 CSV 로 저장한다 (§44).
    checkpoint 가 없는 실험은 건너뛰고, annotation 이 없는 metric 은 NaN 으로 남긴다.
    """
    image_paths = list(image_paths) if image_paths is not None else load_eval_images(cfg)
    rows, viz_data = [], {}

    for exp in experiments:
        try:
            extractor, tag = load_ssl_backbone(cfg, exp)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exp}: {exc}")
            continue
        except RuntimeError as exc:
            print(f"[SKIP] {exp}: {exc}")
            continue
        print(f"[model] {exp} ({tag} backbone) — {MODEL_DESCRIPTION.get(exp, '')}")
        row, feats, labels = evaluate_representation(
            extractor, cfg, image_paths,
            save_dir=experiment_run_dir(cfg, exp), model_name=exp,
        )
        rows.append(row)
        if feats is not None:
            viz_data[exp] = (feats, labels)
        del extractor
        torch.cuda.empty_cache()

    if include_rad_dino:
        try:
            rad = load_rad_dino(cfg)
            row, feats, labels = evaluate_representation(
                rad, cfg, image_paths,
                save_dir=os.path.join(cfg.output_root, "RAD-DINO"), model_name="RAD-DINO",
            )
            rows.append(row)
            if feats is not None:
                viz_data["RAD-DINO"] = (feats, labels)
            del rad
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] RAD-DINO: {exc}")

    if not rows:
        raise RuntimeError(
            "비교할 모델이 없습니다. 최소 한 개 실험의 checkpoint 가 필요합니다 "
            f"(확인 경로: {cfg.output_root}/<E0|E1|E2|E3>/{cfg.eval_checkpoint_name})."
        )

    df = pd.DataFrame(rows)
    for col in COMPARISON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COMPARISON_COLUMNS + [c for c in df.columns if c not in COMPARISON_COLUMNS]]
    out_csv = os.path.join(cfg.output_root, "phase1_representation_comparison.csv")
    os.makedirs(cfg.output_root, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[comparison] saved -> {out_csv}")

    if visualize and viz_data:
        for name, (feats, labels) in viz_data.items():
            visualize_embeddings(
                feats, labels, cfg, title=name,
                save_path=os.path.join(cfg.output_root, "figures", f"embedding_{name}.png"),
            )
    elif visualize:
        print("[viz] annotation 기반 label 이 없어 embedding 시각화를 건너뜁니다.")
    return df


# ---------------------------------------------------------------------------
# 사용 예시 — 아래 코드는 주석 상태이며 자동으로 학습을 시작하지 않는다 (§43).
# ---------------------------------------------------------------------------
USAGE_TEXT = """
사용 순서
---------
1) 경로 설정
   cfg.data_root   = "/path/to/MSK_PNG"
   cfg.output_root = "/NHNHOME/WORKSPACE/26msit005_B/users/kbw1994/outputs/msk_ssl"

2) smoke test (실험별로 명시적 호출)
   run_smoke_test(cfg, experiment="E0")
   run_smoke_test(cfg, experiment="E1")
   run_smoke_test(cfg, experiment="E2")
   run_smoke_test(cfg, experiment="E3")

3) 하나의 실험만 학습 (E0~E3 자동 연속 실행 금지)
   cfg.experiment = "E0"
   trainer = Trainer(cfg)
   # trainer.fit()          # <- 사용자가 직접 주석을 해제해서 실행

4) representation 평가
   df = run_phase1_comparison(cfg, experiments=("E0","E1","E2","E3"), include_rad_dino=True)
   display(df)

.py 로 실행할 때 (MobaXterm 세션이 끊겨도 계속 학습)
   tmux new -s e0
   cd /NHNHOME/WORKSPACE/26msit005_B/users/kbw1994
   ./venv/bin/python3 -u train.py --experiment E0 2>&1 | tee -a outputs/msk_ssl/E0/console.log
   (Ctrl-b d 로 detach, tmux attach -t e0 로 재접속)
"""

if __name__ == "__main__" or "ipykernel" in sys.modules:
    print(USAGE_TEXT)

# cfg.experiment = "E0"
# trainer = Trainer(cfg)
# trainer.fit()   # <- 명시적으로 호출할 때만 학습이 시작된다
