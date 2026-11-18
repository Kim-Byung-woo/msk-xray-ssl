"""
frozen backbone 로더 모음 — 평가 스크립트(.py) 공용

03 평가 노트북의 Cell 0 / R / R2 에 흩어져 있던 로더를 한곳에 모은 것이다 (동작은 같다).
  - 우리가 학습한 모델 : ssl/<폴더>/best.pt, teacher backbone, bf16 끔
  - RAD-DINO          : msk_ssl_phase1.load_rad_dino (microsoft/rad-dino, ViT-B/14), 입력 448
  - DINOv3 (공개)     : facebook/dinov3-vitb16-pretrain-lvd1689m (ViT-B/16), ImageNet mean/std, 입력 512
  - DINOv2 (공개)     : torch.hub dinov2_vits14, ImageNet mean/std, 입력 448
모든 모델의 token grid 는 32×32 이다 (448/14, 512/16).

GPU 번호는 코드에 적지 않는다. 반드시 CUDA_VISIBLE_DEVICES 를 지정해서 실행한다.
  (msk_ssl_phase1/2 는 import 할 때 setdefault 로 다른 서버의 GPU 번호(7, 5)를 넣기 때문에,
   지정하지 않으면 엉뚱한 GPU 를 쓰게 된다 → 여기서 먼저 막는다.)

사용:
    import msk_backbones as MB
    backbones = MB.get_backbones(["DINOv3", "RAD-DINO", "D3-E0"])   # {이름: (loader_fn, input_size)}
"""
import os

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    raise RuntimeError("CUDA_VISIBLE_DEVICES 를 지정해서 실행하세요. 예) CUDA_VISIBLE_DEVICES=0 python ...")

from dataclasses import replace
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError(f"GPU 를 찾지 못했습니다 (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}). "
                       "CPU 로 조용히 넘어가지 않도록 멈춥니다.")

import msk_ssl_phase1 as lib_p1   # noqa: E402  (환경변수 확인 뒤 import)
import msk_ssl_phase2 as lib_p2   # noqa: E402

DEVICE = torch.device("cuda:0")
SSL_ROOT = "/workspace/msk/src/ssl"
CHECKPOINT = "best.pt"

# 표시 이름 -> 폴더 (Phase 1, DINOv2 방식 ViT-B/14, 입력 448)
P1_DIRS = {"E0": "E0", "E1c": "E1_fn_safe", "E2d": "E2_indep_part_gram", "E3": "E3"}
# 표시 이름 -> (experiment, run_tag) (Phase 2, DINOv3 방식 ViT-B/16, 입력 512). 폴더 = {experiment}_{run_tag}
P2_RUNS = {"D3-E0": ("D3-E0", ""),
           "D3-E1c": ("D3-E1", "e1c"),
           "D3-E1c-within": ("D3-E1", "e1c_within"),
           "D3-E2-Adam": ("D3-E2", "adam_bf05"),
           "D3-E2-Adam-ep99": ("D3-E2", "adam_bf05")}           # 같은 폴더, checkpoint 만 다름 (아래)
# 기본(best.pt) 대신 다른 checkpoint 파일을 쓰는 모델. 이름이 달라 결과 CSV 에서 서로 섞이지 않는다.
#   D3-E2-Adam 의 best.pt 는 monitor(L_base) 때문에 epoch 21 → 마지막 epoch(99)은 latest.pt
CKPT_OVERRIDE = {"D3-E2-Adam-ep99": "latest.pt"}
P1_INPUT, P2_INPUT = 448, 512

DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_INPUT = 512
DINOV2_VARIANT = "dinov2_vits14"
REF_INPUT = 448

_cfg_p1 = lib_p1.Config(output_root=SSL_ROOT, eval_checkpoint_name=CHECKPOINT,
                        eval_use_teacher=True, use_bf16=False)
_cfg_p2 = lib_p2.Config(output_root=SSL_ROOT, eval_checkpoint_name=CHECKPOINT,
                        eval_use_teacher=True, use_bf16=False)


# ---------------------------------------------------------------------------- 우리 모델
def ckpt_name(name):
    return CKPT_OVERRIDE.get(name, CHECKPOINT)


def ckpt_path(name):
    if name in P1_DIRS:
        return Path(SSL_ROOT) / P1_DIRS[name] / ckpt_name(name)
    exp, tag = P2_RUNS[name]
    return Path(lib_p2.experiment_run_dir(replace(_cfg_p2, run_tag=tag), exp)) / ckpt_name(name)


def ckpt_epoch(name):
    """checkpoint 에 저장된 epoch (로그로 어떤 checkpoint 를 평가했는지 남기기 위함)"""
    return torch.load(ckpt_path(name), map_location="cpu", weights_only=False, mmap=True).get("epoch")


def load_ssl(name):
    if name in P1_DIRS:
        ex, tag = lib_p1.load_ssl_backbone(_cfg_p1, P1_DIRS[name], checkpoint_name=ckpt_name(name))
    else:
        exp, run_tag = P2_RUNS[name]
        ex, tag = lib_p2.load_ssl_backbone(replace(_cfg_p2, run_tag=run_tag), exp, checkpoint_name=ckpt_name(name))
    ex.use_bf16 = False
    ex.name = name
    print(f"[backbone] {name} ({tag}) dim={ex.embed_dim} patch={ex.patch_size} {ckpt_path(name)} "
          f"epoch {ckpt_epoch(name)}")
    return ex


# ---------------------------------------------------------------------------- 공개 모델
class DINOv3HFExtractor(torch.nn.Module):
    """입력 [B,3,H,W] (0–1) → {'cls': [B,D], 'patch': [B,N,D]}  (register token 제외, 최종 LayerNorm 적용)"""

    def __init__(self, model, name="DINOv3", device=DEVICE):
        super().__init__()
        self.model, self.name, self.device = model, name, device
        c = model.config
        ps = c.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.embed_dim = int(c.hidden_size)
        self.n_reg = int(getattr(c, "num_register_tokens", 0))
        self.use_bf16 = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad = False

    def grid_size(self, input_size):
        if input_size % self.patch_size:
            raise ValueError(f"input_size={input_size} 가 patch_size={self.patch_size} 로 나누어떨어지지 않음")
        return input_size // self.patch_size

    @torch.no_grad()
    def forward(self, x):
        x = (x.to(self.device, dtype=torch.float32, non_blocking=True) - self.mean) / self.std
        h = self.model(pixel_values=x).last_hidden_state           # [B, 1 + n_reg + N, D]
        g_h, g_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patch = h[:, 1 + self.n_reg:]
        assert patch.shape[1] == g_h * g_w, f"patch token 수 불일치: {patch.shape[1]} vs {g_h * g_w}"
        return {"cls": h[:, 0], "patch": patch}


def load_dinov3(local_dir=None):
    from transformers import AutoModel
    if local_dir:
        model = AutoModel.from_pretrained(local_dir, local_files_only=True)
    else:
        model = AutoModel.from_pretrained(DINOV3_ID, token=os.environ.get("HF_TOKEN"))
    ex = DINOv3HFExtractor(model)
    print(f"[backbone] DINOv3 ({local_dir or DINOV3_ID}) dim={ex.embed_dim} patch={ex.patch_size} reg={ex.n_reg}")
    return ex


class DINOv2Extractor(torch.nn.Module):
    """02 노트북 Cell 7 과 동일. 입력 [B,3,H,W] (0–1) → {'cls': [B,D], 'patch': [B,N,D]}"""

    def __init__(self, model, name="DINOv2", device=DEVICE):
        super().__init__()
        self.model, self.name, self.device = model, name, device
        self.embed_dim = int(model.embed_dim)
        ps = model.patch_embed.patch_size
        self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        self.use_bf16 = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad = False

    def grid_size(self, input_size):
        if input_size % self.patch_size:
            raise ValueError(f"input_size={input_size} 가 patch_size={self.patch_size} 로 나누어떨어지지 않음")
        return input_size // self.patch_size

    @torch.no_grad()
    def forward(self, x):
        x = (x.to(self.device, dtype=torch.float32, non_blocking=True) - self.mean) / self.std
        out = self.model.forward_features(x)
        return {"cls": out["x_norm_clstoken"], "patch": out["x_norm_patchtokens"]}


def load_dinov2(variant=DINOV2_VARIANT):
    ex = DINOv2Extractor(torch.hub.load("facebookresearch/dinov2", variant), name="DINOv2")
    print(f"[backbone] DINOv2 ({variant}) dim={ex.embed_dim} patch={ex.patch_size}")
    return ex


def load_raddino():
    ex = lib_p1.load_rad_dino(_cfg_p1)
    ex.use_bf16 = False
    print(f"[backbone] RAD-DINO dim={ex.embed_dim} patch={ex.patch_size}")
    return ex


# ---------------------------------------------------------------------------- 등록표
REF_MODELS = {"DINOv3": (load_dinov3, DINOV3_INPUT),
              "RAD-DINO": (load_raddino, REF_INPUT),
              "DINOv2": (load_dinov2, REF_INPUT)}
ALL_NAMES = list(REF_MODELS) + list(P1_DIRS) + list(P2_RUNS)


def get_backbones(names):
    """{이름: (loader_fn, input_size)} — LE.run_pipeline 등과 같은 형식. checkpoint 가 없으면 바로 멈춘다."""
    out = {}
    for n in names:
        if n in REF_MODELS:
            out[n] = REF_MODELS[n]
        elif n in P1_DIRS or n in P2_RUNS:
            p = ckpt_path(n)
            if not p.is_file():
                raise FileNotFoundError(f"{n}: checkpoint 없음 {p}")
            out[n] = ((lambda n=n: load_ssl(n)), P1_INPUT if n in P1_DIRS else P2_INPUT)
        else:
            raise ValueError(f"모르는 모델 이름: {n}. 사용 가능: {ALL_NAMES}")
    return out
