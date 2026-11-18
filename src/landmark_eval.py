"""
landmark_eval.py — dataset-agnostic evaluation of frozen representations for landmark localization.

hip (MTDDH) 분석 노트북 Cell 10–14, 13b 의 로직을 데이터셋에 독립적인 형태로 옮긴 모듈.

구성
----
1. DatasetSpec + adapters      : load_hip / load_foot / load_spine  → 공통 형식
2. prepare / run_qc            : 좌표 정규화·좌우 flip·annotation 이상 검사
3. run_pipeline                : 모델마다 token 캐시 1회 →
                                 (a) label-efficiency decoder 학습/평가 (paired, resume)
                                 (b) prototype similarity margin (학습 없음)
4. analyze                     : 오차 분해(대형 실패/정밀도), paired vs baseline,
                                 margin ↔ 대형 실패 AUROC (image-cluster bootstrap),
                                 모델 간 순위, distractor 유형
5. plot_cases                  : prototype 유사도 지도 사례 figure

hip 재현성
----------
decoder 구조, loss, heatmap target, subpixel decoding, split, subset 추출, 초기값 seed,
학습 스케줄은 노트북 Cell 03–06 / Cell 12 와 동일하게 구현했다.
(GPU 비결정성 때문에 소수점 이하 수치는 run 마다 조금 다를 수 있다.)

backbone 인터페이스
-------------------
backbones = {"E0": (loader_fn, input_size), ...}
loader_fn() -> extractor,  extractor(images[B,3,S,S] in [0,1]) -> {"patch": [B, N, D]},
extractor.grid_size(S) -> int, extractor.embed_dim -> int
(노트북의 load_frozen_backbone 을 그대로 쓸 수 있다.)
"""
from __future__ import annotations

import ast
import copy
import gc
import glob
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import stats

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    roc_auc_score = None


# =============================================================================
# 1. Spec / config
# =============================================================================
@dataclass
class DatasetSpec:
    """
    records: list of dict
        file   : 영상 절대 경로
        kps    : [K,2] float — 픽셀 좌표 (kps_normalized=True 면 0–1 정규화 좌표)
        vis    : [K] bool
        group  : str (하위 집단; 없으면 "all")
        flip   : bool (True 면 좌우 반전해 canonical 방향으로 맞춤)
        meta   : dict (선택, 데이터셋 고유 정보 — 예: foot 의 HVA/IMA)
    kp_structure : landmark 가 속한 구조물 (hip: 좌/우 고관절, spine: 척추뼈, foot: 뼈)
    kp_type      : 구조물 안에서의 역할 (hip: e/y/c/h, spine: TL/TR/BL/BR, foot: prox/dist)
                   → "같은 type, 다른 structure" = 반복 구조 혼동 (hip 좌우 동명, spine 인접 척추뼈)
    """
    name: str
    records: List[Dict]
    kp_names: List[str]
    kp_structure: List[str]
    kp_type: List[str]
    kps_normalized: bool = False
    exclude: Set[int] = field(default_factory=set)
    analysis_group: Optional[str] = None           # margin/대형실패 주 분석 대상 group (None=전체)
    cat_t: float = 0.05                             # 대형 실패 기준: 오차 > diag × cat_t
    margin_radii: Sequence[float] = (2, 3, 4, 6, 8) # token 단위, "GT 로부터 R 밖" hard negative
    primary_margin: str = "m_far_R6"                # 대표 margin 지표 이름
    distractor_radius_tok: float = 6.0
    landmark_near_tok: float = 1.5
    lr_order_pairs: Optional[List[Tuple[str, str]]] = None   # (r_name, l_name): canonical 에서 r.x < l.x
    image_metrics: Optional[Callable] = None        # (pred[K,2], gt[K,2], vis[K], meta) -> dict
    qc_extra: Optional[Callable] = None             # (spec, arrays) -> DataFrame
    notes: str = ""


@dataclass
class EvalConfig:
    seeds: Sequence[int] = (42, 43, 44)
    fractions: Sequence[float] = (0.05, 0.10, 0.25, 1.00)
    n_draws: Dict[float, int] = field(default_factory=lambda: {0.05: 3, 0.10: 3, 0.25: 2, 1.00: 1})
    low_label_fracs: Sequence[float] = (0.05, 0.10)
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    min_train: int = 8
    total_steps: int = 3000
    warmup_steps: int = 150
    val_every: int = 200
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    heatmap_size: int = 128
    sigma: float = 2.0
    cache_batch: int = 16
    num_workers: int = 4
    n_boot: int = 1000
    baseline: str = "E0"
    run_label_efficiency: bool = True
    run_margin: bool = True
    device: str = "cuda"


# =============================================================================
# 2. Adapters
# =============================================================================
def _classify_hip_pose(kps, vis, kp_names):
    pairs = [(kp_names.index("lc"), kp_names.index("lh")), (kp_names.index("rc"), kp_names.index("rh"))]
    dxs, dys = [], []
    for ic, ih in pairs:
        if vis[ic] and vis[ih]:
            dxs.append(abs(kps[ic, 0] - kps[ih, 0]))
            dys.append(abs(kps[ic, 1] - kps[ih, 1]))
    if not dxs:
        return "unknown"
    return "standard" if np.mean(dxs) / max(np.mean(dys), 1e-6) < 1.0 else "spread"


def load_hip(annotation_json: str, image_root: str, exclude: Set[int] = frozenset({451})) -> DatasetSpec:
    """MTDDH Dataset1 keypoints (COCO). 노트북 Cell 02 와 동일한 순서/규칙."""
    with open(annotation_json) as f:
        data = json.load(f)
    id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
    kp_names = None
    for cat in data.get("categories", []):
        if "keypoints" in cat:
            kp_names = list(cat["keypoints"])
            break
    n_kp = data["annotations"][0]["num_keypoints"]
    kp_names = kp_names or [f"landmark_{i}" for i in range(n_kp)]
    records = []
    for ann in data["annotations"]:
        fname = id_to_file.get(ann["image_id"])
        if fname is None:
            continue
        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(n_kp, 3)
        vis = kps[:, 2] > 0
        records.append({
            "file": os.path.join(image_root, fname),
            "kps": kps[:, :2].copy(), "vis": vis,
            "group": _classify_hip_pose(kps[:, :2], vis, kp_names),
            "flip": False, "meta": {"file_name": fname},
        })
    return DatasetSpec(
        name="hip_MTDDH", records=records, kp_names=kp_names,
        kp_structure=[("right" if n[0] == "r" else "left") for n in kp_names],
        kp_type=[n[1:] for n in kp_names],
        exclude=set(exclude), analysis_group="standard",
        margin_radii=(2, 3, 4, 6, 8), primary_margin="m_far_R6",
        distractor_radius_tok=6.0, landmark_near_tok=1.5,
        lr_order_pairs=[("r" + t, "l" + t) for t in ("e", "y", "c", "h")],
        notes="MTDDH Dataset1 keypoints; pose group via (lc,lh),(rc,rh) dx/dy ratio",
    )


def _parse_line(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (list, tuple)):
        return [float(t) for t in x]
    s = str(x).strip()
    for parser in (ast.literal_eval, json.loads):
        try:
            v = parser(s)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                return [float(t) for t in v]
        except Exception:
            pass
    import re
    s2 = re.sub(r"np\.float(16|32|64)", "", s)
    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", s2)
    return [float(t) for t in nums] if len(nums) == 4 else None


def _foot_angles(pts):
    """pts: [6,2] (gt_prox, gt_dist, mt1_prox, mt1_dist, mt2_prox, mt2_dist) → HVA, IMA (deg, 0–180)."""
    def ang(a, b):
        va, vb = pts[a[1]] - pts[a[0]], pts[b[1]] - pts[b[0]]
        c = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        a = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        return min(a, 180.0 - a)      # 선분 방향(끝점 순서)과 무관한 축 사이 각도 (0–90)
    return ang((0, 1), (2, 3)), ang((2, 3), (4, 5))


def _canon_foot_lines(lines, w=1.0, h=1.0):
    """
    끝점 순서를 해부학적으로 통일 (영상 방향과 무관):
      중족골(mt1, mt2): 엄지 선분 중심에 가까운 끝점 = dist
      엄지(gt)       : mt1 dist 에 가까운 끝점 = prox
    거리는 픽셀 비율(w, h)로 계산한다. 반환: 3×[2,2] (prox, dist), 뒤집은 선분 수.
    """
    s = np.array([w, h], float)
    L = [np.array(l, float).reshape(2, 2) for l in lines]
    n_sw = 0
    toe_c = L[0].mean(0)
    for j in (1, 2):
        if np.linalg.norm((L[j][0] - toe_c) * s) < np.linalg.norm((L[j][1] - toe_c) * s):
            L[j] = L[j][::-1].copy(); n_sw += 1
    mt1_dist = L[1][1]
    if np.linalg.norm((L[0][1] - mt1_dist) * s) < np.linalg.norm((L[0][0] - mt1_dist) * s):
        L[0] = L[0][::-1].copy(); n_sw += 1
    return L, n_sw


def _foot_image_metrics(pred, gt, vis, meta):
    if not vis.all():
        return {}
    hva_p, ima_p = _foot_angles(pred)
    hva_g, ima_g = _foot_angles(gt)
    return {"HVA_pred": hva_p, "HVA_gtpts": hva_g, "abs_err_HVA": abs(hva_p - hva_g),
            "IMA_pred": ima_p, "IMA_gtpts": ima_g, "abs_err_IMA": abs(ima_p - ima_g)}


def _foot_qc(spec, arr):
    rows = []
    for i, r in enumerate(spec.records):
        k = arr["kps"][i]
        hva, ima = _foot_angles(k)
        rows.append({
            "image_index": i, "file": os.path.basename(r["file"]), "group": r["group"],
            "HVA_from_points": hva, "HVA_csv": r["meta"].get("HVA", np.nan),
            "IMA_from_points": ima, "IMA_csv": r["meta"].get("IMA", np.nan),
            # canonical(flip 후) 방향에서 각 선분의 첫 점이 둘째 점보다 아래(근위)인가
            "prox_below_dist_all": bool(all(k[2 * j, 1] > k[2 * j + 1, 1] for j in range(3))),
        })
    df = pd.DataFrame(rows)
    df["abs_diff_HVA"] = (df["HVA_from_points"] - df["HVA_csv"]).abs()
    df["abs_diff_IMA"] = (df["IMA_from_points"] - df["IMA_csv"]).abs()
    return df


def load_foot(root: str, canonical_flip: bool = True, flip_label: str = "left") -> DatasetSpec:
    """
    HVAngleEst. root 아래에서 dataset.csv / datasets.csv 와 images 폴더를 찾는다.
    선분 [x1,y1,x2,y2] (0–1 정규화) × 3 → landmark 6개 (각 선분의 1번점=prox, 2번점=dist 로 명명;
    실제 해부학적 정의는 QC 의 prox_below_dist_all 과 데이터셋 문서로 확인할 것).
    canonical_flip=True 면 label==flip_label 영상을 좌우 반전해 방향을 통일한다.
    """
    cands = sorted(glob.glob(os.path.join(root, "**", "dataset.csv"), recursive=True)
                   + glob.glob(os.path.join(root, "**", "datasets.csv"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"dataset.csv / datasets.csv 를 {root} 아래에서 찾지 못했습니다.")
    csv_path = cands[0]
    img_dir = os.path.join(os.path.dirname(csv_path), "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"images 폴더 없음: {img_dir}")
    df = pd.read_csv(csv_path)
    need = {"filename", "great_toe", "first_metatarsal", "second_metatarsal"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{csv_path}: 필요한 컬럼 없음 {sorted(miss)} (있는 컬럼: {list(df.columns)})")
    bones = ["great_toe", "first_metatarsal", "second_metatarsal"]
    # 좌/우 컬럼: 대소문자·공백 무시하고 탐색
    def _side(v):
        s = str(v).strip().lower()
        has_l, has_r = "left" in s, "right" in s
        return "left" if has_l and not has_r else "right" if has_r and not has_l else "unknown"
    # 좌/우 컬럼: 이름 후보 중 left/right 값이 실제로 들어 있는 컬럼
    lab_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("label", "labels", "side", "laterality", "lr"):
            if df[c].map(_side).isin(["left", "right"]).any():
                lab_col = c
                break
    print(f"[foot] csv = {csv_path}")
    print(f"[foot] columns = {list(df.columns)}")
    for c in ("label", "labels", "properties"):
        if c in df.columns:
            print(f"[foot] '{c}' 예시 = {df[c].astype(str).head(3).tolist()}")
    print(f"[foot] 좌/우 컬럼 = {lab_col!r}"
          + ("" if lab_col is None else f"  값 분포 = {df[lab_col].map(_side).value_counts().to_dict()}"))
    if canonical_flip and lab_col is None:
        print("[foot] 경고: 좌/우 컬럼이 없어 canonical flip 을 적용하지 못합니다.")
    records, skipped, n_swapped = [], 0, 0
    for _, row in df.iterrows():
        lines = [_parse_line(row[b]) for b in bones]
        if any(l is None for l in lines):
            skipped += 1
            continue
        L, sw = _canon_foot_lines(lines)
        n_swapped += sw
        kps = np.concatenate(L, 0).astype(np.float32)    # gt_prox, gt_dist, mt1_prox, mt1_dist, mt2_prox, mt2_dist
        label = _side(row[lab_col]) if lab_col is not None else "unknown"
        records.append({
            "file": os.path.join(img_dir, row["filename"]), "kps": kps,
            "vis": np.ones(6, bool), "group": label,
            "flip": bool(canonical_flip and label == flip_label),
            "meta": {"file_name": row["filename"],
                     "HVA": float(row["HVA"]) if "HVA" in df.columns else np.nan,
                     "IMA": float(row["IMA"]) if "IMA" in df.columns else np.nan},
        })
    if skipped:
        print(f"[foot] 선분 파싱 실패로 {skipped}행 제외")
    print(f"[foot] 끝점 순서를 뒤집은 선분 {n_swapped} / {3 * len(records)}")
    if not records:
        raise ValueError(f"[foot] 유효한 행이 없습니다: {csv_path}")
    names = [f"{b}_{p}" for b in ("gt", "mt1", "mt2") for p in ("prox", "dist")]
    return DatasetSpec(
        name="foot_HVAngleEst", records=records, kp_names=names,
        kp_structure=[n.split("_")[0] for n in names], kp_type=[n.split("_")[1] for n in names],
        kps_normalized=True, analysis_group=None,
        margin_radii=(0.75, 1.0, 1.5, 2.0, 3.0), primary_margin="m_lm_sametype",
        distractor_radius_tok=1.5, landmark_near_tok=1.0,
        image_metrics=_foot_image_metrics, qc_extra=_foot_qc,
        notes=f"csv={csv_path}; canonical_flip={canonical_flip} (flip '{flip_label}')",
    )


def _spine_qc(spec, arr):
    rows = []
    for i in range(len(spec.records)):
        k = arr["kps"][i].reshape(17, 4, 2)
        tl_left = (k[:, 0, 0] < k[:, 1, 0]).mean()        # TL.x < TR.x
        bl_left = (k[:, 2, 0] < k[:, 3, 0]).mean()        # BL.x < BR.x
        top_above = (k[:, 0, 1] < k[:, 2, 1]).mean()      # TL.y < BL.y
        order = (np.diff(k[:, :, 1].mean(1)) > 0).mean()  # 척추뼈가 위→아래 순서
        rows.append({"image_index": i, "file": os.path.basename(spec.records[i]["file"]),
                     "frac_TL_left_of_TR": tl_left, "frac_BL_left_of_BR": bl_left,
                     "frac_top_above_bottom": top_above, "frac_vertebra_order_down": order})
    return pd.DataFrame(rows)


def _spine_outlier_corners(p: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    [68,2] → [68] bool. 자기 척추뼈의 나머지 3 모서리 중앙값에서 (factor × 영상의 척추뼈 폭 중앙값)
    보다 멀리 떨어진 모서리 = 주석 오류로 보고 True.
    """
    v = p.reshape(17, 4, 2)
    width = np.median(np.linalg.norm(v[:, 1] - v[:, 0], axis=-1))
    bad = np.zeros((17, 4), bool)
    for j in range(17):
        for c in range(4):
            others = np.delete(v[j], c, axis=0)
            bad[j, c] = np.linalg.norm(v[j, c] - np.median(others, 0)) > factor * width
    return bad.reshape(68)


def load_spine(root: str, split: str = "training", drop_outlier_corners: bool = True,
               outlier_factor: float = 2.0, sort_vertebrae: bool = True) -> DatasetSpec:
    """
    AASCE 형식: root/data/<split>/*.jpg, root/labels/<split>/<image>.mat ('p2': 68×2 px).
    drop_outlier_corners=True 면 자기 척추뼈에서 멀리 떨어진 모서리(주석 오류)를 vis=False 로 둔다
    (학습·prototype·평가에서 모두 제외, 같은 영상의 나머지 점은 그대로 사용).
    """
    import scipy.io
    lab_dir = os.path.join(root, "labels", split)
    img_dir = os.path.join(root, "data", split)
    mats = sorted(glob.glob(os.path.join(lab_dir, "*.mat")))
    if not mats:
        raise FileNotFoundError(f".mat 없음: {lab_dir}")
    records, skipped, bad_rows, reordered = [], 0, [], []
    names = [f"v{v + 1:02d}_{c}" for v in range(17) for c in ["TL", "TR", "BL", "BR"]]
    for m in mats:
        img_name = os.path.basename(m)[:-4]
        img_path = os.path.join(img_dir, img_name)
        if not os.path.isfile(img_path):
            skipped += 1
            continue
        p = np.asarray(scipy.io.loadmat(m)["p2"], dtype=np.float32)
        if p.shape != (68, 2):
            skipped += 1
            continue
        vis = np.ones(68, bool)
        if drop_outlier_corners:
            bad = _spine_outlier_corners(p, outlier_factor)
            vis &= ~bad
        if sort_vertebrae:
            # 척추뼈 순서를 중심 y(보이는 모서리 기준) 로 위 → 아래 정렬. 모서리 순서는 그대로.
            v, vv = p.reshape(17, 4, 2), vis.reshape(17, 4)
            # 모서리가 전부 제외된 척추뼈는 위치를 믿을 수 없으므로 원래 번호에 고정하고,
            # 나머지 척추뼈만 남은 번호 자리 안에서 위 → 아래로 정렬한다.
            ok = vv.any(1)
            slots = np.nonzero(ok)[0]
            cy = np.array([v[j, vv[j], 1].mean() for j in slots])
            o = np.arange(17)
            o[slots] = slots[np.argsort(cy, kind="stable")]
            if not np.array_equal(o, np.arange(17)):
                reordered.append({"image_index": len(records), "file": img_name,
                                  "old_order": o.tolist()})
                p, vis = v[o].reshape(68, 2).copy(), vv[o].reshape(68).copy()
        if drop_outlier_corners:
            for k in np.nonzero(~vis)[0]:
                bad_rows.append({"image_index": len(records), "file": img_name, "landmark": names[k]})
        records.append({"file": img_path, "kps": p, "vis": vis,
                        "group": "all", "flip": False, "meta": {"file_name": img_name}})
    if skipped:
        print(f"[spine] 영상 없음/형식 불일치로 {skipped}개 제외")
    if drop_outlier_corners:
        bd = pd.DataFrame(bad_rows, columns=["image_index", "file", "landmark"])
        print(f"[spine] 주석 오류로 제외한 점 {len(bd)}개 / 영상 {bd.image_index.nunique()}개")
        if len(bd):
            print(bd.groupby(["image_index", "file"]).landmark.apply(list).to_string())
    if sort_vertebrae:
        print(f"[spine] 척추뼈 순서를 위→아래로 다시 정렬한 영상 {len(reordered)}개")
        for r in reordered:
            moved = [f"v{o + 1:02d}→v{n + 1:02d}" for n, o in enumerate(r["old_order"]) if o != n]
            print(f"  {r['image_index']:4d} {r['file']}: {', '.join(moved)}")
    return DatasetSpec(
        name="spine_AASCE", records=records, kp_names=names,
        kp_structure=[n.split("_")[0] for n in names], kp_type=[n.split("_")[1] for n in names],
        analysis_group=None,
        margin_radii=(0.5, 0.75, 1.0, 1.5, 2.0), primary_margin="m_lm_sametype",
        distractor_radius_tok=0.75, landmark_near_tok=0.5,
        qc_extra=_spine_qc,
        notes="68 = 17 vertebrae × (TL,TR,BL,BR); order assumed from sample — see QC",
    )


# =============================================================================
# 3. Geometry prep & QC
# =============================================================================
def prepare(spec: DatasetSpec) -> Dict[str, np.ndarray]:
    """영상 크기를 읽고 좌표를 canonical 픽셀 좌표로 변환 (정규화 해제, flip 반영)."""
    N, K = len(spec.records), len(spec.kp_names)
    kps = np.zeros((N, K, 2), np.float32)
    vis = np.zeros((N, K), bool)
    size = np.zeros((N, 2), np.float32)
    for i, r in enumerate(spec.records):
        with Image.open(r["file"]) as im:
            w, h = im.size
        k = np.asarray(r["kps"], np.float32).copy()
        if spec.kps_normalized:
            k[:, 0] *= w
            k[:, 1] *= h
        if r.get("flip", False):
            k[:, 0] = w - k[:, 0]
        kps[i], vis[i], size[i] = k, np.asarray(r["vis"], bool), (w, h)
    return {"kps": kps, "vis": vis, "size": size,
            "group": np.array([r.get("group", "all") for r in spec.records]),
            "flip": np.array([bool(r.get("flip", False)) for r in spec.records])}


def run_qc(spec: DatasetSpec, arr: Dict, out_dir: str, grid: int = 32) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    N, K = arr["vis"].shape
    kps, vis, size = arr["kps"], arr["vis"], arr["size"]

    # (a) 기본 요약
    inb = (kps[..., 0] >= 0) & (kps[..., 0] < size[:, None, 0]) & \
          (kps[..., 1] >= 0) & (kps[..., 1] < size[:, None, 1])
    summ = {"dataset": spec.name, "n_images": N, "n_landmarks": K,
            "n_excluded": len(spec.exclude), "visible_rate": float(vis.mean()),
            "out_of_bounds_visible": int((vis & ~inb).sum()),
            "img_w_median": float(np.median(size[:, 0])), "img_h_median": float(np.median(size[:, 1])),
            "px_per_token_x_median": float(np.median(size[:, 0] / grid)),
            "px_per_token_y_median": float(np.median(size[:, 1] / grid)),
            "cat_threshold_px_median": float(np.median(np.hypot(size[:, 0], size[:, 1]) * spec.cat_t))}
    for g, c in pd.Series(arr["group"]).value_counts().items():
        summ[f"group_{g}"] = int(c)
    out["summary"] = pd.DataFrame([summ])

    # (b) landmark 간 최근접 거리 (token 단위) — margin 반경 선택 근거
    uv = kps / size[:, None, :] * grid
    rows = []
    for k in range(K):
        same_type_other = [j for j in range(K) if spec.kp_type[j] == spec.kp_type[k]
                           and spec.kp_structure[j] != spec.kp_structure[k]]
        other_struct = [j for j in range(K) if spec.kp_structure[j] != spec.kp_structure[k]]
        same_struct = [j for j in range(K) if j != k and spec.kp_structure[j] == spec.kp_structure[k]]
        def nn_dist(js):
            if not js:
                return np.nan
            d = np.linalg.norm(uv[:, js, :] - uv[:, k:k + 1, :], axis=-1)
            d = np.where(vis[:, js] & vis[:, k:k + 1], d, np.inf).min(1)
            d = d[np.isfinite(d)]
            return float(np.median(d)) if len(d) else np.nan
        rows.append({"landmark": spec.kp_names[k],
                     "median_tok_to_nearest_same_structure": nn_dist(same_struct),
                     "median_tok_to_nearest_other_structure": nn_dist(other_struct),
                     "median_tok_to_nearest_same_type_other_structure": nn_dist(same_type_other)})
    out["landmark_spacing_tok"] = pd.DataFrame(rows)

    # (c) 기하학적 이상치 점수 (landmark 쌍 거리 / 영상 내 median, robust z) — 노트북 Cell 11-F
    pairs = list(itertools.combinations(range(K), 2))
    if K > 40:   # spine: 인접 structure 쌍만 사용해 계산량 축소
        struct_ids = {s: i for i, s in enumerate(dict.fromkeys(spec.kp_structure))}
        sid = np.array([struct_ids[s] for s in spec.kp_structure])
        pairs = [(a, b) for a, b in pairs if abs(sid[a] - sid[b]) <= 1]
    D = np.full((N, len(pairs)), np.nan, np.float32)
    for p, (a, b) in enumerate(pairs):
        d = np.linalg.norm(kps[:, a] - kps[:, b], axis=-1)
        D[:, p] = np.where(vis[:, a] & vis[:, b], d, np.nan)
    D = D / np.nanmedian(D, axis=1, keepdims=True)
    med = np.nanmedian(D, 0)
    mad = np.nanmedian(np.abs(D - med), 0) * 1.4826 + 1e-8
    Z = np.abs(D - med) / mad
    img_score = np.nanmedian(Z, axis=1)
    out["shape_outlier"] = (pd.DataFrame({"image_index": np.arange(N),
                                          "file": [os.path.basename(r["file"]) for r in spec.records],
                                          "group": arr["group"], "shape_score": img_score})
                            .sort_values("shape_score", ascending=False))

    # (d) 좌우 순서 검사 (hip 등)
    if spec.lr_order_pairs:
        rows = []
        for rn, ln in spec.lr_order_pairs:
            i_r, i_l = spec.kp_names.index(rn), spec.kp_names.index(ln)
            bad = np.nonzero(vis[:, i_r] & vis[:, i_l] & (kps[:, i_r, 0] >= kps[:, i_l, 0]))[0]
            for b in bad:
                rows.append({"image_index": int(b), "file": os.path.basename(spec.records[b]["file"]),
                             "pair": f"{rn}/{ln}", "r_x": float(kps[b, i_r, 0]), "l_x": float(kps[b, i_l, 0])})
        out["lr_order_violations"] = pd.DataFrame(rows, columns=["image_index", "file", "pair", "r_x", "l_x"])

    if spec.qc_extra is not None:
        out["dataset_specific"] = spec.qc_extra(spec, arr)

    for k, v in out.items():
        v.to_csv(os.path.join(out_dir, f"qc_{k}.csv"), index=False)
    print(f"[QC] {spec.name}: " + ", ".join(f"{k}={len(v)}" for k, v in out.items())
          + f" -> {out_dir}")
    return out


# =============================================================================
# 4. Decoder / loss / decoding  (노트북 Cell 04–05 와 동일)
# =============================================================================
class HeatmapDecoder(nn.Module):
    def __init__(self, embed_dim, grid, n_landmarks, heatmap_size=128, hidden=256):
        super().__init__()
        self.grid = grid
        n_up = int(round(math.log2(heatmap_size / grid)))
        if 2 ** n_up * grid != heatmap_size:
            raise ValueError(f"heatmap_size({heatmap_size}) 가 grid({grid}) 의 2의 거듭제곱 배수가 아닙니다.")
        layers = [nn.Conv2d(embed_dim, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True)]
        ch = hidden
        for _ in range(n_up):
            nxt = max(ch // 2, 64)
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(ch, nxt, 3, padding=1), nn.BatchNorm2d(nxt), nn.ReLU(inplace=True)]
            ch = nxt
        layers += [nn.Conv2d(ch, n_landmarks, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, tokens):
        B, N, D = tokens.shape
        return self.net(tokens.transpose(1, 2).reshape(B, D, self.grid, self.grid))


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        super().__init__()
        self.omega, self.theta, self.epsilon, self.alpha = omega, theta, epsilon, alpha

    def forward(self, pred, target, weight=None):
        delta = (target - pred).abs()
        a_pow = self.alpha - target
        A = (self.omega * (1 / (1 + (self.theta / self.epsilon) ** a_pow))
             * a_pow * ((self.theta / self.epsilon) ** (a_pow - 1)) / self.epsilon)
        C = self.theta * A - self.omega * torch.log1p((self.theta / self.epsilon) ** a_pow)
        loss = torch.where(delta < self.theta,
                           self.omega * torch.log1p((delta / self.epsilon) ** a_pow), A * delta - C)
        loss = loss.mean(dim=(2, 3))
        if weight is not None:
            return (loss * weight).sum() / weight.sum().clamp(min=1.0)
        return loss.mean()


def decode_heatmaps(heatmaps):
    B, K, H, W = heatmaps.shape
    idx = heatmaps.reshape(B, K, -1).argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    hm = heatmaps
    for b in range(B):
        for k in range(K):
            x, y = int(xs[b, k]), int(ys[b, k])
            if 1 <= x < W - 1:
                l, c, r = hm[b, k, y, x - 1], hm[b, k, y, x], hm[b, k, y, x + 1]
                d = (l - 2 * c + r)
                if d.abs() > 1e-6:
                    xs[b, k] = x + 0.5 * (l - r) / d
            if 1 <= y < H - 1:
                u, c, d_ = hm[b, k, y - 1, x], hm[b, k, y, x], hm[b, k, y + 1, x]
                d = (u - 2 * c + d_)
                if d.abs() > 1e-6:
                    ys[b, k] = y + 0.5 * (u - d_) / d
    return torch.stack([xs, ys], dim=-1)


def _gaussian_heatmaps_np(kps, visible, w, h, H, sigma):
    """노트북 LandmarkDataset(train=False) + make_gaussian_heatmaps 를 그대로 옮긴 것 (numpy, float32)."""
    kps = np.asarray(kps, np.float32)
    x0, y0, cw, ch = 0.0, 0.0, float(w), float(h)
    coords = np.stack([(kps[:, 0] - x0) / cw * H, (kps[:, 1] - y0) / ch * H], axis=1)
    K = coords.shape[0]
    hm = np.zeros((K, H, H), dtype=np.float32)
    weight = np.asarray(visible).astype(np.float32).copy()
    ax = np.arange(H, dtype=np.float32)
    for k in range(K):
        if not visible[k]:
            continue
        x, y = coords[k]
        if not (0 <= x < H and 0 <= y < H):
            weight[k] = 0.0
            continue
        gx = np.exp(-((ax - x) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((ax - y) ** 2) / (2 * sigma ** 2))
        hm[k] = np.outer(gy, gx)
    return hm, weight


def gaussian_targets(kps, vis, size, H, sigma, device):
    """kps [B,K,2] px → heatmap [B,K,H,H], weight [B,K]. Cell 12 처럼 fp16 으로 한 번 반올림한다."""
    out = [_gaussian_heatmaps_np(kps[i], vis[i], size[i, 0], size[i, 1], H, sigma) for i in range(len(kps))]
    hm = torch.from_numpy(np.stack([o[0] for o in out])).half().float().to(device)
    w = torch.from_numpy(np.stack([o[1] for o in out])).to(device)
    return hm, w


# =============================================================================
# 5. Splits (노트북 split_records / Cell 12 와 동일)
# =============================================================================
def split_indices(n_all, seed, train_ratio=0.6, val_ratio=0.2, exclude=()):
    idx = np.random.default_rng(seed).permutation(n_all)
    n_tr, n_va = int(n_all * train_ratio), int(n_all * val_ratio)
    ex = set(exclude)
    f = lambda a: np.array([i for i in a if i not in ex], dtype=int)
    return f(idx[:n_tr]), f(idx[n_tr:n_tr + n_va]), f(idx[n_tr + n_va:])


def subset_indices(tr_idx, seed, frac, draw, min_train=8):
    if frac >= 1.0:
        return np.sort(tr_idx)
    n = max(int(round(frac * len(tr_idx))), min_train)
    rng = np.random.default_rng([int(seed), int(round(frac * 1000)), int(draw)])
    return np.sort(rng.choice(tr_idx, size=n, replace=False))


def init_seed_of(seed, frac, draw):
    return int(seed) * 1000 + int(round(frac * 100)) * 10 + int(draw)


# =============================================================================
# 6. Token cache
# =============================================================================
def _load_view(path, flip, input_size):
    with Image.open(path) as im:
        im = im.convert("L").convert("RGB")
        im.load()
    if flip:
        im = ImageOps.mirror(im)
    im = im.resize((input_size, input_size), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)


class _ViewDS(torch.utils.data.Dataset):
    def __init__(self, files, flips, input_size):
        self.files, self.flips, self.s = files, flips, input_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return _load_view(self.files[i], self.flips[i], self.s)


@torch.no_grad()
def build_token_cache(extractor, spec, arr, input_size, cfg: EvalConfig):
    files = [r["file"] for r in spec.records]
    ld = torch.utils.data.DataLoader(_ViewDS(files, arr["flip"], input_size), cfg.cache_batch,
                                     shuffle=False, num_workers=cfg.num_workers)
    toks = [extractor(x)["patch"].half().cpu() for x in ld]
    return torch.cat(toks)


@torch.no_grad()
def normalize_tokens(tok, bs=64):
    out = torch.empty_like(tok)
    for s in range(0, tok.shape[0], bs):
        out[s:s + bs] = F.normalize(tok[s:s + bs].float(), dim=-1).half()
    return out


@torch.no_grad()
def gt_features(tok_n, grid, arr, device, bs=64):
    kps, size = arr["kps"], arr["size"]
    N = tok_n.shape[0]
    out = []
    for s in range(0, N, bs):
        e = min(s + bs, N)
        t = tok_n[s:e].to(device).float()
        fmap = t.transpose(1, 2).reshape(e - s, -1, grid, grid)
        gx = kps[s:e, :, 0] / size[s:e, 0:1] * 2 - 1
        gy = kps[s:e, :, 1] / size[s:e, 1:2] * 2 - 1
        g = torch.from_numpy(np.stack([gx, gy], -1)).float().to(device)[:, None]
        f = F.grid_sample(fmap, g, mode="bilinear", align_corners=False)[:, :, 0]
        out.append(F.normalize(f.transpose(1, 2), dim=-1).cpu())
    return torch.cat(out)


# =============================================================================
# 7. Label-efficiency (노트북 Cell 12 와 동일한 학습 절차)
# =============================================================================
@torch.no_grad()
def _predict(decoder, tok, idxs, arr, cfg, device, bs=32):
    decoder.eval()
    out = []
    for s in range(0, len(idxs), bs):
        b = idxs[s:s + bs]
        hm = decoder(tok[b].to(device).float())
        c = decode_heatmaps(hm.float().cpu()).numpy()
        size = arr["size"][b]
        out.append(np.stack([c[..., 0] / cfg.heatmap_size * size[:, None, 0],
                             c[..., 1] / cfg.heatmap_size * size[:, None, 1]], -1))
    return np.concatenate(out)


def _train_decoder(tok, grid, embed_dim, K, arr, train_idx, val_idx, init_seed, cfg, device):
    torch.manual_seed(init_seed)
    np_rng = np.random.default_rng(init_seed)
    decoder = HeatmapDecoder(embed_dim, grid, K, cfg.heatmap_size).to(device)
    crit = AdaptiveWingLoss()
    opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step):
        if step < cfg.warmup_steps:
            return cfg.lr * step / cfg.warmup_steps
        p = (step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * p))

    best_val, best_state, order, ptr = math.inf, None, np.array([], int), 0
    bs = min(cfg.batch_size, len(train_idx))
    for step in range(cfg.total_steps):
        if ptr + bs > len(order):
            order, ptr = np_rng.permutation(train_idx), 0
        b = order[ptr:ptr + bs]
        ptr += bs
        decoder.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        hm, w = gaussian_targets(arr["kps"][b], arr["vis"][b], arr["size"][b],
                                 cfg.heatmap_size, cfg.sigma, device)
        pred = decoder(tok[b].to(device, non_blocking=True).float())
        loss = crit(torch.sigmoid(pred), hm, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        opt.step()
        if (step + 1) % cfg.val_every == 0 or step == cfg.total_steps - 1:
            pv = _predict(decoder, tok, val_idx, arr, cfg, device)
            err = np.linalg.norm(pv - arr["kps"][val_idx], axis=-1)
            v = float(err[arr["vis"][val_idx]].mean())
            if v < best_val:
                best_val, best_state = v, copy.deepcopy(decoder.state_dict())
    decoder.load_state_dict(best_state)
    return decoder, best_val


def _smean(x):
    return float(np.mean(x)) if np.size(x) else np.nan


def _smed(x):
    return float(np.median(x)) if np.size(x) else np.nan


# =============================================================================
# 8. Margin (노트북 Cell 13 / 13b 일반화)
# =============================================================================
@torch.no_grad()
def _margins(tok_n, gtf, grid, spec, arr, train_idx, test_idx, device, bs=16):
    K = len(spec.kp_names)
    vis = arr["vis"]
    mu = []
    for k in range(K):
        sel = train_idx[vis[train_idx, k]]
        mu.append(gtf[sel, k].mean(0) if len(sel) else torch.zeros(gtf.shape[-1]))
    mu = F.normalize(torch.stack(mu), dim=-1).to(device)                          # [K,D]
    rr, cc = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    centers = torch.from_numpy(np.stack([cc.ravel() + 0.5, rr.ravel() + 0.5], -1)).float().to(device)
    struct = np.array(spec.kp_structure)
    ktype = np.array(spec.kp_type)
    other_struct = torch.from_numpy(struct[:, None] != struct[None, :]).to(device)          # [K(j),K(k)]
    same_type_other = torch.from_numpy((ktype[:, None] == ktype[None, :])
                                       & (struct[:, None] != struct[None, :])).to(device)
    eye = torch.eye(K, dtype=torch.bool, device=device)
    names = np.array(spec.kp_names)
    frames = []
    for s in range(0, len(test_idx), bs):
        b = test_idx[s:s + bs]
        B = len(b)
        S = tok_n[b].to(device).float() @ mu.T                                   # [B,T,K]
        size_b = arr["size"][b]
        gt_uv = torch.from_numpy(arr["kps"][b] / size_b[:, None, :] * grid).float().to(device)
        vis_b = torch.from_numpy(vis[b]).to(device)
        f_gt = gtf[b].to(device)
        c_gt = (f_gt * mu[None]).sum(-1)                                        # [B,K]
        P = f_gt @ mu.T                                                          # [B,a,c]: feat@GT_a · proto_c
        m_point = c_gt - P.masked_fill(eye[None], -1e9).max(-1).values        # feat@GT_k vs 다른 proto
        Pv = P.masked_fill(~vis_b[:, :, None], -1e9)
        def lm_margin(mask):
            mx = Pv.masked_fill(~mask[None], -1e9).max(1).values                # [B,k]
            return torch.where(mx > -1e8, c_gt - mx, torch.nan)
        m_lm_other = lm_margin(other_struct)
        m_lm_sametype = lm_margin(same_type_other)

        dist = torch.cdist(centers[None].expand(B, -1, -1), gt_uv)               # [B,T,K]
        cols = {}
        for r in spec.margin_radii:
            mx = S.masked_fill(dist <= r, -1e9).max(1).values
            cols[f"m_far_R{r:g}"] = torch.where(mx > -1e8, c_gt - mx, torch.nan)
        arg_all = S.argmax(1)
        scale = torch.from_numpy(size_b / grid).float().to(device)[:, None, :]
        zs_px = ((centers[arg_all] - gt_uv) * scale).norm(dim=-1)
        diag = torch.from_numpy(np.hypot(size_b[:, 0], size_b[:, 1])).float().to(device)[:, None]
        # distractor: distractor_radius 밖 최고 유사도 위치가 어느 landmark 근처인가
        mxo, argo = S.masked_fill(dist <= spec.distractor_radius_tok, -1e9).max(1)
        d2 = torch.cdist(centers[argo], gt_uv).masked_fill(eye[None], 1e9)
        d2 = d2.masked_fill(~vis_b[:, None, :], 1e9)
        near_d, near_j = d2.min(-1)

        cpu = lambda t: t.detach().cpu().numpy()
        c_gt_, m_point_, m_o, m_st = cpu(c_gt), cpu(m_point), cpu(m_lm_other), cpu(m_lm_sametype)
        zs_px_, zs_n = cpu(zs_px), cpu(zs_px / diag)
        near_d_, near_j_ = cpu(near_d), cpu(near_j)
        cols_ = {k: cpu(v) for k, v in cols.items()}
        ii, kk = np.nonzero(vis[b])
        jj = near_j_[ii, kk]
        dtype_ = np.where(near_d_[ii, kk] > spec.landmark_near_tok, "non_landmark",
                 np.where(struct[jj] == struct[kk], "same_structure",
                 np.where(ktype[jj] == ktype[kk], "same_type_other_structure", "other_landmark")))
        df = pd.DataFrame({"image_index": b[ii], "landmark": names[kk],
                           "c_gt": c_gt_[ii, kk], "m_point": m_point_[ii, kk],
                           "m_lm_other": m_o[ii, kk], "m_lm_sametype": m_st[ii, kk],
                           "zs_err_px": zs_px_[ii, kk], "zs_err_norm": zs_n[ii, kk],
                           "distractor_type": dtype_, "distractor_landmark": names[jj]})
        for k, v in cols_.items():
            df[k] = v[ii, kk]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 9. Pipeline
# =============================================================================
KEYS = ["model", "split_seed", "fraction", "draw", "image_index", "landmark"]


def _rkey(m, s, f, d):
    return f"{m}|{int(s)}|{float(f):.2f}|{int(d)}"


def _done_keys(path):
    if not os.path.isfile(path):
        return set()
    d = pd.read_csv(path, usecols=["model", "split_seed", "fraction", "draw"])
    return {_rkey(*r) for r in d.itertuples(index=False)}


def _append(df, path):
    df.to_csv(path, mode="a", header=not os.path.isfile(path), index=False)


def run_pipeline(spec: DatasetSpec, backbones: Dict[str, Tuple[Callable, int]], out_dir: str,
                 cfg: EvalConfig = EvalConfig(), arr: Optional[Dict] = None):
    """모델마다 token 캐시 1회 → label-efficiency 학습/평가 + margin 계산. 끝난 run 은 건너뛴다."""
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    arr = arr or prepare(spec)
    N, K = arr["vis"].shape
    runs_csv = os.path.join(out_dir, "le_runs.csv")
    inst_csv = os.path.join(out_dir, "le_instances.csv")
    img_csv = os.path.join(out_dir, "le_image_metrics.csv")
    mg_csv = os.path.join(out_dir, "margin_instances.csv")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"dataset": spec.name, "notes": spec.notes, "n_images": N, "n_landmarks": K,
                   "exclude": sorted(spec.exclude), "cat_t": spec.cat_t,
                   "margin_radii": list(spec.margin_radii), "primary_margin": spec.primary_margin,
                   "backbones": {k: v[1] for k, v in backbones.items()},
                   "cfg": {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.__dict__.items()}},
                  f, indent=2, default=str)

    configs = [(s, f, d) for s in cfg.seeds for f in cfg.fractions for d in range(cfg.n_draws[f])]
    mg_configs = [(s, f, d) for s in cfg.seeds for f in cfg.fractions for d in range(cfg.n_draws[f])]
    splits = {s: split_indices(N, s, cfg.train_ratio, cfg.val_ratio, spec.exclude) for s in cfg.seeds}
    t_all = time.time()
    for model, (loader_fn, input_size) in backbones.items():
        done_le = _done_keys(runs_csv)
        done_mg = _done_keys(mg_csv)
        todo_le = [c for c in configs if cfg.run_label_efficiency and _rkey(model, *c) not in done_le]
        todo_mg = [c for c in mg_configs if cfg.run_margin and _rkey(model, *c) not in done_mg]
        if not todo_le and not todo_mg:
            print(f"[{spec.name}] {model}: all done")
            continue
        t0 = time.time()
        extractor = loader_fn()
        grid = extractor.grid_size(input_size)
        embed_dim = extractor.embed_dim
        tok = build_token_cache(extractor, spec, arr, input_size, cfg)
        del extractor
        gc.collect(); torch.cuda.empty_cache()
        print(f"[{spec.name}] {model}: tokens {tuple(tok.shape)} grid={grid} ({time.time() - t0:.0f}s)")

        for (s, f, d) in todo_le:
            t1 = time.time()
            tr, va, te = splits[s]
            sub = subset_indices(tr, s, f, d, cfg.min_train)
            dec, best_val = _train_decoder(tok, grid, embed_dim, K, arr, sub, va,
                                           init_seed_of(s, f, d), cfg, device)
            pred = _predict(dec, tok, te, arr, cfg, device)
            gt = arr["kps"][te]
            err = np.linalg.norm(pred - gt, axis=-1)
            diag = np.hypot(arr["size"][te, 0], arr["size"][te, 1])[:, None]
            errn = err / diag
            vis = arr["vis"][te]
            cat = errn > spec.cat_t
            grp = arr["group"][te][:, None].repeat(K, 1)
            row = {"model": model, "split_seed": s, "fraction": f, "draw": d,
                   "n_train": len(sub), "n_test_images": len(te), "val_MRE_px": best_val,
                   "MRE_px": _smean(err[vis]), "median_px": _smed(err[vis]),
                   "catastrophic_rate": _smean(cat[vis]), "n_catastrophic": int(cat[vis].sum()),
                   "catastrophic_contrib_px": float(err[vis & cat].sum() / max(vis.sum(), 1)),
                   "inlier_median_px": _smed(err[vis & ~cat]), "inlier_mean_px": _smean(err[vis & ~cat]),
                   "sec": round(time.time() - t1, 1)}
            for g in np.unique(arr["group"]):
                m = vis & (grp == g)
                row[f"MRE_px_{g}"] = _smean(err[m])
                row[f"catastrophic_rate_{g}"] = _smean(cat[m])
            ii, kk = np.nonzero(vis)
            inst = pd.DataFrame({"model": model, "split_seed": s, "fraction": f, "draw": d,
                                 "image_index": te[ii], "landmark": np.array(spec.kp_names)[kk],
                                 "group": arr["group"][te][ii], "error_px": err[ii, kk],
                                 "error_norm": errn[ii, kk], "catastrophic": cat[ii, kk],
                                 "failure_2pct": errn[ii, kk] > 0.02,
                                 "pred_x": pred[ii, kk, 0], "pred_y": pred[ii, kk, 1]})
            if spec.image_metrics is not None:
                im_rows = []
                for n_i, gi in enumerate(te):
                    mtr = spec.image_metrics(pred[n_i], gt[n_i], vis[n_i], spec.records[gi].get("meta", {}))
                    if mtr:
                        im_rows.append({"model": model, "split_seed": s, "fraction": f, "draw": d,
                                        "image_index": gi, "group": arr["group"][gi], **mtr})
                if im_rows:
                    imdf = pd.DataFrame(im_rows)
                    _append(imdf, img_csv)
                    for c in [c for c in imdf.columns if c.startswith("abs_err_")]:
                        row[f"mean_{c}"] = imdf[c].mean()
                        row[f"median_{c}"] = imdf[c].median()
            _append(pd.DataFrame([row]), runs_csv)
            _append(inst, inst_csv)
            print(f"  {model:6s} seed {s} frac {f:.2f} draw {d} n_train={len(sub):3d} | "
                  f"MRE {row['MRE_px']:.2f} median {row['median_px']:.2f} "
                  f"cat {row['n_catastrophic']} ({row['sec']:.0f}s)")
            del dec

        if todo_mg:
            tok_n = normalize_tokens(tok)
            gtf = gt_features(tok_n, grid, arr, device)
            for (s, f, d) in todo_mg:
                tr, _, te = splits[s]
                mg = _margins(tok_n, gtf, grid, spec, arr, subset_indices(tr, s, f, d, cfg.min_train),
                              te, device)
                for c, v in zip(["draw", "fraction", "split_seed", "model"], [d, f, s, model]):
                    mg.insert(0, c, v)
                _append(mg, mg_csv)
            print(f"  {model}: margins done ({len(todo_mg)} configs)")
            del tok_n, gtf
        del tok
        gc.collect(); torch.cuda.empty_cache()
    print(f"[{spec.name}] pipeline finished in {(time.time() - t_all) / 60:.1f} min -> {out_dir}")
    return arr


# =============================================================================
# 10. Analysis
# =============================================================================
def _auc(y, score):
    ok = np.isfinite(score)
    y, score = np.asarray(y)[ok], np.asarray(score)[ok]
    if roc_auc_score is None or y.sum() < 5 or (~y).sum() < 5:
        return np.nan
    return float(roc_auc_score(y, score))


def _cluster_boot_auc(y, score, clusters, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    codes = pd.factorize(clusters)[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(codes.max() + 1)]
    n = len(groups)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, n, n)])
        vals.append(_auc(y[idx], score[idx]))
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    return tuple(np.percentile(vals, [2.5, 97.5])) if len(vals) > 50 else (np.nan, np.nan)


def analyze(spec: DatasetSpec, out_dir: str, cfg: EvalConfig = EvalConfig(),
            models: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    res = {}
    runs = pd.read_csv(os.path.join(out_dir, "le_runs.csv"))
    runs["fraction"] = runs["fraction"].round(2)
    models = models or list(dict.fromkeys(runs["model"]))
    base = cfg.baseline if cfg.baseline in models else models[0]

    # --- A. label-efficiency summary
    mets = [c for c in ["MRE_px", "median_px", "catastrophic_rate", "n_catastrophic",
                        "inlier_median_px", "catastrophic_contrib_px"] if c in runs.columns]
    mets += [c for c in runs.columns if c.startswith(("mean_abs_err_", "MRE_px_"))]
    res["A_le_summary"] = (runs.groupby(["fraction", "model"])[mets].agg(["mean", "std"])
                           .reindex(pd.MultiIndex.from_product([sorted(runs.fraction.unique()), models])))
    res["A_le_catastrophic_total"] = runs.pivot_table(index="fraction", columns="model",
                                                      values="n_catastrophic", aggfunc="sum"
                                                      ).reindex(columns=models)

    # --- B. paired vs baseline (같은 subset/초기값)
    K_ = ["split_seed", "fraction", "draw"]
    b = runs[runs.model == base].set_index(K_)
    rows = []
    for m in [x for x in models if x != base]:
        c = runs[runs.model == m].set_index(K_)
        for f in sorted(runs.fraction.unique()):
            idx = [k for k in b.index.intersection(c.index) if np.isclose(k[1], f)]
            if not idx:
                continue
            row = {"model": m, "fraction": f, "n_runs": len(idx)}
            for met in [x for x in ["MRE_px", "median_px", "catastrophic_rate", "n_catastrophic"]
                        + [c for c in runs.columns if c.startswith("mean_abs_err_")] if x in runs.columns]:
                dlt = c.loc[idx, met].to_numpy() - b.loc[idx, met].to_numpy()
                row[f"d_{met}_mean"] = dlt.mean()
                row[f"d_{met}_n_better"] = int((dlt < 0).sum())
            rows.append(row)
    res["B_paired_vs_baseline"] = pd.DataFrame(rows)

    # --- C. margin ↔ 대형 실패
    mg_path = os.path.join(out_dir, "margin_instances.csv")
    if os.path.isfile(mg_path):
        inst = pd.read_csv(os.path.join(out_dir, "le_instances.csv"))
        mg = pd.read_csv(mg_path)
        for t in (inst, mg):
            t["fraction"] = t["fraction"].round(2)
        J = inst.merge(mg, on=KEYS, how="inner")
        if spec.analysis_group is not None:
            J = J[J["group"] == spec.analysis_group]
        J["cluster"] = J["split_seed"].astype(str) + "_" + J["image_index"].astype(str)
        L = J[J["fraction"].isin([round(f, 2) for f in cfg.low_label_fracs])].copy()
        margin_cols = ["c_gt", "m_point", "m_lm_other", "m_lm_sametype"] + \
                      [c for c in mg.columns if c.startswith("m_far_R")]
        sign = {c: -1 for c in margin_cols}
        sign["zs_err_norm"] = 1
        rows = []
        for m in models:
            s_ = L[L.model == m]
            y = s_["catastrophic"].to_numpy(bool)
            row = {"model": m, "n": len(s_), "n_catastrophic": int(y.sum())}
            for c in margin_cols + ["zs_err_norm"]:
                row[f"AUROC_{c}"] = _auc(y, sign[c] * s_[c].to_numpy())
            pc = spec.primary_margin
            lo, hi = _cluster_boot_auc(y, sign[pc] * s_[pc].to_numpy(), s_["cluster"].to_numpy(), cfg.n_boot)
            row[f"AUROC_{pc}_ci_low"], row[f"AUROC_{pc}_ci_high"] = lo, hi
            rows.append(row)
        res["C_margin_auroc"] = pd.DataFrame(rows)

        rows = []
        for m in models:
            s_ = L[L.model == m]
            row = {"model": m, "decoder_catastrophic_pct": s_["catastrophic"].mean() * 100,
                   "zs_catastrophic_pct": (s_["zs_err_norm"] > spec.cat_t).mean() * 100,
                   "zs_median_err_px": s_["zs_err_px"].median()}
            for c in ["m_lm_other", "m_lm_sametype"] + [c for c in mg.columns if c.startswith("m_far_R")]:
                row[f"neg_pct_{c}"] = (s_[c] < 0).mean() * 100
            rows.append(row)
        res["D_model_level"] = pd.DataFrame(rows)

        PK = ["split_seed", "fraction", "draw", "image_index", "landmark"]
        e0 = L[L.model == base].set_index(PK)
        rows = []
        for m in [x for x in models if x != base]:
            cm = L[L.model == m].set_index(PK)
            idx = e0.index.intersection(cm.index)
            a, bb = e0.loc[idx, "catastrophic"].to_numpy(bool), cm.loc[idx, "catastrophic"].to_numpy(bool)
            grp = np.select([a & ~bb, ~a & bb], ["rescued", "newly_failed"], "other")
            for c in [spec.primary_margin, "m_lm_other", "m_lm_sametype"]:
                dlt = cm.loc[idx, c].to_numpy() - e0.loc[idx, c].to_numpy()
                for g in ["rescued", "newly_failed"]:
                    mk = grp == g
                    if mk.any():
                        rows.append({"model": m, "metric": c, "group": g, "n": int(mk.sum()),
                                     "frac_margin_up": float(np.nanmean(dlt[mk] > 0))})
        res["E_paired_margin"] = pd.DataFrame(rows)

        rows = []
        for m in models:
            s_ = L[L.model == m]
            for name, mk in [("catastrophic", s_["catastrophic"].to_numpy(bool)),
                             ("all", np.ones(len(s_), bool))]:
                vc = s_.loc[mk, "distractor_type"].value_counts(normalize=True)
                rows.append({"model": m, "subset": name, "n": int(mk.sum()),
                             **{k: round(v, 3) for k, v in vc.items()}})
        res["F_distractor_types"] = pd.DataFrame(rows).fillna(0)
        J.to_csv(os.path.join(out_dir, "margin_linked_instances.csv"), index=False)

    for k, v in res.items():
        v.to_csv(os.path.join(out_dir, f"{k}.csv"))
    print(f"[analyze] {spec.name}: saved {list(res)} -> {out_dir}")
    return res


# =============================================================================
# 11. Case figures (노트북 Cell 14 일반화)
# =============================================================================
def spine_vertebra_offset(spec: DatasetSpec, out_dir: str, arr: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    spine 전용. decoder 예측이 '몇 번째 척추뼈'의 같은 모서리에 가장 가까운지로 척추뼈 번호 오차를 센다.
      offset = (예측이 가장 가까운 척추뼈 번호) − (정답 척추뼈 번호),  0 = 올바른 척추뼈
    le_instances.csv 에 pred_x/pred_y 가 있어야 한다 (이 버전 이후 실행분).
    """
    arr = arr or prepare(spec)
    I = pd.read_csv(os.path.join(out_dir, "le_instances.csv"))
    if "pred_x" not in I.columns:
        raise ValueError("le_instances.csv 에 pred_x/pred_y 가 없습니다. 수정된 모듈로 다시 실행하세요.")
    I["fraction"] = I["fraction"].round(2)
    kidx = {n: i for i, n in enumerate(spec.kp_names)}
    k = I["landmark"].map(kidx).to_numpy()
    vtrue, corner = k // 4, k % 4
    gi = I["image_index"].to_numpy()
    cand = arr["kps"][gi].reshape(len(I), 17, 4, 2)[np.arange(len(I)), :, corner]      # [n,17,2]
    cvis = arr["vis"][gi].reshape(len(I), 17, 4)[np.arange(len(I)), :, corner]         # [n,17]
    pr = I[["pred_x", "pred_y"]].to_numpy()[:, None, :]
    d = np.linalg.norm(cand - pr, axis=-1)
    d[~cvis] = np.inf
    I["vert_offset"] = d.argmin(1) - vtrue
    I["offset_cls"] = pd.cut(I["vert_offset"].abs(), [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"])
    I.to_csv(os.path.join(out_dir, "spine_vertebra_offset_instances.csv"), index=False)
    tab = (I.groupby(["fraction", "model"])["offset_cls"].value_counts(normalize=True)
             .mul(100).unstack().round(1))
    tab_cat = (I[I["catastrophic"]].groupby(["fraction", "model"])["offset_cls"]
                 .value_counts(normalize=True).mul(100).unstack().round(1))
    right = I[I["vert_offset"] == 0]
    prec = (right.groupby(["fraction", "model"])["error_px"].median().unstack().round(1))
    out = {"offset_all_pct": tab, "offset_in_catastrophic_pct": tab_cat,
           "median_err_when_right_vertebra_px": prec}
    for n_, v in out.items():
        v.to_csv(os.path.join(out_dir, f"S_{n_}.csv"))
    return out


def plot_cases(spec: DatasetSpec, backbones: Dict[str, Tuple[Callable, int]], out_dir: str,
               cfg: EvalConfig = EvalConfig(), arr: Optional[Dict] = None, n_cases: int = 8,
               seed: Optional[int] = None):
    """대형 실패 instance 중 primary margin 이 가장 낮은 사례 n_cases 개의 prototype 유사도 지도."""
    import matplotlib
    import matplotlib.pyplot as plt
    arr = arr or prepare(spec)
    J = pd.read_csv(os.path.join(out_dir, "margin_linked_instances.csv"))
    L = J[J["fraction"].isin([round(f, 2) for f in cfg.low_label_fracs])]
    pc = spec.primary_margin
    L = L.assign(pct=L.groupby("model")[pc].rank(pct=True))
    agg = (L.groupby(["image_index", "landmark"])
             .agg(any_cat=("catastrophic", "max"), pct=("pct", "mean"), seed=("split_seed", "first"))
             .reset_index())
    cases = agg[agg.any_cat].sort_values("pct").head(n_cases)
    if cases.empty:
        print("[plot_cases] 대형 실패 사례 없음")
        return
    fig_dir = os.path.join(out_dir, "case_figures")
    os.makedirs(fig_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    N = len(spec.records)
    kidx = {n: i for i, n in enumerate(spec.kp_names)}
    maps = {}
    for model, (loader_fn, input_size) in backbones.items():
        ex = loader_fn()
        grid = ex.grid_size(input_size)
        tok_n = normalize_tokens(build_token_cache(ex, spec, arr, input_size, cfg))
        del ex
        gtf = gt_features(tok_n, grid, arr, device)
        for s_ in sorted(cases.seed.unique()):
            tr, _, _ = split_indices(N, int(s_), cfg.train_ratio, cfg.val_ratio, spec.exclude)
            for _, r in cases[cases.seed == s_].iterrows():
                k = kidx[r.landmark]
                sel = tr[arr["vis"][tr, k]]
                mu = F.normalize(gtf[sel, k].mean(0), dim=0)
                S = (tok_n[int(r.image_index)].float() @ mu).numpy().reshape(grid, grid)
                maps[(model, int(r.image_index), r.landmark)] = (S, float(gtf[int(r.image_index), k] @ mu), grid)
        del tok_n, gtf
        gc.collect(); torch.cuda.empty_cache()
    models = list(backbones)
    per = L.groupby(["image_index", "landmark", "model"]).agg(cat_rate=("catastrophic", "mean"),
                                                                err_med=("error_px", "median"))
    for _, r in cases.iterrows():
        i, lm = int(r.image_index), r.landmark
        rec = spec.records[i]
        with Image.open(rec["file"]) as im:
            img = im.convert("L")
            if rec.get("flip", False):
                img = ImageOps.mirror(img)
            img = np.asarray(img)
        h, w = img.shape
        k = kidx[lm]
        gx, gy = arr["kps"][i, k]
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2 * h / w + 0.8))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, models):
            S, c_gt, G = maps[(m, i, lm)]
            ax.imshow(img, cmap="gray")
            ax.imshow(S, cmap="jet", alpha=0.45, vmin=np.percentile(S, 5), vmax=S.max(),
                      extent=[0, w, h, 0], interpolation="bilinear")
            X, Y = (np.arange(G) + 0.5) / G * w, (np.arange(G) + 0.5) / G * h
            if S.max() > c_gt:
                ax.contour(X, Y, S, levels=[c_gt], colors="white", linewidths=1.0)
            for j in range(len(spec.kp_names)):
                if arr["vis"][i, j] and j != k:
                    ax.plot(*arr["kps"][i, j], "o", ms=3, mfc="none", mec="lime", mew=0.8)
            ax.plot(gx, gy, "*", ms=13, mfc="yellow", mec="black", mew=0.7)
            a = np.unravel_index(S.argmax(), S.shape)
            ax.plot((a[1] + 0.5) / G * w, (a[0] + 0.5) / G * h, "x", color="red", ms=10, mew=2.2)
            try:
                st = per.loc[(i, lm, m)]
                info = f"cat {st['cat_rate']:.0%} err_med {st['err_med']:.0f}px"
            except KeyError:
                info = ""
            ax.set_title(f"{m} | c_gt {c_gt:.3f}\n{info}", fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{spec.name} | img {i} {os.path.basename(rec['file'])} | {lm} | group {rec.get('group')}"
                     f"\n★ GT  × zero-shot argmax  white = similarity ≥ c_gt", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"img{i}_{lm}.png"), dpi=110)
        plt.close(fig)
    print(f"[plot_cases] {len(cases)} figures -> {fig_dir}")
