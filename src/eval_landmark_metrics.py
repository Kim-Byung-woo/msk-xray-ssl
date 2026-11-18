r"""
지정한 모델들의 landmark 표현 지표 3가지를 계산한다 — 고관절(MTDDH) / 족부(HVAngleEst) / 척추(AASCE)

  1. 클러스터링 (cluster)
     GT 가 속한 patch token 1개 → L2 정규화 → 768차원(DINOv2 384) feature 그대로
       kNN Acc   : k=5, StratifiedGroupKFold 5 (group = 영상 → 같은 영상의 점이 train/test 에 섞이지 않음)
       Silhouette: 같은 landmark 끼리 뭉치고 다른 landmark 와 떨어지는가 (점이 1만 개 넘으면 표본 1만 개)
     t-SNE 는 그림으로만 쓴다 (지표는 t-SNE 좌표가 아니라 원래 feature 로 계산).
     03 노트북 Cell L 과 같은 계산 → 기존 landmark_clustering_dense_patch*.csv 와 같은 값이 나와야 한다.
     원인 분석용: landmark 별 kNN 정확도·Silhouette, 혼동 행렬(어느 landmark 와 헷갈리는가).
     ※ 고관절은 standard 자세 영상이 주 결과. 모든 자세(standard + spread) 값도 계산하지만
       기존 결과(모든 자세로 계산됨)와 대조하는 용도로만 쓴다 (cluster.csv 의 image_group 열).

  2. Data efficiency (de)
     label 5 / 10 / 25 / 100 % (draw 0), seed 42/43/44 → MRE(px)·오탐 % 를 seed 평균 ± SD. 고관절은 standard 자세만.
     decoder 설정은 옵션으로 바꿀 수 있다: --de-steps (기본 12000), --de-decoder heatmap|linear, --de-hidden (기본 256)
       - 기본 설정이면 기존 결과({부위}_full12k)를 읽기만 한다 (학습 없음, GPU·가중치 불필요).
       - --de-steps 3000 (heatmap, hidden 256) 이면 기존 3000 step 결과({부위}/, draw 3/3/2/1)를 읽는다.
         draw 가 여러 개면 seed 안에서 draw 평균 → seed 평균 ± SD (03 노트북 Cell S2 와 같은 집계).
         --train-de 로 새로 학습할 때도 이 두 설정은 기존과 같은 draw 수로 학습한다 (12000: 1개, 3000: 3/3/2/1).
       - 그 밖의 설정이면 기존 결과는 쓰지 않는다. --train-de 로 새 폴더(<tag>/<부위>_de_<설정>)에 학습한 결과만 쓴다.
       - 이전 실행에서 학습한 비기본 설정 결과는 --de-source <이전 tag> 로 다시 읽는다 (de_setting.json 이 같을 때만).
     완료 판정: run 이 le_runs.csv 에 있고, le_instances 의 (영상, landmark) 쌍이 중복 없이
       그 seed 의 test split visible 쌍 집합과 정확히 같아야 완료 (행 수만 같은 run 은 인정하지 않음).
       (모델, label 비율) 칸마다 seed 3개가 모두 완료된 폴더 하나를 골라 그 폴더의 run 만 쓴다 (폴더·run 을 섞지 않음).
       그런 폴더가 없으면 그 칸은 "미완료 (n/3 seed)" — 부분 seed 평균은 계산하지 않는다.
     원인 분석용: landmark 별 MRE, 중앙값(소수의 큰 오차가 MRE 를 끌어올리는지), 기준 모델과 seed 별 짝 비교.

  3. Zero-shot error (zeroshot)
     decoder 없이: train 영상 GT 위치 feature(bilinear, L2 정규화) 평균 → landmark 별 기준 feature
       → test 영상의 모든 patch 와 cosine 유사도 → 최댓값 patch 중심 = 예측
       → 오차 > 영상 대각선 5% 이면 오탐. seed 42/43/44 split 마다 계산 → seed 평균 ± SD. 고관절은 standard 자세만.
     기준 feature 를 만드는 train 영상 두 가지를 모두 계산한다.
       train_full    : train split 전체                     (주 지표, 기존 margin_instances.csv fraction=1.0 과 같음)
       label_5_10    : label 5% / 10% 부분집합 (draw 0)       (기존 D_model_level.csv zs_catastrophic_pct 와 같음)
     원인 분석용: landmark 별 오탐 %, 오탐일 때 예측이 간 곳
       (same_type_other_structure = 다른 구조물의 같은 종류 landmark, 예: 반대쪽 고관절 / other_landmark /
        same_structure / non_landmark), margin 음수 비율.

기존 결과 폴더(src/outputs/ssl/landmark_eval/)는 읽기만 한다. 결과는 새 폴더에만 쓴다:
  src/outputs/ssl/metrics_py/<tag>/   (이미 있으면 멈춘다 — 덮어쓰기 없음)
    run_config.json, summary.md, check_vs_existing.csv
    <부위>/cluster.csv, cluster_per_class.csv, cluster_confusion_<label>.csv, tsne.png, tsne_coords.csv
    <부위>/zeroshot_instances.csv, zeroshot_summary.csv, zeroshot_paired.csv, zeroshot_where.csv, zeroshot_per_landmark.csv
    <부위>/de_summary.csv, de_paired.csv, de_per_landmark.csv
    <부위>_de_<설정>/  (--train-de 로 학습했을 때만: le_runs.csv, le_instances.csv, de_setting.json)

실행 (/workspace/msk 에서):
  # 세 지표 모두 (GPU 필요, 번호는 직접 지정)
  CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python src/eval_landmark_metrics.py \
      --models DINOv3 RAD-DINO D3-E0 D3-E1c D3-E1c-within D3-E2-Adam \
      --parts hip foot spine --metrics cluster de zeroshot --tag 2026-11-16_d3_6models
  # data efficiency 만, 기존 12k 결과 읽기 (GPU 불필요)
  /opt/conda/bin/python src/eval_landmark_metrics.py --models D3-E0 D3-E1c --metrics de --tag de_only
  # decoder 를 바꿔 새로 학습 (GPU, 오래 걸림)
  CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python src/eval_landmark_metrics.py --models D3-E0 D3-E1c \
      --metrics de --de-decoder linear --de-steps 3000 --train-de --tag de_linear_3k
  # 위에서 학습한 결과를 나중에 다시 집계 (GPU 불필요)
  /opt/conda/bin/python src/eval_landmark_metrics.py --models D3-E0 D3-E1c \
      --metrics de --de-decoder linear --de-steps 3000 --de-source de_linear_3k --tag de_linear_3k_re
"""
import argparse
import contextlib
import datetime
import functools
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch          # GPU 는 쓰지 않는다 (GPU 확인은 msk_backbones 를 불러올 때만)

SRC =os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

DATA = "/workspace/msk/data"
EXISTING = os.path.join(SRC, "outputs/ssl/landmark_eval")     # 기존 결과 (읽기만)
OUT_ROOT = os.path.join(SRC, "outputs/ssl/metrics_py")

SEEDS = (42, 43, 44)
DE_FRACS = (0.05, 0.10, 0.25, 1.00)
ZS_DEFS = {"train_full": (1.00,), "label_5_10": (0.05, 0.10)}   # 기준 feature 를 만드는 train 영상
ZS_MAIN = "train_full"
CLU_SEED = 42
SIL_SAMPLE = 10000
TSNE_MAX = 8000
SPINE_EXCLUDE = {394, 397, 420, 445}     # 03 노트북 Cell P 와 같음 (주석 불확실 영상)


# =============================================================================
# 데이터
# =============================================================================
def load_part(LE, part):
    """03 노트북 Cell P 와 같은 설정."""
    if part == "hip":
        spec = LE.load_hip(f"{DATA}/MTDDH/Dataset1/Keypoints/Keypoints_Train.json",
                           f"{DATA}/MTDDH/Dataset1/Keypoints/Train/")
    elif part == "foot":
        spec = LE.load_foot(f"{DATA}/HVAngleEst", canonical_flip=True, flip_label="left")
    elif part == "spine":
        spec = LE.load_spine(f"{DATA}/Spine", split="training")
        spec.exclude = set(SPINE_EXCLUDE)
    else:
        raise ValueError(part)
    return spec, LE.prepare(spec)


def label_sets(part, kp_names):
    """{label 이름: (k → class 번호, landmark 이름 → class 이름)}. label 이름은 기존 csv 와 같다."""
    K = len(kp_names)
    if part == "spine":
        return {f"corner({K})": (lambda k: k, lambda n: n),
                f"vertebra({K // 4})": (lambda k: k // 4, lambda n: n.split("_")[0]),
                "corner_type(4)": (lambda k: k % 4, lambda n: n.split("_")[1])}
    return {f"landmark({K})": (lambda k: k, lambda n: n)}


# =============================================================================
# 1. 클러스터링 (03 노트북 Cell L 과 같은 계산)
# =============================================================================
def nearest_gt_tokens(torch, tok, grid, arr, idx):
    kps, size, vis = arr["kps"][idx], arr["size"][idx], arr["vis"][idx]
    col = np.clip((kps[..., 0] / size[:, None, 0] * grid).astype(int), 0, grid - 1)
    row = np.clip((kps[..., 1] / size[:, None, 1] * grid).astype(int), 0, grid - 1)
    ii, kk = np.nonzero(vis)
    t = torch.as_tensor(row[ii, kk] * grid + col[ii, kk])
    f = tok[torch.as_tensor(idx[ii]), t].float()
    f = torch.nn.functional.normalize(f, dim=-1).numpy()
    return f, idx[ii], kk


def cluster_one(model, part, f, img, kk, kp_names, diagnose=True):
    from sklearn.metrics import (calinski_harabasz_score, confusion_matrix, davies_bouldin_score,
                                 silhouette_samples, silhouette_score)
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    rows, per_class, confusions = [], [], {}
    for lab, (fn, namer) in label_sets(part, kp_names).items():
        y = np.vectorize(fn)(kk)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CLU_SEED)
        knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), f, y, groups=img, cv=cv, scoring="accuracy")
        ss = SIL_SAMPLE if len(y) > SIL_SAMPLE else None
        rows.append({"label": lab, "model": model, "kNN Acc": knn.mean(), "kNN SD": knn.std(),
                     "Silhouette": silhouette_score(f, y, sample_size=ss, random_state=CLU_SEED),
                     "Davies-Bouldin": davies_bouldin_score(f, y),
                     "Calinski-Harabasz": calinski_harabasz_score(f, y),
                     "n_points": len(y), "n_classes": len(np.unique(y)), "silhouette_sampled": ss is not None})
        if not diagnose:
            continue

        # ---- 원인 분석: class 별 kNN 정확도 / 가장 많이 헷갈린 class / class 별 Silhouette
        pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5), f, y, groups=img, cv=cv)
        classes = np.unique(y)
        cname = {c: namer(kp_names[int(np.nonzero(np.vectorize(fn)(np.arange(len(kp_names))) == c)[0][0])])
                 for c in classes}
        cm = confusion_matrix(y, pred, labels=classes)
        cm_pct = 100 * cm / cm.sum(1, keepdims=True)
        names = [cname[c] for c in classes]
        confusions[lab] = pd.DataFrame(cm_pct, index=pd.Index(names, name="true"),
                                       columns=pd.Index(names, name="pred")).assign(model=model)
        sub = (np.random.default_rng(CLU_SEED).choice(len(y), SIL_SAMPLE, replace=False)
               if len(y) > SIL_SAMPLE else np.arange(len(y)))
        sil_s = silhouette_samples(f[sub], y[sub])
        for i, c in enumerate(classes):
            off = cm_pct[i].copy()
            off[i] = -1
            j = int(off.argmax())
            m = y[sub] == c
            per_class.append({"label": lab, "model": model, "class": cname[c], "n": int((y == c).sum()),
                              "kNN_acc_pct": cm_pct[i, i], "most_confused_with": names[j],
                              "most_confused_pct": cm_pct[i, j],
                              "silhouette_mean": float(sil_s[m].mean()) if m.any() else np.nan})
    return rows, per_class, confusions


def tsne_coords(f, kk):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    keep = np.arange(len(kk))
    if len(kk) > TSNE_MAX:                                   # class 비율을 유지해 표본 추출 (03 노트북 Cell J 와 같음)
        rng = np.random.default_rng(CLU_SEED)
        keep = np.sort(np.concatenate([
            rng.choice(np.nonzero(kk == c)[0], size=max(1, round((kk == c).sum() * TSNE_MAX / len(kk))),
                       replace=False) for c in np.unique(kk)]))
    f50 = PCA(n_components=min(50, f.shape[1]), random_state=CLU_SEED).fit_transform(f[keep])
    xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
              random_state=CLU_SEED).fit_transform(f50)
    return xy, kk[keep]


def plot_tsne(part, kp_names, coords, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    if any("NanumGothic" in x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
    models = list(coords)
    n = len(models)
    if part == "spine":                                       # 위: 척추뼈 번호, 아래: 모서리 종류
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8.4), squeeze=False)
        vcm = plt.get_cmap("viridis")
        ccol = ["#e6194b", "#f58231", "#3cb44b", "#4363d8"]
        nv = len(kp_names) // 4
        for c, m in enumerate(models):
            xy, kk = coords[m]
            axes[0, c].scatter(xy[:, 0], xy[:, 1], c=vcm((kk // 4) / max(nv - 1, 1)), s=2, alpha=0.7, linewidths=0)
            axes[1, c].scatter(xy[:, 0], xy[:, 1], c=[ccol[k % 4] for k in kk], s=2, alpha=0.7, linewidths=0)
            axes[0, c].set_title(m, fontsize=11)
        axes[0, 0].set_ylabel("척추뼈 번호 (위→아래)")
        axes[1, 0].set_ylabel("모서리 종류 TL/TR/BL/BR")
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        hd = [plt.Line2D([], [], marker="o", ls="", color=ccol[i], label=t) for i, t in enumerate(["TL", "TR", "BL", "BR"])]
        axes[1, -1].legend(handles=hd, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    else:
        ncol = min(n, 3)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.2 * nrow), squeeze=False)
        cmap = plt.get_cmap("tab10")
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        for ax, m in zip(axes.ravel(), models):
            xy, kk = coords[m]
            ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(k) for k in kk], s=2, alpha=0.7, linewidths=0)
            ax.set_title(m, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
        hd = [plt.Line2D([], [], marker="o", ls="", color=cmap(k), label=nm) for k, nm in enumerate(kp_names)]
        fig.legend(handles=hd, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle(f"{part} — dense_patch t-SNE (지표는 t-SNE 가 아니라 원래 feature 로 계산)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 3. Zero-shot (기준 feature·유사도·margin 은 landmark_eval._margins 를 그대로 사용)
# =============================================================================
def zs_argmax_target(torch, tok_n, gtf, grid, spec, arr, train_idx, test_idx, device, bs=16):
    """유사도 최댓값 위치(= zero-shot 예측)와, 그 위치에서 가장 가까운 '다른' landmark (원인 분석용).
    기준 feature 는 _margins 와 같은 방식으로 만든다 → 예측 위치가 _margins 의 zs_err 와 같다."""
    F = torch.nn.functional
    K = len(spec.kp_names)
    vis = arr["vis"]
    mu = []
    for k in range(K):
        sel = train_idx[vis[train_idx, k]]
        mu.append(gtf[sel, k].mean(0) if len(sel) else torch.zeros(gtf.shape[-1]))
    mu = F.normalize(torch.stack(mu), dim=-1).to(device)
    rr, cc = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    centers = torch.from_numpy(np.stack([cc.ravel() + 0.5, rr.ravel() + 0.5], -1)).float().to(device)
    eye = torch.eye(K, dtype=torch.bool, device=device)
    names, struct, ktype = np.array(spec.kp_names), np.array(spec.kp_structure), np.array(spec.kp_type)
    frames = []
    with torch.no_grad():
        for s in range(0, len(test_idx), bs):
            b = test_idx[s:s + bs]
            S = tok_n[b].to(device).float() @ mu.T                                  # [B,T,K]
            uv = centers[S.argmax(1)]                                               # [B,K,2] token 단위
            size_b = arr["size"][b]
            gt_uv = torch.from_numpy(arr["kps"][b] / size_b[:, None, :] * grid).float().to(device)
            vis_b = torch.from_numpy(vis[b]).to(device)
            d = torch.cdist(uv, gt_uv).masked_fill(eye[None], 1e9).masked_fill(~vis_b[:, None, :], 1e9)
            nd, nj = d.min(-1)                                                      # [B,K]
            uv_, nd_, nj_ = uv.cpu().numpy(), nd.cpu().numpy(), nj.cpu().numpy()
            ii, kk = np.nonzero(vis[b])
            jj = nj_[ii, kk]
            ttype = np.where(nd_[ii, kk] > spec.landmark_near_tok, "non_landmark",
                    np.where(struct[jj] == struct[kk], "same_structure",
                    np.where(ktype[jj] == ktype[kk], "same_type_other_structure", "other_landmark")))
            frames.append(pd.DataFrame({
                "image_index": b[ii], "landmark": names[kk],
                "zs_pred_x": uv_[ii, kk, 0] / grid * size_b[ii, 0], "zs_pred_y": uv_[ii, kk, 1] / grid * size_b[ii, 1],
                "zs_target_type": ttype, "zs_target_landmark": np.where(ttype == "non_landmark", "", names[jj]),
                "zs_target_dist_tok": nd_[ii, kk]}))
    return pd.concat(frames, ignore_index=True)


def zeroshot_one(torch, LE, model, tok, grid, spec, arr, cfg, device):
    tok_n = LE.normalize_tokens(tok)
    gtf = LE.gt_features(tok_n, grid, arr, device)
    N = len(spec.records)
    frames = []
    for s in SEEDS:
        tr, _, te = LE.split_indices(N, s, cfg.train_ratio, cfg.val_ratio, spec.exclude)
        for zdef, fracs in ZS_DEFS.items():
            for f in fracs:
                sub = LE.subset_indices(tr, s, f, 0, cfg.min_train)
                mg = LE._margins(tok_n, gtf, grid, spec, arr, sub, te, device)
                tg = zs_argmax_target(torch, tok_n, gtf, grid, spec, arr, sub, te, device)
                mg = mg.merge(tg, on=["image_index", "landmark"], how="left", validate="one_to_one")
                mg.insert(0, "proto_def", zdef)
                mg.insert(0, "n_proto_images", len(sub))
                mg.insert(0, "fraction", f)
                mg.insert(0, "split_seed", s)
                mg.insert(0, "model", model)
                frames.append(mg)
    del tok_n, gtf
    df = pd.concat(frames, ignore_index=True)
    df["group"] = arr["group"][df["image_index"].to_numpy()]
    df["zs_fp"] = df["zs_err_norm"] > spec.cat_t                                  # 오탐
    return df


def _mean_sd(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else np.nan), len(x)


def zeroshot_tables(Z, spec, baseline):
    if spec.analysis_group is not None:
        Z = Z[Z["group"] == spec.analysis_group]
    pm = spec.primary_margin
    rows = []
    for (m, zdef), g in Z.groupby(["model", "proto_def"], sort=False):
        per_seed = g.groupby("split_seed").agg(fp=("zs_fp", lambda x: 100 * x.mean()),
                                               med=("zs_err_px", "median"))
        mu, sd, n = _mean_sd(per_seed["fp"])
        rows.append({"model": m, "proto_def": zdef, "zs_오탐_pct_mean": mu, "zs_오탐_pct_sd": sd, "n_seed": n,
                     "zs_오탐_pct_pooled": 100 * g["zs_fp"].mean(),
                     "zs_median_err_px": per_seed["med"].mean(),
                     "neg_pct_m_lm_sametype": 100 * (g["m_lm_sametype"] < 0).mean(),
                     f"neg_pct_{pm}": 100 * (g[pm] < 0).mean(), "n_instances": len(g)})
    summ = pd.DataFrame(rows)

    # 기준 모델과 seed 별 짝 비교 (같은 split, 같은 test 영상)
    main = Z[Z["proto_def"] == ZS_MAIN]
    ps = main.groupby(["model", "split_seed"])["zs_fp"].mean().mul(100).unstack()
    paired = []
    if baseline in ps.index:
        for m in [x for x in ps.index if x != baseline]:
            dlt = (ps.loc[m] - ps.loc[baseline]).dropna()
            mu, sd, n = _mean_sd(dlt)
            paired.append({"model": m, "baseline": baseline, "d_zs_오탐_pct_mean": mu, "d_zs_오탐_pct_sd": sd,
                           "n_seed_better": int((dlt < 0).sum()), "n_seed": n})
    paired = pd.DataFrame(paired)

    fp = main[main["zs_fp"]]
    where = (fp.groupby("model")["zs_target_type"].value_counts(normalize=True).mul(100)
               .unstack(fill_value=0).round(1))
    where.insert(0, "n_오탐", fp.groupby("model").size())

    per_lm = []
    for (m, lm), g in main.groupby(["model", "landmark"], sort=False):
        f_ = g[g["zs_fp"]]
        top = (f_["zs_target_landmark"].replace("", "non_landmark").value_counts(normalize=True)
               if len(f_) else pd.Series(dtype=float))
        per_lm.append({"model": m, "landmark": lm, "n": len(g), "zs_오탐_pct": 100 * g["zs_fp"].mean(),
                       "zs_median_err_px": g["zs_err_px"].median(),
                       "오탐시_가장많이_간곳": top.index[0] if len(top) else "",
                       "그_비율_pct": 100 * top.iloc[0] if len(top) else np.nan,
                       "neg_pct_m_lm_sametype": 100 * (g["m_lm_sametype"] < 0).mean()})
    return summ, paired, where.reset_index(), pd.DataFrame(per_lm)


# =============================================================================
# 2. Data efficiency (완료된 결과 읽기, 필요하면 새 폴더에 학습)
# =============================================================================
DE_COLS = ["model", "split_seed", "fraction", "draw", "image_index", "landmark", "group", "error_px", "catastrophic"]
RUN_KEY = ["model", "split_seed", "fraction", "draw"]
DE_DEFAULT = {"steps": 12000, "decoder": "heatmap", "hidden": 256}   # 기존 {부위}_full12k 결과의 설정
NEW_DRAWS = {round(f, 2): 1 for f in DE_FRACS}                    # 새로 학습할 때: label 비율마다 draw 1개
# 설정 이름 → (기존 결과 폴더, label 비율별 draw 수). 이 설정일 때만 기존 결과를 읽는다.
EXISTING_DE = {"s12000_heatmap_h256": ("{part}_full12k", NEW_DRAWS),                          # 03 노트북 Cell M
               "s3000_heatmap_h256": ("{part}", {0.05: 3, 0.10: 3, 0.25: 2, 1.00: 1})}       # 03 노트북 Cell S


class LinearProbeDecoder(torch.nn.Module):
    """token 마다 선형층 하나(1×1 conv) → K 채널 32×32 지도 → bilinear 로 heatmap 크기까지 확대.
    decoder 용량을 최소로 줄여 'feature 자체에 위치 정보가 얼마나 있는가'를 보는 용도.
    LE.HeatmapDecoder 와 같은 인자·입출력 형식."""

    def __init__(self, embed_dim, grid, n_landmarks, heatmap_size=128):
        super().__init__()
        self.grid, self.heatmap_size = grid, heatmap_size
        self.proj = torch.nn.Conv2d(embed_dim, n_landmarks, 1)

    def forward(self, tokens):
        B, N, D = tokens.shape
        x = self.proj(tokens.transpose(1, 2).reshape(B, D, self.grid, self.grid))
        return torch.nn.functional.interpolate(x, size=(self.heatmap_size, self.heatmap_size),
                                               mode="bilinear", align_corners=False)


DECODERS = {"heatmap": "LE.HeatmapDecoder: 1×1 conv(hidden) → [2배 확대 + 3×3 conv] × 2 → 128×128 (기존 설정)",
            "linear": "LinearProbeDecoder: 1×1 conv 하나 + bilinear 확대 (decoder 용량 최소)"}


def de_setting_name(st):
    return f"s{st['steps']}_{st['decoder']}" + (f"_h{st['hidden']}" if st["decoder"] == "heatmap" else "")


def de_setting_text(st):
    ex = EXISTING_DE.get(de_setting_name(st))
    return (f"decoder {st['decoder']}" + (f" (hidden {st['hidden']})" if st["decoder"] == "heatmap" else "")
            + f", {st['steps']} step" + (f" — 기존 {ex[0].format(part='<부위>')} 결과와 같은 설정" if ex else ""))


def setting_draws(st):
    """이 설정으로 학습·집계할 때 label 비율별 draw 수.
    기존 결과가 있는 설정(12000 / 3000 step)은 기존과 같은 draw 수 → 새로 학습한 모델도 같은 조건으로 비교된다.
    그 밖의 설정은 draw 1개."""
    return EXISTING_DE.get(de_setting_name(st), (None, NEW_DRAWS))[1]


def resolve_de_source(x):
    """--de-source 값 → 폴더 경로. 경로가 아니면 metrics_py 아래 tag 이름으로 본다."""
    if os.path.isdir(x):
        return os.path.abspath(x)
    p = os.path.join(OUT_ROOT, x)
    if os.path.isdir(p):
        return p
    sys.exit(f"[멈춤] --de-source 폴더를 찾지 못했습니다: {x} (경로 또는 {OUT_ROOT}/<tag>)")


def _source_setting_ok(folder, st):
    """이전 실행 폴더의 de_setting.json 이 지금 설정과 같은지. 없거나 다르면 (False, 이유)."""
    p = os.path.join(folder, "de_setting.json")
    if not os.path.isfile(p):
        return False, "de_setting.json 없음 (설정을 확인할 수 없음)"
    with open(p) as f:
        saved = json.load(f)
    if de_setting_name(saved) != de_setting_name(st) or int(saved.get("steps", -1)) != st["steps"]:
        return False, f"설정 다름 (저장: {de_setting_text(saved)})"
    return True, ""


def de_sources(part, out_dir, st, extra=()):
    """읽을 (폴더, label 비율별 draw 수) 목록 (앞쪽 우선):
         1) 기존 결과 폴더 — 설정이 EXISTING_DE 에 있을 때만 (12000 step → {부위}_full12k, 3000 step → {부위})
         2) --de-source 로 준 이전 실행 폴더의 {부위}_de_<설정>  — de_setting.json 이 지금 설정과 같을 때만
         3) 이번 실행의 새 폴더 (--train-de 로 학습할 곳)"""
    name = f"{part}_de_{de_setting_name(st)}"
    srcs = []
    if de_setting_name(st) in EXISTING_DE:
        sub, draws = EXISTING_DE[de_setting_name(st)]
        srcs.append((os.path.join(EXISTING, sub.format(part=part)), draws))
    for root in extra:
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            print(f"[{part}] --de-source {root}: {name}/ 없음 → 건너뜀")
            continue
        ok, why = _source_setting_ok(d, st)
        if ok:
            srcs.append((d, setting_draws(st)))
        else:
            print(f"[{part}] --de-source {d}: {why} → 건너뜀")
    return srcs + [(os.path.join(out_dir, name), setting_draws(st))]


@contextlib.contextmanager
def decoder_override(LE, st):
    """LE._train_decoder 가 만드는 decoder 를 이 블록 안에서만 바꾼다 (landmark_eval.py 파일은 수정하지 않음).
    LE._train_decoder 는 호출할 때 모듈의 HeatmapDecoder 이름을 찾으므로, 이름만 잠시 바꾸고 끝나면 되돌린다."""
    orig = LE.HeatmapDecoder
    if st["decoder"] == "linear":
        LE.HeatmapDecoder = LinearProbeDecoder
    elif st["hidden"] != DE_DEFAULT["hidden"]:
        LE.HeatmapDecoder = functools.partial(orig, hidden=st["hidden"])
    try:
        yield
    finally:
        LE.HeatmapDecoder = orig


def _expected_keys(LE, spec, arr, cfg):
    """seed 별로 완료된 run 하나가 가져야 할 instance 집합.
    test split 의 visible (영상, landmark) 쌍 (모든 자세 포함) 을 code = image_index × K + landmark 번호 로 표시.
    반환: DataFrame[split_seed, code]"""
    N, K = len(spec.records), len(spec.kp_names)
    rows = []
    for s in SEEDS:
        te = LE.split_indices(N, s, cfg.train_ratio, cfg.val_ratio, spec.exclude)[2]
        ii, kk = np.nonzero(arr["vis"][te])
        rows.append(pd.DataFrame({"split_seed": s, "code": te[ii].astype(np.int64) * K + kk}))
    return pd.concat(rows, ignore_index=True)


def _valid_runs(src, models, expected, kp_names, n_draws):
    """src 폴더에서 '완료된' run 만 고른다. 완료 = 아래를 모두 만족:
         - le_runs.csv 에 있다
         - le_instances 의 (영상, landmark) 쌍에 중복이 없다
         - 그 쌍의 집합이 그 seed 의 test split 에서 기대되는 집합과 정확히 같다
           (행 수만 같고 다른 영상·중복 행으로 채워진 run, 중간에 끊긴 run 은 제외)
    (model, label 비율, seed) 는 draw 0 … n_draws[비율]-1 이 모두 완료여야 완료.
    반환: (완료 run 의 instance 행, {(model, fraction): {완료된 seed}})"""
    rp, ip = os.path.join(src, "le_runs.csv"), os.path.join(src, "le_instances.csv")
    if not (os.path.isfile(rp) and os.path.isfile(ip)):
        return None, {}
    runs = pd.read_csv(rp, usecols=RUN_KEY)
    inst = pd.read_csv(ip, usecols=DE_COLS)
    fr = [round(f, 2) for f in DE_FRACS]
    out = []
    for t in (runs, inst):
        t["fraction"] = t["fraction"].round(2)
        out.append(t[t.model.isin(models) & t.split_seed.isin(SEEDS) & t.fraction.isin(fr)
                     & (t.draw < t.fraction.map(n_draws))])
    runs, inst = out
    k_of = {n: i for i, n in enumerate(kp_names)}
    lm = inst["landmark"].map(k_of)
    inst = inst.assign(code=np.where(lm.notna(), inst["image_index"].astype(np.int64) * len(kp_names)
                                     + lm.fillna(0).astype(np.int64), -1))
    n_exp = expected.groupby("split_seed").size().rename("n_expected")
    hit = inst.merge(expected.drop_duplicates(), on=["split_seed", "code"], how="inner")
    n = (inst.groupby(RUN_KEY).agg(n_rows=("code", "size"), n_unique=("code", "nunique"))
             .join(hit.groupby(RUN_KEY)["code"].nunique().rename("n_matched")).fillna({"n_matched": 0})
             .reset_index())
    n = n.merge(runs.drop_duplicates(), on=RUN_KEY, how="inner")
    n = n.join(n_exp, on="split_seed")
    n = n[(n.n_rows == n.n_expected) & (n.n_unique == n.n_expected) & (n.n_matched == n.n_expected)]
    inst = inst.drop(columns="code")
    nd = n.groupby(["model", "fraction", "split_seed"]).size().reset_index(name="n_draw_ok")
    nd = nd[nd.n_draw_ok == nd.fraction.map(n_draws)]               # 그 seed 의 draw 가 모두 완료
    ok = {}
    for r in nd.itertuples():
        ok.setdefault((r.model, r.fraction), set()).add(int(r.split_seed))
    return inst.merge(n[RUN_KEY], on=RUN_KEY, how="inner"), ok


def de_load(LE, spec, arr, cfg, models, sources):
    """(모델, label 비율) 칸마다 seed 3개가 모두 완료된 폴더 하나를 골라 그 폴더의 run 만 쓴다.
    폴더·run 을 섞지 않는다. 그런 폴더가 없으면 그 칸은 '미완료' (부분 seed 값은 쓰지 않음)."""
    expected = _expected_keys(LE, spec, arr, cfg)
    loaded = [(src, *_valid_runs(src, models, expected, spec.kp_names, nd)) for src, nd in sources]
    picks, status = [], []
    for m in models:
        for f in DE_FRACS:
            key = (m, round(f, 2))
            chosen, best = None, ("", 0)
            for src, inst, ok in loaded:
                n_ok = len(ok.get(key, ()))
                if n_ok == len(SEEDS):
                    chosen = (src, inst)
                    break
                if n_ok > best[1]:
                    best = (src, n_ok)
            if chosen:
                src, inst = chosen
                picks.append(inst[(inst.model == m) & (inst.fraction == key[1])].assign(source=src))
                status.append({"model": m, "fraction": key[1], "상태": "OK", "n_seed": len(SEEDS), "source": src})
            else:
                status.append({"model": m, "fraction": key[1], "상태": "미완료", "n_seed": best[1],
                               "source": best[0]})
    I = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=DE_COLS + ["source"])
    return I, pd.DataFrame(status)


def de_tables(I, status, spec, models, baseline):
    """완료된 칸(상태 OK)만 값을 계산한다. 미완료 칸은 값 없이 상태·완료 seed 수만 남긴다."""
    if spec.analysis_group is not None:
        I = I[I["group"] == spec.analysis_group]
    per_seed = (I.groupby(["model", "fraction", "split_seed", "draw"])
                  .agg(MRE=("error_px", "mean"), median=("error_px", "median"),
                       fp=("catastrophic", lambda x: 100 * x.mean()))
                  .groupby(["model", "fraction", "split_seed"]).mean())          # draw 평균 → seed 단위
    rows = []
    for st in status.itertuples():
        row = {"model": st.model, "fraction": st.fraction, "MRE_px_mean": np.nan, "MRE_px_sd": np.nan,
               "median_px_mean": np.nan, "오탐_pct_mean": np.nan, "오탐_pct_sd": np.nan,
               "n_seed": st.n_seed, "상태": st.상태, "source": st.source}
        if st.상태 == "OK":
            g = per_seed.loc[(st.model, st.fraction)]
            row["MRE_px_mean"], row["MRE_px_sd"], _ = _mean_sd(g["MRE"])
            row["오탐_pct_mean"], row["오탐_pct_sd"], _ = _mean_sd(g["fp"])
            row["median_px_mean"] = g["median"].mean()
        rows.append(row)
    summ = pd.DataFrame(rows)

    ok = set(zip(summ.model[summ.상태 == "OK"], summ.fraction[summ.상태 == "OK"]))
    paired = []
    if baseline in models:
        for m in [x for x in models if x != baseline]:
            for f in [round(x, 2) for x in DE_FRACS]:
                if (m, f) not in ok or (baseline, f) not in ok:          # 둘 다 완료된 칸만 비교
                    continue
                dlt = per_seed.loc[(m, f)]["MRE"] - per_seed.loc[(baseline, f)]["MRE"]
                mu, sd, n = _mean_sd(dlt)
                paired.append({"model": m, "fraction": f, "baseline": baseline,
                               "d_MRE_px_mean": mu, "d_MRE_px_sd": sd, "n_seed_better": int((dlt < 0).sum()),
                               "n_seed": n})
    per_lm = (I.groupby(["model", "fraction", "landmark", "split_seed", "draw"])["error_px"].mean()
                .groupby(["model", "fraction", "landmark", "split_seed"]).mean()
                .groupby(["model", "fraction", "landmark"]).mean().rename("MRE_px").reset_index())
    return summ, pd.DataFrame(paired), per_lm


# =============================================================================
# 기존 결과와 대조 (같은 계산이므로 차이 ≈ 0 이어야 한다)
# =============================================================================
def check_cluster(part, clu):
    pieces = []
    d0 = os.path.join(EXISTING, part)
    if not os.path.isdir(d0):
        print(f"[{part}] 기존 클러스터링 결과 폴더 없음 → 대조 건너뜀: {d0}")
        return []
    for fn in sorted(os.listdir(d0)):
        if fn.startswith("landmark_clustering_dense_patch_") and fn.endswith(".csv") and \
                (fn == "landmark_clustering_dense_patch_all.csv" or fn.startswith("landmark_clustering_dense_patch_D3-")):
            d = pd.read_csv(os.path.join(EXISTING, part, fn))
            if "kNN Acc↑" in d.columns:
                pieces.append(d)
    if not pieces:
        return []
    old = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["label", "model"], keep="last")
    j = clu[clu.image_group == "all"].merge(old, on=["label", "model"])     # 기존 결과는 모든 자세로 계산됨
    rows = []
    for _, r in j.iterrows():
        for new_c, old_c in [("kNN Acc", "kNN Acc↑"), ("Silhouette", "Silhouette↑")]:
            rows.append({"part": part, "metric": f"cluster {r['label']} {new_c} (모든 자세)", "model": r["model"],
                         "new": r[new_c], "existing": r[old_c], "abs_diff": abs(r[new_c] - r[old_c]),
                         "existing_file": "landmark_clustering_dense_patch_all*.csv"})
    return rows


def check_zeroshot(part, spec, Z, models):
    rows = []
    mg_p = os.path.join(EXISTING, f"{part}_full12k", "margin_instances.csv")
    if os.path.isfile(mg_p):
        old = pd.read_csv(mg_p, usecols=["model", "split_seed", "fraction", "draw", "image_index", "zs_err_norm"])
        old = old[old.model.isin(models) & (old.draw == 0)]
        old["fraction"] = old["fraction"].round(2)
        for zdef, fracs in ZS_DEFS.items():
            o = old[old.fraction.isin([round(f, 2) for f in fracs])].copy()
            n = Z[Z.proto_def == zdef]
            if spec.analysis_group is not None:
                n = n[n.group == spec.analysis_group]
                grp = dict(zip(Z.image_index, Z.group))
                o = o[o.image_index.map(grp) == spec.analysis_group]
            for m in models:
                a, b = n[n.model == m], o[o.model == m]
                if len(b) == 0:
                    continue
                va, vb = 100 * (a.zs_err_norm > spec.cat_t).mean(), 100 * (b.zs_err_norm > spec.cat_t).mean()
                rows.append({"part": part, "metric": f"zeroshot {zdef} 오탐 % (pooled)", "model": m,
                             "new": va, "existing": vb, "abs_diff": abs(va - vb),
                             "existing_file": f"{part}_full12k/margin_instances.csv"})
    d_p = os.path.join(EXISTING, f"{part}_full12k", "D_model_level.csv")
    if os.path.isfile(d_p):
        d = pd.read_csv(d_p, index_col=0).set_index("model")
        n = Z[Z.proto_def == "label_5_10"]
        if spec.analysis_group is not None:
            n = n[n.group == spec.analysis_group]
        for m in models:
            if m in d.index:
                va = 100 * n[n.model == m].zs_fp.mean()
                rows.append({"part": part, "metric": "zeroshot label_5_10 vs D_model_level zs_catastrophic_pct",
                             "model": m, "new": va, "existing": d.loc[m, "zs_catastrophic_pct"],
                             "abs_diff": abs(va - d.loc[m, "zs_catastrophic_pct"]),
                             "existing_file": f"{part}_full12k/D_model_level.csv"})
    return rows


def check_de(part, de):
    p = os.path.join(EXISTING, f"{part}_full12k", "M_summary_12k.csv")
    if not os.path.isfile(p):
        return []
    old = pd.read_csv(p)
    old["fraction"] = old["fraction"].round(2)
    de = de[(de.상태 == "OK") & (de.source == os.path.join(EXISTING, f"{part}_full12k"))]   # 기존 폴더에서 읽은 칸만
    j = de.merge(old, on=["model", "fraction"])
    return [{"part": part, "metric": f"de MRE frac {r.fraction:.2f}", "model": r.model, "new": r.MRE_px_mean,
             "existing": r.MRE_mean, "abs_diff": abs(r.MRE_px_mean - r.MRE_mean),
             "existing_file": f"{part}_full12k/M_summary_12k.csv"} for r in j.itertuples()]


# =============================================================================
# 보고서
# =============================================================================
def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                    else ("" if isinstance(v, float) and np.isnan(v) else str(v)) for v in r) + " |")
    return "\n".join(out)


def pm(m, s, d=2):
    if not np.isfinite(m):
        return "미완료"
    return f"{m:.{d}f} ± {s:.{d}f}" if np.isfinite(s) else f"{m:.{d}f}"


def write_summary(out_dir, args, res, checks):
    L = [f"# Landmark 표현 지표 — {args.tag}", "",
         f"- 모델: {', '.join(args.models)}  /  기준 모델(짝 비교): {args.baseline}",
         f"- 부위: {', '.join(args.parts)}  /  지표: {', '.join(args.metrics)}",
         "- 값은 seed 42/43/44 평균 ± SD. 고관절은 standard 자세만 집계 (클러스터링의 '모든 자세' 값은 기존 결과 대조용).",
         "- 오탐 = 예측이 GT 에서 영상 대각선 5% 를 넘게 벗어난 경우.",
         "- 미완료 = seed 3개가 모두 끝나지 않은 칸. 부분 seed 값은 표시하지 않는다.", ""]
    for part, r in res.items():
        L += [f"## {part}", ""]
        if "cluster" in r:
            c = r["cluster"]
            c = c[c.image_group == r["cluster_main_group"]][["label", "model", "kNN Acc", "kNN SD", "Silhouette"]]
            L += [f"### 1. 클러스터링 — 영상: {r['cluster_main_group']} (출처: cluster.csv, 혼동: cluster_per_class.csv)",
                  "", md_table(c, "{:.4f}"), ""]
        if "de" in r:
            d = r["de"]
            v = [pm(a, b) if s == "OK" else f"미완료 ({n}/{len(SEEDS)} seed)"
                 for a, b, s, n in zip(d.MRE_px_mean, d.MRE_px_sd, d.상태, d.n_seed)]
            w = d.assign(v=v).pivot(index="model", columns="fraction", values="v").reindex(args.models)
            w.columns = [f"MRE {f:.0%}" for f in w.columns]
            L += [f"### 2. Data efficiency — MRE px, {de_setting_text(r['de_setting'])} (출처: de_summary.csv)", "",
                  md_table(w.reset_index()), ""]
            if len(r["de_paired"]):
                p = r["de_paired"]
                w = p.assign(v=[f"{pm(a, b)} ({k}/{n})" for a, b, k, n in
                                zip(p.d_MRE_px_mean, p.d_MRE_px_sd, p.n_seed_better, p.n_seed)]).pivot(
                    index="model", columns="fraction", values="v").reindex(
                    [m for m in args.models if m != args.baseline])
                w.columns = [f"ΔMRE {f:.0%}" for f in w.columns]
                L += [f"{args.baseline} 대비 seed 별 MRE 차이 (음수 = 좋아짐, 괄호 = 좋아진 seed 수 / seed 수, "
                      "둘 다 완료된 칸만)", "",
                      md_table(w.reset_index()), ""]
        if "zs" in r:
            z = r["zs"]
            w = z.assign(v=[pm(a, b) for a, b in zip(z["zs_오탐_pct_mean"], z["zs_오탐_pct_sd"])]).pivot(
                index="model", columns="proto_def", values="v").reindex(args.models)
            w = w[[c for c in ZS_DEFS if c in w.columns]]
            w["margin 음수 % (m_lm_sametype, train_full)"] = (
                z[z.proto_def == ZS_MAIN].set_index("model")["neg_pct_m_lm_sametype"].reindex(w.index).round(2))
            L += ["### 3. Zero-shot 오탐 % (출처: zeroshot_summary.csv)",
                  f"train_full = train split 전체로 기준 feature (주 지표) / label_5_10 = 기존 슬라이드 정의", "",
                  md_table(w.reset_index()), ""]
            if len(r["zs_paired"]):
                p = r["zs_paired"]
                L += [f"{args.baseline} 대비 seed 별 차이 (train_full, 음수 = 좋아짐)", "",
                      md_table(pd.DataFrame({"model": p.model,
                                             "Δ zs 오탐 %": [pm(a, b) for a, b in zip(p.d_zs_오탐_pct_mean, p.d_zs_오탐_pct_sd)],
                                             "좋아진 seed": [f"{k}/{n}" for k, n in zip(p.n_seed_better, p.n_seed)]})), ""]
            L += ["오탐일 때 예측이 간 곳 % (출처: zeroshot_where.csv)", "", md_table(r["zs_where"], "{:.1f}"), ""]
    if checks is not None and len(checks):
        mx = checks.groupby(["part", "metric"])["abs_diff"].max().reset_index()
        L += ["## 기존 결과와 대조 (같은 계산 → 0 에 가까워야 함, 출처: check_vs_existing.csv)", "",
              md_table(mx, "{:.4f}"), ""]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(L))


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--parts", nargs="+", default=["hip", "foot", "spine"], choices=["hip", "foot", "spine"])
    ap.add_argument("--metrics", nargs="+", default=["cluster", "de", "zeroshot"], choices=["cluster", "de", "zeroshot"])
    ap.add_argument("--tag", required=True, help="출력 폴더 이름 (src/outputs/ssl/metrics_py/<tag>, 이미 있으면 멈춤)")
    ap.add_argument("--baseline", default="D3-E0", help="짝 비교 기준 모델")
    ap.add_argument("--no-tsne", action="store_true", help="t-SNE 그림 생략 (CPU 시간 절약)")
    ap.add_argument("--num-workers", type=int, default=2)
    g = ap.add_argument_group("data efficiency decoder (기본값 = 기존 _full12k 결과와 같은 설정 → 기존 결과를 읽음)")
    g.add_argument("--de-steps", type=int, default=DE_DEFAULT["steps"], help="decoder 학습 step 수 (기본 12000)")
    g.add_argument("--de-decoder", default=DE_DEFAULT["decoder"], choices=list(DECODERS),
                   help="; ".join(f"{k}: {v}" for k, v in DECODERS.items()))
    g.add_argument("--de-hidden", type=int, default=DE_DEFAULT["hidden"], help="heatmap decoder 첫 층 채널 수 (기본 256)")
    g.add_argument("--train-de", action="store_true",
                   help="완료된 결과가 없는 모델을 위 설정으로 새 폴더에 학습 (GPU, 모델·부위당 수십 분~수 시간)")
    g.add_argument("--de-source", nargs="+", default=[], metavar="TAG_OR_DIR",
                   help="이전 실행에서 학습한 결과를 다시 읽을 폴더 (metrics_py 아래 tag 이름 또는 경로, 여러 개 가능). "
                        "그 안의 <부위>_de_<설정>/ 중 de_setting.json 이 지금 설정과 같은 것만 읽는다 (읽기만 함)")
    args = ap.parse_args()
    de_st = {"steps": args.de_steps, "decoder": args.de_decoder, "hidden": args.de_hidden}
    de_extra = [resolve_de_source(x) for x in args.de_source]

    out_dir = os.path.join(OUT_ROOT, args.tag)
    if os.path.exists(out_dir):
        sys.exit(f"[멈춤] 출력 폴더가 이미 있습니다 (덮어쓰지 않음): {out_dir}\n  --tag 를 바꿔서 실행하세요.")
    if any(os.path.abspath(x) == os.path.abspath(out_dir) for x in de_extra):
        sys.exit("[멈춤] --de-source 가 이번 출력 폴더와 같습니다.")

    import landmark_eval as LE

    # GPU·가중치는 backbone 을 실제로 돌릴 때만 요구한다 (--metrics de 만이면 CSV 만 읽음)
    need_gpu = bool({"cluster", "zeroshot"} & set(args.metrics)) or ("de" in args.metrics and args.train_de)
    if need_gpu:
        import msk_backbones as MB                      # CUDA_VISIBLE_DEVICES·GPU 확인은 여기서 한다
        backbones = MB.get_backbones(args.models)       # checkpoint 없으면 여기서 멈춤
        device = MB.DEVICE
        print(f"[OK] device={device} ({torch.cuda.get_device_name(0)}), "
              f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        ckpts = {m: {"path": str(MB.ckpt_path(m)), "epoch": MB.ckpt_epoch(m)}
                 for m in args.models if m in MB.P1_DIRS or m in MB.P2_RUNS}
        for m, c in ckpts.items():
            print(f"  checkpoint {m:16s} epoch {c['epoch']}  {c['path']}")
    else:
        backbones, device, ckpts = None, torch.device("cpu"), {}
        print("[info] GPU·가중치 없이 기존 결과 CSV 만 읽습니다.")
    cfg = LE.EvalConfig(device=str(device), num_workers=args.num_workers)
    if "de" in args.metrics and de_setting_name(de_st) not in EXISTING_DE:
        print(f"[info] data efficiency 설정({de_setting_text(de_st)})에 맞는 기존 결과가 없음 → 기존 결과는 쓰지 않음."
              + ("" if args.train_de or de_extra else
                 " --train-de 나 --de-source 가 없으면 모두 미완료로 표시됩니다."))

    os.makedirs(out_dir)
    try:
        commit = subprocess.run(["git", "-C", SRC, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "started": datetime.datetime.now().isoformat(timespec="seconds"),
                   "git_commit": commit, "python": sys.executable, "torch": torch.__version__,
                   "seeds": SEEDS, "de_fracs": DE_FRACS, "de_setting": de_st, "de_source": de_extra,
                   "zs_defs": ZS_DEFS, "checkpoints": ckpts,
                   "existing_root": EXISTING,
                   "eval_cfg": {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.__dict__.items()}},
                  f, indent=2, ensure_ascii=False, default=str)

    res, checks = {}, []
    t_all = time.time()
    for part in args.parts:
        print(f"\n==================== {part}")
        spec, arr = load_part(LE, part)
        pdir = os.path.join(out_dir, part)
        os.makedirs(pdir)
        res[part] = {}

        # ---- 1·3: backbone token 을 모델마다 한 번만 뽑아 클러스터링·zero-shot 에 같이 쓴다
        if {"cluster", "zeroshot"} & set(args.metrics):
            idx_all = np.array([i for i in range(len(spec.records)) if i not in set(spec.exclude)])
            # 클러스터링 영상: 고관절은 standard 자세가 주 결과, '모든 자세'는 기존 결과 대조용으로 같이 계산
            main_g = spec.analysis_group or "all"
            clu_sets = {"all": idx_all}
            if spec.analysis_group is not None:
                clu_sets = {main_g: idx_all[arr["group"][idx_all] == main_g], "all": idx_all}
            clu_rows, clu_cls, clu_conf, coords, Zs = [], [], {}, {}, []
            for m in args.models:
                t0 = time.time()
                loader_fn, input_size = backbones[m]
                ex = loader_fn()
                grid = ex.grid_size(input_size)
                tok = LE.build_token_cache(ex, spec, arr, input_size, cfg)
                del ex
                gc.collect(); torch.cuda.empty_cache()
                print(f"[{part}] {m}: tokens {tuple(tok.shape)} grid={grid} ({time.time() - t0:.0f}s)")
                if "cluster" in args.metrics:
                    for grp, idx in clu_sets.items():
                        is_main = grp == main_g
                        f, img, kk = nearest_gt_tokens(torch, tok, grid, arr, idx)
                        rows, pc, conf = cluster_one(m, part, f, img, kk, spec.kp_names, diagnose=is_main)
                        clu_rows += [dict(r, image_group=grp) for r in rows]
                        if is_main:
                            clu_cls += [dict(p, image_group=grp) for p in pc]
                            for lab, c in conf.items():
                                clu_conf.setdefault(lab, []).append(c)
                            if not args.no_tsne:
                                coords[m] = tsne_coords(f, kk)
                        print(f"  cluster [{grp}]: " + ", ".join(
                            f"{r['label']} kNN {r['kNN Acc']:.4f} Sil {r['Silhouette']:.4f}" for r in rows))
                        del f
                if "zeroshot" in args.metrics:
                    Z = zeroshot_one(torch, LE, m, tok, grid, spec, arr, cfg, device)
                    Zs.append(Z)
                    zm = Z[(Z.proto_def == ZS_MAIN) & ((Z.group == spec.analysis_group) if spec.analysis_group else True)]
                    print(f"  zero-shot 오탐 (train_full, pooled): {100 * zm.zs_fp.mean():.2f}%")
                del tok
                gc.collect(); torch.cuda.empty_cache()
                print(f"  {m} done ({time.time() - t0:.0f}s)")

            if "cluster" in args.metrics:
                clu = pd.DataFrame(clu_rows)
                clu.to_csv(os.path.join(pdir, "cluster.csv"), index=False)
                pd.DataFrame(clu_cls).to_csv(os.path.join(pdir, "cluster_per_class.csv"), index=False)
                for lab, cs in clu_conf.items():
                    safe = lab.replace("(", "_").replace(")", "")
                    pd.concat(cs).to_csv(os.path.join(pdir, f"cluster_confusion_{safe}.csv"))
                if coords:
                    plot_tsne(part, spec.kp_names, coords, os.path.join(pdir, "tsne.png"))
                    pd.concat([pd.DataFrame({"model": m, "x": xy[:, 0], "y": xy[:, 1], "kp": kk})
                               for m, (xy, kk) in coords.items()]).to_csv(os.path.join(pdir, "tsne_coords.csv"),
                                                                          index=False)
                res[part].update(cluster=clu, cluster_main_group=main_g)
                checks += check_cluster(part, clu)

            if "zeroshot" in args.metrics:
                Z = pd.concat(Zs, ignore_index=True)
                Z.to_csv(os.path.join(pdir, "zeroshot_instances.csv"), index=False)
                summ, paired, where, per_lm = zeroshot_tables(Z, spec, args.baseline)
                summ.to_csv(os.path.join(pdir, "zeroshot_summary.csv"), index=False)
                paired.to_csv(os.path.join(pdir, "zeroshot_paired.csv"), index=False)
                where.to_csv(os.path.join(pdir, "zeroshot_where.csv"), index=False)
                per_lm.to_csv(os.path.join(pdir, "zeroshot_per_landmark.csv"), index=False)
                res[part].update(zs=summ, zs_paired=paired, zs_where=where)
                checks += check_zeroshot(part, spec, Z, args.models)
                del Z, Zs

        # ---- 2: data efficiency
        if "de" in args.metrics:
            sources = de_sources(part, out_dir, de_st, de_extra)
            I, status = de_load(LE, spec, arr, cfg, args.models, sources)
            missing = list(dict.fromkeys(status.model[status.상태 != "OK"]))
            if missing and args.train_de:
                new_dir = sources[-1][0]                    # 이 실행의 새 폴더 (기존 폴더에는 쓰지 않음)
                os.makedirs(new_dir, exist_ok=True)
                with open(os.path.join(new_dir, "de_setting.json"), "w") as f:
                    json.dump({**de_st, "decoder_desc": DECODERS[de_st["decoder"]],
                               "n_draws": {str(k): v for k, v in setting_draws(de_st).items()}},
                              f, indent=2, ensure_ascii=False)
                cfg_de = replace(cfg, seeds=SEEDS, fractions=DE_FRACS, n_draws=dict(setting_draws(de_st)),
                                 total_steps=de_st["steps"], run_margin=False)
                print(f"[{part}] decoder 학습 ({de_setting_text(de_st)}) → {new_dir}: {missing}")
                with decoder_override(LE, de_st):
                    LE.run_pipeline(spec, {m: backbones[m] for m in missing}, new_dir, cfg_de, arr)
                I, status = de_load(LE, spec, arr, cfg, args.models, sources)
            elif missing:
                print(f"[{part}] 완료된 결과가 없는 칸이 있는 모델 → 미완료로 표시: {missing} (--train-de 로 학습 가능)")
            de, de_p, de_lm = de_tables(I, status, spec, args.models, args.baseline)
            de.to_csv(os.path.join(pdir, "de_summary.csv"), index=False)
            de_p.to_csv(os.path.join(pdir, "de_paired.csv"), index=False)
            de_lm.to_csv(os.path.join(pdir, "de_per_landmark.csv"), index=False)
            res[part].update(de=de, de_paired=de_p, de_setting=de_st)
            if de_st == DE_DEFAULT:
                checks += check_de(part, de)

    chk = pd.DataFrame(checks)
    if len(chk):
        chk.to_csv(os.path.join(out_dir, "check_vs_existing.csv"), index=False)
        print("\n=== 기존 결과와 대조 (최대 차이) ===")
        print(chk.groupby(["part", "metric"])["abs_diff"].max().to_string())
    write_summary(out_dir, args, res, chk)
    print(f"\n[done] {(time.time() - t_all) / 60:.1f} min -> {out_dir}/summary.md")


if __name__ == "__main__":
    main()
