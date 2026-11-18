# Phase 2: DINOv3 방식 자기지도 학습

[프로젝트 소개로 돌아가기](../README.md)

## 어떤 파일인가요?

`src/msk_ssl_phase2.py`는 DINOv3 방식의 ViT-B/16과 DINO·iBOT·KoLeo 목표를 기반으로 한 자기지도 학습 코드입니다. 위치 InfoNCE 또는 전체·부분 관계 목표를 추가하는 실험을 포함합니다.

현재 구현된 `experiment`는 `D3-E0`, `D3-E1`, `D3-E2`입니다. D3-E3와 Gram 확장은 파일 주석에 등장하지만 현재 구현된 실험으로 안내하지 않습니다. Phase 2 파일에는 Phase 1에서 재사용한 데이터·증강 코드가 들어 있으며, Phase 1 가중치로 초기화하는 것이 기본 동작은 아닙니다.

근거: `src/msk_ssl_phase2.py:3`, `:102`, `:1689`.

## 실험 대응

| 평가 이름 | experiment | run_tag | 핵심 차이 |
|---|---|---|---|
| D3-E0 | D3-E0 | 빈 문자열 | 기본 자기지도 학습 |
| D3-E1c | D3-E1 | e1c | 위치 InfoNCE, 같은 영상 음성 32개 + 다른 영상 음성 64개 |
| D3-E1c-within | D3-E1 | e1c_within | 위치 InfoNCE, 같은 영상 음성 32개, 다른 영상 음성 0개 |
| D3-E2-Adam | D3-E2 | adam_bf05 | `partwhole_method="adam"`, 배치의 50%에 전체·부분 목표 적용 |

여기서 음성은 contrastive learning의 negative 예시입니다. `D3-E2-Adam`의 Adam은 이 코드의 전체·부분 목표 방식 이름이며, optimizer 설정 자체는 부록에서 별도로 확인합니다.

출력 폴더는 `output_root / (experiment + "_" + run_tag)`입니다. run_tag가 비어 있으면 experiment 이름만 사용합니다. 실제 평가 폴더는 `D3-E0`, `D3-E1_e1c`, `D3-E1_e1c_within`, `D3-E2_adam_bf05`입니다.

## 저장된 주요 설정

| 항목 | 값 |
|---|---|
| backbone | ViT-B/16, embed_dim 768 |
| 학습 global / local crop | 256 / 112 |
| batch_size | 256 |
| epochs / steps_per_epoch | 100 / 2,000 |
| max_train_images | null |
| seed | 42 |
| lr / weight_decay | 0.0005 / 0.04 |
| use_bf16 | true |
| E1c 계열 InfoNCE temperature | 0.2 |
| E1c 계열 negative overlap 기준 | 0.05 |
| D3-E2-Adam part 크기 / crop 수 | 128 / 2 |
| D3-E2-Adam 목표 시작 epoch | 10 |
| D3-E2-Adam 표현 / 손실 | patch_mean / nmse |
| D3-E2-Adam monitor | epoch_mean_base_loss |

landmark 평가 입력은 512입니다. 위 학습 crop 크기 및 학습 설정에 저장된 `eval_input_size=256`과 구분합니다. 전체 설정은 아래 부록을 참고하세요.

## checkpoint 선택

평가 이름 `D3-E2-Adam`은 `best.pt`를 사용합니다. `D3-E2-Adam-ep99`는 같은 폴더의 `latest.pt`를 선택하는 별도 평가 이름입니다. 파일 이름만 보고 같은 결과로 합치지 마세요. 실제 epoch는 checkpoint와 실행 기록을 확인해야 합니다.

근거: `src/msk_backbones.py:43`의 `P2_RUNS`, `:50`의 `CKPT_OVERRIDE`.

## 학습 실행

`src/train_phase2.py`는 문서 작성 도중 추가로 확인되었습니다. 이 문서의 예제는 부록의 전체 설정을 사용하기 위해 `Config`, `Trainer` API를 직접 호출합니다. 별도 실행기는 `--gpu` 옵션으로 `CUDA_VISIBLE_DEVICES`를 덮어쓰므로 사용할 때 GPU 선택을 반드시 명시해야 합니다.

아래 부록에서 원하는 실험 JSON 하나를 복사해 저장소 최상위의 `local_config.json`으로 저장합니다. 학습 데이터와 출력 경로를 자신의 환경에 맞추세요. 이 파일은 공개 업로드 목록에 포함하지 않습니다.

[설치 문서](setup.md)대로 GPU를 선택한 다음 저장소 최상위에서 실행합니다. 장시간 학습이므로 학습 서버에서 사용자가 직접 실행하며, 문서 작성 과정에서는 실행하지 않았습니다.

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
import json
import sys
from pathlib import Path
import torch
assert torch.cuda.is_available(), "CUDA GPU가 필요합니다."
import msk_ssl_phase2 as train

settings = json.loads(Path("local_config.json").read_text())
cfg = train.Config(**settings)
cfg.python_executable = sys.executable
cfg.work_dir = str(Path.cwd())
cfg.run_tag = input("새 실행 태그: ").strip()
assert cfg.run_tag and "/" not in cfg.run_tag and "\\" not in cfg.run_tag
assert Path(cfg.data_root).is_dir(), "학습 이미지 경로를 확인하세요."
assert not Path(cfg.run_dir).exists(), "기존 출력 폴더입니다. 새 태그를 사용하세요."
train.set_seed(cfg.seed, cfg.deterministic)
trainer = train.Trainer(cfg)
trainer.fit()
PY
```

D3-E2를 직접 설정할 때는 `Config(experiment="D3-E2", partwhole_method="adam", ...)`처럼 처음부터 두 값을 함께 넣어야 합니다. 생성 시 검증하므로 experiment만 먼저 지정하면 오류가 납니다. 부록의 D3-E2-Adam 설정에는 두 값이 들어 있습니다.

새 태그로 학습한 모델을 평가하려면 평가 로더가 그 새 폴더를 가리키도록 연결해야 합니다. 새 실행을 기존 실험 이름의 원본 가중치로 덮어넣지 마세요.

## 설정 부록 읽는 법

아래는 저장된 실험 설정입니다. 서버 절대 경로를 범용 경로로 바꾸고 `dataset_cache_list`를 `null`로 바꾼 문서용 복사본입니다. 원본은 수정하지 않았습니다. 실제 데이터·출력 경로는 사용자가 입력합니다.

저장 설정에서 현재 `Config`가 모르는 키는 발견되지 않았습니다. 이전 E0·E1 설정에는 나중에 추가된 비활성 옵션 필드가 없어 현재 코드의 기본값이 사용됩니다. 이는 키 호환성 확인이며 과거 코드와 수치 재현을 보증하지 않습니다.

<!-- saved-config:D3-E0 -->
<details>
<summary>D3-E0: 저장 설정 전체 보기</summary>

출처: `src/ssl/D3-E0/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": true,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
  "centering": "sinkhorn_knopp",
  "clip_grad": 3.0,
  "contrast_jitter": 0.4,
  "crop_max_attempts": 10,
  "crop_ratio": [
    0.75,
    1.3333333333333333
  ],
  "data_root": "/path/to/MSK_images",
  "dataset_cache_list": null,
  "depth": 12,
  "deterministic": false,
  "drop_last": true,
  "drop_path_rate": 0.1,
  "embed_dim": 768,
  "epochs": 100,
  "eval_batch_size": 16,
  "eval_checkpoint_name": "best.pt",
  "eval_input_size": 256,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "D3-E0",
  "ffn_bias": true,
  "foreground_aware_crop": true,
  "foreground_map_size": 64,
  "foreground_max_attempts": 10,
  "foreground_threshold": 0.02,
  "freeze_last_layer_epochs": 1,
  "global_blur_prob": [
    1.0,
    0.1
  ],
  "global_crop_scale": [
    0.4,
    1.0
  ],
  "global_crop_size": 256,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": false,
  "hflip_prob": 0.5,
  "ibot_compute_on_masked_only": true,
  "image_extensions": [
    ".png",
    ".jpg",
    ".jpeg"
  ],
  "in_chans": 3,
  "knn_k": 5,
  "koleo_eps": 1e-08,
  "lambda_comp": 0.1,
  "lambda_decomp": 0.1,
  "lambda_dino": 1.0,
  "lambda_ibot": 1.0,
  "lambda_koleo": 0.1,
  "lambda_pos": 0.1,
  "layerscale_init": 1e-05,
  "layerwise_decay": 0.9,
  "ln_eps": 1e-05,
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 112,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
  "mask_k_bias": true,
  "mask_ratio_max": 0.5,
  "mask_ratio_min": 0.1,
  "mask_sample_prob": 0.5,
  "max_train_images": null,
  "max_vis_samples": 2000,
  "min_foreground_global": 0.3,
  "min_foreground_local": 0.15,
  "min_lr": 1e-06,
  "mlp_ratio": 4.0,
  "monitor": "epoch_mean_total_loss",
  "monitor_mode": "min",
  "n_storage_tokens": 4,
  "negative_threshold": 0.05,
  "normalize_mean": [
    0.2495,
    0.2495,
    0.2495
  ],
  "normalize_std": [
    0.2609,
    0.2609,
    0.2609
  ],
  "num_global_crops": 2,
  "num_heads": 12,
  "num_local_crops": 6,
  "num_workers": 24,
  "output_root": "./new_training_outputs/phase2",
  "part_grid_size": 4,
  "patch_center_momentum": 0.9,
  "patch_embed_lr_mult": 0.2,
  "patch_size": 16,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_overlap_threshold": 0.5,
  "position_center_momentum": 0.9,
  "position_pair_chunk_size": 16384,
  "position_student_temp": 0.1,
  "position_teacher_temp": 0.04,
  "position_topk": 1,
  "positive_threshold": 0.5,
  "pre_resize": 512,
  "prefetch_factor": 2,
  "probe_batch_size": 64,
  "probe_epochs": 50,
  "probe_lr": 0.001,
  "probe_train_ratio": 0.8,
  "probe_weight_decay": 0.0,
  "progress_bar": true,
  "proj_bias": true,
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "rope_base": 100.0,
  "rope_dtype": "fp32",
  "rope_normalize_coords": "separate",
  "rope_rescale_coords": 2.0,
  "run_tag": "",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "sk_n_iterations": 3,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.992,
  "teacher_momentum_final": 1.0,
  "teacher_temp_final": 0.07,
  "teacher_temp_start": 0.04,
  "teacher_temp_warmup_epochs": 30,
  "use_bf16": true,
  "use_compile": false,
  "warmup_epochs": 10,
  "weight_decay": 0.04,
  "weight_decay_end": 0.4,
  "work_dir": "."
}
```

</details>


<!-- saved-config:D3-E1c -->
<details>
<summary>D3-E1c: 저장 설정 전체 보기</summary>

출처: `src/ssl/D3-E1_e1c/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": true,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
  "centering": "sinkhorn_knopp",
  "clip_grad": 3.0,
  "contrast_jitter": 0.4,
  "crop_max_attempts": 10,
  "crop_ratio": [
    0.75,
    1.3333333333333333
  ],
  "data_root": "/path/to/MSK_images",
  "dataset_cache_list": null,
  "depth": 12,
  "deterministic": false,
  "drop_last": true,
  "drop_path_rate": 0.1,
  "embed_dim": 768,
  "epochs": 100,
  "eval_batch_size": 16,
  "eval_checkpoint_name": "best.pt",
  "eval_input_size": 256,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "D3-E1",
  "ffn_bias": true,
  "foreground_aware_crop": true,
  "foreground_map_size": 64,
  "foreground_max_attempts": 10,
  "foreground_threshold": 0.02,
  "freeze_last_layer_epochs": 1,
  "global_blur_prob": [
    1.0,
    0.1
  ],
  "global_crop_scale": [
    0.4,
    1.0
  ],
  "global_crop_size": 256,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": false,
  "hflip_prob": 0.5,
  "ibot_compute_on_masked_only": true,
  "image_extensions": [
    ".png",
    ".jpg",
    ".jpeg"
  ],
  "in_chans": 3,
  "knn_k": 5,
  "koleo_eps": 1e-08,
  "lambda_comp": 0.1,
  "lambda_decomp": 0.1,
  "lambda_dino": 1.0,
  "lambda_ibot": 1.0,
  "lambda_koleo": 0.1,
  "lambda_pos": 0.1,
  "layerscale_init": 1e-05,
  "layerwise_decay": 0.9,
  "ln_eps": 1e-05,
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 112,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
  "mask_k_bias": true,
  "mask_ratio_max": 0.5,
  "mask_ratio_min": 0.1,
  "mask_sample_prob": 0.5,
  "max_train_images": null,
  "max_vis_samples": 2000,
  "min_foreground_global": 0.3,
  "min_foreground_local": 0.15,
  "min_lr": 1e-06,
  "mlp_ratio": 4.0,
  "monitor": "epoch_mean_total_loss",
  "monitor_mode": "min",
  "n_storage_tokens": 4,
  "negative_threshold": 0.05,
  "normalize_mean": [
    0.2495,
    0.2495,
    0.2495
  ],
  "normalize_std": [
    0.2609,
    0.2609,
    0.2609
  ],
  "num_global_crops": 2,
  "num_heads": 12,
  "num_local_crops": 6,
  "num_workers": 24,
  "output_root": "./new_training_outputs/phase2",
  "part_grid_size": 4,
  "patch_center_momentum": 0.9,
  "patch_embed_lr_mult": 0.2,
  "patch_size": 16,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_overlap_threshold": 0.5,
  "position_center_momentum": 0.9,
  "position_infonce_temp": 0.2,
  "position_loss_type": "infonce",
  "position_n_cross_negatives": 64,
  "position_n_within_negatives": 32,
  "position_negative_ios_threshold": 0.05,
  "position_pair_chunk_size": 4096,
  "position_student_temp": 0.1,
  "position_teacher_temp": 0.04,
  "position_topk": 1,
  "positive_threshold": 0.5,
  "pre_resize": 512,
  "prefetch_factor": 2,
  "probe_batch_size": 64,
  "probe_epochs": 50,
  "probe_lr": 0.001,
  "probe_train_ratio": 0.8,
  "probe_weight_decay": 0.0,
  "progress_bar": true,
  "proj_bias": true,
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "rope_base": 100.0,
  "rope_dtype": "fp32",
  "rope_normalize_coords": "separate",
  "rope_rescale_coords": 2.0,
  "run_tag": "e1c",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "sk_n_iterations": 3,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.992,
  "teacher_momentum_final": 1.0,
  "teacher_temp_final": 0.07,
  "teacher_temp_start": 0.04,
  "teacher_temp_warmup_epochs": 30,
  "use_bf16": true,
  "use_compile": false,
  "warmup_epochs": 10,
  "weight_decay": 0.04,
  "weight_decay_end": 0.4,
  "work_dir": "."
}
```

</details>


<!-- saved-config:D3-E1c-within -->
<details>
<summary>D3-E1c-within: 저장 설정 전체 보기</summary>

출처: `src/ssl/D3-E1_e1c_within/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": true,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
  "centering": "sinkhorn_knopp",
  "clip_grad": 3.0,
  "contrast_jitter": 0.4,
  "crop_max_attempts": 10,
  "crop_ratio": [
    0.75,
    1.3333333333333333
  ],
  "data_root": "/path/to/MSK_images",
  "dataset_cache_list": null,
  "depth": 12,
  "deterministic": false,
  "drop_last": true,
  "drop_path_rate": 0.1,
  "embed_dim": 768,
  "epochs": 100,
  "eval_batch_size": 16,
  "eval_checkpoint_name": "best.pt",
  "eval_input_size": 256,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "D3-E1",
  "ffn_bias": true,
  "foreground_aware_crop": true,
  "foreground_map_size": 64,
  "foreground_max_attempts": 10,
  "foreground_threshold": 0.02,
  "freeze_last_layer_epochs": 1,
  "global_blur_prob": [
    1.0,
    0.1
  ],
  "global_crop_scale": [
    0.4,
    1.0
  ],
  "global_crop_size": 256,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": false,
  "hflip_prob": 0.5,
  "ibot_compute_on_masked_only": true,
  "image_extensions": [
    ".png",
    ".jpg",
    ".jpeg"
  ],
  "in_chans": 3,
  "knn_k": 5,
  "koleo_eps": 1e-08,
  "lambda_comp": 0.1,
  "lambda_decomp": 0.1,
  "lambda_dino": 1.0,
  "lambda_ibot": 1.0,
  "lambda_koleo": 0.1,
  "lambda_pos": 0.1,
  "layerscale_init": 1e-05,
  "layerwise_decay": 0.9,
  "ln_eps": 1e-05,
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 112,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
  "mask_k_bias": true,
  "mask_ratio_max": 0.5,
  "mask_ratio_min": 0.1,
  "mask_sample_prob": 0.5,
  "max_train_images": null,
  "max_vis_samples": 2000,
  "min_foreground_global": 0.3,
  "min_foreground_local": 0.15,
  "min_lr": 1e-06,
  "mlp_ratio": 4.0,
  "monitor": "epoch_mean_total_loss",
  "monitor_mode": "min",
  "n_storage_tokens": 4,
  "negative_threshold": 0.05,
  "normalize_mean": [
    0.2495,
    0.2495,
    0.2495
  ],
  "normalize_std": [
    0.2609,
    0.2609,
    0.2609
  ],
  "num_global_crops": 2,
  "num_heads": 12,
  "num_local_crops": 6,
  "num_workers": 24,
  "output_root": "./new_training_outputs/phase2",
  "part_grid_size": 4,
  "patch_center_momentum": 0.9,
  "patch_embed_lr_mult": 0.2,
  "patch_size": 16,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_overlap_threshold": 0.5,
  "position_center_momentum": 0.9,
  "position_infonce_temp": 0.2,
  "position_loss_type": "infonce",
  "position_n_cross_negatives": 0,
  "position_n_within_negatives": 32,
  "position_negative_ios_threshold": 0.05,
  "position_pair_chunk_size": 4096,
  "position_student_temp": 0.1,
  "position_teacher_temp": 0.04,
  "position_topk": 1,
  "positive_threshold": 0.5,
  "pre_resize": 512,
  "prefetch_factor": 2,
  "probe_batch_size": 64,
  "probe_epochs": 50,
  "probe_lr": 0.001,
  "probe_train_ratio": 0.8,
  "probe_weight_decay": 0.0,
  "progress_bar": true,
  "proj_bias": true,
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "rope_base": 100.0,
  "rope_dtype": "fp32",
  "rope_normalize_coords": "separate",
  "rope_rescale_coords": 2.0,
  "run_tag": "e1c_within",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "sk_n_iterations": 3,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.992,
  "teacher_momentum_final": 1.0,
  "teacher_temp_final": 0.07,
  "teacher_temp_start": 0.04,
  "teacher_temp_warmup_epochs": 30,
  "use_bf16": true,
  "use_compile": false,
  "warmup_epochs": 10,
  "weight_decay": 0.04,
  "weight_decay_end": 0.4,
  "work_dir": "."
}
```

</details>


<!-- saved-config:D3-E2-Adam -->
<details>
<summary>D3-E2-Adam: 저장 설정 전체 보기</summary>

출처: `src/ssl/D3-E2_adam_bf05/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": true,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
  "centering": "sinkhorn_knopp",
  "clip_grad": 3.0,
  "contrast_jitter": 0.4,
  "crop_max_attempts": 10,
  "crop_ratio": [
    0.75,
    1.3333333333333333
  ],
  "data_root": "/path/to/MSK_images",
  "dataset_cache_list": null,
  "depth": 12,
  "deterministic": false,
  "drop_last": true,
  "drop_path_rate": 0.1,
  "embed_dim": 768,
  "epochs": 100,
  "eval_batch_size": 16,
  "eval_checkpoint_name": "best.pt",
  "eval_input_size": 256,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "D3-E2",
  "ffn_bias": true,
  "foreground_aware_crop": true,
  "foreground_map_size": 64,
  "foreground_max_attempts": 10,
  "foreground_threshold": 0.02,
  "freeze_last_layer_epochs": 1,
  "global_blur_prob": [
    1.0,
    0.1
  ],
  "global_crop_scale": [
    0.4,
    1.0
  ],
  "global_crop_size": 256,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": false,
  "hflip_prob": 0.5,
  "ibot_compute_on_masked_only": true,
  "image_extensions": [
    ".png",
    ".jpg",
    ".jpeg"
  ],
  "in_chans": 3,
  "knn_k": 5,
  "koleo_eps": 1e-08,
  "lambda_comp": 0.1,
  "lambda_decomp": 0.1,
  "lambda_dino": 1.0,
  "lambda_ibot": 1.0,
  "lambda_koleo": 0.1,
  "lambda_pos": 0.1,
  "layerscale_init": 1e-05,
  "layerwise_decay": 0.9,
  "ln_eps": 1e-05,
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 112,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
  "mask_k_bias": true,
  "mask_ratio_max": 0.5,
  "mask_ratio_min": 0.1,
  "mask_sample_prob": 0.5,
  "max_train_images": null,
  "max_vis_samples": 2000,
  "min_foreground_global": 0.3,
  "min_foreground_local": 0.15,
  "min_lr": 1e-06,
  "mlp_ratio": 4.0,
  "monitor": "epoch_mean_base_loss",
  "monitor_mode": "min",
  "n_storage_tokens": 4,
  "negative_threshold": 0.05,
  "normalize_mean": [
    0.2495,
    0.2495,
    0.2495
  ],
  "normalize_std": [
    0.2609,
    0.2609,
    0.2609
  ],
  "num_global_crops": 2,
  "num_heads": 12,
  "num_local_crops": 6,
  "num_workers": 24,
  "output_root": "./new_training_outputs/phase2",
  "part_grid_size": 4,
  "partwhole_batch_frac": 0.5,
  "partwhole_embed": "patch_mean",
  "partwhole_empty_fg_thr": 0.05,
  "partwhole_head_hidden": 2048,
  "partwhole_loss": "nmse",
  "partwhole_method": "adam",
  "partwhole_n_crops": 2,
  "partwhole_part_size": 128,
  "partwhole_start_epoch": 10,
  "partwhole_student_whole": "unmasked",
  "patch_center_momentum": 0.9,
  "patch_embed_lr_mult": 0.2,
  "patch_size": 16,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_overlap_threshold": 0.5,
  "position_center_momentum": 0.9,
  "position_infonce_temp": 0.2,
  "position_loss_type": "infonce",
  "position_n_cross_negatives": 64,
  "position_n_within_negatives": 32,
  "position_negative_ios_threshold": 0.05,
  "position_pair_chunk_size": 4096,
  "position_student_temp": 0.1,
  "position_teacher_temp": 0.04,
  "position_topk": 1,
  "positive_threshold": 0.5,
  "pre_resize": 512,
  "prefetch_factor": 2,
  "probe_batch_size": 64,
  "probe_epochs": 50,
  "probe_lr": 0.001,
  "probe_train_ratio": 0.8,
  "probe_weight_decay": 0.0,
  "progress_bar": true,
  "proj_bias": true,
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "rope_base": 100.0,
  "rope_dtype": "fp32",
  "rope_normalize_coords": "separate",
  "rope_rescale_coords": 2.0,
  "run_tag": "adam_bf05",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "sk_n_iterations": 3,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.992,
  "teacher_momentum_final": 1.0,
  "teacher_temp_final": 0.07,
  "teacher_temp_start": 0.04,
  "teacher_temp_warmup_epochs": 30,
  "use_bf16": true,
  "use_compile": false,
  "warmup_epochs": 10,
  "weight_decay": 0.04,
  "weight_decay_end": 0.4,
  "work_dir": "."
}
```

</details>
