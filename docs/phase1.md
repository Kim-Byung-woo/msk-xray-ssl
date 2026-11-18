# Phase 1: DINOv2 방식 자기지도 학습

[프로젝트 소개로 돌아가기](../README.md)

## 어떤 파일인가요?

`src/msk_ssl_phase1.py`는 ViT-B/14 backbone으로 MSK X-ray를 자기지도 학습하는 코드입니다. 기본 목표는 DINO·iBOT·KoLeo이며, 위치 대응과 전체·부분 관계를 추가하는 실험 분기가 있습니다.

현재 코드의 `Config.experiment`가 받는 값은 `E0`, `E1`, `E2`, `E3`입니다. 문서나 평가 결과에서 부르는 E1c·E2d는 저장된 실험을 구별하는 이름입니다.

## 실험 대응과 현재 한계

| 평가 이름 | 저장된 experiment | 저장된 run_tag | 평가 폴더 | 설명 |
|---|---|---|---|---|
| E0 | E0 | 빈 문자열 | E0 | 기본 자기지도 학습 |
| E1c | E1 | fn_safe | E1_fn_safe | 위치 InfoNCE 실험 |
| E2d | E2 | indep_part_gram | E2_indep_part_gram | 독립 부분 표현과 Gram 보존 설정을 포함한 실험 |
| E3 | E3 | 빈 문자열 | E3 | 현재 코드에서 위치·전체부분 목표를 함께 켜는 분기 |

**현재 파일만으로 E1c·E2d의 학습을 그대로 재현할 수 있다고 보장할 수 없습니다.** 저장된 설정과 현재 `Config`를 비교하면 다음 차이가 있습니다.

- E1c의 저장 설정에는 `position_loss_type="infonce"`, `position_infonce_temp=0.2`, `position_n_within_negatives=32`, `position_n_cross_negatives=64`, `position_negative_ios_threshold=0.05`가 있습니다. 현재 Phase 1 `Config`에는 이 5개 필드가 없습니다.
- E2d의 저장 설정에는 위 필드에 더해 `partwhole_part_source`, `partwhole_local_part_feature`, `partwhole_center_whole`, `partwhole_containment_threshold`, `partwhole_geom_dim`, `local_preservation`, `lambda_gram`, `gram_teacher_ckpt`, `gram_teacher_key`가 있지만 현재 `Config`에는 없습니다.
- 따라서 알 수 없는 키를 버리고 실행하는 방식으로 해당 실험을 재현했다고 처리하면 안 됩니다. 아래에 적은 별도 파일과 당시 실행 코드의 일치를 추가로 확인해야 합니다.
- E0·E3는 저장 설정 키가 현재 `Config`와 일치합니다. 키 일치만 확인했으며 학습 동작 전체나 과거 실행 코드의 일치까지 검증하지는 않았습니다.

근거: `src/msk_ssl_phase1.py:167`의 `Config`, `:383`의 실험 분기와 각 실험의 저장된 `config.json`.

문서 작성 도중 `src/msk_ssl_phase1_e1c.py`, `src/msk_ssl_phase1_e2d.py`가 작업 폴더에 추가로 확인되었습니다. 이들은 기본 파일과 구분해 보존해야 합니다. 이 문서 작성에서는 해당 파일을 생성하거나 수정하지 않았습니다. 각 파일의 `Config` 필드 집합은 대응하는 E1c·E2d 저장 설정의 키 집합과 일치했습니다. 이는 정적 확인이며 학습·가중치 재현은 실행하지 않았습니다. 두 파일을 사용할 때는 각각 해당 모듈을 import해야 하며, 기본 `msk_ssl_phase1`로 자동 연결되는 것은 아닙니다.

## 저장 설정의 주요 공통값

| 항목 | 값 |
|---|---|
| backbone | ViT-B/14, embed_dim 768 |
| 학습 global / local crop | 224 / 98 |
| batch_size | 256 |
| epochs / steps_per_epoch | 100 / 2,000 |
| max_train_images | null: 상한 없이 사용 |
| seed | 42 |
| lr / weight_decay | 0.0005 / 0.04 |
| use_bf16 | true |

위는 저장된 실험 값입니다. 현재 코드의 기본 `batch_size=32`, `max_train_images=300000`과 다릅니다. landmark 평가 입력 448은 학습 crop 크기와 구분합니다.

## 실제 존재하는 API로 학습하기

`python src/msk_ssl_phase1.py --experiment E0`만으로 학습이 시작되는 구조가 아닙니다. 별도 `src/train.py`는 작성 도중 추가로 확인되었으나 기본 `msk_ssl_phase1`을 import하며, 그 파일에 없는 `local_preservation` 등도 사용합니다. 따라서 이 실행기를 E1c·E2d를 포함한 검증 완료 진입점으로 안내하지 않습니다. 아래는 기본 E0 모듈의 `Config`·`Trainer`를 직접 호출하는 예제입니다.

E0를 예로, 아래 부록의 E0 JSON을 복사해 저장소 최상위에 `local_config.json`으로 저장하고 경로를 자신의 환경에 맞춥니다. 부록의 `run_tag`는 원래 기록이므로 실제 실행에서는 새 태그로 바꾸세요. 이 파일을 저장하는 작업은 사용자가 수행하며, 공개 업로드 목록에는 포함하지 않습니다.

[설치 문서](setup.md)에 따라 GPU를 선택한 뒤 저장소 최상위에서 실행합니다. 이 명령은 실제 장시간 학습을 시작하므로 사용자가 학습 서버에서 직접 실행합니다. 문서 작성 과정에서는 실행하지 않았습니다.

```bash
PYTHONPATH="$PWD/src" python - <<'PY'
import json
import sys
from pathlib import Path
import torch
assert torch.cuda.is_available(), "CUDA GPU가 필요합니다."
import msk_ssl_phase1 as train

settings = json.loads(Path("local_config.json").read_text())
assert settings["experiment"] == "E0", "이 예제는 E0용입니다."
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

현재 평가 서버에서는 `python`을 `/opt/conda/bin/python`으로 바꾸어도 사전학습 데이터가 없으므로 이 학습을 진행할 수 없습니다. 기존 출력 디렉터리를 재사용하지 마세요. 학습 코드에는 자동 resume 기능이 있습니다.

## 설정 부록 읽는 법

아래 JSON은 저장된 설정의 복사본이며 서버의 절대 경로는 범용 예시로 치환했습니다. 원본 `config.json`과 가중치는 수정하지 않았습니다. `dataset_cache_list`는 새 데이터의 이전 파일 목록을 재사용하지 않도록 `null`로 바꾸었습니다. 그 밖의 실험 파라미터는 보존했습니다.

`data_root`, `output_root`, 필요 시 `gram_teacher_ckpt`를 실제 환경에 맞추세요. E1c·E2d 부록은 기본 `msk_ssl_phase1.py`에 바로 넣어 학습할 수 있는 설정이 아닙니다. 별도 실험 파일과의 대응을 확인한 뒤 사용해야 합니다.

<!-- saved-config:E0 -->
<details>
<summary>E0: 저장 설정 전체 보기</summary>

출처: `src/ssl/E0/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": false,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
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
  "eval_input_size": 224,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "E0",
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
  "global_crop_size": 224,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": true,
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
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 98,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
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
  "output_root": "./new_training_outputs/phase1",
  "part_grid_size": 4,
  "patch_center_momentum": 0.9,
  "patch_size": 14,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_embed_grid": 16,
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
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "run_tag": "",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.996,
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


<!-- saved-config:E1c -->
<details>
<summary>E1c: 저장 설정 전체 보기</summary>

출처: `src/ssl/E1_fn_safe/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": false,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
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
  "eval_input_size": 224,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "E1",
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
  "global_crop_size": 224,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": true,
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
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 98,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
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
  "output_root": "./new_training_outputs/phase1",
  "part_grid_size": 4,
  "patch_center_momentum": 0.9,
  "patch_size": 14,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_embed_grid": 16,
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
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "run_tag": "fn_safe",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.996,
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


<!-- saved-config:E2d -->
<details>
<summary>E2d: 저장 설정 전체 보기</summary>

출처: `src/ssl/E2_indep_part_gram/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

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
  "eval_input_size": 224,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "E2",
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
  "global_crop_size": 224,
  "gram_teacher_ckpt": "/path/to/gram_teacher.pt",
  "gram_teacher_key": "teacher_backbone",
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": true,
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
  "lambda_gram": 1.0,
  "lambda_ibot": 1.0,
  "lambda_koleo": 0.1,
  "lambda_pos": 0.1,
  "layerscale_init": 1e-05,
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 98,
  "local_preservation": "gram",
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
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
  "output_root": "./new_training_outputs/phase1",
  "part_grid_size": 4,
  "partwhole_center_whole": true,
  "partwhole_containment_threshold": 0.8,
  "partwhole_geom_dim": 8,
  "partwhole_local_part_feature": "patch_mean",
  "partwhole_part_source": "local_crop",
  "patch_center_momentum": 0.9,
  "patch_size": 14,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_embed_grid": 16,
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
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "run_tag": "indep_part_gram",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.996,
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


<!-- saved-config:E3 -->
<details>
<summary>E3: 저장 설정 전체 보기</summary>

출처: `src/ssl/E3/config.json`의 `config`. 서버 경로와 캐시 목록은 위 설명대로 치환했습니다.

```json
{
  "K_cls": 65536,
  "K_patch": 65536,
  "K_pos": 1024,
  "anatomical_annotation_csv": null,
  "auto_resume": false,
  "batch_size": 256,
  "brightness_jitter": 0.4,
  "center_momentum": 0.9,
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
  "eval_input_size": 224,
  "eval_num_images": 256,
  "eval_seed": 12345,
  "eval_use_teacher": true,
  "exclude_dir_names": [
    "keem_ckpt"
  ],
  "experiment": "E3",
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
  "global_crop_size": 224,
  "head_bottleneck_dim": 256,
  "head_hidden_dim": 2048,
  "head_n_layers": 3,
  "head_norm_last_layer": true,
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
  "local_blur_prob": 0.5,
  "local_crop_scale": [
    0.05,
    0.4
  ],
  "local_crop_size": 98,
  "log_interval": 10,
  "lr": 0.0005,
  "lr_linear_scaling": true,
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
  "output_root": "./new_training_outputs/phase1",
  "part_grid_size": 4,
  "patch_center_momentum": 0.9,
  "patch_size": 14,
  "patch_student_temp": 0.1,
  "patch_teacher_temp_final": 0.07,
  "patch_teacher_temp_start": 0.04,
  "persistent_workers": true,
  "pin_memory": true,
  "pos_embed_grid": 16,
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
  "python_executable": "python",
  "qkv_bias": true,
  "rad_dino_local_path": null,
  "rad_dino_model_id": "microsoft/rad-dino",
  "run_tag": "",
  "save_iteration_log": true,
  "scan_workers": 16,
  "seed": 42,
  "solarize_prob": 0.2,
  "start_warmup_lr": 0.0,
  "steps_per_epoch": 2000,
  "student_temp": 0.1,
  "teacher_momentum": 0.996,
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
