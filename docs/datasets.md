# 데이터와 가중치 준비

[프로젝트 소개로 돌아가기](../README.md)

## 학습 데이터

자기지도 학습에는 MSK X-ray 약 349만 장을 사용한 것으로 프로젝트 기록에 남아 있습니다. 학습 파일의 원래 `data_root`는 별도 학습 서버 경로이며, 이 평가 서버에는 그 사전학습 데이터가 없습니다.

저장된 주요 실험 설정의 `max_train_images`는 `null`이고 `steps_per_epoch`는 2,000입니다. 이미지 수와 epoch 정의를 함께 확인해야 합니다. 100 epoch를 전체 이미지 100회 순회라고 해석하면 안 됩니다.

학습 데이터와 그 파일 목록은 이 저장소의 공개 대상에 포함하지 않습니다. 현재 외부 사용자를 위한 학습 데이터 다운로드 경로는 제공되지 않았으므로, 같은 데이터로 재학습하려면 별도 데이터 접근이 필요합니다. 새 데이터에서 학습하는 사용자는 학습 설정의 `data_root`를 자신의 PNG/JPG/JPEG 이미지 폴더로 지정합니다.

## 평가 데이터 구조

아래는 현재 평가 로더가 기대하는 구조입니다. 실제 데이터 원본을 수정·이동·이름 변경하지 마세요.

```text
data/
├── MTDDH/
│   └── Dataset1/Keypoints/
│       ├── Keypoints_Train.json
│       └── Train/                 # 이미지
├── HVAngleEst/
│   └── <배포본의 하위 폴더>/
│       ├── dataset.csv            # 또는 datasets.csv
│       └── images/
└── Spine/
    ├── data/training/             # *.jpg
    └── labels/training/           # 대응하는 *.mat
```

| 부위 | 현재 코드의 입력 조건 |
|---|---|
| 고관절 | MTDDH Dataset1의 COCO 형태 keypoints JSON과 Train 이미지 폴더 |
| 발 | `dataset.csv` 또는 `datasets.csv`와 같은 폴더의 `images/`; `filename`, `great_toe`, `first_metatarsal`, `second_metatarsal` 열 필요 |
| 척추 | AASCE 형식, `.mat`의 `p2`에 68×2 좌표 |

HVAngleEst는 루트 아래의 CSV를 재귀 탐색해 정렬상 첫 파일을 선택하므로 서로 다른 배포본을 한 루트에 섞지 마세요. 발 영상은 좌우 방향을 통일하는 `canonical_flip=True, flip_label="left"` 설정을 사용합니다.

척추는 배포본의 `training` 폴더를 읽은 뒤 연구 코드에서 다시 train/validation/test를 나눕니다. 배포본 공식 test split에 대한 평가라고 해석하면 안 됩니다. 자세한 분할 규칙은 [평가 문서](evaluation.md)를 참고하세요.

다운로드할 원본 배포처와 데이터 이용 조건은 공개자가 추가 확인해야 합니다. 데이터셋 이름만으로 특정 배포본의 URL을 추측해 기재하지 않았습니다.

근거: `src/eval_landmark_metrics.py:98`, `src/landmark_eval.py:137`, `:250`, `:356`.

## 자체 학습 가중치

현재 로더는 다음 상대 구조에 대응합니다. 기본 절대 경로는 `/workspace/msk/src/ssl`이며, 다른 위치에서 쓰는 방법은 [평가 문서](evaluation.md)에 있습니다.

```text
src/ssl/
├── E0/best.pt
├── E1_fn_safe/best.pt
├── E2_indep_part_gram/best.pt
├── E3/best.pt
├── D3-E0/best.pt
├── D3-E1_e1c/best.pt
├── D3-E1_e1c_within/best.pt
└── D3-E2_adam_bf05/
    ├── best.pt
    └── latest.pt                  # D3-E2-Adam-ep99로 평가할 때
```

위는 필요한 파일의 배치 안내입니다. 자체 가중치의 공개 다운로드 링크는 아직 정해지지 않았으며, 저장소를 clone해도 이 파일들은 내려오지 않습니다. 공개자가 가중치를 배포할 때에는 모델 이름, 설정, checkpoint 선택 기준, epoch, SHA-256, 이용 조건을 함께 안내해야 합니다. 기존 가중치를 덮어쓰지 마세요.

자체 가중치의 backbone 평가는 teacher를 사용하고 bf16을 끕니다. 근거: `src/msk_backbones.py:58`.

## 공개 비교 모델

| 평가 이름 | 코드의 모델 식별자 |
|---|---|
| DINOv2 | `facebookresearch/dinov2`의 `dinov2_vits14` |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| RAD-DINO | `microsoft/rad-dino` |

첫 로딩 시 인터넷 연결과 모델별 접근 권한이 필요할 수 있습니다. DINOv3 로더는 환경변수 `HF_TOKEN`을 읽습니다. 토큰 값은 코드·문서·명령 기록에 넣어 공유하지 마세요. 각 모델의 이용 조건은 원 배포처를 확인해야 합니다.

근거: `src/msk_backbones.py:53`, `:128`, `:167`, `src/msk_ssl_phase1.py:175`.

## 공개 범위

이 안내의 landmark 평가는 `JPG_DICOM/`을 요구하지 않습니다. DICOM과 환자 식별 정보, 원본 데이터, 이미지 경로 목록을 공개 코드에 포함하지 않습니다. 원본 데이터가 공개 데이터여도 헤더의 식별 정보를 출력하거나 문서에 옮기지 마세요.
