# MSK X-ray 자기지도 학습과 landmark 평가

MSK X-ray에서 자기지도 학습한 모델이 해부학적 landmark의 위치를 얼마나 잘 표현하는지 평가하는 연구 코드입니다. 학습 코드는 DINOv2 방식의 **Phase 1**과 DINOv3 방식의 **Phase 2**로 나눕니다. 실험별 차이는 문서에서 설명합니다.

## 처음 사용하는 분께

1. [설치와 실행 환경](docs/setup.md)을 확인합니다.
2. [데이터와 가중치 준비](docs/datasets.md)를 확인합니다. 코드만 내려받으면 데이터와 학습된 가중치는 포함되지 않습니다.
3. 학습하려면 [Phase 1](docs/phase1.md) 또는 [Phase 2](docs/phase2.md)를 읽습니다.
4. 모델을 평가하려면 [평가 방법](docs/evaluation.md)을 읽습니다.

저장소를 공개하는 사람은 [GitHub 업로드와 공유 안내](docs/github.md)를 참고하세요.

## 코드 구성

| 파일 | 역할 |
|---|---|
| [src/msk_ssl_phase1.py](src/msk_ssl_phase1.py) | DINOv2 방식 기반 자기지도 학습, ViT-B/14 |
| [src/msk_ssl_phase2.py](src/msk_ssl_phase2.py) | DINOv3 방식 기반 자기지도 학습, ViT-B/16 |
| [src/msk_ssl_phase1_e1c.py](src/msk_ssl_phase1_e1c.py), [src/msk_ssl_phase1_e2d.py](src/msk_ssl_phase1_e2d.py) | 추가로 확인된 Phase 1 세부 실험 코드; 아래 재현 상태 참고 |
| [src/msk_backbones.py](src/msk_backbones.py) | 자체 학습 모델과 공개 비교 모델 불러오기 |
| [src/landmark_eval.py](src/landmark_eval.py) | 데이터 로딩, 분할, decoder 학습과 공통 평가 기능 |
| [src/eval_landmark_metrics.py](src/eval_landmark_metrics.py) | 클러스터링, label efficiency, zero-shot 평가 실행·집계 |

여기서 DINOv2·DINOv3 방식은 학습 구조와 목표를 뜻합니다. 두 학습 파일의 기본 backbone 생성은 자체 초기화 방식이며, 공개 pretrained 가중치를 불러와 fine-tuning하는 절차와 구분됩니다. 공개 DINOv2·DINOv3·RAD-DINO는 별도 비교 모델로 불러옵니다.

## 실험 이름

| 구분 | 문서에서 사용하는 실험 이름 |
|---|---|
| Phase 1 | E0, E1c, E2d, E3 |
| Phase 2 | D3-E0, D3-E1c, D3-E1c-within, D3-E2-Adam |

실험 이름과 코드에서 받는 `experiment` 값이 다른 경우가 있습니다. 각 Phase 문서의 대응표를 확인하세요. Phase 2라는 이름이 Phase 1 가중치를 이어서 학습한다는 뜻은 아닙니다.

## 평가 범위

- 부위: 고관절(MTDDH), 발(HVAngleEst), 척추(AASCE 형식).
- 평가: GT patch feature 클러스터링, label efficiency, zero-shot landmark 예측.
- 오탐: 예측 위치가 정답에서 영상 대각선 길이의 5%를 넘게 벗어난 경우.
- 고관절 주 결과: standard 자세만 집계합니다.
- label efficiency 주 비교: decoder 12,000 step, label 5/10/25/100%, seed 42/43/44의 평균 ± SD.

## 현재 제공 범위와 재현 상태

이 문서는 2026-11-18에 작업 폴더의 코드와 저장된 실험 설정을 확인해 작성했습니다. GPU 학습·평가나 새 환경 설치를 다시 실행한 검증은 수행하지 않았습니다.

- 기본 Phase 1 파일에는 저장된 E1c·E2d 설정의 일부 기능이 없습니다. 문서 작성 도중 해당 이름의 별도 코드가 추가로 확인되어 [Phase 1 문서](docs/phase1.md)에 구분했습니다. 과거 학습 당시 코드와 일치하는지는 아직 확인하지 않았습니다.
- `src/train.py`, `src/train_phase2.py`도 작성 도중 추가로 확인되었습니다. 특히 Phase 1 실행기의 기본 import와 세부 실험 코드 연결은 확인이 더 필요합니다. 문서의 학습 예제는 `Config`와 `Trainer`를 직접 호출합니다.
- 평가 코드는 데이터·가중치의 기본 경로가 현재 서버에 맞춰져 있습니다. 다른 위치에서 실행하는 방법은 [평가 문서](docs/evaluation.md)에 있습니다.
- 문서의 설정 부록은 저장된 `config.json`을 기록한 것입니다. 현재 코드가 과거 학습 당시 코드와 완전히 같음을 보증하지 않습니다.
- 데이터 배포 링크, 자체 가중치 다운로드 링크, 코드 라이선스는 아직 확정되지 않았습니다. 다운로드가 준비되어 있다고 가정하지 마세요.

## 이용 조건

현재 이 문서 작성으로 새 라이선스를 지정하지는 않았습니다. 공개자는 배포 권한과 가져온 코드의 조건을 확인하고 적절한 `LICENSE`를 추가해야 합니다. 데이터와 모델 가중치의 이용 조건은 코드와 별도로 확인해야 합니다.
