# landmark 평가 방법

[프로젝트 소개로 돌아가기](../README.md)

## 필요한 코드

`eval_landmark_metrics.py`, `landmark_eval.py`, `msk_backbones.py`와 두 학습 파일을 `src/`에 함께 둡니다. 평가 로더가 학습 파일의 모델 정의를 사용하므로 평가 실행 파일 하나만 공유하면 충분하지 않습니다.

설치와 GPU 선택은 [환경 문서](setup.md), 데이터·가중치 배치는 [데이터 문서](datasets.md)를 참고하세요.

## 세 가지 평가

| 옵션 | 무엇을 계산하나요? | GPU·가중치 |
|---|---|---|
| `cluster` | GT 위치의 patch feature에 kNN 정확도와 Silhouette 계산 | 필요 |
| `zeroshot` | train GT feature로 prototype을 만들고 test patch 유사도 최댓값으로 위치 예측 | 필요 |
| `de` | 기존 decoder 평가 결과를 읽고 label efficiency 집계 | 집계만 하면 불필요 |
| `de --train-de` | 필요한 decoder를 새로 학습하고 평가 | 필요 |

`de`는 기본적으로 기존 결과를 읽습니다. 코드·가중치만 받은 새 사용자가 `--metrics de`만 실행하면 decoder 결과가 없어서 미완료로 표시될 수 있습니다. 새 결과가 필요하면 `--train-de`를 명시해야 합니다.

공통 평가 모듈 `EvalConfig` 자체는 기본 3,000 step입니다. 여기서 소개하는 실행 스크립트의 `--de-steps` 기본값은 주 비교 기준인 **12,000 step**입니다. 둘을 혼동하지 마세요.

## 현재 서버에서 실행

먼저 사용할 GPU를 정해 `CUDA_VISIBLE_DEVICES`를 설정합니다. 아래는 기존 가중치 두 모델의 클러스터링·zero-shot 실행 예시입니다. `--tag`는 매번 새 이름으로 바꿉니다.

```bash
cd /workspace/msk
/opt/conda/bin/python src/eval_landmark_metrics.py \
  --models D3-E0 D3-E1c \
  --parts hip foot spine \
  --metrics cluster zeroshot \
  --tag my_first_eval_20261118_01
```

기존 12,000 step 결과를 CPU로 집계만 할 때는 다음과 같습니다. 결과 CSV가 있어야 하며, 없으면 미완료로 표시됩니다.

```bash
/opt/conda/bin/python src/eval_landmark_metrics.py \
  --models D3-E0 D3-E1c \
  --metrics de \
  --tag my_de_summary_20261118_01
```

decoder를 새로 학습하는 명령은 아래와 같습니다. GPU 사용을 미리 확인하고 사용자가 직접 실행합니다. 모델·부위 조합에 따라 시간이 오래 걸립니다. 기존에 완료된 결과가 있으면 코드가 그것을 사용할 수 있으므로 이 옵션을 모든 기존 run을 강제로 재학습하는 뜻으로 해석하지 마세요.

```bash
/opt/conda/bin/python src/eval_landmark_metrics.py \
  --models D3-E0 D3-E1c \
  --parts hip foot spine \
  --metrics de --train-de --de-steps 12000 \
  --tag my_de_training_20261118_01
```

## 다른 위치에 clone한 경우

현재 코드에는 `DATA="/workspace/msk/data"`, `SSL_ROOT="/workspace/msk/src/ssl"`이 고정되어 있고 이를 바꾸는 CLI 옵션은 없습니다. 다른 컴퓨터에서는 아래처럼 모듈의 경로를 지정하고 `main()`을 호출할 수 있습니다. 코드 파일을 수정하지 않는 실행 예시입니다.

저장소 최상위에 `data/`, `src/ssl/`을 준비하고, [환경 문서](setup.md)대로 GPU를 선택한 뒤 실행하세요. `python`은 준비한 평가 환경의 실행 파일을 사용합니다.

```bash
PYTHONPATH="$PWD/src" python - \
  --models D3-E0 D3-E1c --parts hip foot spine \
  --metrics cluster zeroshot --tag portable_eval_20261118_01 <<'PY'
from pathlib import Path
from dataclasses import replace
import msk_backbones as backbones
import eval_landmark_metrics as evaluation

root = Path.cwd()
evaluation.DATA = str(root / "data")
backbones.SSL_ROOT = str(root / "src" / "ssl")
backbones._cfg_p1 = replace(backbones._cfg_p1, output_root=backbones.SSL_ROOT)
backbones._cfg_p2 = replace(backbones._cfg_p2, output_root=backbones.SSL_ROOT)
evaluation.main()
PY
```

같은 명령의 `--metrics cluster zeroshot`을 `--metrics de --train-de`로 바꾸면 새 decoder 학습을 포함합니다. 기존 결과 집계만 CPU로 하려면 위 예제의 `msk_backbones` import와 관련 경로 설정을 빼고 `evaluation.DATA`를 지정한 뒤 `--metrics de`로 호출합니다.

## 분할과 보고 규칙

- seed는 42/43/44입니다. 기본 분할은 이미지 단위 train 60%, validation 20%, test 20%입니다. 제외 대상은 분할 후에 제거하므로 최종 건수가 정확히 이 비율이 아닐 수 있습니다. 환자 단위 독립성을 보장하는 분할은 아닙니다.
- 고관절 주 결과는 standard 자세만 집계합니다.
- label 비율은 train split의 5/10/25/100%입니다. 최소 학습 이미지 수 설정이 있어 작은 데이터에서는 비율 그대로의 건수가 아닐 수 있습니다.
- 주 비교는 decoder 12,000 step, 각 비율 draw 1개, seed 평균 ± SD입니다. 3,000 step 보조 설정은 draw 수가 다릅니다.
- test 정답으로 모델·하이퍼파라미터·threshold를 선택하지 않습니다.
- 오탐은 오차가 영상 대각선의 5%를 넘는 경우입니다.
- 3개 seed가 모두 완료되지 않은 칸은 미완료로 표시합니다. 부분 결과로 모델 우열을 결론 내리지 않습니다.
- 고관절 기본 제외 index는 451, 척추 실행 스크립트의 제외 index는 394/397/420/445입니다. index는 로더의 기록 순서를 따르므로 다른 배포본이나 파일 순서에 그대로 적용하기 전에 확인해야 합니다.
- 클러스터링 kNN은 영상 단위 group을 사용하는 5-fold 비교이며 decoder의 train/validation/test 평가와 다른 절차입니다. t-SNE는 그림용이고 지표는 원래 feature 공간에서 계산합니다.

근거: `src/landmark_eval.py:97`, `:632`, `:640`, `src/eval_landmark_metrics.py:85`, `:92`, `:137`, `:749`.

## 출력 확인

출력은 `src/outputs/ssl/metrics_py/<tag>/`에 저장됩니다. 같은 tag가 이미 있으면 중단합니다.

| 파일 또는 폴더 | 내용 |
|---|---|
| `run_config.json` | 실행 옵션, Git commit, 환경과 checkpoint 기록 |
| `summary.md` | 실행 결과 요약 |
| `check_vs_existing.csv` | 기존 결과와의 대조 |
| `<부위>/cluster.csv` | 클러스터링 결과 |
| `<부위>/zeroshot_summary.csv` | zero-shot 결과 |
| `<부위>/de_summary.csv` | label efficiency 결과 |
| `<부위>_de_<설정>/` | 새로 학습한 decoder 평가 결과와 설정 |

실행한 지표에 해당하는 파일만 생깁니다. 기존 대조 결과가 없어 대조를 건너뛰는 것은 새 평가 자체의 실패와 구분하세요. 결과를 보고할 때는 실패·미완료 항목도 함께 적고, 수치의 출처 파일과 코드 버전, checkpoint를 연결합니다.

Git commit 기록만으로 커밋되지 않은 변경을 복원할 수는 없습니다. 공유용 재현 결과를 만들 때는 승인받아 코드를 커밋하고 작업 상태를 확인한 뒤 실행하는 것이 좋습니다.
