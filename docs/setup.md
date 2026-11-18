# 설치와 실행 환경

[프로젝트 소개로 돌아가기](../README.md)

## 평가에 사용한 환경

아래는 2026-11-18 현재 평가 서버의 `/opt/conda/bin/python`에서 패키지 메타데이터로 확인한 버전입니다. 과거 학습 환경과는 다릅니다.

| 항목 | 버전 |
|---|---|
| Python | 3.10.13 |
| torch | 2.4.1+cu118 |
| torchvision | 0.19.1+cu118 |
| transformers | 4.57.6 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.8 |
| Pillow | 10.2.0 |
| tqdm | 4.65.0 |
| safetensors | 0.7.0 |
| huggingface-hub | 0.36.2 |

기존 평가 서버에서는 base 환경을 그대로 사용합니다. `msk_v2` 환경으로 바꾸거나 base의 패키지를 재설치하지 않습니다.

## 다른 컴퓨터에서 평가 환경 준비

다음은 Python 3.10과 NVIDIA GPU를 사용하는 Linux용 설치 예시입니다. 새 환경에서의 전체 설치·실행 검증은 아직 수행하지 않았습니다. GPU·드라이버가 CUDA 11.8용 PyTorch와 호환되어야 합니다.

저장소 최상위에서 실행합니다. `python3.10`은 실제 설치된 Python 3.10 실행 파일에 맞추세요.

```bash
python3.10 -m venv .venv_eval
source .venv_eval/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install transformers==4.57.6 numpy==1.26.4 pandas==2.3.3 scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.8 Pillow==10.2.0 tqdm==4.65.0 safetensors==0.7.0 huggingface-hub==0.36.2
```

PyTorch 설치 명령은 [공식 이전 버전 설치 안내](https://pytorch.org/get-started/previous-versions/)의 2.4.1 / CUDA 11.8 조합을 기준으로 합니다. `timm`, `umap-learn`은 Phase 1의 선택 패키지이며, 위의 공통 평가 설치 예시에는 포함하지 않았습니다.

생성한 가상환경 폴더는 GitHub 업로드 대상에서 제외하세요. 현재 프로젝트의 `.gitignore`에 모든 가상환경 이름이 등록되어 있는 것은 아닙니다.

## GPU 선택과 확인

먼저 사용 현황을 보고 사용할 GPU 번호를 사용자 또는 서버 관리자와 정합니다.

```bash
nvidia-smi
```

결정한 번호를 아래 입력으로 받습니다. 코드나 예제에 특정 GPU 번호를 고정하지 않습니다.

```bash
read -r -p "사용하기로 정한 GPU 번호: " MSK_GPU_ID
export CUDA_VISIBLE_DEVICES="${MSK_GPU_ID:?GPU 번호를 입력하세요}"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU를 찾지 못했습니다. 실행을 중단합니다."
print("device:", torch.device("cuda:0"))
print("GPU:", torch.cuda.get_device_name(0))
PY
```

환경변수로 선택한 물리 GPU는 프로세스 안에서 `cuda:0`으로 표시됩니다. 기존 평가 서버에서 위 Python 명령을 실행할 때는 `python` 대신 `/opt/conda/bin/python`을 사용하세요.

두 학습 파일은 import 시에도 CUDA와 패키지를 확인합니다. GPU 환경변수는 import 전에 설정해야 합니다. 기본 GPU 설정은 Phase 1 파일의 11행, Phase 2 파일의 23행에 남아 있습니다.

## 학습 환경은 별도 확인이 필요합니다

저장된 학습 `config.json`의 `environment`에는 Python `3.12.3`, torch `2.13.0+cu130`, torchvision `0.28.0+cu130`, CUDA `13.0`이 기록되어 있습니다. Phase 1 설정에는 transformers `5.17.0`도 기록되어 있습니다.

이 값은 설정 파일에 적힌 기록이며, 이 서버에서 해당 패키지의 설치 가능 여부나 실제 학습 환경과의 일치를 검증한 것은 아닙니다. 이를 근거로 현재 평가 서버 패키지를 바꾸지 마세요. 학습 환경을 공개하려면 학습 서버의 실제 환경 목록을 추가로 확인해야 합니다.

근거: `src/ssl/E0/config.json` 및 Phase 2 각 실험 폴더의 `config.json` 안 `environment`. 원본 파일은 읽기 전용으로 확인했으며 공개 코드에 자동 포함되지 않습니다.
