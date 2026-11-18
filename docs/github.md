# GitHub에 올리고 다른 사람에게 공유하기

[프로젝트 소개로 돌아가기](../README.md)

GitHub는 코드와 설명서를 보관하고 링크로 공유하는 곳입니다. 처음에는 필요한 파일만 별도 공개용 폴더에 복사해 올리는 방법을 권합니다. 현재 연구 폴더와 과거 Git 이력은 그대로 보존할 수 있습니다.

이 문서는 사용자가 직접 실행하는 안내입니다. 문서 작성 과정에서는 저장소 생성·커밋·push·공개를 수행하지 않았습니다. 이 프로젝트 규칙상 에이전트는 사용자 승인 없이 커밋하지 않으며 `git push`를 실행하지 않습니다.

## 1. 어떤 파일을 올릴지 정하기

첫 공개의 최소 목록입니다.

```text
README.md
docs/setup.md
docs/datasets.md
docs/phase1.md
docs/phase2.md
docs/evaluation.md
docs/github.md
src/msk_ssl_phase1.py
src/msk_ssl_phase1_e1c.py
src/msk_ssl_phase1_e2d.py
src/msk_ssl_phase2.py
src/msk_backbones.py
src/landmark_eval.py
src/eval_landmark_metrics.py
```

공통 학습·평가 코드와 추가로 확인된 Phase 1 세부 실험 파일을 선택한 목록입니다. 두 세부 파일은 과거 학습 코드와의 대응을 확인한 뒤 공개하세요. `train.py`, `train_phase2.py`는 문서의 API 예제에 필수가 아니므로 첫 목록에서 제외했습니다. 코드·문서는 먼저 내용을 검토하세요. 이번 문서 작성에서 코드 파일 자체나 전체 Git 이력의 비밀정보 검사를 완료한 것은 아닙니다.

`data/`, 가중치·캐시·원본 결과 폴더, 발표자료, 가상환경, `local_config.json`, 원래 작업 폴더의 `.git/`은 복사하지 않습니다. 노트북도 첫 공개에는 필수가 아닙니다. 추가할 때는 출력과 경로를 검토하고 공개용 복사본에서만 출력을 제거하세요. `.gitattributes`가 있다는 이유만으로 GitHub 웹 업로드 때 출력이 자동 제거되는 것은 아닙니다.

## 2. 공개용 폴더에 복사하기

아래 예시는 `/workspace/msk/public_release_v1`이 **아직 없을 때** 사용합니다. 같은 이름이 이미 있으면 새 폴더 이름으로 명령 전체를 바꾸세요. 기존 파일을 지우거나 덮어쓰지 않습니다.

```bash
cd /workspace/msk
mkdir public_release_v1
```

`mkdir`가 실패하면 여기서 중단합니다. 성공했을 때만 아래를 실행하세요.

```bash
mkdir public_release_v1/src public_release_v1/docs
cp -n README.md public_release_v1/
cp -n docs/setup.md docs/datasets.md docs/phase1.md docs/phase2.md docs/evaluation.md docs/github.md public_release_v1/docs/
cp -n src/msk_ssl_phase1.py src/msk_ssl_phase2.py src/msk_backbones.py src/landmark_eval.py src/eval_landmark_metrics.py public_release_v1/src/
cp -n src/msk_ssl_phase1_e1c.py src/msk_ssl_phase1_e2d.py public_release_v1/src/
```

`cp -n`은 대상 파일이 있으면 덮어쓰지 않는 옵션입니다. 별도 공개용 폴더는 원래 저장소의 새로운 미추적 항목으로 보일 수 있습니다. 원래 저장소에서 그 폴더를 통째로 `git add`하지 마세요. 아래 Git 명령은 반드시 공개용 폴더 안에서만 실행합니다.

## 3. GitHub에서 빈 저장소 만들기

1. GitHub에 로그인합니다.
2. 우측 상단 `+` → `New repository`를 선택합니다.
3. 이름을 입력합니다. 예: `msk-xray-ssl`.
4. 준비 중이면 `Private`, 누구나 볼 수 있게 공개하려면 `Public`을 선택합니다.
5. 이 안내처럼 로컬 파일을 가져올 때는 README·.gitignore·license 자동 생성 옵션을 선택하지 않고 빈 저장소를 만듭니다.

공개 전에 데이터 배포처와 가중치 링크의 제공 여부를 명확히 하고, 권한에 맞는 `LICENSE`를 추가해야 합니다. 라이선스 종류는 소유자와 가져온 코드의 조건을 확인해 결정합니다. 이 안내가 특정 라이선스를 대신 선택하지는 않습니다.

공식 안내: [저장소 만들기](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository), [라이선스 안내](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

## 4. 터미널에서 첫 업로드

서버에 있는 파일은 터미널로 올리는 방법이 편합니다. 아래 명령은 **사용자가 직접 실행**합니다.

```bash
cd /workspace/msk/public_release_v1
git init -b main
git add README.md docs/setup.md docs/datasets.md docs/phase1.md docs/phase2.md docs/evaluation.md docs/github.md
git add src/msk_ssl_phase1.py src/msk_ssl_phase2.py src/msk_backbones.py src/landmark_eval.py src/eval_landmark_metrics.py
git add src/msk_ssl_phase1_e1c.py src/msk_ssl_phase1_e2d.py
git status --short
git diff --cached --stat
```

위 목록을 검토하고 올릴 파일이 맞으면 커밋합니다. 직접 작성한 `LICENSE`가 있다면 그 파일도 검토한 뒤 `git add LICENSE`로 별도 선택하세요.

```bash
git commit -m "MSK 자기지도 학습·평가 코드와 사용 문서 첫 공개"
```

작성자 정보가 없다고 나오면 공개용 저장소에서 사용할 이름과 이메일을 `git config user.name`, `git config user.email`로 설정합니다. 이메일은 GitHub에서 제공하는 noreply 주소를 사용할 수 있습니다.

다음 URL의 `YOUR_NAME`과 `YOUR_REPOSITORY`를 GitHub에서 만든 저장소 주소로 바꿉니다. `git remote -v`로 주소를 확인한 후 업로드합니다.

```bash
git remote add origin https://github.com/YOUR_NAME/YOUR_REPOSITORY.git
git remote -v
git push -u origin main
```

HTTPS 인증에는 GitHub 계정 비밀번호 대신 credential manager 또는 개인 액세스 토큰을 사용합니다. 토큰을 URL이나 코드에 넣지 마세요. 이 명령으로 올라가는 것은 현재 공개용 폴더에서 만든 commit과 그 이력입니다.

공식 안내: [로컬 코드를 GitHub에 추가](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github), [원격 저장소와 인증](https://docs.github.com/en/get-started/git-basics/about-remote-repositories).

## 5. 브라우저로 올리는 대안

터미널 인증이 어렵다면 공개용 폴더의 파일을 자신의 컴퓨터로 받아서 올릴 수 있습니다.

1. 빈 GitHub 저장소에서 `uploading an existing file`을 선택합니다. 파일이 이미 있으면 `Add file` → `Upload files`를 선택합니다.
2. 공개용 폴더 안의 `README.md`, `docs/`, `src/`를 올립니다. 최상위에 `public_release_v1/` 폴더가 한 겹 더 생기지 않도록 확인합니다.
3. 업로드 목록을 확인하고 commit합니다. 이 동작도 GitHub에 변경 이력을 남깁니다.

브라우저 업로드에는 로컬 `.gitignore`가 공개 대상 선별을 대신하지 않습니다. 준비한 최소 목록만 올리세요. 터미널 업로드와 브라우저 업로드 중 처음에는 하나를 선택하면 됩니다.

## 6. 링크로 공유하기

| 공유할 내용 | 보낼 링크 예시 |
|---|---|
| 프로젝트 전체 | `https://github.com/YOUR_NAME/YOUR_REPOSITORY` |
| Phase 1 설명 | `https://github.com/YOUR_NAME/YOUR_REPOSITORY/blob/main/docs/phase1.md` |
| Phase 2 설명 | `https://github.com/YOUR_NAME/YOUR_REPOSITORY/blob/main/docs/phase2.md` |
| 평가 설명 | `https://github.com/YOUR_NAME/YOUR_REPOSITORY/blob/main/docs/evaluation.md` |
| 고정된 배포 버전 | `https://github.com/YOUR_NAME/YOUR_REPOSITORY/releases/tag/v1.0.0` |

`Public`이면 저장소 링크를 전달하면 됩니다. `Private`이면 링크 전달 외에 상대방에게 저장소 접근 권한을 부여해야 합니다.

상대방은 GitHub의 `Code` → `Download ZIP` 또는 다음 명령으로 코드를 받을 수 있습니다.

```bash
git clone https://github.com/YOUR_NAME/YOUR_REPOSITORY.git
```

전달할 메시지 예시:

> MSK X-ray 자기지도 학습·평가 코드를 공유합니다. README의 설치와 데이터 준비 안내부터 확인해 주세요. Phase 1과 Phase 2의 실험 차이는 docs에 정리했습니다. 데이터와 자체 학습 가중치는 코드에 포함되지 않으며, 세부 실험의 과거 학습 재현 여부는 문서의 현재 상태를 확인해 주세요.

## 7. 첫 버전을 고정해 두기

준비된 범위를 검토한 뒤 GitHub 저장소의 `Releases` → `Draft a new release`에서 `v1.0.0` 태그를 만들고 설명을 적습니다. 학습·평가 재현이 미확인이라면 릴리스 설명에도 해당 상태를 명시하세요. 준비 단계에는 draft나 pre-release로 둘 수 있습니다.

수정한 코드는 나중에 `v1.0.1`, `v1.1.0` 등 새 버전으로 공유합니다. D3-E0·D3-E1c는 모델 실험 이름이고 `v1.0.0`은 공개 코드 묶음의 버전입니다. 한 릴리스 안에 여러 실험 코드가 함께 들어갑니다.

공식 안내: [Release 만들기](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
