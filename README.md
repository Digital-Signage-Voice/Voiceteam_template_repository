# 🚀 Digital Signage Voice Project (Voice Team)

안녕하세요! Voice팀의 공식 GitHub 리포지토리입니다.
우리 팀은 영상과 음성 신호 처리를 통해 잡음 환경 속에서도 명확한 음성을 추출하는 파이프라인을 개발합니다.

## 📁 프로젝트 파일 구조

프로젝트의 전체 파일 구조는 다음과 같습니다. 이 구조를 기준으로 각자 맡은 위치에서 작업을 진행해 주세요.

```
digital-signage-voice/
│
├── 📂 data/
│   ├── 📂 input/              # (테스트용 원본 영상 파일을 이곳에 넣어주세요)
│   │   ├── cafe_video.mp4
│   │   └── street_video.mp4
│   │
│   └── 📂 output/             # (파이프라인 실행 결과물이 저장되는 곳입니다)
│       ├── cafe_clean_audio.wav
│       └── street_clean_audio.wav
│
├── 📂 src/                    # (우리 팀의 핵심 소스코드가 들어가는 곳입니다)
│   │
│   ├── 📄 video_module.py     # [담당: 현지님] 영상 분석, 발화 구간 탐지 모듈
│   │
│   ├── 📄 audio_module.py     # [담당: 원후님] 음성 잡음 제거 기술 모듈
│   │
│   └── 📄 main_pipeline.py   # [담당: 해찬님] 전체 파이프라인을 실행하는 메인 파일
│
├── 📂 tests/                  # (각 모듈이 잘 작동하는지 테스트하는 코드를 넣는 곳입니다)
│   ├── 📄 test_video_module.py
│   └── 📄 test_audio_module.py
│
├── 📄 .gitignore             # (Git에 올릴 필요가 없는 파일/폴더를 지정합니다)
│
├── 📄 requirements.txt       # (프로젝트에 필요한 모든 라이브러리 목록입니다)
│
└── 📄 README.md              # (바로 이 파일! 프로젝트의 대문입니다)
```

<br>

## 📌 폴더 및 파일 설명

* **`data/`**: 테스트에 사용되는 모든 데이터 파일을 관리하는 폴더입니다.
    * **`input/`**: 우리가 촬영하거나 다운로드한 원본 영상/음성 파일을 저장합니다.
    * **`output/`**: 우리의 파이프라인을 거쳐 최종적으로 생성된 깨끗한 음성 파일을 저장합니다.
* **`src/` (Source Code)**: 우리 프로젝트의 심장입니다. 모든 파이썬 소스 코드는 이곳에 위치합니다.
    * **`video_module.py`**: **(현지님 담당)** 영상 파일을 분석해서, 사람이 말하는 시간 구간을 찾아내는 모든 코드가 들어갑니다.
    * **`audio_module.py`**: **(원후님 담당)** 오디오 데이터의 잡음을 제거하는 모든 기술 코드가 들어갑니다.
    * **`main_pipeline.py`**: **(해찬님 담당)** 현지님과 원후님의 모듈을 가져와, 전체 처리 과정을 순서대로 실행시키는 지휘자 역할을 하는 파일입니다. **프로젝트 실행은 이 파일을 통해 이루어집니다.**
* **`tests/`**: 각 모듈이 독립적으로 잘 작동하는지 확인하기 위한 테스트 코드를 작성하는 공간입니다. (예: `video_module`이 정확한 시간 값을 돌려주는지 테스트)
* **`.gitignore`**: `__pycache__` 폴더나 개인 설정 파일처럼, GitHub에 공유할 필요 없는 것들을 지정하는 파일입니다.
* **`requirements.txt`**: 우리 프로젝트를 실행하는 데 필요한 모든 파이썬 라이브러리(`opencv-python`, `moviepy` 등)와 그 버전을 적어두는 파일입니다. 새로운 팀원이 왔을 때 `pip install -r requirements.txt` 명령어 한 번으로 쉽게 환경을 설정할 수 있습니다.
* **`README.md`**: 프로젝트의 목표, 사용법, 팀원 소개 등을 적는 가장 중요한 문서입니다.

## 💻 실행 방법

1.  이 리포지토리를 `git clone` 합니다.
2.  `pip install -r requirements.txt` 명령어로 필요한 라이브러리를 모두 설치합니다.
3.  `data/input/` 폴더에 테스트할 영상 파일을 넣습니다.
4.  `src/main_pipeline.py` 파일을 실행하면, `data/output/` 폴더에 결과물이 생성됩니다.

```bash
python src/main_pipeline.py
```

<br>

---
*이 README는 Gemini의 도움을 받아 작성되었습니다.*
