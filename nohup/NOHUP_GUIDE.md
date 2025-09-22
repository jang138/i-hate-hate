# nohup 사용 가이드 (백그라운드 학습/추론 실행)

## nohup이란?
- 터미널을 닫아도 프로세스가 계속 실행되도록 하는 유틸리티입니다.
- SIGHUP 신호를 무시하고, 표준출력/표준에러를 파일로 리다이렉트해 로그를 남깁니다.

## 기본 사용법
```
nohup <명령어> > <로그파일> 2>&1 &
```
- `> <로그파일>`: 표준출력 리다이렉트
- `2>&1`: 표준에러를 표준출력으로 합침
- 마지막 `&`: 백그라운드 실행

## Python 학습/추론을 nohup으로 실행하기
`src/main.py`를 다양한 하이퍼파라미터로 실행하는 예시입니다.

사전 준비(로그 디렉터리 생성):
```
mkdir -p logs
```

### 1) 기본 실행 예시
```
nohup python -u src/main.py --run_name "ssac-bert" --lr 5e-4 > logs/run_bert_lr5e-4.out 2>&1 &
```
- `-u`: 버퍼링 비활성화로 로그가 실시간에 가깝게 기록됨

### 2) 러닝레이트 변경 실행
```
nohup python -u src/main.py --run_name "ssac-bert" --lr 5e-3 > logs/run_bert_lr5e-3.out 2>&1 &
```

### 3) 모델 변경 실행
```
nohup python -u src/main.py --model_name klue/roberta-large --run_name "ssac-roberta" --lr 3e-5 > logs/run_roberta.out 2>&1 &
```

### 4) 여러 작업을 동시에 실행(각 로그 분리)
```
nohup python -u src/main.py --run_name exp1 --lr 2e-5 > logs/exp1.out 2>&1 &
nohup python -u src/main.py --run_name exp2 --lr 3e-5 > logs/exp2.out 2>&1 &
```

## 프로세스 관리 팁

### PID 저장하기(나중에 중지 용이)
```
nohup python -u src/main.py --run_name exp3 > logs/exp3.out 2>&1 & echo $! > exp3.pid
```
- `$!`: 방금 백그라운드로 띄운 프로세스의 PID

### 진행 상황 확인
```
tail -f logs/exp3.out
```

### 프로세스 목록 보기
```
ps -ef | grep "python -u src/main.py" | grep -v grep
# 또는
pgrep -fa "python -u src/main.py"
```

### 종료하기
```
# PID 파일로 종료
kill "$(cat exp3.pid)"
# 강제 종료(필요 시)
kill -9 "$(cat exp3.pid)"

# PID를 직접 지정하여 종료
kill <PID>
```

## 환경 변수/가상환경과 함께 사용하기
```
# 가상환경 활성화 예시(pyenv)
pyenv activate my_env

# WandB 비활성화 등 환경 변수 설정 후 실행
WANDB_MODE=disabled nohup python -u src/main.py --run_name exp_no_wandb > logs/exp_no_wandb.out 2>&1 &
```

## 주의사항
- 서버/VM 전원 종료 시에는 nohup 프로세스도 종료됩니다(재부팅 후 재시작 필요).
- 로그 파일이 커질 수 있으니 주기적으로 압축/정리하세요.
- GPU 사용 시, 드라이버/CUDA가 올바르게 설치되어 있어야 합니다(`nvidia-smi`로 확인).

## 빠른 레시피(복사-붙여넣기)
```
mkdir -p logs
nohup python -u src/main.py --run_name "ssac-bert" --lr 5e-4 > logs/run_bert_lr5e-4.out 2>&1 & echo $! > run_bert_lr5e-4.pid
```

## 스크립트로 한 번에 실행하기(run_experiments.sh)
- 위치: `korean_AU/nohup/run_experiments.sh`
- 기능: 여러 실험을 한 번에 nohup으로 실행/상태확인/중지/로그보기 지원
- 로그 디렉터리: `korean_AU/logs/`
- PID 디렉터리: `korean_AU/nohup/pids/`

### CONFIG 섹션에서 실험 편집하기
스크립트 상단에 다음과 같은 CONFIG가 있습니다. 여기서 실험을 간단히 수정/추가하세요.
```
PYTHON_BIN="python -u"
MAIN_PY="<프로젝트_루트>/src/main.py"
EXPERIMENTS=(
  "run_bert_lr5e-4|--run_name ssac-bert --lr 5e-4"
  "run_bert_lr5e-3|--run_name ssac-bert --lr 5e-3"
  "run_roberta|--model_name klue/roberta-large --run_name ssac-roberta --lr 3e-5"
)
```
- 형식: "로컬이름|main.py 옵션들"
- 로컬이름(local_name)은 자동으로 로그/ PID 파일명으로 사용됩니다.
  - 로그: `logs/<local_name>.out`
  - PID: `nohup/pids/<local_name>.pid`

예) 러닝레이트와 이름을 바꾸려면:
```
"exp_new|--run_name exp_new --lr 2e-5"
```

### 사용법
```
# 시작(설정된 모든 실험 실행)
bash nohup/run_experiments.sh start

# 상태 확인
bash nohup/run_experiments.sh status

# 종료(설정된 모든 실험 종료)
bash nohup/run_experiments.sh stop

# 로그 보기
bash nohup/run_experiments.sh tail run_bert_lr5e-4
```

