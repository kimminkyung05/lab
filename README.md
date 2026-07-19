# Lab

해양 운동 시계열의 다변량 예측과 자유 자기회귀(AR) 평가를 위한 Jupyter Notebook 프로젝트입니다.

## 실행 방법

`notebooks` 폴더를 작업 디렉터리로 연 뒤, `00_experiment_runner.ipynb`만 처음부터 실행하세요. 이 노트북이 같은 커널에서 아래 파이프라인을 순서대로 실행합니다.

1. `01_data_preparation.ipynb` — 데이터 로드, 파일 단위 Train/Validation/Test 분할, 스케일링, 선택적 PCA, 시퀀스와 DataLoader 생성
2. `02_models_and_losses.ipynb` — RNN 계열, TCN, Transformer 및 loss 함수 정의
3. `03_training_and_optuna.ipynb` — 모델별 학습, validation, early stopping, checkpoint 저장
4. `04_ar_evaluation_and_figures.ipynb` — 200-step 자유 AR 예측, 지표 계산, 대표 샘플과 결과 그림 출력

`00_experiment_runner.ipynb`가 01~03의 중간 출력을 숨기고, 04의 표와 그래프는 그대로 표시합니다.

## 설정

모든 설정은 `00_experiment_runner.ipynb`의 단일 `config`에 있습니다. 다른 노트북에서 `config`를 새로 정의하거나 수정하지 마세요.

주요 설정은 다음과 같습니다.

- 데이터 및 예측 길이: `input_len`, `model_output_len`, `eval_output_len`, `stride`
- 변수: `state_cols`, `target_cols`, `report_col`
- 모델: `model_names`, RNN/Transformer/TCN 하이퍼파라미터
- 학습: `lr`, `epoch`, `batch_size`, `patience`
- 손실: `loss_type`
- 재현성: `seed`

기본 실행은 `BiLSTM`, `TCN`, `Transformer`를 `db` loss로 학습하며, PCA와 Optuna는 비활성화되어 있습니다.

## Optuna

- `run_optuna=True`: 새 탐색을 실행하고 최적 파라미터를 실험 폴더에 저장합니다.
- `run_optuna=False`, `use_optuna_best_params=True`: 저장된 최적 파라미터를 사용합니다. 파일이 없으면 명확한 오류가 발생합니다.
- 두 값이 모두 `False`: Optuna 파일을 읽지 않고 `config`의 기본값으로 학습합니다.

## 출력 경로

실험 이름(`experiment_name`)별로 출력이 분리됩니다.

- 결과: `results_ar_eval_200/<experiment_name>/`
- checkpoint: `loss_checkpoints/<experiment_name>/`

## ACF / FFT 탐색 노트북

`acf_fft.ipynb`는 데이터 파일별 ACF 및 FFT 분석만 남긴 탐색용 노트북입니다. 모델 학습과 평가는 포함하지 않습니다.
